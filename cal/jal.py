from jax.numpy import array,log,argsort,cumsum,flip,argmin,argmax
from jax.nn import relu
from jax.random import split,key,permutation,bits
from jax import grad,jit

from matplotlib.pyplot import plot,show,xscale,yscale,imshow,hist,\
                              title,xlabel,ylabel
from time import perf_counter
from flax.nnx import relu
from flax.linen import Module,Dense
from typing import Sequence
from optax import adam,apply_updates
from optax.losses import sigmoid_binary_cross_entropy
from jax.lax import scan
from jax.nn import sigmoid
from collections import namedtuple
from cal.rs import resamplers_list,Resampler

@jit
def shuffle_xy(k,x,y):
  shuff=permutation(k,len(y))
  xy=x[shuff],y[shuff]
  return xy

def get_reshuffled(k,X_trn,Y_trn,n_batches,bs,last):
  X,Y=shuffle_xy(k,X_trn,Y_trn)
  return X[0:last].reshape(n_batches,bs,-1),Y[0:last].reshape(n_batches,bs)

get_reshuffled=jit(get_reshuffled,static_argnames=['n_batches','bs','last'])

def shuffle_batched(k,X,Y,bs):
  l=len(Y)
  n_batches=l//bs
  return get_reshuffled(k,X,Y,n_batches,bs,bs*n_batches)

FPFNComplex=namedtuple('FPFNComplex',['cutoff','fp_rate','fn_rate'])

def fp_fn_complex(preds,targs):
  preds=preds.reshape(-1)
  targs=targs.reshape(-1)
  prediction_indices_by_likelihood=argsort(preds)

  targs=targs[prediction_indices_by_likelihood]
    
  d_err_test=(1/len(targs))
  
  fn_rates=cumsum(targs)*d_err_test
  fp_rates=flip(cumsum(flip(~targs)))*d_err_test
  return FPFNComplex(preds[prediction_indices_by_likelihood],fp_rates,fn_rates)

Results=namedtuple('Results',['fpfn_train','fpfn_test','cutoff','res_by_cost_rat'])

def optimal_cutoff(fpfn,cost_ratio):
  i=argmin(fpfn.fp_rate+cost_ratio*fpfn.fn_rate)
  return fpfn.fp_rate[i],fpfn.fn_rate[i],fpfn.cutoff[i]

def get_rates(fpfn,cutoff):
  i=argmax(fpfn.cutoff>cutoff)
  return fpfn.fp_rate[i],fpfn.fn_rate[i]

def results(preds_train,targs_train,preds_test,targs_test,cost_ratios):
  fpfn_train,fpfn_test=fp_fn_complex(preds_train,targs_train),fp_fn_complex(preds_test,targs_test)
  cutoff={}
  expected_cost_test={}
  for cr in cost_ratios:
    fpr,fnr,co=optimal_cutoff(fpfn_train,cr)
    cutoff[cr]=co
    fpr_test,fnr_test=get_rates(fpfn_test,co)
    expected_cost_test[cr]=fpr,fnr,fpr_test,fnr_test,(fpr_test+cr*fnr_test)/(cr**.5)
  return Results(fpfn_train,fpfn_test,cutoff,expected_cost_test)

class NN(Module):
  features:Sequence[int]
  
  def setup(self):
    self.layers=[Dense(f) for f in self.features]
    
  def __call__(self,x):
    x=self.layers[0](x)
    for l in self.layers[1:]:
      x=l(relu(x)) # in general no restruction on float output of NN - treat as log relative likelihood
    return x

  
class NNPL:
  #Saving and loading resampled data seems to be causing problems not sure why, bit annoying!
  def __init__(self,x_train,y_train,x_test,y_test,cost_rats,ds_name,plot_title,
               loss=sigmoid_binary_cross_entropy,x_dt=None,y_dt=None,
               rs_dir=None,bs=128,lr=1e-4,features=[128,64,32,1],seed=0,lo=print,
               n_epochs=100,loss_param=None):#[256,128,64,1]
    self.x_train=array(x_train,dtype=x_dt)
    if y_dt is None:
      y_dt=bool #self.x_train.dtype
    self.y_train=array(y_train,dtype=y_dt)
    self.x_test=array(x_test,dtype=x_dt)
    self.y_test=array(y_test,dtype=y_dt)
    self.p=self.y_train.mean()
    self.p_test=self.y_test.mean()
    self.bs=bs
    self.n_epochs=n_epochs
    self.lr=lr
    self.loss=loss
    
    self.key=key(seed)
    self.m=NN(features=features)
    
    self.init_param=self.m.init(self.getk(),self.x_train[0])
    
    self.t=adam(learning_rate=self.lr)
    
    self.init_state=self.t.init(self.init_param)
    
    self.cost_rats=cost_rats
    
    self.state={}
    self.param={}
    self.log=lo
    self.ds_name=ds_name
    self.rs_dir=rs_dir
    self.pred_train={}
    self.pred_test={}

    self.epochs_time={}
    self.update_rules={}
    self.res={}
    
    self.plot_title=plot_title
    
    self.rs=Resampler(self.x_train,self.y_train,self.rs_dir,self.ds_name,int(bits(self.getk())))

    
  def updates(self,x_batched,y_batched,rs='',loss_param=None):
    if not loss_param in self.update_rules:
      if loss_param is None:
        loss=self.loss
      else:
        loss=self.loss(loss_param)
      dl=grad(lambda par,feat,targ:loss(self.m.apply(par,feat),targ).sum())

      def update(state_param,x_y):
        state,param=state_param
        x,y=x_y
        g=dl(param,x,y)
        upd,state=self.t.update(g,state)
        param=apply_updates(param,upd)
        return (state,param),0
      self.update_rules[loss_param]=jit(lambda state_param,x_b,y_b:scan(update,state_param,(x_b,y_b))[0])
      
    update_rule=self.update_rules[loss_param]
    y_batched=y_batched.astype(x_batched.dtype)
    self.state[rs,loss_param],self.param[rs,loss_param]=update_rule((self.state[rs,loss_param],
                                                                     self.param[rs,loss_param]),
                                                                    x_batched,y_batched)

  def mk_preds(self,rs='',loss_param=None):
    self.pred_train[rs,loss_param]=self.predict_cts(self.x_train,rs=rs,loss_param=loss_param).reshape(-1)
    self.pred_test[rs,loss_param]=self.predict_cts(self.x_test,rs=rs,loss_param=loss_param).reshape(-1)
    
  def fit(self,rs='',loss_param=None):
    if not rs in self.state:
      self.state[rs,loss_param]=self.init_state
      self.param[rs,loss_param]=self.init_param
    self.epochs(rs=rs,loss_param=loss_param)
    self.results(rs=rs,loss_param=loss_param)

  def epochs(self,rs='',n=None,loss_param=None):
    x,y=self.rs.get_resampled(True,rs)
    self.log('Resampling took',self.rs.get_t(True,rs),'seconds')
    
    if n is None:
      n=self.n_epochs
    t0=perf_counter()
    for e in range(n):
      self.log('Running epoch',e+1,'of',n,'...',end='\r')
      x_batched,y_batched=shuffle_batched(self.getk(),x,y,self.bs)
      self.updates(x_batched,y_batched,rs,loss_param=loss_param)
    self.epochs_time[rs,loss_param]=perf_counter()-t0
    self.log('Completed',n,'epochs in',self.epochs_time[rs,loss_param],'seconds')
    self.log('Getting fp-fn characteristic')
    self.mk_preds(rs=rs,loss_param=loss_param)
    #assert self.pred_train[rs,loss_param].min()<self.pred_train[rs,loss_param].max(),\
    #       'Uh oh:'+str(self.pred_train[rs,loss_param].min())+'=='+str(self.pred_train[rs,loss_param].max())

  def predict_cts(self,x,rs='',loss_param=None):
    if not (rs,loss_param) in self.param:
      self.param[rs,loss_param]=self.init_param
    return self.m.apply(self.param[rs,loss_param],x)
  
  def getk(self):
    self.key,k=split(self.key)
    return k

  def predict_bin(self,x,cost_ratio,rs='',loss_param=None): # NN output is relative likelihood so take log
    pred=self.predict_cts(x,rs=rs,loss_param=loss_param)
    if cutoff_rule=='bayes':
      cutoff=-log(cost_ratio)#inverse_sigmoid(cost_ratio)
    elif cutoff_rule=='optimal':
      cutoff=self.get_optimal_cutoff(cost_ratio)
      
    return pred>cutoff
  
  def results(self,rs='',loss_param=None):
    preds_train=self.predict_cts(self.x_train,rs=rs,loss_param=loss_param)
    preds_test=self.predict_cts(self.x_test,rs=rs,loss_param=loss_param)
    self.res[rs,loss_param]=r=results(preds_train,self.y_train,preds_test,self.y_test,self.cost_rats)
    return r
  
  def report(self,rs=None,loss_param=None,compare_cost_ratio=None,header=True,topk=None,
             rs_vs_raw=False,plot_res=False):
    lj=lambda x:str(x).ljust(13)
    if header:
      self.log(lj('cost_ratio')+lj('fp_train')+lj('fn_train')+lj('fp_test')+\
               lj('fn_test')+lj('E(cost|test)')+lj('resampler')+lj('loss param'))

    if compare_cost_ratio is None:
      compare_cost_ratio=list(self.cost_rats)
    if rs is None:
      rs=list({r for r,p in self.res})
    if loss_param is None:
      loss_param=list({p for r,p in self.res})

    if not isinstance(compare_cost_ratio,list):
      compare_cost_ratio=[compare_cost_ratio]
    if not isinstance(rs,list):
      rs=[rs]
    if not isinstance(loss_param,list):
      loss_param=[loss_param]

    res_all=[]
    for c in compare_cost_ratio:
      rep_raw=[]
      rep_rs=[]
      for r in rs:
        for p in loss_param:
          try:
            (fp,fn,fp_test,fn_test,cst)=self.res[r,p].res_by_cost_rat[c]
          except KeyError:
            self.log('resampler and loss param',r,p,'not found')
            continue
          s=lj(str(c))
          s+=lj(fp)+lj(fn)
          s+=lj(fp_test)+lj(fn_test)
          s+=lj(cst)
          s+=lj(r)
          s+=lj(str(p))
          (rep_rs if r else rep_raw).append((cst,s,(r,p)))
      if rs_vs_raw:
        print()
        print('===================== cost(fn)/cost(fp)=',c,'=====================')
        rep=sorted(sorted(rep_raw)[:topk]+sorted(rep_rs)[:topk])
      else:
        rep=sorted(rep_raw+rep_rs)[:topk]
      [self.log(s[1]) for s in rep]
      res_all+=rep
    if plot_res:
      [self.plot(rs=r,loss_param=p) for r,p in [x[2] for x in res_all]]
  
  def plot(self,rs='',loss_param=None):
    show()
    r=self.res[rs,loss_param]
    plot(r.fpfn_train.fp_rate,r.fpfn_train.fn_rate)
    plot([0,1-self.p],[self.p,0])
    title(self.plot_title+' (resampler:'+(rs if rs else 'None')+') (loss params:'+str(loss_param)+')')
    xlabel('False positives')
    ylabel('False negatives')
    show()

    hist(r.fpfn_train.cutoff,label='Raw NN outputs',bins=100)
    show()

def make_fp_perturbed_bce(beta):
  if beta:
    def beta_loss(pred,targ):
      return -targ*log(sigmoid(pred))+(1-targ)*sigmoid(pred*beta)/beta

    return beta_loss
  else:
    return sigmoid_binary_cross_entropy

def make_fp_fn_perturbed_bce(ab):
  a,b=ab
  l_pos=(lambda r: -sigmoid(r*a)/a) if a else (lambda r:-log(sigmoid(r))) #loss if y=+
  l_neg=(lambda r: -sigmoid(-r*b)/b) if b else (lambda r:-log(sigmoid(-r))) #lloss if y=-
  def pet_bce(pred,targ):
    return targ*l_pos(pred)+(1-targ)*l_neg(pred)
  return pet_bce
