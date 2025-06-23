from itertools import count
from numpy import geomspace
from jax.numpy import array,zeros,array,exp,ones,eye,sum as jsm,maximum
from jax.nn import tanh,softmax,relu
from jax.random import uniform,normal,split,key,choice,permutation
from jax.tree import map as jma
from jax import grad,jit,vmap
from jax.lax import scan,while_loop
from jax.lax.linalg import svd
from jax.scipy.special import xlogy
from cal.mt import MultiTrainer

from matplotlib.pyplot import plot,show,xscale,yscale,imshow,legend,colorbar,hist,\
                              subplots,scatter,title,xlabel,ylabel
from jax.random import split,key,bits
from time import perf_counter
from collections import namedtuple
from cal.rs import resamplers_list,Resampler
from flax.nnx import relu
from flax.linen import Module,Dense
from typing import Sequence
from optax import adam,apply_updates
from optax.losses import sigmoid_binary_cross_entropy
from jax import grad,jit
from jax.numpy import array,log,argsort,cumsum,flip,argmin,argmax
from jax.lax import scan
from jax.tree import structure
from jax.nn import sigmoid
from jax.random import split,key,bits
from time import perf_counter
from collections import namedtuple
from cal.rs import resamplers_list,Resampler

class KeyEmitter:
  def __init__(self,seed,parents=['main','vis']):
    if isinstance(parents,dict):
      self.parents=parents
    else:
      k=key(seed)
      if isinstance(parents,list):
        self.parents={a:k for a,k in zip(parents,split(k,len(parents)))}
      elif isinstance(parents,str):
        self.parents={parents:k}
  def emit_key(self,n=None,parent='main',vis=False,report=False):
    if vis or report:
      parent='vis'
    keys=split(self.parents[parent]) if n is None else split(self.parents[parent],n+1)
    self.parents[parent]=keys[0]
    return keys[1] if n is None else keys[1:]

def _forward(w,x,act,layer_norm=False):
  l=w[:-1]
  abl=w[-1]
  for ab in l:
    z=x@ab[0]
    if layer_norm:
      z=ab[2]*((z-z.mean(axis=-1))/(1e-8+z.var(axis=-1)**.5))
    x=act(z+ab[1])
  y=x@abl[0]+abl[1]
  return y

forward=jit(_forward,static_argnames=['act','layer_norm'])
vorward=jit(vmap(_forward,(0,None,None),0),static_argnames=['act'])

activations={'tanh':tanh,'softmax':softmax,'linear':jit(lambda x:x),
             'relul':jit(lambda x:maximum(x,.03*x)),'relu':relu}

def select_initialisation(imp,act):
  if imp in ['fm','resnet']:
    return 'resnet'
  else:
    match act:
      case 'relu':
        return 'glorot_normal'
      case 'tanh':
        return 'glorot_uniform'
      case 'linear':
        return 'ones'
      case _:
        print('Initialising weights to glorot uniform!')
        return 'glorot_uniform'

def init_layers(layer_dimensions,initialisation,k=None,mult_a=1.,n_par=None,bias=0.,
                sqrt_normalise=False,orthonormalise=False,transpose=True,layer_norm=False):
  n_steps=len(layer_dimensions)-1
  if initialisation in ['ones','zeros']:
    w_k,b_k=count(),count()
  else:
    k1,k2=split(k)
    w_k=split(k1,n_steps)
    b_k=split(k2,n_steps)
  wb=[]
  A=[]
  B=[]
  G=[]
  if n_par is None:
    e_dims=tuple()
  else:
    e_dims=(n_par,)
  for i,(k,l,d_i,d_o) in enumerate(zip(w_k,b_k,layer_dimensions,layer_dimensions[1:])):
    match initialisation:
      case 'zeros':
        A.append(zeros(e_dims+(d_i,d_o)))
        B.append(zeros(e_dims+(d_o,))+bias)
      case 'resnet'|'ones':
        A.append(ones(e_dims+(d_i,d_o))*mult_a)
        B.append(zeros(e_dims+(d_o,))+bias)
        initialisation='zeros' if initialisation=='resnet' else 'ones'
      case 'eye':
        if n>1: raise Exception('Not yet done this')
        A.append(eye(d_i,d_o)*mult_a)
        B.append(zeros(d_o)+bias)
      case 'glorot_uniform':
        A.append((2*(6/(d_i+d_o))**.5)*(uniform(shape=e_dims+(d_i,d_o),key=k)-.5)*mult_a)
        B.append(zeros(e_dims+(d_o,))+bias)
      case 'glorot_normal':
        A.append(((2/(d_i+d_o))**.5)*normal(shape=e_dims+(d_i,d_o),key=k)*mult_a)
        B.append(zeros(e_dims+(d_o,))+bias)
      case _:
        raise Exception('Unknown initialisation: '+initialisation)
    if layer_norm:
      G.append(ones(e_dims+(d_o,)))
  if orthonormalise:
    for i,(d_i,d_o) in enumerate(layer_dimensions,layer_dimensions[1:]):
      if n>1: raise Exception('Not yet done this')
      if d_i>d_o:
        A[i]=svd(A[i],full_matrices=False)[0]
      else:
        A[i]=svd(A[i],full_matrices=False)[2]
  if sqrt_normalise:
    if n>1: raise Exception('Not yet done this')
    for i,d_i in enumerate(layer_dimensions[1:]):
      A[i]/=d_i**.5
  AB=[A,B]
  if layer_norm:
    AB.append(G)
  if transpose:
    return list(list(ab) for ab in zip(*AB))
  else:
    return AB

def init_layers_adam(*a,**kwa):
  w=init_layers(*a,**kwa)
  kwa['bias']=kwa.pop('bias',0)
  return {'w':w,'m':init_layers(a[0],'zeros',**kwa),
          'v':init_layers(a[0],'zeros',**kwa),'t':zeros(kwa['n_par']) if 'n_par' in kwa else 0}

@jit
def shuffle_xy(k,x,y):
  shuff=permutation(k,len(y))
  xy=x[shuff],y[shuff]
  return xy
 
def ewma(a,b,rate):return (1-rate)*a+rate*b

def ad_diff(m,v,eps):return m/(v**.5+eps)

def _upd_adam_no_bias(dw,w,m,v,t,beta1,beta2,lr,reg,eps):
  m=jma(lambda old,upd:ewma(upd,old,beta1),m,dw)
  v=jma(lambda old,upd:ewma(upd**2,old,beta2),v,dw)
  t+=1
  w=jma(lambda W,M,V:(1-reg)*(W-lr*ad_diff(M/(1-beta1**t),V/(1-beta2**t),eps)),w,m,v)
  return (w,m,v,t)
upd_adam_no_bias=_upd_adam_no_bias

def dict_adam_no_bias(dw,state,consts):
  state['w'],state['m'],state['v'],state['t']=upd_adam_no_bias(dw,state['w'],state['m'],state['v'],
                                                               state['t'],consts['beta1'],consts['beta2'],
                                                               consts['lr'],consts['reg'],consts['eps'])
  return state

def binsearch_step(r,tfpfn,y,yp_smooth):
  a,b=r
  mid=(a+b)/2
  yp=yp_smooth-mid>0
  fpfn=((yp&~y).mean()/(y&~yp).mean())
  excess_fps=fpfn>tfpfn
  return a*(~excess_fps)+excess_fps*mid,b*(excess_fps)+(~excess_fps)*mid,(fpfn-tfpfn)**2

@jit
def search_cutoff(y,yp_smooth,tfpfn,adapcutoff_tol):
  yp_min_max=yp_smooth.min(),yp_smooth.max()
  cutoff_range=while_loop(lambda r:(r[2]>adapcutoff_tol)&(r[3]<1/adapcutoff_tol),
                          lambda r:binsearch_step(r[:2],tfpfn,y,yp_smooth)+(r[3]+1,),
                          (*yp_min_max,1,0))[:2]
  return (cutoff_range[0]+cutoff_range[1])/2

def _set_fp_fn_weights(consts,fp,fn):
  return exp(consts['lrfpfn']*(-fp/consts['tfp']+fn/consts['tfn']))
set_fp_fn_weights=jit(vmap(_set_fp_fn_weights,(0,0,0),0))#,None

def get_reshuffled(k,X_trn,Y_trn,n_batches,bs,last):
  X,Y=shuffle_xy(k,X_trn,Y_trn)
  return X[0:last].reshape(n_batches,bs,-1),Y[0:last].reshape(n_batches,bs)

get_reshuffled=jit(get_reshuffled,static_argnames=['n_batches','bs','last'])

def shuffle_batched(k,X,Y,bs):
  l=len(Y)
  n_batches=l//bs
  return get_reshuffled(k,X,Y,n_batches,bs,bs*n_batches)

def loss(w,x,y,fp_fn_weights,act):
  yp=forward(w,x,act)
  return jsm((fp_fn_weights*y+(1-y)/fp_fn_weights)*(1-2*y)*yp)

dl=grad(loss)
def _upd(x,y,state,consts,act):
  return dict_adam_no_bias(dl(state['w'],x,y,state['fp_fn_weights'],act),state,consts)

upd=vmap(_upd,(None,None,0,0,None),0)

def _steps(states,consts,X,Y,act):
  return scan(lambda s,xy:(upd(xy[0],xy[1],s,consts,act),0),states,(X,Y))[0]
steps=jit(_steps,static_argnames=['act'])

def _y_smooth_fp_fn(w,X,Y,act):
  yps=forward(w,X,act)
  Yp_smooth=yps.flatten()
  Yp=Yp_smooth>0.
  Ypb=Yp_smooth>0
  fp,fn=(Ypb&~Y).mean(),(Y&~Ypb).mean()
  return Yp_smooth,fp,fn

y_smooth_fp_fn=jit(_y_smooth_fp_fn,static_argnames=['act'])

calc_fp_fn=lambda w,X,Y,act:y_smooth_fp_fn(w,X,Y,act)[1:]

def _calc_cutoff(w,tfpfn,X,Y,act,tol=1e-1):
  Yp_smooth,fp,fn=y_smooth_fp_fn(w,X,Y,act)
  return search_cutoff(Y,Yp_smooth,tfpfn,tol),fp,fn

calc_cutoff=jit(vmap(_calc_cutoff,(0,0,None,None,None),0),static_argnames=['act','tol'])

def nn_epochs(k,n_epochs,bs,X,Y,consts,states,X_raw=None,Y_raw=None,act='relu',
              adap_cutoff=True,logf=None,layer_norm=False,start_epoch=1):
  if X_raw is None:
    X_raw,Y_raw=X,Y
  if isinstance(act,str):
    act=activations[act]
  n_batches=len(X)//bs
  last=n_batches*bs
  ks=split(k,n_epochs)
  adapt_weights=consts['lrfpfn'].sum()
  if adapt_weights:
    print('Variable weights!!!!!!',file=logf,flush=True)

  tfps=consts['tfp']
  tfns=consts['tfn']
  lrfpfns=consts['lrfpfn']
  lrs=consts['lr']
  regs=consts['reg']
  tfpfns=tfps/tfns
  for epoch,k in enumerate(ks,start_epoch):
    X_b,Y_b=get_reshuffled(k,X,Y,n_batches,bs,last)
    states=steps(states,consts,X_b,Y_b,act)
    if adap_cutoff:
      cutoff,fp,fn=calc_cutoff(states['w'],tfpfns,X_raw,Y_raw,act)
      states['w'][-1][1]-=cutoff.reshape(-1,1)
    else:
      fp,fn=calc_fp_fn(states['w'],X_raw,Y_raw,act)
    if adapt_weights:
      states['fp_fn_weights']*=set_fp_fn_weights(consts,fp,fn)
    print('nn epoch',epoch,file=logf,flush=True)
  return states
  return _nn_epochs(ks,n_epochs,bs,X,Y,X_raw,Y_raw,consts,states,act,n_batches,last,
                    logf=logf,layer_norm=layer_norm,start_epoch=start_epoch,adapt_weights=adw)

def vectorise_fl_ls(x,l):
  if isinstance(x,list|tuple):
    return array(x)
  elif isinstance(x,float):
    return ones(l)*x 
  return x

class NNPar(MultiTrainer):
  def __init__(self,seed,p,tfp,tfn,p_resampled=None,lrfpfn=False,reg=1e-6,inner_dims=None,
               in_dim=None,fp_fn_weights=None,start_width=None,end_width=None,depth=None,bs=128,
               act='relu',init='glorot_normal',times=100,lr=1e-4,beta1=.9,beta2=.999,
               eps=1e-8,bias=.1,acc=.1,min_tfpfn=1,max_tfpfn=100,logf=None,n_par=None,
               adap_cutoff=True,layer_norm=False):
    if isinstance(lrfpfn,bool):
      lrfpfn=.003 if lrfpfn else 0.
    super().__init__(times=times,cutoff=0.)
    self.logf=logf
    self.p=p
    self.p_resampled=p if p_resampled is None else p_resampled
    if fp_fn_weights is None:
      fp_fn_weights=1.#=((1-self.p_resampled)/self.p_resampled)**.5
    self.inner_dims=list(geomspace(start_width,end_width,depth+1,dtype=int)) if\
                         inner_dims is None else inner_dims
    self.ke=KeyEmitter(seed)

    self.init=init
    self.bias=bias
    #consts
    self.lrfpfn=lrfpfn
    self.reg=reg
    self.lr=lr
    self.beta1=beta1
    self.beta2=beta2
    self.eps=eps
    self.tfp=tfp
    self.tfn=tfn
    self.bs=bs
    self.act=act
    self.adap_cutoff=adap_cutoff
    self.layer_norm=layer_norm

    #states
    self.states=None
    self.consts=None
    self.n_par=n_par
    self.fp_fn_weights=vectorise_fl_ls(fp_fn_weights,n_par)
  
  def get_states(self):
    return self.states

  def set_consts_states(self,in_dim):
    self.in_dim=in_dim
    self.mod_shape=[self.in_dim]+self.inner_dims+[1]
    if self.consts is None:
      consts=dict(lrfpfn=self.lrfpfn,tfp=self.tfp*(1-self.p_resampled)/(1-self.p),
                  tfn=self.tfn*self.p_resampled/self.p,beta1=self.beta1,beta2=self.beta2,
                  lr=self.lr,reg=self.reg,eps=self.eps)
      if self.n_par is None:
        self.n_par=max((len(x) if hasattr(x,'__len__') else 1 for x in consts.values()))
      self.consts={k:vectorise_fl_ls(v,self.n_par) for k,v in consts.items()}
    if self.states is None:
      self.states=dict(fp_fn_weights=self.fp_fn_weights,
                       **init_layers_adam(self.mod_shape,self.init,k=self.ke.emit_key(),
                                          n_par=self.n_par,bias=self.bias))

  def fit(self,X,Y,X_raw=None,Y_raw=None):
    self.set_consts_states(len(X[0]))
    nl=0
    for n in sorted(self.times):
      self.states=nn_epochs(self.ke.emit_key(),n-nl,self.bs,X,Y,self.consts,
                            self.states,X_raw=X_raw,start_epoch=nl+1,
                            Y_raw=Y_raw,act=self.act,logf=self.logf,
                            adap_cutoff=self.adap_cutoff)
      print('Saving state at time',n,file=self.logf,flush=True)
      self.past_states[n]=self.states
      nl=n

  def predict_smooth(self,X):
    Yp={t:vorward(self.past_states[t]['w'],X,activations[self.act]) for t in self.times}
    return {t:y.reshape(y.shape[:-1]) for t,y in Yp.items()}

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
               rs_dir=None,bs=128,lr=1e-4,features=[128,64,32,1],seed=0,log=print,
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
    self.log=log
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
  
  def report(self,rs='',loss_param=None):
    r=self.res[rs,loss_param]
    just=14
    lj=lambda x:str(x).ljust(just)
    self.log(lj('cost_ratio')+lj('fp_train')+lj('fn_train')+\
             lj('fp_test')+lj('fn_test')+lj('E_test(cost)'))
    for cost_ratio,(fp,fn,fp_test,fn_test,c) in r.res_by_cost_rat.items():
      s=lj(str(cost_ratio))
      s+=lj(fp)+lj(fn)
      s+=lj(fp_test)+lj(fn_test)
      s+=lj(c)
      print(s)
  
  def plot(self,rs='',loss_param=None):
    show()
    r=self.res[rs,loss_param]
    plot(r.fpfn_train.fp_rate,r.fpfn_train.fn_rate)
    plot([0,1-self.p],[self.p,0])
    title(self.plot_title+' '+rs+' '+str(loss_param))
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
