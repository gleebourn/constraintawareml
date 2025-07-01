from itertools import product
from time import perf_counter
from typing import NamedTuple,Sequence,Callable

from numpy import array as nparr,frompyfunc,log2
from numpy.typing import NDArray

from jax import grad,jit
from jax.numpy import array,log,argsort,cumsum,flip,argmin,argmax,diff
from jax.lax import scan
from jax.nn import sigmoid,relu
from jax.random import split,key,permutation,bits

from flax.nnx import relu
from flax.linen import Module,Dense
from optax import adam,apply_updates
from optax.losses import sigmoid_binary_cross_entropy

from matplotlib.pyplot import plot,show,xscale,yscale,imshow,hist,\
                              title,xlabel,ylabel

from cal.rs import resamplers_list,Resampler

def shuffle_and_batch(k,x,y,bs):
  l=len(y)
  shuff=permutation(k,len(y))
  x,y=x[shuff],y[shuff]
  n_batches=l//bs
  last=n_batches*bs
  return x[0:last].reshape(n_batches,bs,-1),y[0:last].reshape(n_batches,bs)

shuffle_and_batch=jit(shuffle_and_batch,static_argnames='bs')

type Cutoff=float
type CostRatio=float
type CostVal=float
type Likelihood=float

intupstr=lambda t:'('+(','.join((str(e) for e in t)))+')'

def fmt_list_trunc(l,trunc):
  return '\n'+('\n'.join((l if len(l)<= trunc else l[:(trunc+1)//2]+'...'+l[-trunc//2:]) if trunc else l))+'\n'
  #return '('+(','.join((l if len(l)<= trunc else l[:(trunc+1)//2]+'...'+l[-trunc//2:]) if trunc else l))+')'

def format_typey_list(arrays,labels,trunc,lambdas=None):
  if lambdas is None:
    lambdas=[str for _ in arrays]
  entries=[intupstr([lam(l) for lam,l in zip(lambdas,ll)]) for ll in zip(*arrays)]
  return '\n'+intupstr(labels)+':'+fmt_list_trunc(entries,trunc)

lj=lambda s:str(s).ljust(15)

class Errors:
  def __init__(self,*errs):
    assert len(errs)==len(self.error_types),'Error arrays not in correspondence with error types!'
    self.e=array(errs)

  @property
  def error_types(self): raise NotImplementedError('Need to specify error types')

  def str(self,trunc=None):
    return format_typey_list(self.e,self.error_types,trunc)

  def __str__(self):
    return self.str(trunc=10)

  def __repr__(self):
    return '\n=='+self.__name__+'=='+self.__str__()

  def __getitem__(self,key):
    return type(self)(*self.e[:,key])

  def __len__(self):
    return self.e.shape[1]

class BinaryErrors(Errors):
  error_types=('fp','fn')

  @property
  def fp(self):
    return self.e[0]

  @property
  def fn(self):
    return self.e[1]

class ErrorFiltration(NamedTuple):
  cutoff:NDArray[float]
  e:BinaryErrors#NDArray[ClassifierError]

  def subfiltration(self,cost_rats:[CostRatio]=None,cutoffs:[Cutoff]=None):
    assert not(cost_rats==cutoffs==None), 'Need to specify how subfiltration is obtained'
    if cost_rats: #find best cutoff for given ratio
      inds=nparr([self.get_cutoff_index(c) for c in cost_rats])
    else: #find indices of given cutoff
      inds=nparr([argmax(self.cutoff>c) for c in cutoffs])
    return ErrorFiltration(self.cutoff[inds],self.e[inds])

  def get_cutoff_index(self,cost_rat:CostRatio)->int:
    return argmin(self.e.fp+cost_rat*self.e.fn)
    
  def results(self,cost_rats:NDArray[CostRatio],
              cutoffs:NDArray[Cutoff]=None)->tuple:
    if cutoffs is None:
      if len(self.fp)>len(cost_rats):
        return self.subfiltration(cost_rats=cost_rats).results(cost_rats=cost_rats)
      return self,self.costs(cost_rats)
    else:
      return self.subfiltration(cutoffs=cutoffs).results(cost_rats=cost_rats)

  def costs(self,cost_rats:[CostRatio])->[CostVal]:
    cost_rats=nparr(cost_rats)
    assert len(cost_rats)==len(self.e)
    
    return (self.e.fn*cost_rats+self.e.fp)/(cost_rats**.5)

  def str(self,trunc=None):
    return format_typey_list((self.cutoff,self.e),('class cutoffs','error rates'),trunc,lambdas=(lj,str))

  def __str__(self):
    return self.str(trunc=10)

  def __repr__(self):
    return '\n==ErrorFiltration=='+self.__str__()
    #return 'ErrorFiltration'+self.__str__()

  @classmethod
  def from_predictions(self,preds:NDArray[Likelihood],targs:nparr):
    preds=preds.reshape(-1)
    targs=targs.reshape(-1)
    sort_by_preds=argsort(preds)
  
    targs=targs[sort_by_preds]
    error_increment=(1/len(targs))
    
    fn_rates=cumsum(targs)*error_increment
    fp_rates=flip(cumsum(flip(~targs)))*error_increment
    return self(cutoff=preds[sort_by_preds],e=BinaryErrors(fp_rates,fn_rates))#,dtype=ClassifierError))

class TT(NamedTuple):
  train:float
  test:float

class RatRes(NamedTuple):
  rat:CostRatio
  res:TT

  @classmethod
  def n(self,a,b,c):
    return self(a,TT(b,c))

class Results(NamedTuple):
  cost_rats:NDArray[CostRatio]
  costs_train:NDArray[float]
  costs_test:NDArray[float]
  cutoffs:NDArray[Cutoff]

  def __iter__(self):
    return (RatRes.n(cr,tr,te) for cr,tr,te in zip(self.cost_rats,self.costs_train,self.costs_test))

  @classmethod
  def from_predictions(self,preds_train,targs_train,preds_test,targs_test,cost_rats,n_epochs=0):
    f_train,f_test=ErrorFiltration.from_predictions(preds_train,targs_train),ErrorFiltration.from_predictions(preds_test,targs_test)
    #f_train,costs_train=f_train.results(cost_rats=cost_rats) #use the training set to select cutoffs
    #f_test,costs_test=f_test.results(cost_rats=cost_rats,cutoffs=cutoffs) #apply cutoffs to test set
    f_train=f_train.subfiltration(cost_rats=cost_rats)
    f_test=f_test.subfiltration(cutoffs=f_train.cutoff)
    return self(cost_rats,f_train.costs(cost_rats),f_test.costs(cost_rats),f_train.cutoff)

  def str(self,trunc=None):
    return format_typey_list((self.cost_rats,self.costs_train,self.costs_test,self.cutoffs),
                             ('cost ratio','E(cost|train)','E(cost|test)','class cutoff'),trunc,lambdas=(lj,lj,lj,lj))
      
  def __str__(self):
    return self.str(trunc=10)
      
  def __repr__(self):
    return '\n==Results=='+self.__str__()
    #return 'Results'+self.__str__()
    
class NNState(NamedTuple):
  time:int #in epochs in easy case
  state:object #adam etc hyperparams
  param:object #nn weights

class UpdateRule(NamedTuple):
  lr:float
  bs:int
  loss_par:tuple
  
  def __str__(self):
    return '(lr:'+str(self.lr)+' bs:'+str(self.bs)+' loss_par:'+str(self.loss_par)+')'

  def __repr__(self):
    return 'UpdateRule'+self.__str__()

class UpdateRuleImplementation:
  def __init__(self,loss:Callable,forward:Callable,rules:dict={},log:Callable=print):
    self.loss=loss
    self.forward=forward
    self.rules=rules
    self.log=log

  def epochs(self,ur,nns,x,y,n_epochs,k):
    if not ur in self.rules:
      if ur.loss_par is None:
        dl=grad(lambda param,x,y:self.loss(self.forward(param,x).reshape(-1),y,).sum())
      else:
        dl=grad(lambda param,x,y:self.loss(self.forward(param,x).reshape(-1),y,ur.loss_par).sum())
      ad=adam(learning_rate=ur.lr).update
      def step(state_param,x_y):
        x,y=x_y
        s,p=state_param
        g=dl(p,x,y)
        upd,state=ad(g,s)
        param=apply_updates(p,upd)
        return (state,param),None
        
      def steps(nns:NNState,x_b,y_b):
        s,p=scan(step,(nns.state,nns.param),(x_b,y_b))[0]
        return NNState(state=s,param=p,time=nns.time+1)

      steps=jit(steps)
    
      def _epochs(nns,x,y,n_epochs,k):
        t0=perf_counter()
        for e,l in enumerate(split(k,n_epochs),1):
          self.log('Running epoch',e,'of',n_epochs,'...',end='\r')
          x_batched,y_batched=shuffle_and_batch(l,x,y,ur.bs)
          nns=steps(nns,x_batched,y_batched)
        t=perf_counter()-t0
        self.log('Completed',n_epochs,'epochs in',t,'seconds')
        return nns
      self.rules[ur]=_epochs#jit(_epochs,static_argnames='n_epochs')
    return self.rules[ur](nns,x,y,n_epochs,k)

class TrainingCheckpoint(NamedTuple):
  n_epochs:int
  ur:UpdateRule
  rs:str
  def __str__(self):
    return '(n_epochs:'+str(self.n_epochs)+' ur:'+str(self.ur)+\
           ' rs:'+(str(self.rs) if self.rs else 'None')+')'

  def __repr__(self):
    return 'TrainingCheckpoint'+self.__str__()

class Outcome(NamedTuple):
  checkpoint:TrainingCheckpoint
  res:TT

  def __lt__(self,other):
    return self.res.test<other.res.test

  def __str__(self):
    return '('+str(self.checkpoint)+'~>'+str(self.res)+')'

  def __repr__(self):
    return 'Outcome'+self.__str__()

class TrainingRule(NamedTuple):
  n_epochs:tuple
  ur:UpdateRule
  rs:str
  
  def train(self,uri:UpdateRuleImplementation,nns:NNState,ds,k):
    x,y=ds.get_resampled(True,self.rs)
    snapshots=[]
    last=0
    for l,current_epoch,next_checkpoint in zip(split(k,len(self.n_epochs)),(0,)+self.n_epochs,self.n_epochs):
      nns=uri.epochs(self.ur,nns,x,y,next_checkpoint-current_epoch,l)
      snapshots.append(nns)
    return snapshots
  def __str__(self):
    return '(n_epochs:'+intupstr(self.n_epochs)+' ur:'+str(self.ur)+\
           ' rs:'+(str(self.rs) if self.rs else 'None')+')'

  def __repr__(self):
    return 'TrainingRule'+self.__str__()

  def to_checkpoints(self):
    return [TrainingCheckpoint(e,self.ur,self.rs) for e in self.n_epochs]
      
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
  def __init__(self,x_train,y_train,x_test,y_test,cost_rats,ds_name,plot_title,
               n_epochs=(1,2,4,8,16,32,64),bs=128,lr=1e-3,rs=[''],loss_param=[None],
               loss=sigmoid_binary_cross_entropy,x_dt=None,y_dt=None,resampler=None,
               rs_dir=None,features=(128,64,32,1),seed=1729,lg=print):#[256,128,64,1]
    self.x_train=array(x_train,dtype=x_dt)
    if y_dt is None:
      y_dt=bool #self.x_train.dtype
    self.y_train=array(y_train,dtype=y_dt)
    self.x_test=array(x_test,dtype=x_dt)
    self.y_test=array(y_test,dtype=y_dt)
    
    self.p=self.y_train.mean()
    self.p_test=self.y_test.mean()
    
    self.nn=NN(features=features)

    self.set_parametric_loss(loss)
    self.update_rules=set()
    self.training_rules=set()
    if lr:
      self.add_training_rules(lr,bs,loss_param,rs,n_epochs)
    self.results={}
    
    self.key=key(seed)

    self.cost_rats=cost_rats
    
    self.log=lg
    self.ds_name=ds_name
    self.rs_dir=rs_dir

    self.trained={}
    self.res={}
    
    self.plot_title=plot_title
    
    self.rs=Resampler(self.x_train,self.y_train,self.rs_dir,self.ds_name,int(bits(self.getk()))) if\
            resampler is None else resampler

  def add_training_rules(self,lr=False,bs=False,loss_param=False,rs=False,n_epochs=False):
    if isinstance(lr,float):lr=[lr]
    if isinstance(bs,int):bs=[bs]
    if isinstance(rs,str):rs=[rs]
    if isinstance(n_epochs,int):n_epochs=(2**i for i in range(int(1+log2(n_epochs))))
    if loss_param is None or isinstance(loss_param,tuple):loss_param=[loss_param]# want to allow passing None for loss function with no params
    if lr:self.lr=lr
    if bs:self.bs=bs
    if loss_param:self.loss_param=loss_param
    if rs:self.rs=rs
    if n_epochs:self.n_epochs=n_epochs
    self.update_rules.update({UpdateRule(l,b,p) for l,b,p in product(self.lr,self.bs,self.loss_param)})
    self.training_rules.update({TrainingRule(ur=ur,rs=r,n_epochs=n_epochs) for r,ur in product(self.rs,self.update_rules)})

  def set_parametric_loss(self,loss):
    self.loss=loss
    self.uri=UpdateRuleImplementation(loss=self.loss,forward=self.nn.apply)
    
  def mk_nns(self,lr):
    param=self.nn.init(self.getk(),self.x_train[0])
    return NNState(0,adam(learning_rate=lr).init(param),param)

  def train(self,tr_rule):
    if tr_rule in self.trained:
      self.log('Already trained',tr_rule,'?!')
      return
    self.trained[tr_rule]=tr_rule.train(self.uri,self.mk_nns(tr_rule.ur.lr),self.rs,self.getk()) # evaluate weights at varying times
    self.log('Training complete for',len(self.trained),'of',len(self.training_rules),'rules')

  def train_all(self):
    self.log('Training according to all',len(self.training_rules),'rules...')
    [self.log(r) for r in self.training_rules]
    [self.train(tr) for tr in self.training_rules]
    
  def update_res(self):
    for tr,nns in [(tr,nns) for tr,nns in self.trained.items() if not tr in self.res]:
      r=[]
      for nn in nns:
        pred_train,pred_test=self.nn.apply(nn.param,self.x_train),self.nn.apply(nn.param,self.x_test)
        r.append(Results.from_predictions(pred_train,self.y_train,pred_test,self.y_test,self.cost_rats))
      self.res[tr]=r
  
  def getk(self):
    self.key,k=split(self.key)
    return k

  def leaderboard(self):
    res_all={}
    
    for rule,res_rule in self.res.items():
      for checkpoint,res_checkpoint in zip(rule.to_checkpoints(),res_rule):
        for res_rat in res_checkpoint:
          res_all[res_rat.rat]=res_all.pop(res_rat.rat,[])+[Outcome(checkpoint,res_rat.res)]
    for cr,res in res_all.items():
      res.sort()
      print('==== cost ratio:',cr,'====')
      [print(r) for r in res]

def make_fp_perturbed_bce(beta):
  if beta:
    def beta_loss(pred,targ):
      return -targ*log(sigmoid(pred))+(1-targ)*sigmoid(pred*beta)/beta

    return beta_loss
  else:
    return sigmoid_binary_cross_entropy

def fp_fn_perturbed_bce(pred,targ,param):
  a,b=param
  l_pos=(lambda r: -sigmoid(r*a)/a) if a else (lambda r:-log(sigmoid(r))) #loss if y=+
  l_neg=(lambda r: -sigmoid(-r*b)/b) if b else (lambda r:-log(sigmoid(-r))) #lloss if y=-
  return targ*l_pos(pred)+(1-targ)*l_neg(pred)
