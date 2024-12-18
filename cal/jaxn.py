from os import mkdir
from sys import stderr

from jax import grad,jit
from jax.nn import sigmoid,relu,softmax
from jax.numpy import dot,vectorize,zeros,square,sqrt,array,\
                      logical_not,save,sum as jsum
from jax.numpy.linalg import qr,svd
from jax.scipy.signal import convolve
from jax.random import normal,key,randint,permutation,split

#sigmoid=lambda x:relu(1-relu(1-x)) #lazy!!

def rand_batch(X,y,batch_size,key):
  indices=randint(key,batch_size,0,y.shape[0])
  return X[indices],y[indices]

default_layer_dims=[32,32]#[4096]#[32,32]
def mk_nlp(layer_dims=default_layer_dims):
  layer_dims=layer_dims+[1]
  l=len(layer_dims)

  @jit
  def nlp_infer(x,params):
    for i in range(l): ##the subtraction of .5 means we "expect the input to have mean 0"
      x=sigmoid(dot(params[('A',i)],x)+params[('b',i)])-.5 #Otherwise: Wignerian behaviour!
    return x[0]+.5
    #for i in range(l-1): ##the subtraction of .5 means we "expect the input to have mean 0"
    #  x=softmax(dot(params[('A',i)],x)+params[('b',i)])#-.5 #Otherwise: Wignerian behaviour!
    #return sigmoid(dot(params[('A',i)],x)+params[('b',i)])[0] #Otherwise: Wignerian behaviour!

  def nlp_make_params(input_dim):
    ret={}
    for i,out_dim in enumerate(layer_dims):
      ret[('A',i)]=zeros(shape=(out_dim,input_dim))
      ret[('b',i)]=zeros(shape=out_dim)
      input_dim=out_dim
    return ret

  return nlp_make_params,vectorize(nlp_infer,excluded=[1],signature='(n)->()')

nlp_params,nlp_infer=mk_nlp()

default_conv_sizes=[5,3,3]
default_dense_dims=[256,32]
def mk_conv(conv_sizes=default_conv_sizes,dense_dims=default_dense_dims,
            activation=sigmoid):
  dense_dims=dense_dims+[1]

  @jit
  def conv_infer(x,params):
    #for i in len(conv_sizes):
    i=0
    while ('K',i) in params:
      x=activation(convolve(x,params[('K',i)],mode='valid'))
      i+=1
    x=x.flatten()
    i=0
    while ('A',i) in params:
      x=activation(dot(params[('A',i)],x)+params[('b',i)])
      i+=1
    return x[0]

  def conv_make_params(input_dims):#expect square matrix input
    #Calculate the number of nodes of last conv layer
    ret={}
    for i,dim in enumerate(conv_sizes):
      ret[('K',i)]=zeros(shape=(dim,dim))

    input_dim=(input_dims[0]-sum(conv_sizes)+len(conv_sizes))**2
    for i,out_dim in enumerate(dense_dims):
      ret[('A',i)]=zeros(shape=(out_dim,input_dim))
      ret[('b',i)]=zeros(shape=out_dim)
      input_dim=out_dim
    return ret

  return conv_make_params,vectorize(conv_infer,excluded=[1],signature='(n,n)->()')

conv_params,conv_infer=mk_conv()

class bin_optimiser:

  def __init__(self,input_dims,make_params=nlp_params,lr=.00001,seed=0,tol=.5,
               implementation=nlp_infer,beta1=.9,beta2=.999,eps=.00000001,batch_size=32,
               max_relative_confusion_importance=.0000001,target_fp=.01,
               target_fn=.01,avg_tol=.1,logf=stderr,threshold=.5,
               sns_dir=None,sns_per_epoch=10,params=None,empirical_fp=None,empirical_fn=None):
    self.logf=open(logf,'w') if isinstance(logf,str) else logf

    self.sns_dir=sns_dir
    if not sns_dir is None:
      try: mkdir(sns_dir)
      except FileExistsError: pass

    try: self.key=key(seed)
    except TypeError: self.key=split(seed)[1]
    self.lr=lr
    self.input_dims=input_dims
    self.make_params=make_params
    if params:
      self.params=params
    else:
      self.params=make_params(input_dims)
      self.randomise_params()
    self.implementation=implementation

    if empirical_fp:
      self.empirical_fp=empirical_fp
    else:
      self.empirical_fp=.25
    if empirical_fn:
      self.empirical_fn=empirical_fn
    else:
      self.empirical_fn=.25

    def _c_fp(X,y,params): return self.c_fp(y,implementation(X,params))

    def _c_fn(X,y,params): return self.c_fn(y,implementation(X,params))

    self.d_fp,self.d_fn=grad(_c_fp,argnums=2),grad(_c_fn,argnums=2)

    self.target_fp=target_fp
    self.target_fn=target_fn
    self.beta_sq=target_fp/target_fn
    self.beta=self.beta_sq**.5
    self.target_fn=target_fn
    self.avg_tol=avg_tol
    self.max_relative_confusion_importance=max_relative_confusion_importance
    self.tol=tol
    self.threshold=threshold
    self.n_steps=0
    self.batch_size=batch_size

    ##Adam params
    #Don't implement time dependent bias removal
    #Unimportant for asymptotic behaviour
    #Eg, for default beta2 beta2^t<.01 when t>4600 or so.
    self.beta1=beta1
    self.beta2=beta2
    self.one_minus_beta1=1-beta1
    self.one_minus_beta2=1-beta2
    try:
      self.m_fp=self.make_params(self.input_dims)
      self.m_fn=self.make_params(self.input_dims)
    except TypeError:
      self.m_fp={k:0.*self.make_params[k] for k in self.make_params}
      self.m_fn={k:0.*self.make_params[k] for k in self.make_params}
    self.v_fp=0
    self.v_fn=0
    self.eps=eps

  #Assume both y and y_pred are floats but y always 1. or 0.
  def b_tp(self,y,y_pred): return jsum((y==1.)&y_pred)/len(y)

  def b_fp(self,y,y_pred): return jsum((y==0.)&y_pred)/len(y)
  
  def b_fn(self,y,y_pred): return jsum((y==1.)&logical_not(y_pred))/len(y)

  def c_tp(self,y,y_pred): return jsum(y_pred[y==1.])/len(y)

  def c_fp(self,y,y_pred): return jsum(y_pred[y==0.])/len(y)

  def c_fn(self,y,y_pred): return jsum(1-y_pred[y==1.])/len(y)

  def update_weights(self,y,y_pseudolikelihoods):

    y_pred_bin=y_pseudolikelihoods>self.threshold
    batch_fp,batch_fn=self.b_fp(y,y_pred_bin),self.b_fn(y,y_pred_bin)
    fp_detection_thresh=self.avg_tol*min(self.empirical_fp,1-self.empirical_fp)
    fn_detection_thresh=self.avg_tol*min(self.empirical_fn,1-self.empirical_fn)
    self.empirical_fp*=1-fp_detection_thresh
    self.empirical_fn*=1-fn_detection_thresh
    self.empirical_fp+=fp_detection_thresh*batch_fp
    self.empirical_fn+=fn_detection_thresh*batch_fn

    self.fp_ok=self.empirical_fp<self.target_fp*self.tol
    self.fn_ok=self.empirical_fn<self.target_fn*self.tol

    batch_target_fp=self.empirical_fp
    batch_target_fn=self.empirical_fn
    if self.fp_ok:
      if self.fn_ok:
        batch_target_fp=batch_target_fn=0
      else:
        batch_target_fn=0
    else:
      batch_target_fp=0
      if not self.fn_ok:
        batch_target_fn=0

    U=(self.empirical_fp-batch_target_fp)/self.target_fp
    V=(self.empirical_fn-batch_target_fn)/self.target_fn
    self.U=max(U,self.max_relative_confusion_importance*V)
    self.V=max(V,self.max_relative_confusion_importance*U)

    ##Incorporated U and V into adam so need to normalise
    #norm=(self.U**2+self.V**2)**.5
    #self.U/=norm
    #self.V/=norm

  def adam_step(self,X,y):
    y_pred=self.implementation(X,self.params)
    self.update_weights(y,y_pred)
    dfp=self.d_fp(X,y,self.params)
    dfn=self.d_fn(X,y,self.params)

    self.v_fp*=self.beta2
    self.v_fn*=self.beta2
    for k in dfp: #Q: scale by U and V here or after this for block?
      self.m_fp[k]*=self.beta1
      self.m_fp[k]+=self.U*self.one_minus_beta1*dfp[k]
      self.v_fp+=self.one_minus_beta2*jsum(square(dfp[k]))

      self.m_fn[k]*=self.beta1
      self.m_fn[k]+=self.V*self.one_minus_beta1*dfn[k]
      self.v_fn+=self.one_minus_beta2*jsum(square(dfn[k]))

    mult_fp=self.lr/(self.eps+sqrt(self.v_fp))
    mult_fn=self.lr/(self.eps+sqrt(self.v_fn))
    for k in dfp:
      self.params[k]-=mult_fp*self.m_fp[k]+mult_fn*self.m_fn[k]
    
    self.n_steps+=1
    return y_pred>self.threshold

  def save_sns(self):
    if not self.sns_dir is None:
      save(self.sns_dir+f'{self.n_steps:09d}',self.params)

  def randomise_params(self,amount=1):
    for k in self.params:
      self.params[k]*=1-amount
      try:
        shape=self.params[k].shape
      except AttributeError as e:
        shape=()
      delta=2*normal(self.key,self.params[k].shape)
      #if self.params[k].ndim==2:#Trying to get free field...
      #  dim=min(self.params[k].shape)
      #  #Random orthogonal weights
      #  #delta=8*qr(delta)[0]
      ## delta*=2
      #  #if delta.shape[0]==delta.shape[1]:
      #  #  for i in range(min(delta.shape)):
      #  #    delta.at[i,i].set(delta[i,i]+32)#32.)
      ##  
      ##  self.params[k]*=2#*self.params[k].shape[1]**-.5
      #  #for i in range(dim):
      ##else:
      ##  delta=0
      self.params[k]+=amount*delta
      self.key=split(self.key)[0]

  def run_epoch(self,X_all,y_all,verbose=True,
                n_reports=4,n_sns=0):
    n_rows=len(y_all)
    n_batches=n_rows//self.batch_size #May miss 0<=t<32 rows

    perm=permutation(self.key,n_rows)
    self.key=split(self.key)[0]
    X_all,y_all=X_all[perm],y_all[perm]

    report_interval=n_batches//n_reports
    sns_interval=n_batches//(n_sns+1)
    for i in range(n_batches):
      self.adam_step(X_all[i:i+self.batch_size],y_all[i:i+self.batch_size])
      if verbose and not i%report_interval:
        print(f'Epoch {100*(i/n_batches):.0f}% complete... training performance:',
              file=self.logf,flush=True)
        self.bench(X_all,y_all)
      if not i%sns_interval:
        self.save_sns()

  def bench(self,X,y,logf=None):
    if logf is None:
      logf=self.logf
    y_pred=self.implementation(X,self.params)
    y_pred_bin=y_pred>self.threshold
    tp=self.b_tp(y,y_pred_bin)
    fp,fn=self.b_fp(y,y_pred_bin),self.b_fn(y,y_pred_bin)
    cts_tp,cts_fp,cts_fn=self.c_tp(y,y_pred),self.c_fp(y,y_pred),self.c_fn(y,y_pred)
    print(f'[bin,cts] tp: [{tp:.8f},{cts_tp:.8f}], bin/cts:{tp/cts_tp:.8f}',
          f'\n[bin,cts] fp: [{fp:.8f},{cts_fp:.8f}], bin/cts:{fp/cts_fp:.8f}',
          f'\n[bin,cts] fn: [{fn:.8f},{cts_fn:.8f}], bin/cts:{fn/cts_fn:.8f}',
          f'\ntargetted [fp,fn]: [{self.target_fp:.8f},{self.target_fn:.8f}]',
          f'\nempirical [fp,fn]: [{self.empirical_fp:.8f},{self.empirical_fn:.8f}]'+\
          f'\nP(+|predicted +)=bin precision~{tp/(tp+fp):.8f}',
          f'\nP(predicted +|+)=bin recall~{tp/(tp+fn):.8f}',
          f'\nGradient update rule: -(adam_dfp({self.U:.8f})+adam_dfn({self.V:.8f}))',
          file=logf,flush=True)
    return fp,fn

  def run_epochs(self,X_train,y_train,X_test=None,y_test=None,n_epochs=50,verbose=2):
    performance_fp=[]
    performance_fn=[]
    for i in range(1,n_epochs+1):
      if verbose: print('Beginning epoch',i,'...',file=self.logf,flush=True)
      self.run_epoch(X_train,y_train,verbose=verbose==2)
      if verbose:
        print('...done!',file=self.logf,flush=True)
        print('Training performance:',file=self.logf,flush=True)
        self.bench(X_train,y_train)
        if not X_test is None:
          print('Testing performance:',file=self.logf,flush=True)
          p_fp,p_fn=self.bench(X_test,y_test)
          performance_fp.append(p_fp)
          performance_fn.append(p_fn)
    print('Completed',n_epochs,'epochs!',file=self.logf,flush=True)
    print(':)',file=self.logf,flush=True)
    return performance_fp,performance_fn

  def inference(self,X):
    return self.implementation(X,self.params)>self.threshold

