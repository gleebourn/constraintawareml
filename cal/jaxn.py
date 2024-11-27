from jax import grad,jit
from jax.nn import sigmoid,relu
from jax.numpy import dot,vectorize,zeros,square,sqrt,array,logical_not,\
                      sum as jsum
from jax.scipy.signal import convolve
from jax.random import normal,key,randint,permutation

from sys import stderr

def rand_batch(X,y,batch_size,key):
  indices=randint(key,batch_size,0,y.shape[0])
  return X[indices],y[indices]

default_layer_dims=[32,32]
def mk_nlp(layer_dims=default_layer_dims):
  layer_dims=layer_dims+[1]
  l=len(layer_dims)

  @jit
  def nlp_infer(x,params):
    for i in range(l):
      x=sigmoid(dot(params[('A',i)],x)+params[('b',i)])
    return x[0]

  def nlp_init_params(input_dim):
    ret=dict()
    for i,out_dim in enumerate(layer_dims):
      ret[('A',i)]=zeros(shape=(out_dim,input_dim))
      ret[('b',i)]=zeros(shape=out_dim)
      input_dim=out_dim
    return ret

  return nlp_init_params,vectorize(nlp_infer,excluded=[1],signature='(n)->()')

nlp_params,nlp_infer=mk_nlp()

default_conv_sizes=[5,4,3,3]
default_dense_dims=[128,64,32]
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

  def conv_init_params(input_dims):#expect square matrix input
    #Calculate the number of nodes of last conv layer
    ret=dict()
    for i,dim in enumerate(conv_sizes):
      ret[('K',i)]=zeros(shape=(dim,dim))
    
    input_dim=(input_dims[0]-sum(conv_sizes)+len(conv_sizes))**2
    for i,out_dim in enumerate(dense_dims):
      ret[('A',i)]=zeros(shape=(out_dim,input_dim))
      ret[('b',i)]=zeros(shape=out_dim)
      input_dim=out_dim
    return ret

  return conv_init_params,vectorize(conv_infer,excluded=[1],signature='(n,n)->()')

conv_params,conv_infer=mk_conv()

class bin_optimiser:

  def __init__(self,input_dims,init_params=nlp_params,lr=.001,seed=0,
               implementation=nlp_infer,beta1=.9,beta2=.999,max_relative_confusion_importance=.001,
               eps=.00000001,confusion_averaging_rate=.99,target_fp=.01,target_fn=.01,outf=stderr):
    self.outf=open(outf,'w') if isinstance(outf,str) else outf
    self.key=key(seed)
    self.lr=lr
    self.input_dims=input_dims
    self.init_params=init_params
    self.params=init_params(input_dims)
    self.randomise_params()
    self.implementation=implementation

    def _c_fp(X,y,params): return self.c_fp(y,implementation(X,params))

    def _c_fn(X,y,params): return self.c_fn(y,implementation(X,params))

    self.d_fp,self.d_fn=grad(_c_fp,argnums=2),grad(_c_fn,argnums=2)

    self.target_fp=target_fp
    self.target_fn=target_fn
    self.beta_sq=target_fp/target_fn
    self.beta=self.beta_sq**.5
    self.confusion_averaging_rate=confusion_averaging_rate
    self.max_relative_confusion_importance=max_relative_confusion_importance
    self.one_minus_confusion_averaging_rate=1-confusion_averaging_rate
    self.threshold=.5
    self.empirical_fp=.5
    self.empirical_fn=.5

    ##Adam params
    #Don't implement time dependent bias removal
    #Unimportant for asymptotic behaviour
    #Eg, for default beta2 beta2^t<.01 when t>4600 or so.
    self.beta1=beta1
    self.beta2=beta2
    self.one_minus_beta1=1-beta1
    self.one_minus_beta2=1-beta2
    self.m=self.init_params(self.input_dims)
    self.v=0
    self.eps=eps

  def d_l(self,X,y,params):
    ret=dict()
    dfp=self.d_fp(X,y,params)
    dfn=self.d_fn(X,y,params)

    #normalising seems to hinder performance.
    #Could reasons be: wrong norm?  Floating point precision loss?
    #dfp_frob_sq=self.eps #Frobenius norm of the derivative
    #dfn_frob_sq=self.eps
    #for k in dfp:
    #  dfp_frob_sq+=jsum(dfp[k]**2)
    #  dfn_frob_sq+=jsum(dfn[k]**2)
    #for k in dfp:
    #  ret[k]=U*dfp[k]/(self.beta*dfp_frob_sq)+self.beta*V*dfn[k]/dfn_frob_sq

    for k in dfp:
      ret[k]=self.U*dfp[k]+self.V*dfn[k]

    return ret

  #Assume both y and y_pred are floats but y always 1. or 0.
  def b_tp(self,y,y_pred): return jsum((y==1.)&y_pred)/len(y)
  
  def b_fp(self,y,y_pred): return jsum((y==0.)&y_pred)/len(y)
  
  def b_fn(self,y,y_pred): return jsum((y==1.)&logical_not(y_pred))/len(y)

  def c_tp(self,y,y_pred): return jsum(y_pred[y==1.])/len(y)

  def c_fp(self,y,y_pred): return jsum(y_pred[y==0.])/len(y)

  def c_fn(self,y,y_pred): return jsum(1-y_pred[y==1.])/len(y)
                    
  def update_fp_w(self,y,y_pseudolikelihoods):

    y_pred_bin=y_pseudolikelihoods>self.threshold
    batch_fp,batch_fn=self.b_fp(y,y_pred_bin),self.b_fn(y,y_pred_bin)
    self.empirical_fp*=self.confusion_averaging_rate
    self.empirical_fn*=self.confusion_averaging_rate
    self.empirical_fp+=self.one_minus_confusion_averaging_rate*batch_fp
    self.empirical_fn+=self.one_minus_confusion_averaging_rate*batch_fn

    batch_target_fp=self.empirical_fp
    batch_target_fn=self.empirical_fn

    fp_too_high=self.empirical_fp>self.target_fp
    fn_too_high=self.empirical_fn>self.target_fn

    if fp_too_high: batch_target_fp=self.target_fp
    if fn_too_high: batch_target_fn=self.target_fn
    elif not(fp_too_high):batch_target_fp=batch_target_fn=0

    U=self.empirical_fp-batch_target_fp
    V=self.empirical_fn-batch_target_fn
    self.U=max(U,self.max_relative_confusion_importance*V)
    self.V=max(V,self.max_relative_confusion_importance*U)

  def adam_step(self,X,y):
    y_pred=self.implementation(X,self.params)
    self.update_fp_w(y,y_pred)
    upd=self.d_l(X,y,self.params)

    self.v*=self.beta2
    for k in upd:
      self.m[k]*=self.beta1
      self.m[k]+=self.one_minus_beta1*upd[k]
      self.v+=self.one_minus_beta2*jsum(square(upd[k]))

    mult=self.lr/(self.eps+sqrt(self.v))
    for k in upd:
      self.params[k]-=mult*self.m[k]

  def randomise_params(self,amount=1):
    for k in self.params:
      self.params[k]*=1-amount
      try:
        shape=self.params[k].shape
      except AttributeError as e:
        shape=()
      self.params[k]+=amount*normal(self.key,shape=shape)

  def run_epoch(self,X_all,y_all,batch_size=32,verbose=True,reports_per_batch=4):
    n_rows=len(y_all)
    n_batches=n_rows//batch_size #May miss 0<=t<32 rows

    perm=permutation(self.key,n_rows)
    X_all,y_all=X_all[perm],y_all[perm]

    report_interval=n_batches//reports_per_batch
    for i in range(n_batches):
      self.adam_step(X_all[i:i+batch_size],y_all[i:i+batch_size])
      if verbose and not(i%report_interval):
        print(f'Epoch {100*(i/n_batches):.0f}% complete... training performance:',
              file=self.outf,flush=True)
        self.bench(X_all,y_all)

  def bench(self,X,y,outf=None):
    if outf is None:
      outf=self.outf
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
          f'\nGradient update rule: -({self.U:.8f}*dfp+{self.V:.8f}*dfn)',file=outf,flush=True)

  def run_epochs(self,X_train,y_train,X_test=None,y_test=None,
                 batch_size=32,n_epochs=50,verbose=2):
    for i in range(1,n_epochs+1):
      if verbose: print('Beginning epoch',i,'...',file=self.outf,flush=True)
      self.run_epoch(X_train,y_train,batch_size=batch_size,verbose=verbose==2)
      if verbose:
        print('...done!',file=self.outf,flush=True)
        print('Training performance:',file=self.outf,flush=True)
        self.bench(X_train,y_train)
        if not(X_test is None):
          print('Testing performance:',file=self.outf,flush=True)
          self.bench(X_test,y_test)

