from jax.nn import sigmoid,relu
from jax.numpy import dot,vectorize,zeros,square,sqrt,array,inf,logical_not,\
                      sum as jsum,max as jmax
def relu1(x): return relu(1-relu(.5-x))
def mixsr(x): return .9*relu1(x)+.1*sigmoid(x)
from jax.scipy.signal import convolve
from jax.random import normal,key,randint,permutation
from jax import grad,jit
from jax.errors import JaxRuntimeError

from sys import stderr
#from concurrent.futures import ProcessPoolExecutor

#@jit #jitting this causes an iteration over 0d array error!!!!
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

#default_conv_sizes=[5,5,3]
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
               implementation=nlp_infer,beta1=.9,beta2=.999,
               eps=.00000001,kappa=.999,beta=.1,nu=10,outf=stderr):
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

    self.beta=beta
    self.beta_sq=beta**2
    self.kappa=kappa
    self.one_minus_kappa=1-kappa
    self.nu=nu
    self.threshold=.5

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

  #def c_tp(self,y,y_pred): return jsum(relu(2*y_pred[y==1.]-1))/len(y)
   
  #def c_fp(self,y,y_pred): return jsum(relu(2*y_pred[y==0.]-1))/len(y)
   
  #def c_fn(self,y,y_pred): return jsum(relu(2*(1-y_pred[y==1.])-1))/len(y)
                    
  def update_fp_w(self,y,y_pseudolikelihoods):
    thresh_lb=0
    thresh_ub=1
    fp_fn_thresh_lb=inf
    fp_fn_thresh_ub=0

    while thresh_ub-thresh_lb>self.eps:
      avg=(thresh_ub+thresh_lb)/2
      y_pred_bin=y_pseudolikelihoods>avg
      fp_fn=self.b_fp(y,y_pred_bin)/(self.b_fn(y,y_pred_bin)+self.eps)
      if abs(fp_fn-self.beta_sq)<self.eps: break
      if fp_fn>self.beta_sq:
        fp_fn_thresh_lb,thresh_lb=fp_fn,avg
      else :
        fp_fn_thresh_ub,thresh_ub=fp_fn,avg

    self.threshold*=self.kappa
    self.threshold+=self.one_minus_kappa*avg

    ##We want threshold to be about .5 really.
    #self.threshold>.5 -> we are cutting off what would be false positive
    #                  -> want model to make fewer false positives
    #self.U=sigmoid(self.nu*(self.threshold-.5))
    self.U=sigmoid(self.nu*(avg-.5))

    self.V=1-self.U

    #When on target, FP/beta=FN*beta
    #Want to decrease FP by 1/beta for every time FN decreases by beta
    self.U*=self.beta
    self.V/=self.beta


  def adam_step(self,X,y):
    y_pred=self.infer(X)
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

  def infer(self,x):
    try:
      return self.implementation(x,self.params)
    except JaxRuntimeError as e:
      print('Could not do vectorised inference...',file=self.outf,flush=True)
      print('Inferring over a loop...',file=self.outf,flush=True)
      y_pred=array([self.infer(array([t])) for t in x])
      print('...done',file=self.outf,flush=True)
      return y_pred

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
    y_pred=self.infer(X)
    y_pred_bin=y_pred>self.threshold
    tp=self.b_tp(y,y_pred_bin)
    fp,fn=self.b_fp(y,y_pred_bin),self.b_fn(y,y_pred_bin)
    cts_tp,cts_fp,cts_fn=self.c_tp(y,y_pred),self.c_fp(y,y_pred),self.c_fn(y,y_pred)
    print(f'[bin,cts] tp: [{tp:.5f},{cts_tp:.5f}], bin/cts:{tp/cts_tp:.5f}',
          f'\n[bin,cts] fp: [{fp:.5f},{cts_fp:.5f}], bin/cts:{fp/cts_fp:.5f}',
          f'\n[bin,cts] fn: [{fn:.5f},{cts_fn:.5f}], bin/cts:{fn/cts_fn:.5f}',
          f'\n[bin,cts,tgt] fp/fn: [{fp/fn:10.5f},',
          f'{cts_fp/cts_fn:10.5f},{self.beta**2:10.5f}]',
          f'\nP(+|predicted +)=bin precision~{tp/(tp+fp):.5f}',
          f'\nP(predicted +|+)=bin recall~{tp/(tp+fn):.5f}',
          f'\nthreshold,U,V: {self.threshold:.5f},{self.U:.5f},{self.V:.5f}',file=outf,flush=True)

  def run_epochs(self,X_train,y_train,X_test=None,y_test=None,
                 batch_size=32,n_epochs=100,verbose=2):
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

