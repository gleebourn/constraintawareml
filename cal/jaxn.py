from jax.nn import sigmoid
from jax.numpy import dot,vectorize,zeros,square,sqrt,array,sum as jsum
from jax.random import normal,key,randint,permutation
from jax import grad,jit
from jax.errors import JaxRuntimeError

#@jit #jitting this causes an iteration over 0d array error!!!!
def rand_batch(X,y,batch_size,key):
  indices=randint(key,batch_size,0,y.shape[0])
  return X[indices],y[indices]

dflt_layer_dims=[128,64]
def mk_nlp_infer(layer_dims=dflt_layer_dims):
  layer_dims.append(1)
  l=len(layer_dims)

  @jit
  def nlp_infer(x,params):
    for i in range(l):
      x=sigmoid(dot(params[('A',i)],x)+params[('b',i)])
    return x[0]
  return vectorize(nlp_infer,excluded=[1],signature='(n)->()')

def mk_nlp_init_params(layer_dims=dflt_layer_dims):
  layer_dims.append(1)
  def nlp_init_params(input_dim):
    ret=dict()
    for i,out_dim in zip(range(len(layer_dims)),layer_dims):
      ret[('A',i)]=zeros(shape=(out_dim,input_dim))
      ret[('b',i)]=zeros(shape=out_dim)
      input_dim=out_dim
    return ret
  return nlp_init_params

def gsum(y):
  try:
    return jsum(y)
  except JaxRuntimeError as e:
    return sum(y)

class bin_optimiser:

  def __init__(self,input_dim,init_params=mk_nlp_init_params(),lr=.001,seed=0,
               implementation=mk_nlp_infer(),beta1=.9,beta2=.999,
               eps=.00000001,kappa=.99,beta=1,nu=100):
    self.key=key(seed)
    self.lr=lr
    self.input_dim=input_dim
    self.init_params=init_params
    self.params=init_params(input_dim)
    self.randomise_params()
    self.implementation=implementation

    def l(X,y,U,V,params): #(1-y)U-yV = U or -V when y=0 or 1 respectively
      return gsum(((1-y)*U-y*V)*implementation(X,params))

    self.dl=grad(l,argnums=4)

    self.beta=beta
    self.kappa=kappa
    self.om_kappa=1-kappa
    self.nu=nu
    self.fp_weight=0#self.fn_weight=.5

    ##Adam params
    #Don't bother with the fairly pointless time dependent bias removal
    #Eg, for default beta2 beta2^t<.01 when t>4600 or so.
    self.beta1=beta1
    self.beta2=beta2
    self.gamma1=1-beta1
    self.gamma2=1-beta2
    self.m=self.init_params(self.input_dim)
    self.v=0
    self.eps=eps

  #Assume both y and y_pred are floats but y always 1. or 0.
  def b_tp(self,y,y_pred):
    return gsum((y==1.)&(y_pred>.5))/len(y)
  
  def b_fp(self,y,y_pred):
    return gsum((y==0.)&(y_pred>.5))/len(y)
  
  def b_fn(self,y,y_pred):
    return gsum((y==1.)&(y_pred<=.5))/len(y)

  def c_tp(self,y,y_pred):
    return gsum(y*y_pred)/len(y)
  
  def c_fp(self,y,y_pred):
    return gsum((1-y)*y_pred)/len(y)
  
  def c_fn(self,y,y_pred):
    return gsum(y*(1-y_pred))/len(y)
                    
  def update_fp_w(self,y,y_pred):
    #Check which way the skew is.
    #We are aiming for fp/fn=beta**2.
    #u>0 -> too many fp
    #u<0 -> too many fn
    u=self.b_fp(y,y_pred)/self.beta-self.b_fn(y,y_pred)*self.beta
    self.fp_weight=self.kappa*self.fp_weight+self.om_kappa*u

    self.U=sigmoid(self.nu*self.fp_weight)
    self.V=1-self.U

  def dfp(self,X,y):
    return self._dfp(X,y,self.params)

  def dfn(self,X,y):
    return self._dfn(X,y,self.params)

  def adam_step(self,X,y):
    y_pred=self.infer(X)
    self.update_fp_w(y,y_pred)
    upd=self.dl(X,y,self.U,self.V,self.params)

    self.v*=self.beta2
    for key in upd:
      self.m[key]*=self.beta1
      self.m[key]+=self.gamma1*upd[key]
      self.v+=self.gamma2*jsum(square(upd[key]))

    mult=self.lr/(self.eps+sqrt(self.v))
    for key in upd:
      self.params[key]-=mult*self.m[key]

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
      print('could not do vectorised inference... gonna be slow...')
      y_pred=array([self.infer(array([t])) for t in x])
      print('...done')
      return y_pred

  def run_epoch(self,X_all,y_all,batch_size=32,verbose=True,reports_per_batch=10):
    n_rows=len(y_all)
    n_batches=n_rows//batch_size #May miss 0<=t<32 rows

    perm=permutation(self.key,n_rows)
    X_all,y_all=X_all[perm],y_all[perm]

    report_interval=n_batches//reports_per_batch
    for i in range(n_batches):
      self.adam_step(X_all[i:i+batch_size],y_all[i:i+batch_size])
      if verbose and not((i%report_interval)and((i-10)%report_interval)):
        print('Epoch',100*(i/n_batches),'% complete... training performance:')
        self.bench(X_all,y_all)

  def bench(self,X,y):
    y_pred=self.infer(X)
    tp,fp,fn=self.b_tp(y,y_pred),self.b_fp(y,y_pred),self.b_fn(y,y_pred)
    cts_tp,cts_fp,cts_fn=self.c_tp(y,y_pred),self.c_fp(y,y_pred),self.c_fn(y,y_pred)
    print(f'[bin,cts] tp: [{tp:.5f},{cts_tp:.5f}]')
    print(f'[bin,cts] fp: [{fp:.5f},{cts_fp:.5f}]')
    print(f'[bin,cts] fn: [{fn:.5f},{cts_fn:.5f}]')
    print(f'[bin,cts,tgt] fp/fn: [{fp/fn:10.5f},{cts_fp/cts_fn:10.5f},{self.beta**2:10.5f}]')
    print(f'bin precision:{tp/(tp+fp):.5f}')
    print(f'bin recall:{tp/(tp+fn):.5f}')
    print(f'fp wgt: {self.fp_weight:.5f}')

  def run_epochs(self,X_train,y_train,X_test=None,y_test=None,batch_size=32,n_epochs=100,verbose=2):
    for i in range(1,n_epochs+1):
      if verbose: print('Beginning epoch',i,'...')
      self.run_epoch(X_train,y_train,batch_size=batch_size,verbose=verbose==2)
      if verbose:
        print('...done!')
        print('Training performance:')
        self.bench(X_train,y_train)
        if not(X_test is None):
          print('Testing performance:')
          self.bench(X_test,y_test)
