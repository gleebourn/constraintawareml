from jax.nn import sigmoid,relu
from jax.numpy import dot,vectorize,zeros,square,sqrt,array,sum as jsum
from jax.random import normal,key,randint,permutation
from jax import grad,jit
from jax.errors import JaxRuntimeError

#@jit #jitting this causes an iteration over 0d array error!!!!
def rand_batch(X,y,batch_size,key):
  indices=randint(key,batch_size,0,y.shape[0])
  return X[indices],y[indices]

@jit
def hlp_infer(x,params):
  A1=params['A1']
  b1=params['b1']
  A2=params['A2']
  b2=params['b2']
  assert len(x.shape)==1 and x.shape[0]==A1.shape[1],\
         'input shape of '+ str(x.shape)+' != '+\
         'required input shape of ('+str(A1.shape[1])+',)' 
  l1=sigmoid(dot(A1,x)+b1)
  return sigmoid(dot(A2,l1)+b2)

hlp_infer_v=vectorize(hlp_infer,excluded=[1],signature='(n)->()')

def hlp_init_params(input_dim,l_dim):
  return dict(A1=zeros(shape=(l_dim,input_dim)),
              b1=zeros(shape=l_dim),
              A2=zeros(shape=l_dim),
              b2=0.)

def gsum(y):
  try:
    return jsum(y)
  except JaxRuntimeError as e:
    return sum(y)


#Assume both y and y_pred are floats but y always 1. or 0.
def b_tp(y,y_pred):
  return gsum((y==1.)&(y_pred>.5))

def b_fp(y,y_pred):
  return gsum((y==0.)&(y_pred>.5))

def b_fn(y,y_pred):
  return gsum((y==1.)&(y_pred<=.5))

def c_fp(y,y_pred):
  return gsum((1-y)*y_pred)

def c_fn(y,y_pred):
  return gsum(y*(1-y_pred))

class bin_optimiser:

  def __init__(self,input_dim,l_dim,lr=.001,seed=0,init_params=hlp_init_params,
               implementation=hlp_infer_v,beta1=.9,beta2=.999,eps=.00000001,
               kappa=.99,beta=1,nu=50):
    self.key=key(seed)
    self.lr=lr
    self.input_dim=input_dim
    self.l_dim=l_dim
    self.init_params=init_params
    self.params=init_params(input_dim,l_dim)
    self.randomise_params()
    self.implementation=implementation

    self.beta=beta
    self.kappa=kappa
    self.om_kappa=1-kappa
    self.nu=nu
    self.fp_weight=0#self.fn_weight=.5
    
    def _fp(X,y,params):
      return c_fp(y,implementation(X,params))

    def _fn(X,y,params):
      return c_fn(y,implementation(X,params))

    
    self._dfp=grad(_fp,argnums=2)
    self._dfn=grad(_fn,argnums=2)

    ##Adam params
    #Don't bother with the fairly pointless time dependent bias removal
    #Eg, for default beta2 beta2^t<.01 when t>4600 or so.
    self.beta1=beta1
    self.beta2=beta2
    self.gamma1=1-beta1
    self.gamma2=1-beta2
    self.m=self.init_params(self.input_dim,self.l_dim)
    self.v=0
    self.eps=eps

  def b_tp(self,y,y_pred):
    return b_tp(y,y_pred)/len(y)

  def b_fp(self,y,y_pred):
    return b_fp(y,y_pred)/len(y)
                  
  def b_fn(self,y,y_pred):
    return b_fn(y,y_pred)/len(y)
                  
  def c_fp(self,y,y_pred):
    return c_fp(y,y_pred)/len(y)
                  
  def c_fn(self,y,y_pred):
    return c_fn(y,y_pred)/len(y)
                  
  def update_fp_w(self,y,y_pred):
    #Check which way the skew is.
    #We are aiming for fp/fn=beta**2.
    #u>0 -> too many fp
    #u<0 -> too many fn
    u=self.c_fp(y,y_pred)/self.beta-self.c_fn(y,y_pred)*self.beta
    #u=sigmoid(self.nu*u)
    self.fp_weight=self.kappa*self.fp_weight+u#self.om_kappa*u
    #self.fn_weight=1-self.fp_weight

  def dfp(self,X,y):
    return self._dfp(X,y,self.params)

  def dfn(self,X,y):
    return self._dfn(X,y,self.params)

  def adam_step(self,X,y):
    y_pred=self.infer(X)
    self.update_fp_w(y,y_pred)

    #upd=self.dfp if self.fp_weight>0 else self.dfn
    #Only update one of two, if borderline do it randomly
    upd=self.dfp if self.fp_weight>normal(self.key) else self.dfn
    upd=upd(X,y)
    self.v*=self.beta2
    for key in upd:
      self.m[key]*=self.beta1
      #upd=self.fp_weight*dfp[key]+self.fn_weight*dfn[key]
      self.m[key]+=self.gamma1*upd[key]
      self.v+=self.gamma2*jsum(square(upd[key]))

    mult=self.lr/(self.eps+sqrt(self.v))
    for key in upd:
      self.params[key]-=mult*self.m[key]

  def randomise_params(self,amount=1):
    self.params['A1']*=1-amount
    self.params['b1']*=1-amount
    self.params['A2']*=1-amount
    self.params['b2']*=1-amount
    self.params['A1']+=amount*normal(self.key,shape=(self.l_dim,self.input_dim))
    self.params['b1']+=amount*normal(self.key,shape=self.l_dim)
    self.params['A2']+=amount*normal(self.key,shape=self.l_dim)
    self.params['b2']+=amount*normal(self.key)

  def infer(self,x):
    try:
      return self.implementation(x,self.params)
    except JaxRuntimeError as e:
      print('could not do vectorised inference... gonna be slow...')
      y_pred=array([self.infer(array([t])) for t in x])
      print('...done')
      return y_pred

  def run_epoch(self,X_all,y_all,batch_size=32,verbose=True):
    n_rows=len(y_all)
    n_batches=n_rows//batch_size #May miss 0<=t<32 rows
    perm=permutation(self.key,n_rows)

    X_all=X_all[perm]
    y_all=y_all[perm]

    report_interval=n_batches//10
    for i in range(n_batches):
      self.adam_step(X_all[i:i+batch_size],y_all[i:i+batch_size])
      if verbose and not(i%report_interval):
        print('Epoch',100*(i/n_batches),'% complete... training performance:')
        self.bench(X_all,y_all)

  def bench(self,X,y):
    y_pred=self.infer(X)
    tp=self.b_tp(y,y_pred)
    fp=self.b_fp(y,y_pred)
    fn=self.b_fn(y,y_pred)
    cts_fp=self.c_fp(y,y_pred)
    cts_fn=self.c_fn(y,y_pred)
    bt2=self.beta**2
    print('|bin tp |bin fp |bin fn |bin fp:fn |tgt fp:fn |cts fp |cts fn |bin pr |bin rc |fp wgt |')
    print(f'|{tp:.5f}|{fp:.5f}|{fn:.5f}|{fp/fn:10.5f}|{bt2:10.5f}|{cts_fp:.5f}|{cts_fn:.5f}|'+\
          f'{tp/(tp+fp):.5f}|{tp/(tp+fn):.5f}|{self.fp_weight:.5f}|')

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


