from jax.nn import sigmoid,relu
from jax.numpy import dot,vectorize,zeros,square,sqrt,sum as jsum
from jax.random import normal,key,randint
from jax import grad,jit

#@jit #jitting this causes an iteration over 0d array error!!!!
def rand_batch(X,y,batch_size,key):
  indices=randint(key,batch_size,0,y.shape[0])
  return X[indices],y[indices]

@jit
def hidden_layer(x,params):
  A1=params['A1']
  b1=params['b1']
  A2=params['A2']
  b2=params['b2']
  assert len(x.shape)==1 and x.shape[0]==A1.shape[1],\
         'input shape of '+ str(x.shape)+' != '+\
         'required input shape of ('+str(A1.shape[1])+',)' 
  l1=sigmoid(dot(A1,x)+b1)
  return sigmoid(dot(A2,l1)+b2)

hidden_layer_v=vectorize(hidden_layer,excluded=[1],signature='(n)->()')

def hlp_init_params(input_dim,l_dim):
  return dict(A1=zeros(shape=(l_dim,input_dim)),
              b1=zeros(shape=l_dim),
              A2=zeros(shape=l_dim),
              b2=0.)

def c_fp(y,y_pred):
  return jsum((1-y)*y_pred)

def c_fn(y,y_pred):
  return jsum(y*(1-y_pred))

class bin_optimiser:

  def __init__(self,input_dim,l_dim,lr=.1,seed=0,init_params=hlp_init_params,
               implementation=hidden_layer_v,beta1=.9,beta2=.999,eps=.00000001):
    self.key=key(seed)
    self.lr=lr
    self.input_dim=input_dim
    self.l_dim=l_dim
    self.init_params=init_params
    self.params=init_params(input_dim,l_dim)
    self.randomise_params()
    self.implementation=implementation
    
    def fp(X,y,params):
      return c_fp(y,implementation(X,params)

    def fn(X,y,params):
      return c_fn(y,implementation(X,params)
    
    self._dfp=grad(fp,argnums=2)
    self._dfn=grad(fn,argnums=2)

    #Don't bother with the fairly pointless time dependent bias removal
    #Eg, for default beta2 beta2^t<.01 when t>4600 or so.
    self.beta1=beta1
    self.beta2=beta2
    self.gamma1=1-beta1
    self.gamma2=1-beta2
    self.m=self.init_params(self.input_dim,self.l_dim)
    self.v=0
    self.eps=eps

  def dfp(X,y):
    return self._dfp(X,y,self.params)

  def dfn(X,y):
    return self._dfn(X,y,self.params)

  def adam_step(self,X,y,mk_batch=False):
    if mk_batch:
      X,y=rand_batch(X,y,mk_batch,self.key)
    if self.lt_weighted_loss:self.eval_batch_upd_u(X,y)
    upd=self.d_params(X,y)
    self.v*=self.beta2
    for key in upd:
      #self.m[key]=self.beta1*self.m[key]+self.gamma1*upd[key]
      self.m[key]*=self.beta1
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
    return self.implementation(x,self.params)

