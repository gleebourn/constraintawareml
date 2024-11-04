from jax.nn import sigmoid,relu
from jax.numpy import dot,vectorize,zeros,square,sqrt,sum as jsum
from jax.random import normal,key,randint
from jax import grad,jit

#@jit #jitting this causes an iteration over 0d array error!!!!
def rand_batch(X,y,batch_size,key):
  indices=randint(key,batch_size,0,y.shape[0])
  return X[indices],y[indices]

@jit
def f(x,params):
  A1=params['A1']
  b1=params['b1']
  A2=params['A2']
  b2=params['b2']
  assert len(x.shape)==1 and x.shape[0]==A1.shape[1],\
         'input shape of '+ str(x.shape)+' != '+\
         'required input shape of ('+str(A1.shape[1])+',)' 
  l1=sigmoid(dot(A1,x)+b1)
  return sigmoid(dot(A2,l1)+b2)

f_v=vectorize(f,excluded=[1],signature='(n)->()')

@jit
def smooth_f_beta_loss(y_pred,y,lamb=.5):
  tp=jsum(y*y_pred)
  fp=jsum((1-y)*y_pred)
  fn=jsum(y*(1-y_pred))
  weighted_false=(lamb*fp+(1-lamb)*fn)
  return weighted_false/(tp+weighted_false)

@jit
def smooth_quotient_loss(y_pred,y,lamb=.5):
  tp=jsum(y*y_pred)
  fp=jsum((1-y)*y_pred)
  fn=jsum(y*(1-y_pred))
  weighted_false=(lamb*fp+(1-lamb)*fn)
  return weighted_false/tp

@jit
def loss_batch(X,y,params):
  y_pred=f_v(X,params)
  #return smooth_f_beta_loss(y_pred,y)
  return smooth_quotient_loss(y_pred,y)
  
d_params=grad(loss_batch,argnums=2)

def hlp_init_params(input_dim,l_dim):
  return dict(A1=zeros(shape=(l_dim,input_dim)),
              b1=zeros(shape=l_dim),
              A2=zeros(shape=l_dim),
              b2=0.)

class hidden_layer_perceptron:

  def __init__(self,input_dim,l_dim,lr=.1,seed=0):
    self.key=key(seed)
    self.lr=lr
    self.input_dim=input_dim
    self.l_dim=l_dim
    self.params=hlp_init_params(input_dim,l_dim)
    self.randomise_params()

  def init_adam(self,beta1=.9,beta2=.999,lr=.01):
    #Don't bother with the fairly pointless time dependent bias removal
    #Eg, for default beta2 beta2^t<.01 when t>4600 or so.
    self.beta1=beta1
    self.beta2=beta2
    self.gamma1=1-beta1
    self.gamma2=1-beta2
    self.lr=lr
    self.m=hlp_init_params(self.input_dim,self.l_dim)
    self.v=0

  def adam_step(self,X,y,mk_batch=False,eps=.00000001):
    if mk_batch:
      X,y=rand_batch(X,y,mk_batch,self.key)
    upd=d_params(X,y,self.params)
    self.v*=self.beta2
    for key in upd:
      self.m[key]=self.beta1*self.m[key]+self.gamma1*upd[key]
      self.v+=self.gamma2*jsum(square(upd[key]))

    mult=self.lr/(eps+sqrt(self.v))
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
    return f(x,self.params)

  def multi_infer(self,X):
    return f_v(X,self.params)
  
  def loss_batch(self,X,y):
    return loss_batch(X,y,self.params)

  def gradient_desc_step(self,X,y,noise=0,mk_batch=False):
    if mk_batch:
      X,y=rand_batch(X,y,mk_batch,self.key)
    upd=d_params(X,y,self.params)
    self.params['A1']-=self.lr*upd['A1']
    self.params['b1']-=self.lr*upd['b1']
    self.params['A2']-=self.lr*upd['A2']
    self.params['b2']-=self.lr*upd['b2']
    if noise:
      score=loss_batch(X,y,self.params)
      self.randomise_params(amount=score*self.lr*noise)

  def repeated_desc_step(self,X,y,lr,n_runs=1000,noise=.01,verbose=False):
    for i in range(n_runs):
      self.gradient_desc_step(X,y,lr,noise=noise)
      if verbose and not(i%100):
        print('Loss(X,y)=',self.loss_batch(X,y))

