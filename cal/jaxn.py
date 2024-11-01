from jax.nn import sigmoid,relu
from jax.numpy import dot,vectorize,zeros,sum as jsum
from jax.random import normal,key
from jax import grad,jit

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
def loss_batch(X,y,params):
  y_pred=f_v(X,params)
  return smooth_f_beta_loss(y_pred,y)
  
d_params=grad(loss_batch,argnums=2)

class hidden_layer_perceptron:

  def __init__(self,input_dim,l_dim,seed=0):
    self.key=key(seed)
    self.input_dim=input_dim
    self.l_dim=l_dim
    self.params=dict(A1=zeros(shape=(l_dim,input_dim)),
                     b1=zeros(shape=l_dim),
                     A2=zeros(shape=l_dim),
                     b2=0.)
    self.randomise_params()

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

  def gradient_desc_step(self,X,y,lr,noise=0):
    score=loss_batch(X,y,self.params)
    upd=d_params(X,y,self.params)
    self.params['A1']-=lr*upd['A1']
    self.params['b1']-=lr*upd['b1']
    self.params['A2']-=lr*upd['A2']
    self.params['b2']-=lr*upd['b2']
    if noise: self.randomise_params(amount=score*lr*noise)

  def repeated_desc_step(self,X,y,lr,n_runs=1000,noise=.01,verbose=False):
    for i in range(n_runs):
      self.gradient_desc_step(X,y,lr,noise=noise)
      if verbose and not(i%100):
        print('Loss(X,y)=',self.loss_batch(X,y))


