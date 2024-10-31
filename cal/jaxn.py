from jax.nn import sigmoid
from jax.numpy import dot,vectorize,zeros,sum as jsum
from jax.random import normal,key
from jax import grad

def f(params):
  assert len(params['x'].shape)==1 and params['x'].shape[0]==params['A1'].shape[1],\
         'input shape of '+ str(params['x'].shape)+' != '+\
         'required input shape of ('+str(params['A1'].shape[1])+',)' 
  l1=sigmoid(dot(params['A1'],params['x'])+params['b1'])
  l1=l1/jsum(l1)
  l2=sigmoid(dot(params['A2'],l1)+params['b2'])
  return sigmoid(jsum(l2))

df=grad(f)

def another_loss(y_pred,y,lamb=.5):
  tp=jsum(y*y_pred)
  fp=jsum((1-y)*y_pred)
  fn=jsum(y*(1-y_pred))
  weighted_false=(lamb*fp+(1-lamb)*fn)
  return weighted_false/(tp+weighted_false)

class hidden_layer_perceptron:

  def __init__(self,input_dim,l1_dim,l2_dim,seed=0):
    self.key=key(seed)
    self.input_dim=input_dim
    self.l1_dim=l1_dim
    self.l2_dim=l2_dim
    A1=zeros(shape=(l1_dim,input_dim))
    b1=zeros(shape=l1_dim)
    A2=zeros(shape=(l2_dim,l1_dim))
    b2=zeros(shape=l2_dim)
    self.params={'A1':A1,'b1':b1,'A2':A2,'b2':b2}
    self.randomise_params()
    self.get_score_grad()

  def randomise_params(self,amount=1):
    self.params['A1']*=1-amount
    self.params['b1']*=1-amount
    self.params['A2']*=1-amount
    self.params['b2']*=1-amount
    self.params['A1']+=amount*normal(self.key,shape=(self.l1_dim,self.input_dim))
    self.params['b1']+=amount*normal(self.key,shape=self.l1_dim)
    self.params['A2']+=amount*normal(self.key,shape=(self.l2_dim,self.l1_dim))
    self.params['b2']+=amount*normal(self.key,shape=self.l2_dim)
    

  def infer(self,params,x):
    return f(dict(params,**{'x':x}))

  def multi_infer(self,params,X):
    return vectorize(lambda x:self.infer(params,x),signature='(n)->()')(X)

  def score_batch(self,params):
    return another_loss(self.multi_infer(params,params['X']),params['y'])

  def get_score_grad(self):
    self.dscore=grad(self.score_batch)

  def gradient_desc_step(self,X,y,lr):
    p=dict(self.params,**{'X':X,'y':y})
    upd=self.dscore(p)
    for lab in ['A1','b1','A2','b2']:
      self.params[lab]-=lr*upd[lab]

  def repeated_desc_step(self,X,y,lr,n_runs=1000,noise=.01):
    for i in range(n_runs):
      self.gradient_desc_step(X,y,lr)
      self.randomise_params(amount=noise)


