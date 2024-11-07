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
def force_ratio_loss(y_pred,y,beta_squared=.5):
  tp=jsum(y*y_pred)
  fp=jsum((1-y)*y_pred)
  fn=jsum(y*(1-y_pred))
  return (fp-beta_squared*fn)**2-tp

def hlp_init_params(input_dim,l_dim):
  return dict(A1=zeros(shape=(l_dim,input_dim)),
              b1=zeros(shape=l_dim),
              A2=zeros(shape=l_dim),
              b2=0.)

def lt_ratio_loss(y_pred,y,u,kappa=.99,beta=.5):
  #tp=jsum(y*y_pred)
  fp=jsum((1-y)*y_pred)/len(y)
  fn=jsum(y*(1-y_pred))/len(y)
  u*=kappa
  u+=(1-kappa)*sigmoid(fp/beta-fn*beta)
  v=1-u
  return u*fp+v*fn,u
class parametric_optimiser:

  def __init__(self,input_dim,l_dim,lr=.1,seed=0,loss=smooth_quotient_loss,
               init_params=hlp_init_params,implementation=hidden_layer_v,
               adam=True,lt_weighted_loss=False,beta1=.9,beta2=.999,eps=.00000001):
    self.loss=loss
    self.key=key(seed)
    self.lr=lr
    self.input_dim=input_dim
    self.l_dim=l_dim
    self.init_params=init_params
    self.params=init_params(input_dim,l_dim)
    self.randomise_params()
    self.implementation=implementation
    self.lt_weighted_loss=lt_weighted_loss
    
    if self.lt_weighted_loss:
      self.u=.5

      @jit
      def _eval_batch(X,y,u,params):
        y_pred=self.implementation(X,params)
        return self.loss(y_pred,y,u) #also return updated u
      
      def _lossonly_batch(X,y,u,params):return _eval_batch(X,y,u,params)[0]

      _d_params=grad(_lossonly_batch,argnums=3)
      
      def __eval_batch(X,y):
        ret,self.u=_eval_batch(X,y,self.u,self.params)
        return ret

      self.d_params=lambda X,y:_d_params(X,y,self.u,self.params)
      self.eval_batch=lambda X,y:_lossonly_batch(X,y,self.u,self.params)
      self.eval_batch_upd_u=__eval_batch

    else:

      @jit
      def _eval_batch(X,y,params):
        y_pred=self.implementation(X,params)
        return self.loss(y_pred,y)

      _d_params=grad(_eval_batch,argnums=2)

      self.d_params=lambda X,y:_d_params(X,y,self.params)
      self.eval_batch=lambda X,y:_eval_batch(X,y,self.params)

    if adam:
      #Don't bother with the fairly pointless time dependent bias removal
      #Eg, for default beta2 beta2^t<.01 when t>4600 or so.
      self.beta1=beta1
      self.beta2=beta2
      self.gamma1=1-beta1
      self.gamma2=1-beta2
      self.m=self.init_params(self.input_dim,self.l_dim)
      self.v=0
      self.eps=eps


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

  def force_ratio_adam_step(self,X,y,mk_batch=False):
    if mk_batch:
      X,y=rand_batch(X,y,mk_batch,self.key)
    upd=self.d_params(X,y,self.u,self.v)
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

  def gradient_desc_step(self,X,y,noise=0,mk_batch=False):
    if mk_batch:
      X,y=rand_batch(X,y,mk_batch,self.key)
    upd=self.d_params(X,y)
    self.params['A1']-=self.lr*upd['A1']
    self.params['b1']-=self.lr*upd['b1']
    self.params['A2']-=self.lr*upd['A2']
    self.params['b2']-=self.lr*upd['b2']
    if noise:
      score=self.eval_batch(X,y,self.params)
      self.randomise_params(amount=score*self.lr*noise)

  def repeated_desc_step(self,X,y,lr,n_runs=1000,noise=.01,verbose=False):
    for i in range(n_runs):
      self.gradient_desc_step(X,y,lr,noise=noise)
      if verbose and not(i%100):
        print('Loss(X,y)=',self.eval_batch(X,y))

