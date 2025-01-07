from numpy import array,sum as nsm
from types import SimpleNamespace

from jax import grad
from jax.nn import sigmoid
from jax.numpy import zeros,dot,vectorize

from jax.random import normal,key

INPUT_DIM=1

def threshold_f(x,thresh=.5):
  return x>thresh
def threshold_b(y_tilde,y):
  y_p=y_tilde>.5
  l=len(y_p)
  return SimpleNamespace(fp=nsm(y_p&(~y))/l,fn=nsm((~y_p)&y)/l,l=l)

#Track moving averages
def moving_averages_b(c,fpfn):
  
  fp_amnt=min(1,fpfn.l*min(c.fp,1-c.fp))*c.binomial_averaging_tolerance
  fn_amnt=min(1,fpfn.l*min(c.fn,1-c.fn))*c.binomial_averaging_tolerance

  c.fp=(1-fp_amnt)*c.fp+fp_amnt*fpfn.fp
  c.fn=(1-fn_amnt)*c.fn+fn_amnt*fpfn.fn
  approach_fp=approach_fn=0
  fp_ok=c.fp<c.target_fp
  fn_ok=c.fn<c.target_fn
  if fp_ok and not fn_ok:
    approach_fp=c.fp
  elif not fp_ok and fn_ok:
    approach_fn=c.fn
  U,V=(c.fp-approach_fp)/c.target_fp,(c.fn-approach_fn)/c.target_fn
  c.U,c.V=max(U,c.max_gradient_ratio*V),max(V,c.max_gradient_ratio*U)
  return c

def init_smooth_params(in_dim,layer_dims):
  n_layers=len(layer_dims)
  w=dict()
  for i,out_dim in enumerate(layer_dims):
    w[('A',i)]=zeros(shape=(out_dim,in_dim))
    w[('b',i)]=zeros(shape=out_dim)
    in_dim=out_dim
  return w

def mk_smooth_f(layer_dims):
  def smooth_f_unbatched(w,x):
    for i,_ in enumerate(layer_dims):
      x=sigmoid(dot(w[('A',i)],x)+w[('b',i)])-.5
    return x[0]+.5
  return vectorize(smooth_f_unbatched,excluded=[0],signature='(n)->()')

  
def mk_smooth_b(layer_dims):
  smooth_f=mk_smooth_f(layer_dims)
  def c_fp(p,x,y):
    return dot(smooth_f(p,x),~y)
  def c_fn(p,x,y):
    return dot(1-smooth_f(p,x),y)
  
  dfp=grad(c_fp,argnums=0)
  dfn=grad(c_fn,argnums=0)
  
  def smooth_b(p,x,y):
    return SimpleNamespace(dfp=dfp(p,x,y),dfn=dfn(p,x,y))
    
  return smooth_b


def reparameterisation_u(wmc):
  #wm.m : adam variables
  wmc.m.v_fp*=wmc.m.beta2
  wmc.m.v_fn*=wmc.m.beta2
  for k in wmc.d.dfp:
    wmc.m.fp[k]*=wmc.m.beta1
    wmc.m.fn[k]*=wmc.m.beta1
    wmc.m.fp[k]+=wmc.c.U*(1-wmc.m.beta1)*wmc.d.dfp[k]
    wmc.m.fn[k]+=wmc.c.V*(1-wmc.m.beta1)*wmc.d.dfn[k]
    wmc.m.v_fp+=(1-wmc.m.beta2)*nsm(wmc.d.dfp[k]**2)
    wmc.m.v_fn+=(1-wmc.m.beta2)*nsm(wmc.d.dfn[k]**2)

  div_fp=(wmc.m.eps+wmc.m.v_fp**.5)
  div_fn=(wmc.m.eps+wmc.m.v_fn**.5)
  for k in wmc.w:
    wmc.w[k]-=wmc.m.lr*(wmc.m.fp[k]/div_fp+wmc.m.fn[k]/div_fn)
  return wmc

def mk_model_f(layer_dims):
  smooth_f=mk_smooth_f(layer_dims)
  def model_f(wmc,x):
    return threshold_f(smooth_f(wmc.w,x))
  return model_f

def mk_model_b(layer_dims):
  smooth_f=mk_smooth_f(layer_dims)
  smooth_b=mk_smooth_b(layer_dims)
  def model_b(wmc,x,y):
    y_tilde=smooth_f(wmc.w,x)
    fpfn=threshold_b(y_tilde,y)
    wmc.c=moving_averages_b(wmc.c,fpfn)
    wmc.d=smooth_b(wmc.w,x,y)
    wmc=reparameterisation_u(wmc)
    return wmc
  return model_b

def randomise_weights(w,k):
  for i in w:
    try:
      w[i]=normal(k,w[i].shape)
    except AttributeError:
      w[i]=normal(k)

def init_model_params(in_dim,layer_dims,k):
  #Smooth map weights
  w_shape=layer_dims
  w=init_smooth_params(in_dim,layer_dims)
  randomise_weights(w,k)

  #Reparameterisation (here:adam) variables
  m=SimpleNamespace()
  m.fp=init_smooth_params(in_dim,layer_dims) #average of gradients
  m.fn=init_smooth_params(in_dim,layer_dims)
  m.v_fn=0. #averages of gradient norms
  m.v_fp=0.
  m.beta1=.9 #gradient and norm averaging rates
  m.beta2=.999
  m.eps=1e-8 #for numerical stability
  m.lr=1e-4 #learning rate

  #Binary performance tracking
  c=SimpleNamespace()
  c.fp=.5
  c.fn=.5
  c.target_fp=.1
  c.target_fn=.001
  c.max_gradient_ratio=1e-6
  c.binomial_averaging_tolerance=.1
  c.U=c.V=0
  return SimpleNamespace(w=w,m=m,c=c,w_shape=w_shape,in_dim=in_dim)

def get_thresh(target_p,weights,batch_size,key,forward,in_dim):
  #Choose a threshold that constrains the proportion of the random data marked +
  # estimated p is roughly p+N(0,p(1-p)/n) so use dataset size n>>1/p
  #Here: take n=int(batch_size/p)
  thresholding_data=normal(key,(int(batch_size/target_p),in_dim))
  raw_target=forward(weights,thresholding_data).sort()
  thresh=raw_target[-int(target_p*batch_size)]
  return thresh
