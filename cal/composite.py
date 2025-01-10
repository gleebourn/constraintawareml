from numpy import array,sum as nsm
from types import SimpleNamespace
from math import log,exp,isnan

from jax import grad
from jax.nn import sigmoid,softmax
from jax.numpy import zeros,dot,vectorize,maximum

from jax.random import normal,key,split

def threshold_f(x,thresh=.5):
    return x>thresh
def threshold_b(y_tilde,y):
  y_p=y_tilde>.5
  l=len(y_p)
  return SimpleNamespace(fp=nsm(y_p&(~y)),fn=nsm((~y_p)&y),
                         tp=nsm(y_p&y),tn=nsm(~(y_p|y)),l=l)

#Track moving averages
def moving_averages_b(c,conf):
  
  c._fp=conf.fp/conf.l
  c._fn=conf.fn/conf.l
  c._tp=conf.tp/conf.l
  c._tn=conf.tn/conf.l
  c.fp_amnt=min(c.binomial_averaging_tolerance,min(c.fp,1-c.fp)**1.5*conf.l)
  c.fn_amnt=min(c.binomial_averaging_tolerance,min(c.fn,1-c.fn)**1.5*conf.l)

  c.fp=(1-c.fp_amnt)*c.fp+c.fp_amnt*c._fp
  c.fn=(1-c.fn_amnt)*c.fn+c.fn_amnt*c._fn
  approach_fp=approach_fn=0
  #fp_ok=c.fp<c.target_fp
  #fn_ok=c.fn<c.target_fn
  #if fp_ok and not fn_ok:
  #  approach_fp=c.fp
  #elif not fp_ok and fn_ok:
  #  approach_fn=c.fn
  #U,V=(c.fp-approach_fp)/c.target_fp,(c.fn-approach_fn)/c.target_fn
  #U,V=c.fp/c.target_fp,c.fn/c.target_fn
  U,V=c.target_fn*c.fp,c.target_fp*c.fn #Rescaling
  #if U<1: U**=1/U #Accelerate decay of learning when in good region
  #if V<1: V**=1/V
  if U/V<c.min_gradient_ratio: #Make smaller
    V=U/c.min_gradient_ratio
  elif U/V>1/c.min_gradient_ratio:
    U=V/c.min_gradient_ratio
  mUmV=exp(U/V-V/U) if min(U,V)<c.target_fp*c.target_fn else 1
  #U**=1/mUmV if U<1 else mUmV
  #V**=mUmV if V<1 else 1/mUmV
  c.U=.01*U/(1+1/mUmV)
  c.V=.01*V/(1+mUmV)
  return c

def init_smooth_params(in_dim,layer_dims):
  n_layers=len(layer_dims)
  w=dict()
  for i,out_dim in enumerate(layer_dims):
    w[('A',i)]=zeros(shape=(out_dim,in_dim))
    w[('b',i)]=zeros(shape=out_dim)
    in_dim=out_dim
  return w

def mk_smooth_f(layer_dims,activation='sigmoid'):
  if activation=='sigmoid':
    its=len(layer_dims)
    def smooth_f_unbatched(w,x):
      for i in range(its):
        x=sigmoid(dot(w[('A',i)],x)+w[('b',i)])-.5
      return x[0]+.5
    ret=vectorize(smooth_f_unbatched,excluded=[0],signature='(n)->()')
  elif activation=='softmax':
    its=len(layer_dims)-1
    def smooth_f_unbatched(w,x):
      for i in range(its):
        x=softmax(dot(w[('A',i)],x)+w[('b',i)])
      return sigmoid(dot(w[('A',its)],x))[0]
    ret=vectorize(smooth_f_unbatched,excluded=[0],signature='(n)->()')
  return ret

  
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
  size_dfp=dict()
  size_dfn=dict()
  for k in wmc.d.dfp:
    wmc.m.dfp[k]*=wmc.m.beta1
    wmc.m.dfn[k]*=wmc.m.beta1
    wmc.m.dfp[k]+=wmc.c.U*(1-wmc.m.beta1)*wmc.d.dfp[k]
    wmc.m.dfn[k]+=wmc.c.V*(1-wmc.m.beta1)*wmc.d.dfn[k]
    size_dfp[k]=nsm(wmc.d.dfp[k]**2)
    size_dfn[k]=nsm(wmc.d.dfn[k]**2)
    wmc.m.v_fp+=(1-wmc.m.beta2)*size_dfp[k]
    wmc.m.v_fn+=(1-wmc.m.beta2)*size_dfn[k]

  div_fp=(wmc.m.eps+wmc.m.v_fp)
  div_fn=(wmc.m.eps+wmc.m.v_fn)
  #div_fp=(wmc.m.eps+wmc.m.v_fp**.5)
  #div_fn=(wmc.m.eps+wmc.m.v_fn**.5)
  max_recent_deltas=max(wmc.m.norm_step[-10:])
  min_recent_deltas=max(wmc.m.norm_step[-10:])
  delta_size=0
  for k in wmc.w:
    #delta=wmc.m.lr*(wmc.m.dfp[k]/div_fp+wmc.m.dfn[k]/div_fn)
    delta=wmc.m.dfp[k]/div_fp+wmc.m.dfn[k]/div_fn
    div_l2=(size_dfp[k]+size_dfn[k])**.5
    #if div_l2<1wmc.m.lr**2: #I gradient banishing, decay params
    #  w_abs=maximum(abs(wmc.w[k]),1)/(1+div_l2)
    #  wmc.w[k]*=w_abs**-.1*wmc.m.lr #Nonlinearly punish large coeffs
    #elif div_l2>1: #I gradient banishing, decay params
    #  delta/=div_l2

    delta_size+=nsm(delta*wmc.w[k]) #Related to change in L2 norm
    wmc.w[k]-=delta#/(wmc.m.eps+max_recent_deltas)

  #if max_recent_deltas<wmc.m.lr**2:
  #  mult=(wmc.m.lr_max-wmc.m.lr)
  #  #wmc.m.lr+=wmc.m.lr*(wmc.m.lr_max-wmc.m.lr)**2#keep 0<lr<lrmax
  #  wmc.m.lr*=(1+mult**(1+mult))
  #elif min_recent_deltas**2>wmc.m.lr:
  #  #wmc.m.lr-=(wmc.m.lr_max-wmc.m.lr)*wmc.m.lr**2
  #  wmc.m.lr*=(1-wmc.m.lr**(1+wmc.m.lr))
  #wmc.m.lr*=(1+(wmc.m.lr_max-wmc.m.lr)**(wmc.m.lr_max-wmc.m.lr+1))**(
  #           /(1+wmc.m.lr**(wmx.m.lr+1)))**\
  #           (min_recent_deltas-max_recent_deltas-.5*wmc.m.lr)

  wmc.m.norm_step.append(abs(float(delta_size)))

  if isnan(wmc.m.norm_step[-1] ):
    print('Encountered nans!')
    print(wmc.m)
    exit(1)
  return wmc

def mk_model_f(layer_dims,activation='sigmoid'):
  smooth_f=mk_smooth_f(layer_dims,activation=activation)
  def model_f(wmc,x):
    return threshold_f(smooth_f(wmc.w,x))
  return model_f

def mk_model_b(layer_dims):
  smooth_f=mk_smooth_f(layer_dims)
  smooth_b=mk_smooth_b(layer_dims)
  def model_b(wmc,x,y):
    y_tilde=smooth_f(wmc.w,x)
    conf=threshold_b(y_tilde,y)
    wmc.c=moving_averages_b(wmc.c,conf)
    wmc.d=smooth_b(wmc.w,x,y)
    wmc=reparameterisation_u(wmc)
    return wmc
  return model_b

def randomise_weights(w,k,sigma=1.):
  for i in w:
    try:
      w[i]=normal(k,w[i].shape)*sigma
      #if len(w[i].shape)==2:
      #  w[i]*=w[i].shape[1]**-.5
    except AttributeError:
      w[i]=normal(k)*sigma
    k=split(k)[0]

def empirical_network_weight_variance(forward,w,v,in_dim,key,sigma_min=.1,sigma_max=10):
  w_rescaled=dict()
  #Variance may not be monotone but assume it is close enough for larfe networks.
  key,s=split(key)
  sigmas=[]
  vs=[]
  key,s=split(key)
  for i in range(sigma_max):
    randomise_weights(w,key)
    for j in range(sigma_max**2):
      key,s=split(key)
      x=normal(s,(sigma_max**2,in_dim))
      sigma=sigma_min
      var=0
      var_diff=-1
      sigma_best=sigma
      var_diff_best=1
      while var_diff<0 and sigma<sigma_max:
        var_l=var
        sigma*=1+sigma_min
        for k in w:
          if len(w[k].shape)==2:
            w_rescaled[k]=w[k]*sigma
          else:
            w_rescaled[k]=w[k]
        y=forward(w_rescaled,x)
        var=nsm(y**2)/len(y)-(nsm(y)/len(y))**2
        var_diff=var-v
        if abs(var_diff)<var_diff_best:
          var_diff_best=abs(var_diff)
          sigma_best=sigma
      sigmas.append(sigma_best)
      vs.append(var)
  return sum(sigmas)/len(sigmas)


def init_model_params(in_dim,layer_dims,k,optimise_variance=False):
  #Smooth map weights
  w_shape=layer_dims
  w=init_smooth_params(in_dim,layer_dims)
  randomise_weights(w,k)
  if optimise_variance:
    forward=mk_smooth_f(layer_dims)
    sigma_model=1.#empirical_network_weight_variance(forward,w,.1,in_dim,key)
    for k in w:
      if(w[k].shape)==2:
        w[k]*=sigma_model

  #Reparameterisation (here:adam) variables
  m=SimpleNamespace()
  m.dfp=init_smooth_params(in_dim,layer_dims) #average of gradients
  m.dfn=init_smooth_params(in_dim,layer_dims)
  m.v_fn=0. #averages of gradient norms
  m.v_fp=0.
  m.beta1=.9 #gradient and norm averaging rates
  m.beta2=.999
  m.eps=1e-8 #for numerical stability
  m.norm_step=[1]
  #m.lr=.1# learning rate - use to compute $U$ and $V$
  #m.lr_max=.5

  #Binary performance tracking
  c=SimpleNamespace()
  c.fp=.5
  c.fn=.5
  c.target_fp=.001
  c.target_fn=.1
  c.binomial_averaging_tolerance=.01
  c.min_gradient_ratio=max(c.target_fp,c.target_fn)
  #if c.min_gradient_ratio>0:c.min_gradient_ratio**=-1
  c.U=c.V=0
  return SimpleNamespace(w=w,m=m,c=c,w_shape=w_shape,in_dim=in_dim)

def get_thresh(target_p,weights,tolerance,key,forward,in_dim):
  num_samples=1/(tolerance*target_p**3)
  x=normal(key,(int(num_samples),in_dim))
  y_raw=forward(weights,x).sort()
  thresh=y_raw[-int(target_p*num_samples)]
  return thresh
