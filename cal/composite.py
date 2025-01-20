from numpy import array,sum as nsm
from types import SimpleNamespace
from math import log,exp,isnan

from jax import grad
from jax.nn import sigmoid,softmax
from jax.numpy import zeros,dot,vectorize,maximum,concatenate,max as nmx

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
  #Nonlinear weighted average
  #covers all possible p values but should take care to justify
  #Slow down averaging with the learning rate
  c.fp_amnt=c.binomial_averaging_tolerance*min(1,conf.l*min(c.fp,1-c.fp))#*c.lr
  c.fn_amnt=c.binomial_averaging_tolerance*min(1,conf.l*min(c.fn,1-c.fn))#*c.lr

  c.fp=(1-c.fp_amnt)*c.fp+c.fp_amnt*c._fp
  c.fn=(1-c.fn_amnt)*c.fn+c.fn_amnt*c._fn
  approach_fp=approach_fn=0
  U,V=c.fp/c.target_fp,c.fn/c.target_fn
  #c.lr=4*max(c.fp,c.fn)**.5 May do...
  best_c=min(c.fp,c.fn) # lr stays low if sudden jump in only one confusion statistic
  #c.lr=1#2*min(1,-1/log(max_conf*(1-max_conf)))
  #c.lr=min(1,-2/log(max_conf*(1-max_conf)))
  #c.lr=min(1,-4/log(min_conf*(1-min_conf))) #Decay slowly with performance
  #c.lr=1e-3*min(1,-2/log(best_c*(1-best_c))) #Decay slowly with performance
  #c.lr=.1
  #worst=max(U,V)
  #if worst<1:
  #  c.lr*=exp(1-1/worst)

  best=min(U,V)
  if best<1: #Use softmax if one of the quantities are in the desired zone
    s=softmax(array([-1/U,-1/V]))
    U,V=best*U+(1-best)*s[0],best*V+(1-best)*s[0]
  euc=(U**2+V**2)**.5
  U,V=U/euc,V/euc
  #c.U=c.lr*U#*U#/(1+1/mUmV)
  #c.V=c.lr*V#*V#/(1+mUmV)
  c.U=c.lr*U#*U#/(1+1/mUmV)
  c.V=c.lr*V#*V#/(1+mUmV)
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
    #wmc.m.dfp[k]+=(1-wmc.m.beta1)*wmc.d.dfp[k]
    #wmc.m.dfn[k]+=(1-wmc.m.beta1)*wmc.d.dfn[k]
    size_dfp[k]=nsm(wmc.d.dfp[k]**2)
    size_dfn[k]=nsm(wmc.d.dfn[k]**2)
    wmc.m.v_fp+=(1-wmc.m.beta2)*size_dfp[k]
    wmc.m.v_fn+=(1-wmc.m.beta2)*size_dfn[k]
  #div_fp=(wmc.m.eps+wmc.m.v_fp)
  #div_fn=(wmc.m.eps+wmc.m.v_fn)
  div=(wmc.m.eps+wmc.m.v_fn+wmc.m.v_fp)
  #mult_fp=wmc.c.U/(wmc.m.eps+wmc.m.v_fp**.5)
  #mult_fn=wmc.c.V/(wmc.m.eps+wmc.m.v_fn**.5)
  max_recent_deltas=max(wmc.m.norm_step[-10:])
  min_recent_deltas=max(wmc.m.norm_step[-10:])
  delta_size=0
  #delta_max_next=0
  dw_l2=0
  for k in wmc.w:
    #delta=wmc.c.U*wmc.m.dfp[k]/div_fp+wmc.c.V*wmc.m.dfn[k]/div_fn
    delta=(wmc.m.dfp[k]+wmc.m.dfn[k])/div
    #delta=wmc.m.dfp[k]*mult_fp+wmc.m.dfn[k]*mult_fn
    wmc.m.w_l2[k]=nsm(wmc.w[k]**2)
    wmc.m.dw_l2[k]=nsm(delta**2)
    dw_l2+=wmc.m.dw_l2[k]
    wmc.w[k]-=delta
  wmc.m.norm_step.append(dw_l2)
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

def randomise_weights(w,k,sigma_w=1.,sigma_b=1.):
  k=split(k)[0]
  for i in w:
    w[i]=normal(k,w[i].shape)
    try:
      #w[i]/=w[i].shape[1]#*=w[i].shape[1]**-.5
      w[i]*=sigma_w*w[i].shape[1]**-.5
    except (AttributeError,IndexError):
      w[i]*=sigma_b
    k=split(k)[0]

def empirical_network_weight_variance(forward,w,v,in_dim,key,sigma_min=.1,sigma_max=10):
  w_rescaled=dict()
  #Variance may not be monotone but assume it is close enough for large networks.
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


def init_model_params(in_dim,layer_dims,k,sigma_w=1.,sigma_b=1.,optimise_variance=False,
                      target_fp=.001,target_fn=.1):
  #Smooth map weights
  w_shape=layer_dims
  w=init_smooth_params(in_dim,layer_dims)
  randomise_weights(w,k,sigma_w=sigma_w,sigma_b=sigma_b)

  #Reparameterisation (here:adam) variables
  m=SimpleNamespace()
  m.dfp=init_smooth_params(in_dim,layer_dims) #average of gradients
  m.dfn=init_smooth_params(in_dim,layer_dims)
  m.w_l2={k:0. for k in w}
  m.dw_l2={k:0. for k in w}
  m.delta_max=1
  m.v_fn=0. #averages of gradient norms
  m.v_fp=0.
  m.beta1=.9 #gradient and norm averaging rates
  m.beta2=.9
  m.eps=1e-8 #for numerical stability
  m.norm_step=[1]

  #Binary performance tracking
  c=SimpleNamespace()
  c.fp=.5
  c.fn=.5
  c.target_fp=target_fp
  c.target_fn=target_fn
  c.target_max=max(c.target_fp,c.target_fn)
  c.target_min=min(c.target_fp,c.target_fn)
  c.binomial_averaging_tolerance=.1#.01#.01#.1#.2
  c.min_gradient_ratio=c.target_fp*c.target_fn
  c.U=c.V=0
  c.lr=.001#.1#.01
  return SimpleNamespace(w=w,m=m,c=c,w_shape=w_shape,in_dim=in_dim)

def get_thresh(target_p,weights,tolerance,key,forward,in_dim):
  bs=10000
  num_samples=1/(tolerance**2*target_p)
  y_raw=[]
  for i in range(int(num_samples/bs)+1):
    key=split(key)[0]
    x=normal(key,(bs,in_dim))
    y_raw.append(forward(weights,x))
  y_raw=concatenate(y_raw)
  thresh=y_raw.sort()[-int(target_p*num_samples)]
  return thresh
