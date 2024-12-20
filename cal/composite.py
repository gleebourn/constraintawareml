from pl import PL
from numpy import sum as nsm
from types import SimpleNamespace

#No parameters, pass back false positives and negatives
def threshold_b(p,y_tilde,y,yo=None):
  if yo:
    return None,SimpleNamespace(fp=nsm(yo&(~y)),fn=nsm(~y&(yo)))
  else:
    return None,SimpleNamespace(fp=nsm(y_tilde*(~y)),fn=nsm((1-y_tilde)*y))

threshold=PL(f=lambda p,y:y>.5,b=threshold_b)

#Track moving averages
binomial_averaging_tolerance=.1
def moving_averages_b(p,x,y):
  
  fp_amnt=min(p.fp,1-p.fp)*binomial_averaging_tolerance
  fn_amnt=min(p.fn,1-p.fp)*binomial_averaging_tolerance

  q=SimpleNamespace(c=SimpleNamespace(),d=SimpleNamespace())
  q.c.fp=(1-fp_amnt)*p.fp+fp_amnt*y.fp
  q.c.fn=(1-fn_amnt)*p.fn+fn_amnt*y.fn
  fp_ok=q.c.fp<p.target_fp
  fn_ok=q.c.fn<p.target_fn
  approach_fp=approach_fn=0
  if fp_ok and not fn_ok:
    approach_fp=q.c.fp
  elif not fp_ok and fn_ok:
    approach_fn=q.c.fn
  U,V=approach_fp/p.target_fp,approach_fn/p.target_fn
  q.d.U,q.d.V=max(U,p.max_gradient_ratio*V),max(V,p.max_gradient_ratio*U)
  return q,None

moving_averages=PL(None,moving_averages_b)

from jax import grad
from jaxn import mk_nlp
mk_params,infer=mk_nlp()
c_fp=lambda p,x,y: infer(x,p)*(~y)
c_fn=lambda p,x,y: (1-infer(x,p))*y
dfp=grad(c_fp,argnums=0)
dfn=grad(c_fn,argnums=0)

def smooth_b(p,x,y):
  return SimpleNamespace(dfp=dfp(p,x,y),dfn=dfn(p,x,y)),None

smooth=PL(lambda p,x:infer(x,p),smooth_b)
