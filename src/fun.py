from learner import FunctionalLearner
from pal import mk_synthetic_linear
import numpy as np

X,y=mk_synthetic_linear(50,1000,scheme='ndwalkingmodel')
print(X[0],y[0])

def linear_inf(p,x):
  return p.dot(x)

def linear_upd(p,x,y,y_pred=None):
  if y_pred is None:
    y_pred=linear_inf(p,x)
  loss=1-y*y_pred
  l2=x.dot(x)**.5
  eta=loss/l2
  return p+eta*y.T.dot(x)
  
def linear_req(p,x,y,y_pred=None)
 return x+(y-y_pred).dot(p.T)