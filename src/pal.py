import numpy as np
import pandas as pd
from numpy import zeros,ones
from sklearn.metrics import classification_report, accuracy_score

from learner import FunctionalLearner,ComposedLearner,Job

def mk_synthetic_linear(n_cols,n_rows,er=.01,scheme='iidunif'):
  if scheme=='iidunif':
    X_synthetic=(2*np.random.rand(n_rows,n_cols)-1)
    actual=(2*np.random.rand(n_cols)-1)
    noise=er*(2*np.random.rand(n_rows)-1)
    y_synthetic=2*(X_synthetic.dot(actual)+noise > 0)-1

  if scheme=='walkingmodel':
    X_synthetic=(2*np.random.rand(n_rows,n_cols)-1)
    y_synthetic=np.zeros(n_rows)
    actual=np.random.normal(size=n_cols)
    actual/=np.linalg.norm(actual)
    for i in range(n_rows):
      y_synthetic[i]=actual.dot(X_synthetic[i,:])
      actual+=er*np.random.normal(size=n_cols)
      actual/=np.linalg.norm(actual)
  return X_synthetic,2*(y_synthetic>0)-1

def print_report(a,a_pred):
  print(f"Test Set Accuracy : {accuracy_score(a[1:], a_pred[1:]) * 100} %\n\n")
  print(f"Classification Report : \n\n{classification_report(a[1:], a_pred[1:])}")

def PAL_inf(p,x):
  return np.sign(p.dot(x))

def PAL_req(p,x,y,y_pred=None): # This isn't used and may need further thought
  if y_pred is None:
    y_pred=PAL_inf(p,x)
  return x+(y-y_pred)*p

def PAL_upd(p,x,y,y_pred=None):
  if y_pred is None:
    y_pred=PAL_inf(p,x)
  loss=1-y*y_pred
  l2=x.dot(x)**.5
  eta=loss/l2
  return p+eta*y*x

def PAL(data,y_vec):
  if not isinstance(data,pd.DataFrame):
    data = pd.DataFrame(data)
    y_vec = pd.Series(y_vec)

  n_features=len(data.columns)
  y_pred=ones(len(y_vec))

  pal=FunctionalLearner(PAL_inf,PAL_req,PAL_upd,zeros(n_features))
  j=Job(pal)

  for i in range(len(data)):
    j.set_x(pal,data.iloc[i,:])
    j.set_y(pal,y_vec.iloc[i])

    y_pred[i] = pal.infer(j)
    pal.update(j)

  return y_pred,pal.p

data,y_vec=mk_synthetic_linear(50,1000,scheme='walkingmodel',er=.001)
y_pred,w = PAL(data,y_vec)

print_report(y_vec,y_pred)