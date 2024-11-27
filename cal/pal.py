import numpy as np
from numpy import zeros
from sklearn.metrics import classification_report, accuracy_score

from cal.learner import FunctionalLearner,ComposedLearner,Job

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
  loss=y!=y_pred
  l2=x.dot(x)**.5
  eta=loss/l2
  return p+eta*y*x

def pa_learner(n_features):
  return FunctionalLearner(PAL_inf,PAL_req,PAL_upd,zeros(n_features))
