from rsynth import mk_synthetic_linear
from pal import print_report,pal_learner
from learner import Job
import pandas as pd
import numpy as np

def PAL(data,y_vec):
  data = np.array(data)
  y_vec = np.array(y_vec)

  n_features=data.shape[1]
  y_pred=np.ones(data.shape[0])

  pal=pal_learner(n_features)
  j=Job(pal)

  for i,(x,y) in enumerate(zip(data,y_vec)):
    j.set_x(pal,x)
    y_pred[i] = pal.infer(j)

    j.set_y(pal,y)
    pal.update(j)

  return y_pred,pal.p

data,y_vec=mk_synthetic_linear(50,1000,scheme='walkingmodel',er=.001)
y_pred,w = PAL(pd.DataFrame(data),pd.Series(y_vec))

print_report(y_vec,y_pred)

print('=====================================================')
print('=====================================================')
print('=====================================================')
print('=====================================================')

