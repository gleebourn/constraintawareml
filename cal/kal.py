from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from numpy import zeros

def get_threshes(tfpfns,y,yp,p):
  ypy=sorted(zip(yp,y))
  tfpfns=sorted(tfpfns)
  delta_p=1/len(ypy)
  fp=1-p
  fn=0
  for thresh,y in ypy:
    if y:
      fn+=delta_p
    else:
      fp-=delta_p
    if fp<tfpfns[-1]*fn:
      yield tfpfns[-1],thresh
      tfpfns=tfpfns[:-1]
      if not tfpfns:
        break

skl={'RandomForestRegressor':RandomForestRegressor,
     'HistGradientBoostingRegressor':HistGradientBoostingRegressor}

class SKL:
  def __init__(self,skm,tfpfn,ska,p,seed=None):
    if isinstance(skm,str):
      skm=skl[skm]
    if not seed is None:
      ska['random_state']=seed

    self.m=skm(**ska)
    self.tfpfn=tfpfn
    self.threshes=zeros(len(self.tfpfn))
    self.p=p

  def fit(self,X,Y,X_raw=None,Y_raw=None):
    if X_raw is None:
      X_raw,Y_raw=X,Y
    self.m.fit(X,Y)
    for tfpfn,thresh in get_threshes(self.tfpfn,Y_raw,self._smooth_predict(X_raw),self.p):
      self.threshes[self.tfpfn==tfpfn]=thresh

  def _smooth_predict(self,X):
    return self.m.predict(X)

  def predict(self,X):
    return self._smooth_predict(X)>self.threshes.reshape(-1,1)

