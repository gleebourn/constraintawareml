from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor,RandomForestClassifier
from sklearn.svm import NuSVC
from numpy import zeros,concatenate
from cal.mt import MultiTrainer

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

skl={'RandomForestRegressor':RandomForestRegressor,'RandomForestClassifier':RandomForestClassifier,
     'HistGradientBoostingRegressor':HistGradientBoostingRegressor,'NuSVC':NuSVC}

class SKL(MultiTrainer):
  def __init__(self,skm,tfpfn,ska,p,seed=None):
    super().__init__('regressor' if 'Regressor' in skm else 'classifier')
    if isinstance(skm,str):
      skm=skl[skm]
    if not seed is None:
      ska['random_state']=seed

    if 'class_weight' in ska: #Can investigate different power law weightings
      clw=ska['class_weight']
      skas=[{**ska,'class_weight':{True:p**-clw[1]*tpn,False:(1-p)**-clw[0]/tpn}} for tpn in tfpfn]
      self.m=[skm(**ska) for ska in skas]
    else:
      self.m=skm(**ska)
    self.tfpfn=tfpfn
    self.threshes=zeros(len(self.tfpfn))
    self.p=p
    self.times=[1]

  def fit(self,X,Y,X_raw=None,Y_raw=None):
    if X_raw is None:
      X_raw,Y_raw=X,Y
    if self.type=='regressor':
      self.m.fit(X,Y)
      for tfpfn,thresh in get_threshes(self.tfpfn,Y_raw,self.predict_smooth(X_raw)[1],self.p):
        self.threshes[self.tfpfn==tfpfn]=thresh
    else:
      [n.fit(X,Y) for n in self.m]

  def predict_smooth(self,X):
    if self.type=='classifier': raise Exception('No smooth output if the underlying sk model is a classifier')
    return {1:self.m.predict(X)}

  def predict(self,X):
    if self.type=='regressor':
      return {1:self.predict_smooth(X)[1]>self.threshes.reshape(-1,1)}
    else:
      return {1:concatenate([n.predict(X).reshape(1,-1) for n in self.m])}

