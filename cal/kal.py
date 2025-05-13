from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.svm import SVR
from numpy import zeros,concatenate
from cal.mt import MultiTrainer

def get_cutoffs(tfpfns,y,yp,p):
  ypy=sorted(zip(yp,y))
  tfpfns=sorted(tfpfns)
  delta_p=1/len(ypy)
  fp=1-p
  fn=0
  for cutoff,y in ypy:
    if y:
      fn+=delta_p
    else:
      fp-=delta_p
    if fp<tfpfns[-1]*fn:
      yield tfpfns[-1],cutoff
      tfpfns=tfpfns[:-1]
      if not tfpfns:
        break

#Focus on regressors and find a likelihood with no class weighting
sk_spec={'RandomForestRegressor':(RandomForestRegressor,{'n_jobs':-1}),
         'HistGradientBoostingRegressor':(HistGradientBoostingRegressor,{'n_jobs':-1}),
         'SVR':(SVR,{'kernel':'poly'})}

skl={k:v[0] for k,v in sk_spec.items()}
skp={k:v[1] for k,v in sk_spec.items()}

class SKL(MultiTrainer):
  def __init__(self,skm,tfpfn,ska,p,seed):
    super().__init__()
    if isinstance(skm,str):
      skm=skl[skm]
    ska['random_state']=seed
    self.m=skm(**ska)
    self.tfpfn=tfpfn
    self.cutoffs=.5
    self.p=p
    self.times=[1]

  def fit(self,X,Y,X_raw=None,Y_raw=None):
    if X_raw is None:
      X_raw,Y_raw=X,Y
    self.m.fit(X,Y)
    cutoff=zeros(len(self.tfpfn))
    for tfpfn,co in get_cutoffs(self.tfpfn,Y_raw,self.predict_smooth(X_raw)[1],self.p):
      cutoff[self.tfpfn==tfpfn]=co
    self.cutoff=cutoff.reshape(-1,1)

  def predict_smooth(self,X):
    return {1:self.m.predict(X)}
