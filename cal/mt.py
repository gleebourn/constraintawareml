class MultiTrainer:
  def __init__(self,times=1,cutoff=.5):
    self.times=(times,) if isinstance(times,int) else tuple(times)
    self.naive_cutoff=self.cutoff=cutoff

  def fit(self,X,Y,X_raw=None,Y_raw=None):
    pass

  def predict(self,X,naive_cutoff=False):
    cutoff=self.naive_cutoff if naive_cutoff else self.cutoff
    if not isinstance(cutoff,dict):
      cutoff={t:cutoff for t in self.times}
    Yp_smooth=self.predict_smooth(X)
    return {t:v>cutoff[t] for t,v in Yp_smooth.items()}
    
  def predict_smooth(self,X):
    return {t:0. for t in times}
