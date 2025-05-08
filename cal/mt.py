class MultiTrainer:
  def __init__(self,m_type='regressor'):
    self.type=m_type

  def fit(self,X,Y,X_raw,Y_raw=None):
    pass

  def predict(self,X):
    pass

  def predict_smooth(self,X):
    if self.type!='regressor':
      raise NotImplementedError('Continuous predictions unavailable for algorithm type '+self.type)
