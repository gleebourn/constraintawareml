from pickle import load,dump
from pathlib import Path
from imblearn.combine import SMOTETomek,SMOTEENN
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.under_sampling import NearMiss
from numpy.random import default_rng

resamplers={'SMOTETomek':SMOTETomek,'SMOTEENN':SMOTEENN,'SMOTE':SMOTE,
            'ADASYN':ADASYN,'NearMiss':NearMiss}
class Resampler:
  def __init__(self,X,Y,pkl_dir,ds_name,seed,logf=None,p=None):
    self.logf=logf
    self.X={'':{l:X for l in Y} if not isinstance(X,dict) else X}
    self.Y={'':Y}
    self.pkl_dir=Path(pkl_dir)
    self.current=''
    self.rng=default_rng(seed=seed)
    self.p={} if p is None else {'':p}
    self.ds_name=ds_name

  def get_resampled(self,l):
    return self.X[self.current][l],self.Y[self.current][l]

  def get_p(self,l,resampler=None):
    resampler=self.current if resampler is None else resampler
    if not resampler in self.p:
      self.p[resampler]={l:Y.mean() for l,Y in self.Y[resampler].items()}
    return self.p[resampler][l]

  def set_resampler(self,resampler):
    self.current=resampler
    if resampler in self.X:
      return
    pkl=self.pkl_dir/(self.ds_name+'_'+resampler+'.pkl')
    if pkl.exists():
      with pkl.open('rb') as fd:
        self.X[resampler],self.Y[resampler]=load(fd)
    else:
      kwa={'random_state':self.rng.integers(2**32)}
      self.X[resampler]={}
      self.Y[resampler]={}
      for l in self.X['']:
        try:
          rs=resamplers[resampler](**kwa)
        except:
          print(resampler,'doesn\'t take a random seed...',file=self.logf)
          kwa={}
          rs=resamplers[resampler]()
        self.X[resampler][l],self.Y[resampler][l]=rs.fit_resample(self.X[''][l],self.Y[''][l])
      with pkl.open('wb') as fd:
        dump((self.X[resampler],self.Y[resampler]),fd)
