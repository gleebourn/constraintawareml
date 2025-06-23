from pickle import load,dump
from pathlib import Path
from imblearn.combine import SMOTETomek,SMOTEENN
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.under_sampling import NearMiss
from numpy.random import default_rng
from time import perf_counter

#class NullResampler:
#  def __init__(self):
#    pass
#  def fit_resample(X,Y):
#    return X,Y

resamplers={'NearMiss':NearMiss,'SMOTEENN':SMOTEENN,
            'ADASYN':ADASYN,'SMOTE':SMOTE,'SMOTETomek':SMOTETomek}#,'':NullResampler}

resamplers_list=list(resamplers)+['']

class Resampler:
  def __init__(self,X,Y,pkl_dir=None,ds_name='UNKNOWN',seed=0,logf=None,p=None):
    if not isinstance(Y,dict):
      Y={True:Y}
    self.logf=logf
    self.X={'':{l:X for l in Y} if not isinstance(X,dict) else X}
    self.Y={'':Y}
    self.T={'':{l:0 for l in Y}}
    if pkl_dir:
      self.pkl_dir=Path(pkl_dir)
      self.pkl_dir.mkdir(parents=True,exist_ok=True)
    else:
      self.pkl_dir=None
    self.rng=default_rng(seed=seed)
    self.p={} if p is None else {'':p}
    self.ds_name=ds_name

  def get_resampled(self,l,resampler):
    self.set_resampler(resampler)
    return self.X[resampler][l],self.Y[resampler][l]

  def get_t(self,l,resampler):
    self.set_resampler(resampler)
    return self.T[resampler][l]

  def get_p(self,l,resampler):
    self.set_resampler(resampler)
    if not resampler in self.p:
      self.p[resampler]={l:Y.mean() for l,Y in self.Y[resampler].items()}
    return self.p[resampler][l]

  def set_resampler(self,resampler):
    if resampler in self.X:
      return
    if self.pkl_dir:
      pkl=self.pkl_dir/(self.ds_name+'_'+resampler+'.pkl')
      if pkl.exists():
        with pkl.open('rb') as fd:
          self.X[resampler],self.Y[resampler],self.T[resampler]=load(fd)
        return
    print('Resampling data with method',resampler,'...',file=self.logf)
    kwa={'random_state':self.rng.integers(2**32)}
    self.X[resampler]={}
    self.Y[resampler]={}
    self.T[resampler]={}
    for l in self.X['']:
      t0=perf_counter()
      try:
        rs=resamplers[resampler](**kwa)
      except:
        print(resampler,'doesn\'t take a random seed...',file=self.logf)
        kwa={}
        rs=resamplers[resampler]()
      self.X[resampler][l],self.Y[resampler][l]=rs.fit_resample(self.X[''][l],self.Y[''][l])
      self.T[resampler][l]=perf_counter()-t0
    if self.pkl_dir:
      with pkl.open('wb') as fd:
        dump((self.X[resampler],self.Y[resampler],self.T[resampler]),fd)
