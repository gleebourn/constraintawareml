from pickle import load,dump
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
from numpy import unique as npunique,min as nmn,number,max as nmx,log10 as npl10,log
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pandas import read_pickle,read_parquet,concat,get_dummies,read_csv

def dl_rbd24(data_dir=str(Path.home())+'/data',
             data_url='https://zenodo.org/api/records/13787591/files-archive',
             rm_redundant=True,large_rescale_factor=10,logf=None):
  data_dir=Path(data_dir)
  rbd24_dir=data_dir/'rbd24'
  parquet_dir=rbd24_dir/'parquet'
  if not data_dir.is_dir():
    data_dir.mkdir()
  if not rbd24_dir.is_dir():
    rbd24_dir.mkdir()
  if not parquet_dir.is_dir():
    parquet_dir.mkdir()
    print('Downloading rbd24...',file=logf)
    zip_raw=urlopen(data_url).read()
    with ZipFile(BytesIO(zip_raw),'r') as z:
      print('Extracting zip...',file=logf)
      z.extractall(parquet_dir)
    print('rbd24 extracted successfully',file=logf)
  else:
    print('rbd already extracted',file=logf)
  return rbd24_dir

def unsw(csv_train=Path.home()/'data'/'UNSW_NB15_training-set.csv',
         csv_test=Path.home()/'data'/'UNSW_NB15_testing-set.csv',
         rescale='log',numeric_only=False,verbose=False,logf=None,
         random_split=None,lab_cat=True):
  _df_train=read_csv(csv_train)
  _df_test=read_csv(csv_test)
  if lab_cat:
    y_train=_df_train['attack_cat']
    y_test=_df_test['attack_cat']
    if verbose:
      print('Train:',file=logf)
      print(y_train.value_counts(),file=logf)
      print('Test:',file=logf)
      print(y_test.value_counts(),file=logf)
  else:
    y_train=df_train['label']
    y_test=df_test['label']
  df_train=_df_train.drop(['id','attack_cat'],axis=1)
  df_test=_df_test.drop(['id','attack_cat'],axis=1)
  if numeric_only:
    df_train=df_train.select_dtypes(include=number)
    df_test=df_test.select_dtypes(include=number)
  else:
    df_train=get_dummies(df_train)#.__array__().astype(float)
    df_test=get_dummies(df_test)#.__array__().astype(float)
  x_train=df_train.drop(['label'],axis=1)
  x_test=df_test.drop(['label'],axis=1)

  train_cols=set(x_train.columns)
  test_cols=set(x_test.columns)
  diff_cols=train_cols^test_cols
  common_cols=list(train_cols&test_cols)
  x_test=x_test[common_cols]
  x_train=x_train[common_cols]
  if not(random_split is None):
    x_train,x_test,y_train,y_test=train_test_split(concat([x_train,x_test]),concat([y_train,y_test]),
                                                   test_size=.3,random_state=random_split)
  x_train=x_train.__array__().astype(float)
  x_test=x_test.__array__().astype(float)
  if not lab_cat:
    y_train=y_train.__array__().astype(bool)
    y_test=y_test.__array__().astype(bool)
  if rescale:
    if rescale=='log':
      if verbose:print('rescaling x<-log(1+x)',file=logf)
      x_test=log(1+x_test)
      x_train=log(1+x_train)
    else:
      if verbose:print('rescaling x<(x-E(x))/V(x)**.5',file=logf)
      sc=StandardScaler()
      sc.fit(x_train)
      x_test=sc.transform(x_test)
      x_train=sc.transform(x_train)
  if verbose:
    print('New x_train.min(),x_test.min():',x_train.min(),x_test.min(),file=logf)
    print('New x_train.max(),x_test.max():',x_train.max(),x_test.max(),file=logf)
    print('Differing columns:',list(diff_cols),file=logf)
    print('Common cols:',*common_cols,file=logf)
  return (x_train,y_train),(x_test,y_test),(_df_train,_df_test),sc if rescale else None #=='standard' else rescale)

def rbd24(preproc=True,split_test_train=True,rescale='log',single_dataset=False,random_split=None,
          raw_pickle_file=str(Path.home())+'/data/rbd24/rbd24.pkl',categorical=True,logf=None,
          processed_pickle_file=str(Path.home())+'/data/rbd24/rbd24_proc.pkl',verbose=False):
  processed_pickle_file=Path(processed_pickle_file)
  raw_pickle_file=Path(raw_pickle_file)
  if split_test_train and preproc and rescale=='log' and\
  processed_pickle_file.is_file() and not single_dataset:
    if verbose:print('Loading processed log rescaled pickle...',file=logf)
    with processed_pickle_file.open('rb') as fd:
      return load(fd)
  elif raw_pickle_file.is_file():
    if verbose:print('Loading raw pickle...',file=logf)
    df=read_pickle(raw_pickle_file)
  else:
    rbd24_dir=dl_rbd24()
    categories=(rbd24_dir/'parquet').listdir()
    dfs=[read_parquet(rbd24_dir/'parquet'/n) for n in categories]
    for df,n in zip(dfs,categories):
      df['category']=n.split('.')[0]
    df=concat(dfs)
    if verbose:print('Writing raw pickle...',file=logf)
    df.to_pickle(raw_pickle_file)

  if single_dataset:
    df=df[df.category==single_dataset].drop(['category'],axis=1)
  if preproc:
    df=preproc_rbd24(df,rescale_log=rescale=='log',verbose=verbose)
  if not split_test_train:
    return df
  if categorical:
    x=get_dummies(df.drop(['label','user_id','timestamp'],axis=1))
  else:
    x=df.drop(['entity','user_id','timestamp',
               'ssl_version_ratio_v20','ssl_version_ratio_v30',
               'label'],axis=1)
  x_cols=x.columns
  x=x.__array__().astype(float)
  y=df.label.__array__().astype(bool)
  if random_split is None:
    l=len(y)
    split_point=int(l*.7)
    x_train,x_test=x[:split_point],x[split_point:]
    y_train,y_test=y[:split_point],y[split_point:]
  else:
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=random_split)
  if split_test_train and preproc and rescale=='log' and not single_dataset:
    if verbose:print('Saving processed log rescaled pickle...',file=logf)
    with processed_pickle_file.open('wb') as fd:
      dump(((x_train,y_train),(x_test,y_test),(df,x_cols)),fd)
  if rescale=='standard':
    sc=StandardScaler()
    sc.fit(x_train)
    x_train=sc.transform(x_train)
    x_test=sc.transform(x_test)
  return (x_train,y_train),(x_test,y_test),(df,sc)

def load_ds(dataset,rescale='standard',verbose=False,random_split=None,categorical=True,
            single_dataset=None,lab_cat=False,logf=None):
  if dataset=='unsw':
    (X_trn,Y_trn),(X_tst,Y_tst),_,sc=unsw(rescale=rescale,verbose=verbose,logf=logf,
                                          random_split=random_split,lab_cat=lab_cat)
  elif dataset=='rbd24':
    if lab_cat:
      print('WARNING: lab_cat will have no effect on rbd24',file=logf)
    (X_trn,Y_trn),(X_tst,Y_tst),(_,sc)=rbd24(rescale=rescale,random_split=random_split,logf=logf,
                                             categorical=categorical,single_dataset=single_dataset)
  return X_trn,Y_trn,X_tst,Y_tst,sc

def preproc_rbd24(df,split_test_train=True,rm_redundant=True,check_large=False,
                  check_redundant=False,rescale_log=False,verbose=False,logf=None):
  n_cols=len(df.columns)
  if rm_redundant: check_redundant=True
  if rescale_log: check_large=True
  if check_redundant or check_large:
    if rm_redundant: redundant_cols=[]
    if rescale_log:
      large_cols=[]
      maximums=[]
      distinct=[]
    for c in [c for c in df.columns if df.dtypes[c] in [int,float]]:
      feat=df[c].__array__().astype(float)
      a=nmn(feat)
      b=nmx(feat)
      if rm_redundant and a==b:
        redundant_cols.append(c)
      elif rescale_log and b>1:
        large_cols.append(c)
        maximums.append(float(b))
        distinct.append(len(npunique(feat)))
    if check_redundant and verbose:
      print('Redundant columns (all values==0):',file=logf)
      print(', '.join(redundant_cols),file=logf)
    if check_large and verbose:
      print('Large columns (max>1):',file=logf)
      print('name,maximum,n_distinct',file=logf)
      for c,m,distinct in zip(large_cols,maximums,distinct):
        print(c,',',m,',',distinct,file=logf)
  if rm_redundant:
    df=df.drop(redundant_cols,axis=1)
  if rescale_log:
    max_logs=max([npl10(1+m) for m in maximums])
    if verbose and max_logs>10:
      print('Note that rescale factor will not map values to be <=1',file=logf)
      print('Largest value:',max_logs,'>',10,file=logf)
    for c in large_cols:
      df[c]=npl10(1+df[c].__array__())/10
  return df.sort_values('timestamp')
