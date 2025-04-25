from pickle import load,dump
from itertools import count
from os.path import isdir,isfile
from os import mkdir,listdir,get_terminal_size
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from pathlib import Path
from select import select
from sys import stdin,stdout
from time import perf_counter
from json import dump as jump
from numpy import inf,unique as npunique,array as nparr,min as nmn,number,zeros as npzer,\
                  max as nmx,sum as nsm,log10 as npl10,round as rnd,geomspace
from numpy.random import default_rng #Only used for deterministic routines
from sklearn.preprocessing import StandardScaler
from jax.numpy import array,zeros,log,log10,maximum,minimum,\
                      concat,exp,ones,linspace,array_split,reshape,eye,\
                      concatenate,sum as jsm,max as jmx
from jax.nn import tanh,softmax
from jax.random import uniform,normal,split,key,choice,binomial,permutation
from jax.tree import map as jma,reduce as jrd
from jax import grad,value_and_grad,jit,config,vmap
from jax.lax import scan,while_loop,switch
from jax.lax.linalg import svd
from sklearn.utils.extmath import cartesian
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imshow,legend,show,scatter,xlabel,ylabel,\
                              gca,plot,title,savefig,close,rcParams,yscale
from matplotlib.patches import Patch
rcParams['font.family'] = 'monospace'
from pandas import read_pickle,read_parquet,concat,get_dummies,read_csv
from traceback import format_exc
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from csv import reader,writer,DictWriter
from imblearn.combine import SMOTETomek,SMOTEENN
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.under_sampling import NearMiss

def set_jax_cache():
  config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
  config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  config.update("jax_persistent_cache_min_compile_time_secs", 0)
  config.update("jax_persistent_cache_enable_xla_caches",
                "xla_gpu_per_fusion_autotune_cache_dir")

class KeyEmitter:
  def __init__(self,seed=1729,parents=['main','vis']):
    if isinstance(parents,dict):
      self.parents=parents
    else:
      k=key(seed)
      if isinstance(parents,list):
        self.parents={a:k for a,k in zip(parents,split(k,len(parents)))}
      elif isinstance(parents,str):
        self.parents={parents:k}
  def emit_key(self,n=None,parent='main',vis=False,report=False):
    if vis or report:
      parent='vis'
    keys=split(self.parents[parent]) if n is None else split(self.parents[parent],n+1)
    self.parents[parent]=keys[0]
    return keys[1] if n is None else keys[1:]

class TimeStepper:
  def __init__(self,clock_avg_rate=.01,time_avgs={}):
    self.clock_avg_rate=clock_avg_rate
    self.tl=perf_counter()
    self.time_avgs=time_avgs
    self.lab_len=max([0]+[len(lab) for lab in time_avgs])
  def get_timestep(self,label=False,start_immed=False):
    t=perf_counter()
    if label:
      try:
        self.time_avgs[label]+=(1+self.clock_avg_rate)*\
                               self.clock_avg_rate*\
                               float(t-self.tl)
      except:
        print('New time waypoint',label)
        self.time_avgs[label]=(1+self.clock_avg_rate)*\
                              float(t-self.tl) if\
                              start_immed else 1e-8
        self.lab_len=max(len(label),self.lab_len)
      self.time_avgs[label]*=(1-self.clock_avg_rate)
    self.tl=t
  def report(self,p=False,prec=3):
    tai=self.time_avgs.items()
    lj=max(self.lab_len,prec+5)+1
    #tsr='\n'.join([str(k)+':'+str(log10(v)) for k,v in tai])
    tsr=f_to_str(self.time_avgs,lj=lj,prec=prec)+'\n'+\
        f_to_str(self.time_avgs.values(),lj=lj,prec=prec)
    if p:
      print(tsr)
    tsrx='\n'.join(['\\texttt{'+k.replace('_','\\_')+'}&'+\
                    f_to_str(log10(v))+'\\\\\n' for k,v in tai])
    tsrx+='\\end{tabular}'
    return tsr,tsrx

def _forward(w,x,act,batchnorm=False,get_transform=False,ll_act=False,transpose=True):
  if get_transform:
    A=[]
    B=[]
  if transpose:
    l=w[:-1]
    al,bl=w[-1]
  else:
    l=zip(w[0][:-1],w[1][:-1])
    al=w[0][-1]
    bl=w[1][-1]
  for a,b in l:
    x=act(x@a+b)
    if batchnorm:
      ta=(1e-8+x.var(axis=0))**-.5
      tb=x.mean(axis=0)#*ta
      if get_transform:
        A.append(ta)
        B.append(tb)
      x=(x-tb)*ta
  y=x@al+bl
  if ll_act:
    y=act(y)
  if get_transform:
    if transpose:
      A=[a*(ta.reshape(-1,1)) for (a,_),ta in zip(w[1:],A)]
      trans=w[:1]+[(ta,b-tb@ta) for (_,b),ta,tb in zip(w[1:],A,B)]
    else:
      A=[a*(ta.reshape(-1,1)) for a,ta in zip(w[0][1:],A)]
      trans=w[0][:1]+A,w[1][:1]+[b-tb@ta for b,ta,tb in zip(w[1][1:],A,B)]
    return y,trans
  return y

forward=jit(_forward,static_argnames=['batchnorm','get_transform','ll_act','transpose','act'])
vorward=jit(vmap(_forward,(0,None,None),0),static_argnames=['act'])

activations={'tanh':tanh,'softmax':softmax,'linear':jit(lambda x:x),
             'relu':jit(lambda x:maximum(x,.03*x))}

def select_initialisation(imp,act):
  if imp in ['fm','resnet']:
    return 'resnet'
  else:
    match act:
      case 'relu':
        return 'glorot_normal'
      case 'tanh':
        return 'glorot_uniform'
      case 'linear':
        return 'ones'
      case _:
        print('Initialising weights to glorot uniform!')
        return 'glorot_uniform'

def f_to_str(X,lj=None,prec=2,p=False):
  if lj is None:
    lj=prec+6
  if not(isinstance(X,str)):
    try:
      X=float(X)
      X='{0:.{1}g}'.format(X,prec)
    except:
      pass
  if isinstance(X,str):
    if lj:
      X=X.ljust(lj)
  else:
    X=''.join((f_to_str(x,lj=lj,prec=prec) for x in X))
  if p:
    print(X)
  return X

@jit
def l2(w):
  return jrd(lambda x,y:x+(y**2).sum(),w,initializer=0.)

dl2=jit(value_and_grad(l2))

def init_layers(layer_dimensions,initialisation,k=None,mult_a=1.,n_par=None,bias=0.,
                sqrt_normalise=False,orthonormalise=False,transpose=True):
  n_steps=len(layer_dimensions)-1
  if initialisation in ['ones','zeros']:
    w_k,b_k=count(),count()
  else:
    k1,k2=split(k)
    w_k=split(k1,n_steps)
    b_k=split(k2,n_steps)
  wb=[]
  A=[]
  B=[]
  if n_par is None:
    e_dims=tuple()
  else:
    e_dims=(n_par,)
  for i,(k,l,d_i,d_o) in enumerate(zip(w_k,b_k,layer_dimensions,layer_dimensions[1:])):
    match initialisation:
      case 'zeros':
        A.append(zeros(e_dims+(d_i,d_o)))
        B.append(zeros(e_dims+(d_o,))+bias)
      case 'resnet'|'ones':
        A.append(ones(e_dims+(d_i,d_o))*mult_a)
        B.append(zeros(e_dims+(d_o,))+bias)
        initialisation='zeros' if initialisation=='resnet' else 'ones'
      case 'eye':
        if n>1: raise Exception('Not yet done this')
        A.append(eye(d_i,d_o)*mult_a)
        B.append(zeros(d_o)+bias)
      case 'glorot_uniform':
        A.append((2*(6/(d_i+d_o))**.5)*(uniform(shape=e_dims+(d_i,d_o),key=k)-.5)*mult_a)
        B.append(zeros(e_dims+(d_o,))+bias)
      case 'glorot_normal':
        A.append(((2/(d_i+d_o))**.5)*normal(shape=e_dims+(d_i,d_o),key=k)*mult_a)
        B.append(zeros(e_dims+(d_o,))+bias)
      case _:
        raise Exception('Unknown initialisation: '+initialisation)
  if orthonormalise:
    for i,(d_i,d_o) in enumerate(layer_dimensions,layer_dimensions[1:]):
      if n>1: raise Exception('Not yet done this')
      if d_i>d_o:
        A[i]=svd(A[i],full_matrices=False)[0]
      else:
        A[i]=svd(A[i],full_matrices=False)[2]
  if sqrt_normalise:
    if n>1: raise Exception('Not yet done this')
    for i,d_i in enumerate(layer_dimensions[1:]):
      A[i]/=d_i**.5
  if transpose:
    return list([a,b] for a,b in zip(A,B))
  else:
    return [A,B]

def init_layers_adam(*a,**kwa):
  w=init_layers(*a,**kwa)
  kwa['bias']=kwa.pop('bias',0)
  return {'w':w,'m':init_layers(a[0],'zeros',**kwa),
          'v':init_layers(a[0],'zeros',**kwa),'t':zeros(kwa['n_par']) if 'n_par' in kwa else 0}

@jit
def shuffle_xy(k,x,y):
  shuff=permutation(k,len(y))
  xy=x[shuff],y[shuff]
  return xy
 
def ewma(a,b,rate):return (1-rate)*a+rate*b

def ad_diff(m,v,eps):return m/(v**.5+eps)

def ad_fpfn(w,mp,vp,mn,vn,eps,lr,pn):
  return  w-lr*(ad_diff(mp,vp,eps)/pn+ad_diff(mn,vn,eps)*pn)

@jit
def upd_ad_fpfn(w,mp,vp,mn,vn,dfp,dfn,beta1,beta2,lr,pn,reg,eps):
  mp=jma(lambda old,upd:ewma(upd,old,beta1),mp,dfp)
  vp=jma(lambda old,upd:ewma(upd**2,old,beta2),vp,dfp)
  mn=jma(lambda old,upd:ewma(upd,old,beta1),mn,dfn)
  vn=jma(lambda old,upd:ewma(upd**2,old,beta2),vn,dfn)
  w=jma(lambda x,mP,vP,mN,vN:(1-reg)*(x-ad_fpfn(x,mP,vP,mN,vN,eps,lr,pn)),w,mp,vp,mn,vn)
  return w,mp,vp,mn,vn

@jit
def upd_adam(w,m,v,dw,beta1,beta2,lr,reg,eps):
  m=jma(lambda old,upd:ewma(upd,old,beta1),m,dw)
  v=jma(lambda old,upd:ewma(upd**2,old,beta2),v,dw)
  w=jma(lambda W,M,V:(1-reg)*(W-lr*ad_diff(M,V,eps)),w,m,v)
  return w,m,v

def _upd_adam_no_bias(dw,w,m,v,t,beta1,beta2,lr,reg,eps):
  m=jma(lambda old,upd:ewma(upd,old,beta1),m,dw)
  v=jma(lambda old,upd:ewma(upd**2,old,beta2),v,dw)
  t+=1
  w=jma(lambda W,M,V:(1-reg)*(W-lr*ad_diff(M/(1-beta1**t),V/(1-beta2**t),eps)),w,m,v)
  return (w,m,v,t)
upd_adam_no_bias=_upd_adam_no_bias

def dict_adam_no_bias(dw,state,consts):
  state['w'],state['m'],state['v'],state['t']=upd_adam_no_bias(dw,state['w'],state['m'],state['v'],
                                                               state['t'],consts['beta1'],consts['beta2'],
                                                               consts['lr'],consts['reg'],consts['eps'])
  return state

def dl_rbd24(data_dir=str(Path.home())+'/data',
             data_url='https://zenodo.org/api/records/13787591/files-archive',
             rm_redundant=True,large_rescale_factor=10):
  rbd24_dir=data_dir+'/rbd24'
  parquet_dir=rbd24_dir+'/parquet'
  if not isdir(data_dir):
    mkdir(data_dir)
  if not isdir(rbd24_dir):
    mkdir(rbd24_dir)
  if not isdir(parquet_dir):
    mkdir(parquet_dir)
    print('Downloading rbd24...')
    zip_raw=urlopen(data_url).read()
    with ZipFile(BytesIO(zip_raw),'r') as z:
      print('Extracting zip...')
      z.extractall(parquet_dir)
    print('rbd24 extracted successfully')
  else:
    print('rbd already extracted')
  return rbd24_dir

def unsw(csv_train=Path.home()/'data'/'UNSW_NB15_training-set.csv',
         csv_test=Path.home()/'data'/'UNSW_NB15_testing-set.csv',
         rescale='log',numeric_only=False,verbose=False,
         random_split=None,lab_cat=False):
  _df_train=read_csv(csv_train)
  _df_test=read_csv(csv_test)
  if lab_cat:
    y_train=_df_train['attack_cat']
    y_test=_df_test['attack_cat']
    if verbose:
      print('Train:')
      print(y_train.value_counts())
      print('Test:')
      print(y_test.value_counts())
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
  if rescale=='log':
    if verbose:print('rescaling x<-log(1+x)')
    x_test=log(1+x_test)
    x_train=log(1+x_train)
  elif rescale=='standard':
    if verbose:print('rescaling x<(x-E(x))/V(x)**.5')
    sc=StandardScaler()
    sc.fit(x_train)
    x_test=sc.transform(x_test)
    x_train=sc.transform(x_train)
  if verbose:
    print('New x_train.min(),x_test.min():',x_train.min(),x_test.min())
    print('New x_train.max(),x_test.max():',x_train.max(),x_test.max())
    print('Differing columns:',f_to_str(list(diff_cols)))
    print('Common cols:',*common_cols)
  return (x_train,y_train),(x_test,y_test),(_df_train,_df_test),(sc if rescale=='standard' else rescale)

def rbd24(preproc=True,split_test_train=True,rescale='log',single_dataset=False,random_split=None,
          raw_pickle_file=str(Path.home())+'/data/rbd24/rbd24.pkl',categorical=True,
          processed_pickle_file=str(Path.home())+'/data/rbd24/rbd24_proc.pkl',verbose=False):
  if split_test_train and preproc and rescale=='log' and\
  isfile(processed_pickle_file) and not single_dataset:
    if verbose:print('Loading processed log rescaled pickle...')
    with open(processed_pickle_file,'rb') as fd:
      return load(fd)
  elif isfile(raw_pickle_file):
    if verbose:print('Loading raw pickle...')
    df=read_pickle(raw_pickle_file)
  else:
    rbd24_dir=dl_rbd24()
    categories=listdir(rbd24_dir+'/parquet')
    dfs=[read_parquet(rbd24_dir+'/parquet/'+n) for n in categories]
    for df,n in zip(dfs,categories):
      df['category']=n.split('.')[0]
    df=concat(dfs)
    if verbose:print('Writing raw pickle...')
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
    if verbose:print('Saving processed log rescaled pickle...')
    with open(processed_pickle_file,'wb') as fd:
      dump(((x_train,y_train),(x_test,y_test),(df,x_cols)),fd)
  if rescale=='standard':
    sc=StandardScaler()
    sc.fit(x_train)
    x_train=sc.transform(x_train)
    x_test=sc.transform(x_test)
  return (x_train,y_train),(x_test,y_test),(df,sc)

def load_ds(dataset,rescale='standard',verbose=False,random_split=None,categorical=True,
            single_dataset=None,lab_cat=False):
  if dataset=='unsw':
    (X_trn,Y_trn),(X_tst,Y_tst),_,sc=unsw(rescale=rescale,verbose=verbose,
                                          random_split=random_split,lab_cat=lab_cat)
  elif dataset=='rbd24':
    if lab_cat:
      print('WARNING: lab_cat will have no effect on rbd24')
    (X_trn,Y_trn),(X_tst,Y_tst),(_,sc)=rbd24(rescale=rescale,random_split=random_split,
                                             categorical=categorical,single_dataset=single_dataset)
  return X_trn,Y_trn,X_tst,Y_tst,sc

def preproc_rbd24(df,split_test_train=True,rm_redundant=True,check_large=False,
                  check_redundant=False,rescale_log=True,verbose=False):
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
      print('Redundant columns (all values==0):')
      print(', '.join(redundant_cols))
    if check_large and verbose:
      print('Large columns (max>1):')
      print('name,maximum,n_distinct')
      for c,m,distinct in zip(large_cols,maximums,distinct):
        print(c,',',m,',',distinct)
  if rm_redundant:
    df=df.drop(redundant_cols,axis=1)
  if rescale_log:
    max_logs=max([npl10(1+m) for m in maximums])
    if verbose and max_logs>10:
      print('Note that rescale factor will not map values to be <=1')
      print('Largest value:',max_logs,'>',10)
    for c in large_cols:
      df[c]=npl10(1+df[c].__array__())/10
  return df.sort_values('timestamp')

def binsearch_step(r,tfpfn,y,yp_smooth):
  a,b=r
  mid=(a+b)/2
  yp=yp_smooth-mid>0
  fpfn=((yp&~y).mean()/(y&~yp).mean())
  excess_fps=fpfn>tfpfn
  return a*(~excess_fps)+excess_fps*mid,b*(excess_fps)+(~excess_fps)*mid,(fpfn-tfpfn)**2

@jit
def search_thresh(y,yp_smooth,tfpfn,adapthresh_tol):
  yp_min_max=yp_smooth.min(),yp_smooth.max()
  thresh_range=while_loop(lambda r:(r[2]>adapthresh_tol)&(r[3]<1/adapthresh_tol),
                          lambda r:binsearch_step(r[:2],tfpfn,y,yp_smooth)+(r[3]+1,),
                          (*yp_min_max,1,0))[:2]
  return (thresh_range[0]+thresh_range[1])/2

def _set_bet(consts,fp,fn):
  return exp(consts['lrfpfn']*(-fp/consts['tfp']+fn/consts['tfn']))
set_bet=jit(vmap(_set_bet,(0,0,0),0))#,None

def get_reshuffled(k,X_trn,Y_trn,n_batches,bs,last):
  X,Y=shuffle_xy(k,X_trn,Y_trn)
  return X[0:last].reshape(n_batches,bs,-1),Y[0:last].reshape(n_batches,bs)

get_reshuffled=jit(get_reshuffled,static_argnames=['n_batches','bs','last'])

def loss(w,x,y,bet,act):
  yp=forward(w,x,act)
  return jsm((bet*y+(1-y)/bet)*(1-2*y)*yp)

dl=grad(loss)
def _upd(x,y,state,consts,act):
  return dict_adam_no_bias(dl(state['w'],x,y,state['bet'],act),state,consts)

upd=vmap(_upd,(None,None,0,0,None),0)

def _steps(states,consts,X,Y,act):
  return scan(lambda s,xy:(upd(xy[0],xy[1],s,consts,act),0),states,(X,Y))[0]
steps=jit(_steps,static_argnames=['act'])

def _calc_thresh(w,tfpfn,X,Y,act,tol=1e-1):
  yps=forward(w,X,act)
  Yp_smooth=yps.flatten()
  Yp=Yp_smooth>0.
  Ypb=Yp_smooth>0
  return search_thresh(Y,Yp_smooth,tfpfn,tol),(Ypb&~Y).mean(),(Y&~Ypb).mean()

calc_thresh=jit(vmap(_calc_thresh,(0,0,None,None,None),0),static_argnames=['act','tol'])

def _nn_epochs(ks,n_epochs,bs,X,Y,X_raw,Y_raw,consts,states,act,n_batches,last):
  tfps=consts['tfp']
  tfns=consts['tfn']
  lrfpfns=consts['lrfpfn']
  lrs=consts['lr']
  regs=consts['reg']
  tfpfns=tfps/tfns
  for epoch,k in enumerate(ks,1):
    X_b,Y_b=get_reshuffled(k,X,Y,n_batches,bs,last)
    states=steps(states,consts,X_b,Y_b,act)
    thresh,fp,fn=calc_thresh(states['w'],tfpfns,X_raw,Y_raw,act)
    states['w'][-1][1]-=thresh.reshape(-1,1)
    states['bet']*=set_bet(consts,fp,fn)
    print('nn epoch',epoch)
  return states

#_nn_epochs=jit(_nn_epochs,static_argnames=['n_epochs','n_batches','bs','last','act'])

def nn_epochs(k,n_epochs,bs,X,Y,consts,states,X_raw=None,Y_raw=None,act='relu'):
  if X_raw is None:
    X_raw,Y_raw=X,Y
  if isinstance(act,str):
    act=activations[act]
  n_batches=len(X)//bs
  last=n_batches*bs
  ks=split(k,n_epochs)
  return _nn_epochs(ks,n_epochs,bs,X,Y,X_raw,Y_raw,consts,states,act,n_batches,last)

def vectorise_fl_ls(x,l):
  if isinstance(x,list):
    return nparr(x)
  elif isinstance(x,float):
    return ones(l)*x 
  return x

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

imbl_resamplers={'SMOTETomek':SMOTETomek,'SMOTEENN':SMOTEENN,'SMOTE':SMOTE,
                 'ADASYN':ADASYN,'NearMiss':NearMiss}
skl={'RandomForestRegressor':RandomForestRegressor,
     'HistGradientBoostingRegressor':HistGradientBoostingRegressor}

class SKL:
  def __init__(self,skm,tfpfn,ska,p,p_resampled,seed=None):
    if isinstance(skm,str):
      skm=skl[skm]
    if not seed is None:
      ska['random_state']=seed

    self.m=skm(**ska)
    self.tfpfn=tfpfn
    self.threshes=npzer(len(self.tfpfn))
    self.p=p
    self.p_resampled=p_resampled

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


class ModelEvaluation:
  def __init__(self,directory,ds='unsw',seed=1729,lab_cat=True,sds=False,categorical=True,out_f=None,
               fixed_params={'nn':dict(n_epochs=100,start_width=128,end_width=32,depth=2,n_par=8,bs=128,
                                       bias=.1,act='relu',init='glorot_normal',eps=1e-8,beta1=.9,beta2=.999),
                             'sk':{}},
                             varying_params={'sk':[{'ska':dict(max_depth=i,
                                                               **({'n_jobs':-1} if\
                                                               rg=='RandomForestRegressor' else {})),\
                                                    'regressor':rg} for i in range(2,15,2) for rg in
                                                   ('RandomForestRegressor','HistGradientBoostingRegressor')],
                               'nn':[{'lr_ad':la,'reg':rl*la,'lrfpfn':lf}for la in [.0001] for\
                                     rl in [1e-2] for lf in geomspace(.0002,.03,4)]}):
    self.directory=Path(directory)
    if not self.directory.exists():
      self.directory.mkdir()
    #else:
    #  print('Directory exists.  Continue? [y/N]')
    #  if not 'y' in stdin.readline().lower():
    #    exit()
    self.parent_seed=seed
    self.rs_seeds={}
    self.rng=default_rng(seed)
    self.varying_params=varying_params
    self.fixed_params=fixed_params
    if out_f is None:
      out_f=ds+'_'+'_'+('lc_' if lab_cat else '')+'s_'+str(seed)+'_model_res'
    self.out_f=out_f
    if isinstance(ds,str):
      self.X_trn,Y_trn,self.X_tst,Y_tst,self.sc=load_ds(ds,random_split=self.seed(),
                                                        lab_cat=lab_cat,single_dataset=sds)
    else:
      self.X_trn,Y_trn,self.X_tst,Y_tst,self.sc=ds
    if lab_cat:
      Y_labs=set(Y_trn)
      self.Y_trn={l:(Y_trn==l).to_numpy() for l in Y_labs}
      self.Y_tst={l:(Y_tst==l).to_numpy() for l in Y_labs}
    else:
      self.Y_trn={'all':Y_trn}
      self.Y_tst={'all':Y_tst}

    self.p_trn={l:yt.mean() for l,yt in self.Y_trn.items()}
    self.p_tst={l:yt.mean() for l,yt in self.Y_tst.items()}
    self.regressors={}
    self.thresholds={}
    self.res={}
    self.resample('')

  def seed(self):
    return self.rng.integers(2**32)
  
  def set_targets(self,tfps=None,tfns=None,n_targets=8,min_tfpfn=1.,max_tfpfn=100.,e0=.1):
    if tfps is None:
      gar=geomspace(min_tfpfn**.5,max_tfpfn**.5,n_targets)
      tfp0=e0*gar
      tfn0=e0/gar
      self.targets={l:[(tfp*p,tfn*p,tfp/tfn) for (tfp,tfn) in zip(tfp0,tfn0)]\
                    for l,p in self.p_trn.items()}
    elif isinstance(tfp0s,list):
      n_targets=len(tfp0s)
      self.targets={l:[(tfp0*p,tfn0*p,tfp0/tfn0) for (tfp0,tfn0) in zip(tfp0s,tfn0s)] for\
                    l,p in self.p_trn.items()}
    self.tfps={l:nparr([t[0] for t in v]) for l,v in self.targets.items()}
    self.tfns={l:nparr([t[1] for t in v]) for l,v in self.targets.items()}
    self.tfpfns={l:nparr([t[2] for t in v]) for l,v in self.targets.items()}
    self.n_targets=n_targets

  def define_jobs(self,methods=['nn','sk'],resamplers=['']+list(imbl_resamplers),check_done=True):
    self.methods=methods
    self.resamplers=resamplers
    self.jobs=[(method,i,resampler) for resampler in resamplers for method in methods for i in\
               range(len(self.varying_params[method]))]
    self.n_complete=0
    self.n_remaining=len(self.jobs)
    print('Scheduled',self.n_remaining,'jobs:')
    [print(*j) for j in self.jobs]
    if check_done:
      done_fp=self.directory/'done.csv'
      if done_fp.exists():
        with done_fp.open('r') as fd:
          done_jobs=[(r[0],int(r[1]),r[2]) for r in reader(fd)]
          self.jobs=[j for j in self.jobs if not j in done_jobs]
          self.n_complete=len(done_jobs)
          print('Already completed:')
          [print(*dj) for dj in done_jobs]
          print('Still to complete:')
          [print(*j) for j in self.jobs]
          self.n_remaining-=self.n_complete

  def run_jobs(self,of=stdout):
    while self.n_remaining:
      job=self.jobs.pop()
      if of:
        m,i,rs=job
        rs=rs or 'none'
        print('Initialising model...',file=of)
        print('method:',m,file=of)
        print('method index:',i,file=of)
        print('resampling:',rs,file=of)
        print('Globally fixed parameters:',file=of)
        [print(k,v) for k,v in self.fixed_params[m].items()]
        print('Locally fixed parameters:',file=of)
        [print(k,v) for k,v in self.varying_params[m][i].items()]
      self.init_model(job)
      if of: print('Benchmarking...',file=of)
      self.benchmark_model(job)
      if of: print('Updating results...',file=of)
      self.update_results(job)
      if of: print('Removing job...',file=of)
      self.rm_job(job)
      if of: print(self.n_complete,'experiments completed with',self.n_remaining,'to go...',file=of)

  def rm_job(self,job):
    self.regressors.pop(job)
    self.res.pop(job)
    self.n_complete+=1
    self.n_remaining-=1

  def resample(self,resampler):
    rs_fp=(self.directory/(resampler+'.pkl'))
    self.resampler=resampler
    if resampler:
      if rs_fp.exists():
        print('Resampling by',resampler,'...')
        print('Loading data resampled by',resampler,'...')
        with rs_fp.open('rb') as fd:
          self.X_trn_resamp,self.Y_trn_resamp=load(fd)
      else:
        self.Y_trn_resamp={}
        self.X_trn_resamp={}
        for l,Y in self.Y_trn.items():
          seed=self.seed()
          try:
            sm=imbl_resamplers[resampler](random_state=seed)
          except:
            print(resampler,'doesn\'t take a random seed...')
            sm=imbl_resamplers[resampler]()
          self.X_trn_resamp[l],self.Y_trn_resamp[l]=sm.fit_resample(self.X_trn,Y)
        with rs_fp.open('wb') as fd:
          dump((self.X_trn_resamp,self.Y_trn_resamp),fd)
    else:
      self.Y_trn_resamp=self.Y_trn
      self.X_trn_resamp={l:self.X_trn for l in self.Y_trn}
      print('No resampling...')

    self.p_resampled={l:Y.mean() for l,Y in self.Y_trn_resamp.items()}
    print('#Training rows:',len(self.X_trn),'~~>')
    [print(k,':',len(x)) for k,x in self.Y_trn_resamp.items()]

  def init_model(self,job):
    method,i,resampler=job
    pv=self.varying_params[method][i]
    pf=self.fixed_params[method]
    match method:
      case 'sk':
        j={l:SKL(skl[pv['regressor']],self.tfpfns[l],pv['ska'],p,self.p_resampled[l],
                 seed=self.seed()) for l,p in self.p_trn.items()}
      case 'nn':
        j={l:NNPar(self.seed(),p,self.tfps[l],self.tfns[l],p_resampled=self.p_resampled[l],
                   init=pf['init'],lrfpfn=pv['lrfpfn'],reg=pv['reg'],start_width=pf['start_width'],
                   end_width=pf['end_width'],depth=pf['depth'],lr=pv['lr_ad'],beta1=pf['beta1'],
                   beta2=pf['beta2'],eps=pf['eps'],bias=pf['bias']) for l,p in self.p_trn.items()}
      case _:
        raise NotImplementedError('Method',method,'not found')
    self.regressors[job]=j

  def benchmark_model(self,job):
    method,param_index,resampler=job
    if resampler!=self.resampler:self.resample(resampler)
    regressor=self.regressors[job]
    ms=self.fixed_params[method]
    res={}
    for l,reg in regressor.items():
      print('Training for label',l,'...')
      r={'trn':{},'tst':{},'tgt':{targs[2]:dict(zip(('fp','fn','fpfn'),targs)) for targs in self.targets[l]}}
      reg.fit(self.X_trn_resamp[l],self.Y_trn_resamp[l],X_raw=self.X_trn,Y_raw=self.Y_trn[l])

      for stage,X,Y in [('tst',self.X_tst,self.Y_tst[l]),('trn',self.X_trn,self.Y_trn[l])]:
        Yp=reg.predict(X)
        fps=(Yp&~Y).mean(axis=1)
        fns=((~Yp)&Y).mean(axis=1)
        for tfpfn,fp,fn in zip(self.tfpfns[l],fps,fns):
          r[stage][tfpfn]=dict(fp=fp,fn=fn,fpfn=fp/fn)
      res[l]=r
    self.res[job]=res

  def update_results(self,single=False):
    done_fp=self.directory/'done.csv'
    res_fp=self.directory/'res.csv'
    param_js=self.directory/'fixed_params.json'
    param_vp=self.directory/'varying_params.csv'
    append_done=done_fp.exists()
    if single:
      newly_done={single}
    elif append:
      with done_fp.open('r') as fd:
        previously_done=set(list(reader(fd)))
      newly_done=set(self.res)-previously_done
    else:
      newly_done=self.res

    append_res=res_fp.exists()
    with res_fp.open('a' if append_res else 'w') as fd:
      w=DictWriter(fd,['cat_lab','p','method','parameter_index','resampler','fp_target','fn_target',
                      'fp_train','fn_train','fp_test','fn_test'])
      if not append_res:w.writeheader()
      [w.writerow(dict(method=method,parameter_index=i,resampler=resampler,cat_lab=l,p=self.p_trn[l],
                       **{k+lab:self.res[method,i,resampler][l][s][tfpfn][k] for k in ('fp','fn') for\
                       (lab,s) in (('_target','tgt'),('_train','trn'),('_test','tst'))})) for\
                       (method,i,resampler) in newly_done for l in self.p_trn\
                       for _,_,tfpfn in self.targets[l]]

    with done_fp.open('a' if append_done else 'w') as fd:
      w=writer(fd)
      [w.writerow(r) for r in newly_done]
    if not param_js.exists():
      with param_js.open('w') as fd:
        jump({'fixed':self.fixed_params,'methods':self.methods,'resamplers':self.resamplers},fd)
    if not param_vp.exists():
      with param_vp.open('w') as fd:
        w=DictWriter(fd,['param_index','method']+\
                        sum([list(self.varying_params[m][0]) for m in self.methods],[]))
        w.writeheader()
        for m in self.methods:
          [w.writerow(dict(param_index=i,method=m,**vps)) for i,vps in enumerate(self.varying_params[m])]

class NNPar:
  def __init__(self,seed,p,tfp,tfn,p_resampled=None,lrfpfn=.03,reg=1e-6,mod_shape=None,
               in_dim=None,bet=None,start_width=None,end_width=None,depth=None,bs=128,
               act='relu',init='glorot_normal',n_epochs=100,lr=1e-4,beta1=.9,beta2=.999,
               eps=1e-8,bias=.1,acc=.1,min_tfpfn=1,max_tfpfn=100):
    self.n_par=len(tfp)
    self.p=p
    self.p_resampled=p if p_resampled is None else p_resampled
    if bet is None:
      bet=((1-self.p_resampled)/self.p_resampled)**.5
    self.bias=bias
    self.mod_shape0=list(geomspace(start_width,end_width,depth+1,dtype=int))+[1] if\
                         mod_shape is None else mod_shape
    self.ke=KeyEmitter(seed)
    lrfpfn,reg,lr,beta1,beta2,eps,tfp,tfn,self.bet=(vectorise_fl_ls(x,self.n_par) for x in\
                                                    (lrfpfn,reg,lr,beta1,beta2,eps,tfp,tfn,bet))

    self.tfp_actual=tfp
    self.tfn_actual=tfn
    self.bs=bs

    self.init=init
    self.n_epochs=n_epochs
    self.consts=dict(lrfpfn=lrfpfn,tfp=tfp*(1-self.p_resampled)/(1-self.p),tfn=tfn*self.p_resampled/self.p,
                     beta1=beta1,beta2=beta2,lr=lr,reg=reg,eps=eps)
    self.act=act
    self.states=None
  
  def init_states(self,in_dim):
    if self.p_resampled:
      self.states=dict(bet=self.bet,**init_layers_adam([in_dim]+self.mod_shape0,self.init,
                                                       k=self.ke.emit_key(),n_par=self.n_par,bias=self.bias))

  def fit(self,X,Y,X_raw=None,Y_raw=None):
    if not self.states:
      self.init_states(X.shape[1])
    self.states=nn_epochs(self.ke.emit_key(),self.n_epochs,self.bs,X,Y,self.consts,self.states,
                          X_raw=X_raw,Y_raw=Y_raw,act=self.act)

  def predict(self,X):
    Yp=vorward(self.states['w'],X,activations[self.act])
    return Yp.reshape(Yp.shape[:-1])>0
