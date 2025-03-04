from argparse import ArgumentParser
from pickle import load,dump
from types import SimpleNamespace
from csv import writer
from os.path import isdir,isfile
from os import mkdir,listdir
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from pathlib import Path
from select import select
from sys import stdin
from time import perf_counter
from numpy import inf,unique as npunique,array as nparr,min as nmn,\
                  max as nmx,sum as nsm,log10 as npl10,round as rnd
from numpy.random import default_rng #Only used for deterministic routines
from jax.numpy import array,vectorize,zeros,log,log10,flip,maximum,minimum,pad,\
                      concat,exp,ones,linspace,array_split,reshape,corrcoef,eye,\
                      concatenate,unique,cov,expand_dims,identity,\
                      diag,average,triu_indices,sum as jsm,max as jmx
from jax.numpy.linalg import svdvals
from jax.scipy.signal import convolve
from jax.nn import tanh,softmax
from jax import grad,value_and_grad,jit,config
from jax.random import uniform,normal,split,key,choice,binomial,permutation
from sklearn.utils.extmath import cartesian
from pandas import read_csv
from matplotlib.pyplot import imshow,legend,show,scatter,xlabel,ylabel,\
                              gca,plot,title,savefig,close
from matplotlib.patches import Patch
from matplotlib.cm import jet
from pandas import read_pickle,read_parquet,concat,get_dummies

def set_jax_cache():
  config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
  config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  config.update("jax_persistent_cache_min_compile_time_secs", 0)
  config.update("jax_persistent_cache_enable_xla_caches",
                "xla_gpu_per_fusion_autotune_cache_dir")

class TimeStepper:
  def __init__(self,clock_avg_rate=.01):
    self.clock_avg_rate=clock_avg_rate
    self.tl=perf_counter()
    self.time_avgs={}
  def get_timestep(self,label):
    t=perf_counter()
    try:
      self.time_avgs[label]+=(1+self.clock_avg_rate)*\
                             self.clock_avg_rate*\
                             float(t-self.tl)
    except:
      self.time_avgs[label]=(1+self.clock_avg_rate)*\
                            float(t-self.tl)
      self.time_avgs[label]*=(1-self.clock_avg_rate)
    self.tl=t
  def report(self,p=False):
    tai=self.time_avgs.items()
    tsr='\n'.join([str(k)+':'+str(log10(v)) for k,v in tai])
    if p:
      print(tsr)
      return
    tsrx='\n'.join(['\\texttt{'+k.replace('_','\\_')+'}&'+\
                    f_to_str(log10(v))+'\\\\\n' for k,v in tai])
    tsrx+='\\end{tabular}'
    return tsr,tsrx

class OTFBinWeights:
  def __init__(self,avg_rate,target_fp,target_fn,adaptive_thresh_rate,
               tp0=.25,tn0=.25,fp0=.25,fn0=.25,nl_avg=False,
               sm_UV=False,imb=.5,p_scale=1.):
    self.thresh=self.pre_thresh=self.d_pre_thresh=0.
    self.adaptive_thresh_rate=adaptive_thresh_rate
    self.target_fp=target_fp
    self.target_fn=target_fn
    self.avg_rate=avg_rate
    self.om_avg_rate=1-avg_rate
    self.tp=tp0
    self.tn=tn0
    self.fp=fp0
    self.fn=fn0
    self.nl_avg=nl_avg
    self.sm_UV=sm_UV
    self.U=.5
    self.V=.5
    self.imb=imb
    self.p_scale=p_scale

  def upd(self,y,yp):
    n_samps=len(y)
    tp_b=jsm(yp&y)/n_samps
    tn_b=jsm(~(yp|y))/n_samps
    fp_b=jsm(yp&~y)/n_samps
    fn_b=1.-tp_b-tn_b-fp_b
    if self.nl_avg:
      fp_amnt=self.avg_rate*min(1,n_samps*self.fp)
      fn_amnt=self.avg_rate*min(1,n_samps*self.fn)
      self.fp*=(1-fp_amnt)
      self.fn*=(1-fn_amnt)
      self.fp+=fp_amnt*fp_b
      self.fn+=fn_amnt*fn_b
    else:
      self.tp*=self.om_avg_rate
      self.tn*=self.om_avg_rate
      self.fp*=self.om_avg_rate
      self.fn*=self.om_avg_rate
      self.tp+=self.avg_rate*tp_b
      self.tn+=self.avg_rate*tn_b
      self.fp+=self.avg_rate*fp_b
      self.fn+=self.avg_rate*fn_b
    self.pre_thresh+=(max(0,(fp_b-self.target_fp))**2-\
                      max(0,(fn_b-self.target_fn))**2)*\
                      self.adaptive_thresh_rate
    self.pre_thresh*=(1-self.pre_thresh**2)**.25
    #self.pre_thresh+=(max(0,fp_b/self.target_fp-1)-\
    #                  max(0,fn_b/self.target_fn-1))*\
    #                  self.adaptive_thresh_rate
    #self.thresh=self.pre_thresh/\
    #            (1+self.pre_thresh**2)**.5
    #self.thresh=tanh(self.pre_thresh)
    self.thresh=self.pre_thresh
    U=self.fp/self.target_fp
    V=self.fn/(self.target_fn*self.imb**self.p_scale)
    if self.sm_UV:
      self.U,self.V=softmax(array([U,V]))
    else:
      sUV=U+V
      self.U,self.V=U/sUV,V/sUV

  def report(self,p='\r'):
    rep='tp:'+f_to_str(self.tp)+'tn:'+f_to_str(self.tn)+\
        'fp:'+f_to_str(self.fp)+'fn:'+f_to_str(self.fn)+\
        'threshold:'+\
        f_to_str(self.thresh)+\
        'pre_threshold:'+\
        f_to_str(self.pre_thresh)+\
        'U:'+f_to_str(self.U)+'V:'+f_to_str(self.V)
    if p:
      print(rep,end=p)
    return rep

def read_input_if_ready():
  return stdin.readline().lower() if stdin in select([stdin],[],[],0)[0] else ''

leg=lambda t=None,h=None,l='upper right':legend(fontsize='x-small',loc=l,
                                                handles=h,title=t)
class cyc:
  def __init__(self,n):
    self.list=[None]*n
    self.n=n
    self.virt_len=0

  def __getitem__(self,k):
    if isinstance(k,slice):
      return [self.list[i%self.n] for i in range(k.start,k.stop)]
    return self.list[k%self.n]

  def __setitem__(self,k,v):
    self.list[k%self.n]=v
    self.virt_len=min(k+1,self.n)

  def avg(self):
    return sum(self.list[:self.virt_len])/self.virt_len if self.virt_len else 0

def fm(w,x,act=tanh):
  for a,b in zip(*w):
    x=act(x*(x@a)+b)
  return tanh(jsm(x,axis=1))

def f(w,x,act=tanh):
  for a,b in zip(*w):
    x=act(x@a+b)
  return x

def pad_or_trunc(x,n):
  return x[:n] if len(x)>=n else pad(x,n-len(x))

def resnet(w,x,act=tanh,first_layer_no_skip=True):
  if first_layer_no_skip:
    x=act(x@w[0][0]+w[1][0])
    A=w[0][1:]
    B=w[1][1:]
  for a,b in zip(*w):
    #x=pad_or_trunc(x,len(b))+act(x@a+b)
    x+=act(x@a+b)
  return jsm(x,axis=1) # final layer: sum components, check + or -.

activations={'tanh':tanh,'softmax':softmax,'linear':jit(lambda x:x),
             'relu':jit(lambda x:minimum(1,maximum(-1,x)))}
implementations={'mlp':f,'resnet':resnet,'fm':fm,'linear':lambda w,x,_:x@w[0]+w[1],
                  'casewise_linear':lambda w,x,_:dot_general(x,(x@w[0]),((1,1),(0,0)))}

def implementation(imp,act=None):
  return jit(lambda w,x:implementations[imp](w,x,act))

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

def init_ensemble():
  ap=ArgumentParser()
  n_exps=5
  target_fps=[int(50*10**(1-i/n_exps))/1000 for i in range(n_exps+1)]
  target_fns=[int(10*10**(i/n_exps))/1000 for i in range(n_exps+1)]
  ap.add_argument('mode',default='all',
                  choices=['all','adaptive_lr','imbalances',
                           'unsw','gmm','mnist','rbd24'])
  ap.add_argument('-no_U_V',action='store_true')
  ap.add_argument('-no_epochs',action='store_true')
  ap.add_argument('-rbd24_no_rescale_log',action='store_true')
  ap.add_argument('-rbd24_no_shuffle',action='store_true') #Very easy - sorted by label!!!!!!
  ap.add_argument('-rbd24_sort_timestamp',action='store_true')
  ap.add_argument('-trad_avg',action='store_true')
  ap.add_argument('-resnet',default=0,type=int)
  ap.add_argument('-resnet_dim',default=64,type=int)
  ap.add_argument('-fm',default=0,type=int)
  ap.add_argument('-nnpca',action='store_true')
  ap.add_argument('-single_layer_upd',action='store_true')
  ap.add_argument('-p_scale',default=1.,type=float)#.5
  ap.add_argument('-scale_before_sm',action='store_true')
  #ap.add_argument('-no_softmax_U_V',action='store_true')
  ap.add_argument('-softmax_U_V',action='store_true')
  ap.add_argument('-window_avg',action='store_true')
  ap.add_argument('-n_gaussians',default=4,type=int)
  ap.add_argument('-loss',default='loss',choices=list(losses))
  ap.add_argument('-reporting_interval',default=1000,type=int)
  ap.add_argument('-print_step_interval',default=100,type=int)
  ap.add_argument('-gmm_spread',default=.05,type=float)
  ap.add_argument('-gmm_scatter_samples',default=10000,type=int)
  ap.add_argument('-no_gmm_compensate_variances',action='store_true')
  ap.add_argument('-gmm_min_dist',default=4,type=float)
  ap.add_argument('-force_batch_cost',default=0.,type=float)
  ap.add_argument('-adam',action='store_true')
  ap.add_argument('-no_adam',action='store_true')
  ap.add_argument('-adaptive_threshold',action='store_true')
  ap.add_argument('-activation',choices=list(activations),default='tanh')
  ap.add_argument('-implementation',type=str,
                  choices=list(implementations),default='mlp')
  ap.add_argument('--seed',default=20255202,type=int)
  ap.add_argument('-n_splits_img',default=100,type=int)
  ap.add_argument('-rbd24_single_dataset',default='',type=str)
  ap.add_argument('-rbd24_no_categorical',action='store_true')
  ap.add_argument('-rbd24_no_preproc',action='store_true')
  ap.add_argument('-imbalances',type=float,
                  default=[0.1,0.06812921,0.04641589,0.03162278,0.02154435,
                           0.01467799,0.01,0.00681292,0.00464159,0.00316228,
                           0.00215443,0.0014678,0.001],nargs='+')
  ap.add_argument('-in_dim',default=2,type=int)
  ap.add_argument('-unsw_cat_thresh',default=.1,type=int)
  ap.add_argument('-lr_resolution',default=16,type=int)
  ap.add_argument('-all_resolution',default=5,type=int)
  ap.add_argument('-history_len',default=16384,type=int)
  ap.add_argument('-weight_normalisation',default=0.,type=float)
  ap.add_argument('-no_bias',action='store_true')
  ap.add_argument('-orthonormalise',action='store_true')
  ap.add_argument('-saving_interval',default=1000,type=int)
  ap.add_argument('-lr_init_min',default=1e-4,type=float)
  ap.add_argument('-lr_init_max',default=1e-2,type=float)
  ap.add_argument('-lr_min',default=1e-5,type=float)
  ap.add_argument('-lr_max',default=1e-1,type=float)
  ap.add_argument('-lrs',default=[1e-2],type=float,nargs='+')
  ap.add_argument('-lr_update_interval',default=1000,type=int)
  ap.add_argument('-recent_memory_len',default=1000,type=int)
  ap.add_argument('-beta1',default=.9,type=float) #1- beta1 in adam
  ap.add_argument('-beta2',default=.999,type=float)
  ap.add_argument('-avg_rate',default=.1,type=float)#binom avg parameter
  ap.add_argument('-unsw_test',default='~/data/UNSW_NB15_testing-set.csv')
  ap.add_argument('-unsw_train',default='~/data/UNSW_NB15_training-set.csv')
  ap.add_argument('-model_inner_dims',default=[],type=int,nargs='+')
  ap.add_argument('-bs',default=0,type=int)
  ap.add_argument('-res',default=1000,type=int)
  ap.add_argument('-outf',default='thlay')
  ap.add_argument('-model_resid',default=False,type=bool)
  ap.add_argument('-p',default=.1,type=float)
  ap.add_argument('-target_fp',default=.01,type=float)
  ap.add_argument('-target_fn',default=.01,type=float)
  ap.add_argument('-clock_avg_rate',default=.1,type=float) #Track timings
  ap.add_argument('-threshold_accuracy_tolerance',default=.1,type=float)
  ap.add_argument('-fpfn_ratios',default=[],nargs='+',type=float)
  ap.add_argument('-target_fps',default=target_fps,nargs='+',type=float)
  ap.add_argument('-target_fns',default=target_fns,nargs='+',type=float)
  ap.add_argument('-target_tolerance',default=.5,type=float)
  ap.add_argument('-stop_on_target',action='store_true')
  #Silly
  ap.add_argument('-x_max',default=10.,type=float)
  ap.add_argument('-sqrt_normalise_a',action='store_true')
  ap.add_argument('-initialisation',type=str,
                  choices=['glorot_uniform','glorot_normal','eye',
                           'ones','zeros','resnet','casewise_linear'],default='')
  ap.add_argument('-reproduce_llpal',action='store_true')
  ap.add_argument('-mult_a',default=1.,type=float)
  
  a=ap.parse_args()
  a.epochs=not a.no_epochs
  ##a.softmax_U_V=not a.no_softmax_U_V
  if not isdir('reports'):
    mkdir('reports')
  a.outf='reports/'+a.outf
  a.report_dir=a.outf+'_report'
  a.n_imb=len(a.imbalances)
  if a.nnpca:
    a.loss='nn_pca_loss'
  elif a.loss=='distribution_flow_cost':
    a.w_init=1.
    a.target_increment=.9

  if not a.implementation:
    if a.resnet:
      a.implementation='resnet'
    elif a.fm:
      a.implementation='fm'
    else:
      a.implementation='mlp'

  if not a.activation and a.implementation not in ['linear','casewise_linear']:
    a.activation='tanh'

  if not a.initialisation:
    a.initialisation=select_initialisation(a.implementation,a.activation)

  a.gmm_compensate_variances=not a.no_gmm_compensate_variances
  if a.adam and a.no_adam:
    print('Cannot have adam and no_adam!')
    exit(1)
  elif not a.adam and not a.no_adam:
    if a.implementation=='casewise_linear' or a.activation=='linear':
      a.no_adam=True
    else:
      a.adam=True
  if a.fpfn_ratios:
    a.target_fpfns=array([[a*b,a] for a,b in zip(a.target_fns,a.target_fpfns)])
  else:
    a.target_fpfns=array(list(zip(a.target_fps,a.target_fns)))
  a.lrs=array(a.lrs)
  a.out_dir=a.outf+'_report'
  a.step=0
  if len(a.lrs)==1:
    a.lr=a.lrs[0]
  if not a.bs:
    if a.mode=='mnist':
      a.bs=128
    else:
      a.bs=1
  if not a.model_inner_dims and not a.fm:
    if a.mode=='gmm':
      a.model_inner_dims=[32,16]
    elif a.mode==['mnist','rbd24'] and a.act not in ['linear','relu']:
      a.model_inner_dims=[64,32,16]
    else:
      a.model_inner_dims=[64,32]
  return a

f_to_str=lambda x,p=True:f'{x:.3g}'.ljust(9) if p else f'{x:.3g}'

exp_to_str=lambda e:'exp_p'+f_to_str(e.p,p=False)+'fpt'+f_to_str(e.target_fp,p=False)+\
                    'fnt'+f_to_str(e.target_fn,p=False)+'lr'+f_to_str(e.lr,p=False)

fpfnp_lab=lambda e:'FP_t,FN_t,p='+f_to_str(e.target_fp,p=False)+','+\
                   f_to_str(e.target_fn,p=False)+','+f_to_str(e.p,p=False)

@jit
def svd_cost(w,x,act=tanh,top_growth=array([2,1]),imp=resnet,
             eps=1e-8,dimension=4,vol_growth=0.):
  if dimension is None: dimension=x.shape[1]
  iden=identity(x_dim)
  sqs=jsm(x**2,axis=1)
  dists_init=sqs+expand_dims(sqs,-1)-2*x@x.T
  x=imp(*w,x)
  sqs=jsm(x**2,axis=1)
  dists_final=sqs+expand_dims(sqs,-1)-2*x@x.T
  distortion=dists_final/(dists_init+iden)
  growth=log(svdvals(distortion)[:dimension])
  return (top_growth-growth[:len(top_growth)])**2+(jsm(growth)-vol_growth)**2

@jit
def dot_penalty(z,y):
  target_directions=1-2*y^expand_dims(y,-1)
  sqs=jsm(z**2,axis=1)
  return jsm((sqs+expand_dims(sqs,-1)-2*z@z.T)*target_directions)

@jit
def dist_penalty(z,y):
  n_pos=int(jsm(y))
  n_neg=len(y)-n_pos
  U=n_neg
  V=n_pos
  pn=y^expand_dims(y,-1)
  pp=y&expand_dims(y,-1)
  nn=~(pp|pn)
  sqs=jsm(z**2,axis=1)
  dists=sqs+expand_dims(sqs,-1)-2*z@z.T
  return (U*jsm(dists[pp])+V*jsm(dists[nn]))/(1e-8+jsm(dists[pn]))

@jit
def metric_cost(w,x,y,act=tanh,imp=resnet):
  z=imp(*w,x)
  return dist_penalty(z,y)

@jit
def coalescence_cost(w,x,y,U,V,act=tanh,tol=1e-4):
  bs=len(y)
  iden=identity(bs)
  n_pos=jsm(y)
  n_neg=bs-n_pos
  target_expansion=0
  yT=expand_dims(y,-1)
  ny=~y
  if n_pos and n_neg:
    target_expansion=(U*n_neg/n_pos+V*n_pos/n_neg)*(y^yT)
  target_growth=target_expansion-U*(~(y|yT))-V*(y&yT)+diag(ny*U+y*V)
  sqs=jsm(x**2,axis=1)
  old_ldists=log(iden+tol+sqs+expand_dims(sqs,-1)-2*x@x.T)
  ret=0.
  for a,b in zip(*w):
    x=act(x@a+b)
    iden=identity(bs)
    sqs=jsm(x**2,axis=1)
    dists=iden+tol+sqs+expand_dims(sqs,-1)-2*x@x.T
    ldists=log(dists)
    ret+=jsm((old_ldists-ldists)*target_growth/dists) #Don't worry so much about far pts
    old_ldists=ldists
  return (ret+y@log(tol+1-x)+ny@log(tol+1+x))[0]

@jit
def nn_cost_expansion(w,xp,xn,contraction=False,act=tanh,tol=1e-2,imp=f):
  ret=0.
  dists_init=jsm((xp-xn)**2,axis=1)
  xp,xn=imp(w,xp,act=act),imp(w,xn,act=act)
  dists_final=jsm((xp-xn)**2,axis=1)
  expansion=jsm((tol+dists_final)/(tol+dists_init))
  return expansion if contraction else -expansion

'''
def nn_cost_contraction(A,B,x,act=tanh,tol=1e-8,imp=f):
  ret=0.
  dists_init=jsm((expand_dims(x,axis=0)-expand_dims(x,axis=1))**2,axis=2)
  x=imp(A,B,xp,act=act)
  dists_final=jsm((expand_dims(x,axis=0)-expand_dims(x,axis=1))**2,axis=2)
  return jsm(dists_final/dists_init)/len(x)
'''

@jit
def coalescence_res_cost(w,x,y,U,V,act=tanh,tol=1e-2):
  bs=len(y)
  #l=1/len(A)
  iden=identity(bs)
  n_pos=jsm(y)
  n_neg=bs-n_pos
  target_expansion=0
  yT=expand_dims(y,-1)
  ny=~y
  #if n_pos and n_neg:
  r=1.#U*n_neg/(1+n_pos)+V*n_pos/(1+n_neg)
  norm=(n_pos*(n_pos-1)*V**2+n_neg*(n_neg-1)*U**2+2*n_pos*n_neg*r)**.5
  u=U/norm
  v=V/norm
  r/=norm
  #target_growth=#((y^yT)-U*(~(y|yT))-V*(y&yT)+diag(ny*U+y*V))/\
  #target_growth=r*(y^yT)-u*(~(y|yT))-v*(y&yT)+diag(ny*u+y*v)
  target_lyap=-r*(y^yT)+u*(~(y|yT))+v*(y&yT)+diag(ny*u+y*v)
  #target_growth=(1.*(y^yT)-1.*(y==yT)+iden)/(bs*(bs-1))**.5
  ret=0.
  sqs=jsm(x**2,axis=1)
  old_ldists=log(tol+sqs+expand_dims(sqs,-1)-2*x@x.T)
  for a,b in zip(*w):
    dx=act(x@a+b)
    x+=dx
    sqs=jsm(x**2,axis=1)
    dists=sqs+expand_dims(sqs,-1)-2*x@x.T
    similarities=exp(-dists)#+iden)#tol
    ldists=log(tol+dists)
    #ret+=jsm((dx@dx.T-target_growth)**2) #same class->similar directions
    #xdx=x@dx.T # 
    #ret+=jsm((((1-iden)*(-xdx-xdx.T)-target_growth)**2)/dists)
    #delx=(-xdx-xdx.T)/dists
    #delx=-xdx/dists #>0 when dist growing, roughly
    #ret+=log(tol+(jsm(delx**2)-jsm(delx*target_growth)**2))
    #grow_dirs=xdx*target_direction
    #grow_dirs/=tol+jsm(xdx**2)**.5
    ret+=jsm(target_lyap*(ldists-old_ldists)/similarities)
    old_ldists=ldists
  #x=act(jsm(x,axis=1))
  return ret+jsm(V*y@(tol+1-x)+U*ny@(tol+1+x))
  #return log(1+ret)+jsm(V*y*(1-x)+U*ny*(1+x))

@jit
def distribution_flow_cost(*w,x,y,U,V,w_init,act=tanh,tol=1e-8):
  ret=0.
  n=len(A)
  bs=len(y)
  n_pos=jsm(y)
  n_neg=bs-n_pos
  end_dim=len(B[-1])
  end_dists=log(2)*(y!=expand_dims(y,-1))
  end_l2=(n_pos**2+n_neg**2)*log(2)**2

  sqs=jsm(x**2,axis=1)
  init_dists=log(1+sqs+expand_dims(sqs,-1)-2*x@x.T)
  #init_l2=jsm(init_dists)**.5
  init_l2=jsm(init_dists**2)
  #init_dists**=.5
  UV_mask=U*(~y)+V*y
  UV_mask=UV_mask*expand_dims(UV_mask,-1)
  for i,(a,b) in enumerate(zip(*w)):
    x=act(x@a+b)
    sqs=jsm(x**2,axis=1)
    dists=log(1+sqs+expand_dims(sqs,-1)-2*x@x.T)
    dists_l2=jsm(dists**2)
    dec=tol**(i/n)
    #ret+=w_init*dec*log(1+init_l2*dists_l2-jsm(dists*init_dists)**2)
    ret+=w_init*dec*jsm((dists-init_dists)**2)
    #dists*=UV_mask
    #dists_l2=jsm(dists**2)
    #ret+=(1-w_init)*(tol/dec)*log(1+end_l2*dists_l2-jsm(dists*end_dists)**2)
    ret+=(1-w_init)*(tol/dec)*jsm(UV_mask*(dists-end_dists)**2)
  return ret/n+(1-w_init)*jsm(U*(~y)*(1+x)+V*y*(1-x))

@jit
def resnet_cost(w,x,y,U,V,act=tanh,eps=1e-8): #Already vectorised
  c=0
  n=len(A)
  UmV=y*(U+V)-U #weight positives and negatives by importance
  for a,b in zip(*w):
    dx=act(x@a.T+b)
    #sg_dx=dx.T*UmV
    sg_dx=dx.T*(2*y-1)
    #c-=log(eps+jsm(sg_dx.T@sg_dx)) #force + and - difference direction correlations
    #c-=log(eps+jsm(sg_dx.T@sg_dx))
    #c-=jsm(log(1+eps+sg_dx.T@sg_dx))
    #c-=jsm(sg_dx.T@sg_dx)
    x+=dx
  return c-jsm(UmV*x.T) # final layer: sum components, check + or -.

@jit
def resnet_cost_layer(*w,i,c,d,x,y,U,V,act=tanh,eps=1e-8): #Already vectorised
  ret=0.
  UmV=y*(U+V)-U
  for j,(a,b) in enumerate(zip(*w)):
    if i==j:
      dx=act(x@(a.T+c.T)+b+d)
    else:
      dx=act(x@a.T+b)
    sg_dx=dx.T*(2*y-1)#UmV
    #sg_dx=dx.T*(2*y-1)
    #c-=log(eps+jsm(sg_dx.T@sg_dx))
    #c-=log(eps+jsm(sg_dx.T@sg_dx))
    #c-=jsm(log(1+eps+sg_dx.T@sg_dx))
    #ret-=jsm(sg_dx.T@sg_dx)#/jsm(dx.T@dx)
    ret-=jsm(log(1+eps+sg_dx.T@sg_dx))
    x+=dx
  return ret-jsm(UmV*x.T)

@jit
def l2(w):
  return sum(jsm(a**2) for a in w[0])+\
         sum(jsm(a**2) for a in w[1])
dl2=jit(value_and_grad(l2))

'''
def mk_l1_fm(imp=f):
  @jit
  def l1(w,x,y,U,V,eps=.1):
    y_smooth=imp(w,x)
    y_diffs=1-y_smooth*(2*y-1)
    l=(x*(x@A[0]))**2
    sparsity=jsm(l)-jmx(l) #Foce layer 1 to be typically dominated by 1 term
    return jsm(((V-U)*y+U)*y_diffs)+sparsity #U*cts_fp+V*cts_fn
  return l1,value_and_grad(l1)

def mk_l1_orth(act=tanh,imp=f):
  @jit
  def l1(w,x,y,U,V,eps=.1):
    y_smooth=imp(w,x,act=act)
    y_diffs=1-y_smooth*(2*y-1)
    #non_orthog=sum([jsm(((a.T@a)**2)[triu_indices(a.shape[1],k=1)]) for a in A])
    non_orthog=[jsm(((a.T@a)**2)[triu_indices(a.shape[1],k=1)]) for a in A]
    noth=0
    for a in non_orthog:
      noth+=a

    #a_p,a_n=y_smooth[y],y_smooth[~y]
    #cts_fp=jsm(1.+a_n)
    #cts_fn=jsm(1.-a_p)
    return jsm(((V-U)*y+U)*y_diffs)+noth#non_orthog #U*cts_fp+V*cts_fn
  return l1,value_and_grad(l1)

def mk_waqas(imp,eps=1e-5):
  @jit
  def waqas(w,x,y,U,V,eps=eps):
    y_smooth=imp(w,x)
    y_diffs=maximum(0,-y_smooth*(2*y-1)/(jsm(x**2,axis=1)**.5+eps))
    return jsm(y_diffs) #U*cts_fp+V*cts_fn
  dwaqas=value_and_grad(waqas)
  return waqas,dwaqas

def mk_waqas_consaw(imp,eps=1e-5):
  @jit
  def waqas(w,x,y,U,V,eps=eps):
    y_smooth=imp(w,x)
    y_diffs=maximum(0,-y_smooth*(2*y-1)/(jsm(x**2,axis=1)**.5+eps))
    return jsm(((V-U)*y+U)*y_diffs) #U*cts_fp+V*cts_fn
  dwaqas=value_and_grad(waqas)
  return waqas,dwaqas
'''

def mk_hinge(imp,eps=1e-5):
  @jit
  def hinge(w,x,y,U,V,eps=eps):
    y_smooth=imp(w,x,act=act)
    y_diffs=maximum(0,-y_smooth*(2*y-1))
    return jsm(((V-U)*y+U)*y_diffs) #U*cts_fp+V*cts_fn
  dhinge=value_and_grad(hinge)
  return hinge,dhinge

def mk_l1(act=None,imp=f,reg=True):
  def l1(w,x,y,U,V,eps=None):
    y_smooth=imp(w,x)
    y_diffs=1-y_smooth*(2*y-1)
    return jsm(((V-U)*y+U)*y_diffs) #U*cts_fp+V*cts_fn
  if reg:
    def ret(w,x,y,U,V,eps=None,reg=1e-2):
      return l1(w,x,y,U,V,eps=None)+reg*l2(w)
  else:
    ret=l1
  return jit(ret)

def mk_mk_lp(p=2.):
  def mk_l(imp=f,reg=True):
    def lp(w,x,y,U,V,eps=None):
      y_smooth=imp(w,x)
      y_diffs=((2*y-1)-y_smooth)**p
      #a_p,a_n=y_smooth[y],y_smooth[~y]
      #cts_fp=jsm(1.+a_n)
      #cts_fn=jsm(1.-a_p)
      return jsm(((V-U)*y+U)*y_diffs) #U*cts_fp+V*cts_fn
    if reg:
      def ret(w,x,y,U,V,eps=None,reg=.1):
        return lp(w,x,y,U,V)+reg*l2(w)
    else:
      ret=lp
    return jit(ret)
  return mk_l

mk_l2=mk_mk_lp()

@jit
def l1(w,x,y,U,V,act=tanh):
  return l1_soft(w,x,y,U,V,0.,act)

def mk_cross_entropy(act=tanh,imp=f,reg=True):
  def cross_entropy(w,x,y,U,V,eps=1e-8):
    y_smooth=imp(w,x)#,act=act)
    y_winnings=y_smooth*(2*y-1) #+ if same sign, - if different sign
    #return -jsm(((V-U)*y+U)*log(1+eps+y_preloss))
    return -jsm(((V-U)*y+U)*log(1+eps+y_winnings))
  if reg:
    def ret(w,x,y,U,V,eps=1e-8,reg=1.):
      return cross_entropy(w,x,y,U,V,eps)+reg*l2(w)
  else:
    ret=cross_entropy
  return jit(ret)

@jit
def cross_entropy_rn(w,x,y,U,V,act=tanh,eps=1e-8):
  y_smooth=tanh(resnet(w,x,act=act))
  a_p,a_n=y_smooth[y],y_smooth[~y] # 0<y''=(1+y')/2<1
  cts_fn=-jsm(log(eps+1.+a_p)) #y=1 => H(y,y')=-log(y'')=-log((1+y')/2)
  cts_fp=-jsm(log(eps+1.-a_n)) #y=0 => H(y,y')=-log(1-y'')=-log((1-y')/2)
  return U*cts_fp+V*cts_fn

@jit
def cross_entropy_soft(w,x,y,U,V,act=tanh,normalisation=False,softness=.1,eps=1e-8):
  y_smooth=f(w,x,act=act)
  a_p,a_n=y_smooth[y],y_smooth[~y] # 0<y''=(1+y')/2<1
  cts_fn=-(1-softness)*jsm(log(eps+1.+a_p))-softness*jsm(log(eps+1.-a_p))
  cts_fp=-(1-softness)*jsm(log(eps+1.-a_n))-softness*jsm(log(eps+1.+a_n))
  return U*cts_fp+V*cts_fn

@jit
def nn_pca_loss(w_c,b_c,w_e,b_e,x,x_targ,eps=1e-8):
  x_c=f(w_c,b_c,x)
  l=cov(x_c)
  x_c_vars=var(x_c,axis=0)
  #x_c_vars/=jsm(x_c_vars)
  l+=jsm(x_c_vars[1:]/(eps+x_c_vars[:-1]))

  x_p=f(w_e,b_e,x_c)
  l+=jsm((x_p-x_targ)**2)
  return l#log(jsm(w_c[-1]@w_c[-1].T)))

losses={'loss':mk_cross_entropy,'l1':mk_l1,'cross_entropy':mk_cross_entropy,
        'l2':mk_l2,'hinge':mk_hinge}
        #'l1_soft':dl1_soft,'cross_entropy_soft':dcross_entropy_soft,
        #'resnet_cost':dresnet_cost,'resnet_cost_layer':dresnet_cost_layer,
        #'nn_pca_loss':nn_pca_loss,'distribution_flow_cost':ddistribution_flow_cost,
        #'coalescence_cost':dcoalescence_cost,'cross_entropy_rn':dcross_entropy_rn,
        #'coalescence_res_cost':dcoalescence_res_cost}

def init_layers(k,layer_dimensions,initialisation,mult_a=1.,
                sqrt_normalise=False,orthonormalise=False):
  k1,k2=split(k)
  wb=[]
  n_steps=len(layer_dimensions)-1
  w_k=split(k1,n_steps)
  b_k=split(k2,n_steps)
  A=[]
  B=[]
  for i,(k,l,d_i,d_o) in enumerate(zip(w_k,b_k,layer_dimensions,layer_dimensions[1:])):
    match initialisation:
      case 'zeros':
        A.append(zeros((d_i,d_o)))
        B.append(zeros(d_o))
      case 'resnet'|'ones':
        A.append(ones((d_i,d_o))*mult_a)
        B.append(zeros(d_o))
        initialisation='zeros' if initialisation=='resnet' else 'ones'
      case 'eye':
        A.append(eye(d_i,d_o))*mult_a
        B.append(zeros(d_o))
      case 'glorot_uniform':
        A.append((2*(6/(d_i+d_o))**.5)*(uniform(shape=(d_i,d_o),key=k)-.5)*mult_a)
        B.append(zeros(d_o))
      case 'glorot_normal':
        A.append(((2/(d_i+d_o))**.5)*(normal(shape=(d_i,d_o),key=k))*mult_a)
        B.append(zeros(d_o))
      case 'casewise_linear':
        A=ones((d_i,d_i))
        #A=zeros((d_i,d_i))
        B=zeros(shape=d_i)
        return A,B
      case _:
        raise Exception('Unknown initialisation: '+initialisation)
  if orthonormalise:
    for i,(d_i,d_o) in enumerate(layer_dimensions,layer_dimensions[1:]):
      if d_i>d_o:
        A[i]=svd(A[i],full_matrices=False)[0]
      else:
        A[i]=svd(A[i],full_matrices=False)[2]
  if sqrt_normalise:
    for i,d_i in enumerate(layer_dimensions[1:]):
      A[i]/=d_i**.5
  return A,B

def sample_x(bs,key):
  return 2*a.x_max*uniform(shape=(bs,2),key=key)-a.x_max

def colour_rescale(fpfn):
  l=log(array(fpfn))-log(a.fpfn_min)
  l/=log(a.fpfn_max)-log(a.fpfn_min)
  return jet(l)

def mk_experiment(p,thresh,target_fp,target_fn,lr,a,eps=1e-8):
  e=SimpleNamespace()
  e.report_done=False
  e.eps=eps
  e.bs=a.bs
  e.target_tolerance=a.target_tolerance
  e.steps_to_target=False
  e.avg_rate=a.avg_rate
  e.dw_l2=e.w_l2=0
  if a.initialisation=='casewise_linear':
    e.w_model=a.w_model_init[0].copy(),a.w_model_init[1].copy()
  else:
    e.w_model=[v.copy() for v in a.w_model_init[0]],[v.copy() for v in a.w_model_init[1]]
  e.history_len=a.history_len
  e.step=0

  e.adam_V=[u*0. for u in e.w_model[0]],[u*0. for u in e.w_model[1]]
  e.adam_M=0.

  e.lr=float(lr)
  e.p=float(p) #"imbalance"
  #e.p_its=int(1/p) #if repeating minority class iterations
  e.target_fp=target_fp#target_fn*fpfn_ratio
  e.target_fn=target_fn
  e.fpfn_ratio=e.target_fp/e.target_fn
  e.recent_memory_len=int((1/a.avg_rate)*max(1,1/(e.bs*min(e.target_fp,e.target_fn))))

  e.FPs=cyc(e.recent_memory_len)
  e.FNs=cyc(e.recent_memory_len)
  e.loss_vals=cyc(e.recent_memory_len)#a.bs)
  if a.loss=='distribution_flow_cost':
    e.w_init=1.
    e.loss_target=inf
    e.loss_val=a.bs**2*a.model_inner_dims[-1]
  e.fp=(e.target_fp*.5)**.5
  e.fn=(target_fn*.5)**.5#.25 #softer start if a priori assume doing well

  e.U=e.V=1
  e.thresh=thresh
  e.history=SimpleNamespace(FP=[],FN=[],lr=[],cost=[],w=[],dw=[],loss_vals=[],
                            resolution=1,l=0)
  return e

def init_experiments(a,global_key):
  a.global_key=global_key
  k1,k2,k3,k4,k5,k6=split(global_key,6)
  a.time_avgs=dict()
  if a.mode=='rbd24':
    (a.x_train,a.y_train),\
    (a.x_test,a.y_test),\
    (_,a.x_columns)=rbd24(rescale_log=not a.rbd24_no_rescale_log,
                          preproc=not a.rbd24_no_preproc,
                          categorical=not a.rbd24_no_categorical,
                          single_dataset=a.rbd24_single_dataset)
    a.p=sum(a.y_train)/len(a.y_train)
    a.p_test=sum(a.y_test)/len(a.y_test)
    a.in_dim=len(a.x_train[0])
    if a.epochs:
      a.epoch_num=0
      a.offset=inf
      #a.x_train,a.y_train=shuffle_xy(k1,a.x_train,a.y_train)

  if a.mode=='mnist':
    from tensorflow.keras.datasets import mnist
    (a.x_train,a.y_train),(a.x_test,a.y_test)=mnist.load_data()
    y_ones_train=a.y_train==1 #1 detector
    y_ones_test=a.y_test==1 #1 detector
    a.in_dim=784
    a.x_train_pos=reshape(a.x_train[y_ones_train],(-1,a.in_dim))
    a.x_train_neg=reshape(a.x_train[~y_ones_train],(-1,a.in_dim))

    a.x_test_pos=reshape(a.x_test[y_ones_test],(-1,a.in_dim))
    a.x_test_neg=reshape(a.x_test[~y_ones_test],(-1,a.in_dim))

  if a.mode=='all':
    a.target_shape=[2]+[16]*8+[1]

  if a.mode=='unsw':
    df_train=read_csv(a.unsw_train)
    df_test=read_csv(a.unsw_test)
    a.x_test=array(df_test[df_test.columns[(df_test.dtypes==int)|\
                                           (df_test.dtypes==float)]]).T[1:].T
    a.x_train=array(df_test[df_train.columns[(df_train.dtypes==int)|\
                                             (df_train.dtypes==float)]]).T[1:].T
    a.y_train=df_train['attack_cat']
    a.y_test=df_test['attack_cat']
    a.in_dim=len(a.x_train[0])

  #elif a.mode=='all':
  #  a.target_mult_a=.75
  #  a.target_mult_b=2.
  #
  #  a.w_target=init_layers(k1,a.target_shape,a.target_mult_a,a.target_mult_b)

  if a.implementation=='linear':
    a.model_shape=[a.in_dim,1]

  if a.fm:
    a.model_shape=[a.in_dim]*a.fm#+[1]
  elif a.resnet:
    a.model_shape=[a.in_dim]+([a.resnet_dim]*a.resnet)
  else:
    a.model_shape=[a.in_dim]+a.model_inner_dims+[1]

  a.w_model_init=init_layers(k2,a.model_shape,a.initialisation,a.mult_a,
                             sqrt_normalise=a.sqrt_normalise_a,
                             orthonormalise=a.orthonormalise)

  if a.mode=='unsw':
    a.imbalances=(a.y_train.value_counts()+a.y_test.value_counts())/\
                  (len(df_train)+len(df_test))
    a.cats={float(p):s for p,s in zip(a.imbalances,a.y_train.value_counts().index)}
    a.imbalances=a.imbalances[a.imbalances>a.unsw_cat_thresh]
  
  if a.mode in ['unsw','gmm','mnist','rbd24']:
    a.thresholds={float(p):0. for p in a.imbalances}
  else:
    print('Finding thresholds...')
    
    thresholding_sample_size=int(1/(a.threshold_accuracy_tolerance**2*\
                                    a.imbalance_min))
    x_thresholding=sample_x(thresholding_sample_size,k3)
    
    y_t_cts=f(a.w_target[0],a.w_target[1],x_thresholding).flatten()
    y_t_cts_sorted=y_t_cts.sort()
    a.thresholds={float(p):y_t_cts_sorted[-int(p*len(y_t_cts_sorted))]\
                  for p in a.imbalances}
    
    print('Imbalances and thresholds')
    for i,t in a.thresholds.items(): print(i,t)
  
  a.loop_master_key=k4
  a.step=0

  #if a.mode=='all':
  #  a.targets=list(zip(a.fpfn_ratios,a.target_fns))
  #  experiments=[mk_experiment(p,a.thresholds[p],fpfn_ratio,target_fn,a.lr,a)\
  #               for p in a.imbalances for (fpfn_ratio,target_fn) in a.targets\
  #               for lr in a.lrs]
  #elif a.mode=='adaptive_lr':
  #  experiments=[mk_experiment(.1,a.thresholds[.1],1.,.01,lr,a) for\
  #               lr in a.lrs]
  if a.mode in ['unsw','gmm','mnist','rbd24']:
    a.imbalances=None
    experiments=[mk_experiment(a.p,0.,t_fp,t_fn,a.lr,a)\
                 for t_fp,t_fn in a.target_fpfns]
  #elif a.mode in ['imbalances']:
  #  experiments=[mk_experiment(float(p),float(a.thresholds[float(p)]),fpfn_ratio,
  #                             float(p*target_fn),a.lr,a)\
  #               for p in a.imbalances for fpfn_ratio in a.fpfn_ratios\
  #               for target_fn in a.target_fns]

  if a.mode=='gmm':
    min_dist=0
    while min_dist<a.gmm_min_dist:
      a.means=2*a.x_max*uniform(k5,(2*a.n_gaussians,a.in_dim))-a.x_max
      min_dist=min([jsm((A-B)**2) for b,A in enumerate(a.means) for B in a.means[b+1:]])
    a.variances=2*a.x_max*uniform(k6,2*a.n_gaussians)*a.gmm_spread #hmm
  return experiments

def shuffle_xy(k,x,y):
  shuff=permutation(k,len(y))
  x,y=x[shuff],y[shuff]
  return x,y

def get_xy_jit(X,Y,start,bs,k,new_epoch):
  end=start+bs
  if new_epoch:
    new_epoch=True
    k1,k=split(k)
    X,Y=shuffle_xy(k1,X,Y)
    start=0
    end=bs
  return (X,Y),(X[start:end],Y[start:end]),end,new_epoch
get_xy_jit=jit(get_xy_jit,static_argnames='new_epoch')

def get_xy(a,bs,k,imbs=None):
  if type(imbs)==float:
    ret_single=imbs
    imbs=[imbs]
  else:
    ret_single=False

  if a.mode=='gmm':
    ret=dict()
    for p in imbs:
      probs=array(([(1.-p)/a.n_gaussians]*a.n_gaussians)+\
                  ([p/a.n_gaussians]*a.n_gaussians))
      mix=choice(k,2*a.n_gaussians,shape=(bs,),p=probs)
      y=mix>=a.n_gaussians
      k1,k=split(k)
      z=normal(k1,shape=(bs,a.in_dim))
      if a.gmm_compensate_variances:
        z/=(-2*log(p))**.5
      x=z*a.variances[mix,None]+a.means[mix]
      ret[float(p)]=x,y
  elif a.mode=='rbd24':
    if a.epochs: #epochs: sample randomly shuffled dataset without replacement
      next_offset=a.offset+a.bs
      if next_offset>len(a.y_train): #new epoch
        a.epoch_num+=1
        print('Start of epoch',a.epoch_num)
        print('Number of rows in training data:',len(a.y_train))
        if a.rbd24_no_shuffle:
          if a.epoch_num>1:
            print('END OF TIME')
        else:
          k1,k=split(k)
          a.x_train,a.y_train=shuffle_xy(k1,a.x_train,a.y_train)
        a.offset=0
        next_offset=a.bs
      ret={float(a.p):(a.x_train[a.offset:next_offset],
                       a.y_train[a.offset:next_offset])}
      a.offset=next_offset
    else: #sample without replacement
      k1,k=split(k)
      batch_indices=choice(k1,len(a.y_train),shape=(a.bs,))
      ret={float(a.p):(a.x_train[batch_indices],
                      array(a.y_train[batch_indices]))}

  elif a.mode=='unsw':
    k1,k=split(k)
    batch_indices=choice(k1,len(a.y_train),shape=(a.bs,))
    ret={float(p):(a.x_train[batch_indices],
                   array(a.y_train[batch_indices]==a.cats[p])) for p in a.imbalances}
  elif a.mode=='mnist': #force imbalance of mnist dataset
    k1,k2,k3,k4,k=split(k,5)
    n_pos=[int(binomial(kk,a.bs,p)) for kk,p in zip(split(k1,a.n_imb),a.imbalances)]
    x_pos=[choice(kk,a.x_train_pos,shape=(np,)) for kk,np in\
           zip(split(k2,a.n_imb),n_pos)]
    x_neg=[choice(kk,a.x_train_neg,shape=(a.bs-np,)) for kk,np in\
           zip(split(k3,a.n_imb),n_pos)]
    x_all=[(xn if not len(xp) else(xp if not len(xn) else concatenate([xp,xn]))) for\
           xp,xn in zip(x_pos,x_neg)]
    perms=[permutation(kk,a.bs) for kk in split(k4,a.n_imb)]

    ret={p:(x[perm],perm<np) for p,x,np,perm in zip(a.imbalances,x_all,n_pos,perms)}
  else:
    ret=dict()
    for p in imbs:
      k1,k=split(k)
      x=sample_x(a.bs,k1)
      ret[float(p)]=x,f(a.w_target[0],a.w_target[1],x).flatten()

  return ret[ret_single] if ret_single else ret

@jit
def evaluate_fp_fn(bs,avg_rate,target_fp,target_fn,y_p,y_t,fp,fn):
  FP=jsm(y_p&(~y_t))/bs #Stop jax weirdness after ADC
  FN=jsm(y_t&(~y_p))/bs
  cost=maximum(FP/target_fp,FN/target_fn)
  fp_amnt=avg_rate*minimum(1,bs*fp)
  fn_amnt=avg_rate*minimum(1,bs*fn)
  fp*=(1-fp_amnt)
  fp+=fp_amnt*FP
  fn*=(1-fn_amnt)
  fn+=fn_amnt*FN
  return FP,FN,cost,fp,fn


def even_indices(li):
  l=len(li)
  return [li[2*i] for i in range(l//2)]

def update_history(e):
  if not e.step%e.history.resolution: #for plotting purposes
    e.history.FP.append(e.FP)
    e.history.FN.append(e.FN)
    e.history.lr.append(e.lr)
    e.history.w.append(e.w_l2)
    e.history.dw.append(e.dw_l2)
    e.history.cost.append(e.cost)
    e.history.loss_vals.append(e.loss_val)
    e.history.l+=1
    if e.history.l>e.history_len:
      e.history.resolution*=2
      e.history.FP=even_indices(e.history.FP)
      e.history.FN=even_indices(e.history.FN)
      e.history.lr=even_indices(e.history.lr)
      e.history.cost=even_indices(e.history.cost)
      e.history.w=even_indices(e.history.dw)
      e.history.dw=even_indices(e.history.dw)
      e.history.loss_vals=even_indices(e.history.loss_vals)
      e.history.l//=2
  if e.fp<e.target_tolerance*e.target_fp and\
     e.fn<e.target_tolerance*e.target_fn and\
     not e.steps_to_target:
       e.steps_to_target=e.step
       e.report_done=True
    

def compute_U_V(fp,fn,target_fp,target_fn,p,sm=False,p_scale=.5,scale_before_sm=True):
  pp=p**p_scale
  U=fp/target_fp
  V=fn/target_fn
  if sm:
    if scale_before_sm:
      return softmax(array([U,V/pp]))
    else:
      U,V=softmax(array([U,V]))
  V/=pp
  UpV=U+V
  #if UpV>0:#Should be fine, moving average never hits 0
  U/=UpV
  V/=UpV
  return U,V
compute_U_V=jit(compute_U_V,static_argnames=['sm','scale_before_sm'])

def update_lrs(a,experiments,k): 
  k1,k2,k3=split(k,3)
  experiments=sorted(experiments,key=lambda x:x.cost)
  goodnesses=array([1/(1e-8+e.cost) for e in experiments])
  e_lr=v_lr=0.
  for e,g in zip(experiments,goodnesses):
    print('lr,un-normalised goodnesses=',e.lr,g)
    le_lr+=log10(e.lr)
    lv_lr+=(log10(e.lr))**2
  le_lr/=len(experiments)
  lv_lr/=len(experiments)
  lv_lr-=le_le**2
  print('E(log(lr)),V(log(lr))=',le_lr,lv_lr)
  goodnesses/=jsm(goodnesses)
  experiment_indices=array(range(len(experiments)))
  e=experiments[-1]
  parent=experiments[int(choice(k1,experiment_indices,p=goodnesses))]
  e.lr=parent.lr
  w=lambda x,y,z:(x*z,y/z)
  if parent.lr>a.lr_max: rule=lambda x,y:(x,y*exp(-abs(normal(k2))))
  elif parent.lr<a.lr_min: rule=lambda x,y:(x,y*exp(abs(normal(k2))))
  else: rule=lambda x,y:w(x,y,exp(normal(k2)))
  parent.lr,e.lr=rule(parent.lr,e.lr)

  if uniform(k3)<e.cost/(1e-8+parent.cost)-1:
    print('Weight copying')
    e.w_model=([v.copy() for v in parent.w_model[0]],
               [v.copy() for v in parent.w_model[1]])
    e.adam_M=parent.adam_M.copy()
    e.adam_V=parent.adam_V.copy()
    e.fp=float(parent.fp)
    e.fn=float(parent.fn)
  a.lrs=array([e.lr for e in experiments])
  return experiments

@jit
def upd_adam(w,adam_V,adam_M,upd,beta1,beta2,lr,eps=1e-8):
  adva,advb=adam_V
  wa,wb=w
  adam_M*=beta2
  for i,(u,v) in enumerate(zip(*upd)):
    adva[i]*=beta1
    advb[i]*=beta1
    adva[i]+=(1-beta1)*u
    advb[i]+=(1-beta1)*v
    adam_M+=(1-beta2)*(nsm(u**2)+nsm(v**2))
  w_l2=0
  dw_l2=0
  for i,(s,t) in enumerate(zip(adva,advb)):
    delta_u=lr*s/(adam_M**.5+eps)
    delta_v=lr*t/(adam_M**.5+eps)
    wa[i]-=delta_u
    wb[i]-=delta_v
    w_l2+=jsm(wa[i]**2)+jsm(wb[i]**2)
    dw_l2+=jsm(delta_u**2)+jsm(delta_v**2)
  return (wa,wb),(adva,advb),adam_M,w_l2,dw_l2

@jit
def upd_grad(w,upd,lr):
  w=([a-lr*da for a,da in zip(w[0],upd[0])],
     [b-lr*db for b,db in zip(w[1],upd[1])])
  dw_lw=(lr**2)*sum([jsm(da**2)+jsm(db**2) for da,db in zip(*upd)])
  w_l2=sum([jsm(a**2)+jsm(b**2) for a,b in zip(*w)])

  return w,w_l2,dw_l2

#@jit
def update_weights(a,e,upd):#,start=None,end=None):
  if a.initialisation=='casewise_linear':
    wm=list(e.w_model[0]),list(e.w_model[1])
  else:
    wm=e.w_model
  if a.adam:
    wm,e.adam_V,e.adam_M,e.w_l2,e.dw_l2=upd_adam(wm,e.adam_V,e.adam_M,upd,
                                                 a.beta1,a.beta2,e.lr)
  else:
    wm,e.w_l2,e.dw_l2=upd_grad(wm,upd,e.lr)

  if a.no_bias:
    wm=wm[0],[0. for b in wm[1]]

  if a.initialisation=='casewise_linear':
    e.w_model=array(wm[0]),array(wm[1])
  else:
    e.w_model=wm

def plot_stopping_times(experiments,fd_tex,report_dir):
  for e in experiments: e.fpfn_ratio=float(e.fpfn_ratio)
  completed_experiments=[e for e in experiments if e.steps_to_target]
  try:
    fpfn_ratios=list(set([e.fpfn_ratio for e in experiments]))
    for rat in fpfn_ratios:
      x=[log10(e.p) for e in completed_experiments if e.fpfn_ratio==rat]
      y=[log10(e.steps_to_target) for e in completed_experiments if\
         e.fpfn_ratio==rat]
      plot(x,y)
      title('Stopping times for target fp/fn='+f_to_str(rat))
      xlabel('log(imbalance)')
      ylabel('log(Stopping step)')
      if fd_tex:
        savefig(report_dir+'/stopping_times_'+str(rat)+'.png',dpi=500)
        close()
        print('\n\\begin{figure}[H]',file=fd_tex)
        print('\\centering',file=fd_tex)
        print('\\includegraphics[width=.9\\textwidth]'
              '{stopping_times_'+str(rat)+'.png}',file=fd_tex)
        print('\\end{figure}',file=fd_tex)
      else:
        show()
  except AttributeError:
    print('fpfn ratios not found, skipping stopping time analysis')

def plot_2d(experiments,fd_tex,a,imp,line,k):
  if fd_tex:
    print('\\subsection{2d visualisation of classifications}',file=fd_tex)
    print('Here a 2d region is learned.\\\\',file=fd_tex)
  plot_num=0
  for e in experiments:
    if a.mode=='gmm':
      x_t,y_t=get_xy(a,e.p,a.gmm_scatter_samples,k)
      col_mat=[[1.,1,1],[0,0,0]]#fp,fn,tp,tn
      labs=['Predict +','Predict -']
      x_0_max=jmx(x_t[:,0])
      x_0_min=-jmx(-x_t[:,0])
      x_1_max=jmx(x_t[:,1])
      x_1_min=-jmx(-x_t[:,1])
    else:
      x_0_max=x_1_max=a.x_max
      x_0_min=x_1_min=-a.x_max
    x=cartesian([linspace(x_0_min,x_0_max,num=a.res),
                 linspace(x_1_min,x_1_max,num=a.res)])
    x_split=array_split(x,a.n_splits_img)
    y_p=concat([(imp(e.w_model,_x)>0).flatten() for\
                _x in x_split])
    y_p=flip(y_p.reshape(a.res,a.res),axis=1) #?!?!?!

    cm=None
    if 'b' in line: #draw boundary
      cols=abs(convolve(y_p,array([[1,1,1],[1,-8,1],[1,1,1]]))).T
      cols/=-(jmx(cols)+jmx(-cols))
      cols+=jmx(-cols)
      cm='gray'
    else:
      if a.mode=='gmm':
        regions=array([y_p,~y_p]).T
      else:
        y_t=concat([f(a.w_target[0],a.w_target[1],_x)>e.thresh for _x in x_split])\
            .reshape(a.res,a.res)
        fp_img=(y_p&~y_t)
        fn_img=(~y_p&y_t)
        tp_img=(y_p&y_t)
        tn_img=(~y_p&~y_t)
        regions=array([fp_img,fn_img,tp_img,tn_img]).T
        col_mat=[[1.,0,0],[0,1,0],[1,1,1],[0,0,0]]#fp,fn,tp,tn
        labs=['FP','FN','TP','TN']
      cols=regions.dot(array(col_mat))
    imshow(cols,extent=[x_0_min,x_0_max,x_1_min,x_1_max],cmap=cm)
    handles=[Patch(color=c,label=s) for c,s in zip(col_mat,labs)]
    if a.mode=='gmm':
      x_0,x_1=tuple(x_t.T)
      y_p_s=f(e.w_model,x_t,act=act).flatten()>0
      scatter(x_0[y_t&y_p_s],x_1[y_t&y_p_s],c='darkgreen',s=1,label='TP')
      scatter(x_0[y_t&~y_p_s],x_1[y_t&~y_p_s],c='palegreen',s=1,label='FP')
      scatter(x_0[~y_t&~y_p_s],x_1[~y_t&~y_p_s],c='cyan',s=1,label='TN')
      scatter(x_0[~y_t&y_p_s],x_1[~y_t&y_p_s],c='blue',s=1,label='FN')
      handles+=[Patch(color=c,label=s) for c,s in\
                zip(['blue','palegreen','darkgreen','cyan'],['FP','FN','TP','TN'])]
    leg(h=handles,l='lower left')
    title('p='+f_to_str(e.p,p=False)+',target_fp='+f_to_str(e.target_fp,p=False)+\
          ',target_fn='+f_to_str(e.target_fn,p=False)+',lr='+f_to_str(e.lr,p=False))
    if fd_tex:
      if not plot_num%9:
        print('\\begin{figure}',file=fd_tex)
        print('\\centering',file=fd_tex)
      plot_num+=1
      img_name=exp_to_str(e)+'.png'
      savefig(a.report_dir+'/'+img_name,dpi=500)
      print('\\begin{subfigure}{.33\\textwidth}',file=fd_tex)
      print('\\centering',file=fd_tex)
      print('\\includegraphics[width=.9\\linewidth,scale=1]{'+img_name+'}',
            file=fd_tex)
      print('\\end{subfigure}%',file=fd_tex)
      print('\\hfill',file=fd_tex)
      if not plot_num%9:
        print('\\end{figure}',file=fd_tex)
      close()
    else:
      show()
  if fd_tex and plot_num%9: print('\\end{figure}',file=fd_tex)

get_cost=lambda e:log10(1e-8+array(e.history.cost))
get_lr=lambda e:e.history.lr
get_dw=lambda e:log10(1e-8+array(e.history.dw))
def plot_historical_statistics(experiments,fd_tex,a,smoothing=100):
  if fd_tex:
    print('\\subsection{Historical statistics}',file=fd_tex)
  stats=[(get_cost,'log(eps+max(fp/target_fp,fn/target_fn))','Loss'),
         (get_dw,'log(eps+dw)','Change_in_weights')]
  if a.mode=='adaptive_lr':
    stats.append((get_lr,'log(lr)','Learning_rate'))
  for get_var,yl,desc in stats:
    for e in experiments:
      arr=get_var(e)
      if smoothing:
        ker=ones(smoothing)/smoothing
        arr=convolve(array(arr,dtype=float),ker,'same')
      if a.mode=='unsw':
        lab=a.cats[e.p]
      else:
        lab=fpfnp_lab(e)
      plot(arr,label=lab,alpha=.5)
    xlabel('Step number *'+str(e.history.resolution))
    ylabel(yl)
    title(desc.replace('_',' '))
    leg()
    if fd_tex:
      savefig(a.report_dir+'/'+desc+'.png',dpi=500)
      close()
      print('\n\\begin{figure}[H]',file=fd_tex)
      print('\\centering',file=fd_tex)
      print('\\includegraphics[width=.9\\textwidth]{'+desc+'.png}',file=fd_tex)
      print('\\end{figure}',file=fd_tex)
    else:
      show()
  for e in experiments:
    conv_len=min(int(a.step**.5),int(1/e.p))
    ker=ones(conv_len)/conv_len
    smoothed_fp=convolve(array(e.history.FP,dtype=float),ker,'valid')
    smoothed_fn=convolve(array(e.history.FN,dtype=float),ker,'valid')
    plot(log10(smoothed_fp),log10(smoothed_fn),label=fpfnp_lab(e),alpha=.5)
    xlabel('log(fp)')
    ylabel('log(fn)')
    title('FP versus FN rate')
  leg()
  if fd_tex:
    savefig(a.report_dir+'/phase.png',dpi=500)
    close()
    print('\n\\begin{figure}[H]',file=fd_tex)
    print('\\centering',file=fd_tex)
    print('\\includegraphics[width=.9\\textwidth]{phase.png}',file=fd_tex)
    print('\\end{figure}',file=fd_tex)
  else:
    show()

def plot_fpfn_scatter(experiments,fd_tex,a):
  fp_perf=[e.fp/e.target_fp for e in experiments]
  fn_perf=[e.fn/e.target_fn for e in experiments]
  sc=scatter(fp_perf,fn_perf)#,c=colours)#,s=sizes)
  if a.mode=='all': gca().add_artist(cl)
  xlabel('fp/target_fp')
  ylabel('fn/target_fn')
  if fd_tex:
    savefig(a.report_dir+'/scatter.png',dpi=500)
    close()
    print('\\subsection{Comparing performance}',file=fd_tex)
    print('\n\\begin{figure}[H]',file=fd_tex)
    print('\\centering',file=fd_tex)
    print('\\includegraphics[width=.9\\textwidth]{scatter.png}',file=fd_tex)
    print('\\end{figure}',file=fd_tex)
  else:
    show()

def report_progress(a,experiments,line,imp,k):
  try:
    k1,k2=split(k)
    print('|'.join([t.ljust(9) for t in ['p','target_fp','target_fn',
                                          'lr','fp','fn','w','dw','U','V','complete']]))
    for e in experiments:
      print('|'.join([f_to_str(t) for t in [e.p,e.target_fp,e.target_fn,e.lr,
                                            e.fp,e.fn,e.w_l2,e.dw_l2,e.U,e.V]])+'|'+\
            (f_to_str(e.steps_to_target) if e.steps_to_target else 'no'))
    for e in experiments:
      print('Recent FPs:',
            ','.join([str(t) for t in e.FPs[e.step-5:e.step]]))
      print('Recent FNs:',
            ','.join([str(t) for t in e.FNs[e.step-5:e.step]]))
      print('Losses:',e.loss_vals[e.step-5:e.step])

    if ':' in line:
      l=line.split(':')
      if len(l)<3:
        print('Invalid command')
        return
      if l[0]=='f':
        ty=float
      elif line[0]=='i':
        ty=int
      vars(a)[l[1]]=ty(l[2])
      print('a.'+line[1]+'->'+str(ty(l[2])))
      return
    elif '?' in line:
      query_var=line.split('?')[0]
      va=vars(a)
      if query_var in va:
        print('a.'+query_var,':',va[query_var])
      else:
        print('"',line.split('?')[0],'" not in a')
      return
    elif '!' in line:
      query_var=line.split('!')[0]
      found=False
      for i,e in enumerate(experiments):
        ve=vars(e)
        if query_var in ve:
          found=True
          print('experiment_'+str(i)+'.'+query_var,':',ve[query_var])
      if not found:
        print('"',line.split('?')[0],'" not in any experiment')
      return
    elif '*' in line:
      print('a variables:')
      [print(v) for v in vars(a).keys()]
      return

    fd_tex=False
    if 'e' in line and a.mode in ['mnist','rbd24']:
      print('p_train:',a.p)
      print('p_test:',a.p_test)
      for e in experiments:
        if a.mode=='rbd24':
          x_train_pos=a.x_train[a.y_train]
          x_train_neg=a.x_train[~a.y_train]
          x_test_pos=a.x_test[a.y_test]
          x_test_neg=a.x_test[~a.y_test]
          print()
          print('mavg_fp_0,mavg_fn_0:',f_to_str(e.fp),f_to_str(e.fn))
          print('target_fp_0,target_fn_0:',
                f_to_str(e.target_fp),f_to_str(e.target_fn))
          print('fp_train,fn_train:',
                f_to_str(sum([nsm(f(e.w_model,x_train_neg[i:i+a.bs])>0) for\
                              i in range(0,len(x_train_neg),a.bs)])/len(a.x_train)),
                f_to_str(sum([nsm(f(e.w_model,x_train_pos[i:i+a.bs])<=0) for\
                              i in range(0,len(x_train_pos),a.bs)])/len(a.x_train)))
          print('fp_test,fn_test:',
                f_to_str(sum([nsm(f(e.w_model,x_test_neg[i:i+a.bs])>0) for\
                              i in range(0,len(x_test_neg),a.bs)])/len(a.x_test)),
                f_to_str(sum([nsm(f(e.w_model,x_test_pos[i:i+a.bs])<=0) for\
                              i in range(0,len(x_test_pos),a.bs)])/len(a.x_test)))
        else:
          x_train_pos=a.x_train_pos
          x_train_neg=a.x_train_neg
          x_test_pos=a.x_test_pos
          x_test_neg=a.x_test_neg
          print()
          print('mavg_fp_0,mavg_fn_0:',f_to_str((1-.1)*e.fp/(1-e.p)),f_to_str(.1*e.fn/e.p))
          print('target_fp_0,target_fn_0:',
                f_to_str((1-.1)*e.target_fp/(1-e.p)),f_to_str(.1*e.target_fn/e.p))
          print('fp_train_0,fn_train_0:',
                f_to_str(nsm(f(e.w_model,x_train_neg)>0)/len(a.x_train)),
                f_to_str(nsm(f(e.w_model,x_train_pos)<=0)/len(a.x_train)))
          print('fp_test_0,fn_test_0:',
                f_to_str(nsm(f(e.w_model,x_test_neg)>0)/len(a.x_test)),
                f_to_str(nsm(f(e.w_model,x_test_pos)<=0)/len(a.x_test)))
    if 'r' in line:
      line+='clist'
      a.report_dir=a.outf+'_report'
      fd_tex=open(a.outf+'_report/report.tex','w')
      print('\\documentclass[landscape]{article}\n'
            '\\usepackage[margin=0.7in]{geometry}\n'
            '\\usepackage[utf8]{inputenc}\n'
            '\\usepackage{graphicx}\n'
            '\\usepackage{float}\n'
            '\\usepackage{caption}\n'
            '\\usepackage{subcaption}\n'
            '\\usepackage{amsmath,amssymb,amsfonts,amsthm}\n'
            '\n'
            '\\title{Experiment ensemble report: '+a.mode+'}\n'
            '\\author{George Lee}\n'
            '\n'
            '\\begin{document}\n'
            '\\maketitle\n'
            'Report for ensemble labelled '+a.outf.replace('_','\\_')+'\n'
            '\\subsection{Performance after '+str(a.step)+' steps}',file=fd_tex)
      
      with open(a.out_dir+'/performance.csv','w') as fd_csv:
        w=writer(fd_csv)
        n_fps=len(a.fpfn_ratios)
        row=['imbalance','target_fps']+['']*(n_fps-1)+['target_fn','fp']+\
            ['']*(n_fps-1)+['fn']+['']*(n_fps-1)+['steps_to_target']
        if a.mode=='unsw':
          row=['attack_cat']+row
        w.writerow(row)
        if fd_tex:
          conf_fill='r'*n_fps
          ct='l' if a.mode=='unsw' else ''
          print('\\begin{tabular}{l'+ct+'|'+conf_fill+'|r'+(('|'+conf_fill)*4)+'}',
                file=fd_tex)
          print(' & '.join(row).replace('_',' ')+'\\\\',file=fd_tex)
          print('\\hline',file=fd_tex)
        for p in a.imbalances:
          tgt_fps=[]
          report_fps=[]
          report_fns=[]
          steps_to_target=[]
          for e in [e for e in experiments if e.p==p]:
            tgt_fps.append(f_to_str(e.target_fp))
            fp_hist=e.history.FP[-int(10/e.p**2):]
            fn_hist=e.history.FN[-int(10/e.p**2):]
            if a.mode=='mnist':
              y_ones_test=a.y_test==1 #1 detector

              report_fns.append(f_to_str(e.p*nsm(f(e.w_model,
                                                   a.x_test_pos)<=0)/\
                                         (.1*len(a.x_test))))
              report_fps.append(f_to_str((1-e.p)*nsm(f(e.w_model,
                                                       a.x_test_neg)>0)/\
                                         ((1-.1)*len(a.x_test))))
            else:
              report_fps.append(f_to_str(sum(fp_hist)/(e.bs*len(fp_hist))))
              report_fns.append(f_to_str(sum(fn_hist)/(e.bs*len(fn_hist))))
            steps_to_target.append(str(e.steps_to_target) if e.steps_to_target else '-')
          row=[f_to_str(p)]+tgt_fps+[f_to_str(p/10)]+report_fps+report_fns+steps_to_target
          if a.mode=='unsw':
            row=[a.cats[p]]+row
          w.writerow(row)
          if fd_tex:
            print('&'.join(row)+'\\\\',file=fd_tex)
    if fd_tex:
      print('\\end{tabular}\n'
            '\\subsection{Timing analysis of update step}\n'
            'Distinct steps in the algorithm taking on average:\\\\\n'
            '\\begin{tabular}{r|r}\n'
            'step&$\\log(\\overline{\\texttt{T}_\\texttt{step}})$\\\\\n',file=fd_tex)

    print('Timing:')
    tsr,tsrx=a.ts.report()
    print(tsr)
    if fd_tex:print(tsrx,file=fd_tex)

    if a.in_dim==2 and 'c' in line:
      plot_2d(experiments,fd_tex,a,imp,line,k1)

    if 'i' in line:
      model_desc='- Model shape:\n'+('->'.join([str(l) for l in a.model_shape]))+\
                 '\n- Activation: '+a.activation+'\n'+\
                 '- Implementation: '+a.implementation+'\n'+\
                 '- Model initialisation: '+str(a.initialisation)+'\n'\
                 '- Matrix weight multiplier: '+str(a.mult_a)+'\n'\
                 '- Sqrt variance correction:'+str(a.sqrt_normalise_a)+'\n'\
                 '- Batch size:'+str(a.bs)+'\n'\
                 '- learning rate:'+str(a.lr)
      print(model_desc)
      if fd_tex:
        print('\\subsection{Model parameters}',file=fd_tex)
        print('Here the batch size was set to '+str(a.bs)+'.\\\\',file=fd_tex)
        print('\\texttt{'+(model_desc.replace('\n','}\\\\\n\\texttt{'))+'}\\\\',
              file=fd_tex)

    if 's' in line and a.mode=='all':
      plot_fpfn_scatter(experiments,fd_tex,a)
    if 't' in line:
      plot_stopping_times(experiments,fd_tex,a.report_dir)
    if 'l' in line:
      plot_historical_statistics(experiments,fd_tex,a)
    if 'x' in line:
      print('Bye!')
      exit()

    if fd_tex:
      print('\\end{document}',file=fd_tex,flush=True)
      fd_tex.close()
  except Exception as e:
    print('Error reporting on progress:')
    print(e)

def save_ensemble(a,experiments,global_key):
  with open(a.out_dir+'/ensemble.pkl','wb') as fd:
    a.global_key=global_key
    dump((a,experiments,global_key),fd)

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

def rbd24(preproc=True,split_test_train=True,rescale_log=True,single_dataset=False,
          raw_pickle_file=str(Path.home())+'/data/rbd24/rbd24.pkl',categorical=True,
          processed_pickle_file=str(Path.home())+'/data/rbd24/rbd24_proc.pkl'):
  if split_test_train and preproc and rescale_log and isfile(processed_pickle_file) and not single_dataset:
    print('Loading processed log rescaled pickle...')
    with open(processed_pickle_file,'rb') as fd:
      return load(fd)
  elif isfile(raw_pickle_file):
    print('Loading raw pickle...')
    df=read_pickle(raw_pickle_file)
  else:
    rbd24_dir=dl_rbd24()
    categories=listdir(rbd24_dir+'/parquet')
    dfs=[read_parquet(rbd24_dir+'/parquet/'+n) for n in categories]
    for df,n in zip(dfs,categories):
      df['category']=n.split('.')[0]
    df=concat(dfs)
    print('Writing raw pickle...')
    df.to_pickle(raw_pickle_file)

  if single_dataset:
    df=df[df.category==single_dataset].drop(['category'],axis=1)
  if preproc:
    df=preproc_rbd24(df,rescale_log=rescale_log)
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
  l=len(y)
  split_point=int(l*.7)
  x_train,x_test=x[:split_point],x[split_point:]
  y_train,y_test=y[:split_point],y[split_point:]
  if split_test_train and preproc and rescale_log and not single_dataset:
    print('Saving processed log rescaled pickle...')
    with open(processed_pickle_file,'wb') as fd:
      dump(((x_train,y_train),(x_test,y_test),(df,x_cols)),fd)
  return (x_train,y_train),(x_test,y_test),(df,x_cols)

def preproc_rbd24(df,split_test_train=True,rm_redundant=True,plot_xvals=False,
                  check_large=False,check_redundant=False,rescale_log=10):
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
    if check_redundant:
      print('Redundant columns (all values==0):')
      print(', '.join(redundant_cols))
    if check_large:
      print('Large columns (max>1):')
      print('name,maximum,n_distinct')
      for c,m,distinct in zip(large_cols,maximums,distinct):
        print(c,',',m,',',distinct)
  if rm_redundant:
    df=df.drop(redundant_cols,axis=1)
  if rescale_log:
    max_logs=max([npl10(1+m) for m in maximums])
    if max_logs>rescale_log:
      print('Note that rescale factor will not map values to be <=1')
      print('Largest value:',max_logs)
    for c in large_cols:
      df[c]=npl10(1+df[c].__array__())/rescale_log
  if plot_xvals:
    plot_uniques(df)
  return df.sort_values('timestamp')

def plot_uniques(df):
  for col in [c for c in df.columns if df.dtypes[c] in [float,int]]:
    vals=unique(df[col].__array__())
    if len(vals)>2000:
      vals=vals[::len(vals)//2000]
    title('Unique values for '+col)
    plot(linspace(0,1,len(vals)),vals)
  show()

gen=default_rng(1729) #only used for deterministic algorithm so not a problem for reprod
def min_dist(X,Y=None):
  if not Y is None:
    ret_x_y=True
    if len(Y)>len(X):
      X,Y=Y,X
      ret_x_y=False
    if not len(Y):
      return inf,None,None

    X=nparr(X)
    Y=nparr(Y)
    if len(Y)==1:
      m=inf
      y=Y[0]
      for x_cand in X:
        m_cand=nsm((x_cand-y)**2)
        if m_cand<m:
          m,x=m_cand,x_cand
          if not m:
            return (m,x,y) if ret_x_y else (m,y,x)
      return (m,x,y) if ret_x_y else (m,y,x)

    X_c=gen.choice(X,X.shape[0])
    Y_c=gen.choice(Y,X.shape[0])
    dists=nsm((X_c-Y_c)**2,axis=1)
    m=inf
    for m_cand,x_cand,y_cand in zip(dists,X_c,Y_c):
      if m_cand<m:
        m,x,y=m_cand,x_cand,y_cand
        if not m:
          return (m,x,y) if ret_x_y else (m,y,x)
    h={}
    X_r=rnd(X/m)
    Y_r=rnd(Y/m)
    for x,x_r in zip(X,X_r):
      x_r=tuple(x_r)
      if x_r in h:
        h[x_r][0].append(x)
      else:
        h[x_r]=[x],[]
    for y,y_r in zip(Y,Y_r):
      y_r=tuple(y_r)
      if y_r in h:
        h[y_r][1].append(y)
      else:
        h[x_r]=[],[y]
    h_tups_arrs=[(t,nparr(t)) for t in h]
    n_neighbs=len(h_tups_arrs)
    for i in range(X.shape[1]):
      h_tups_arrs.sort(key=lambda x:x[0][i])
    moore_neighbs={k:(list(v[0]),list(v[1])) for k,v in h.items()}
    for i,(i_tup,i_arr) in enumerate(h_tups_arrs):
      for j in range(i+1,n_neighbs):
        j_tup,j_arr=h_tups_arrs[j]
        if nmx(abs(i_arr-j_arr))>1:
          break
        moore_neighbs[i_tup][0].extend(h[j_tup][0])
        moore_neighbs[i_tup][1].extend(h[j_tup][1])
        moore_neighbs[j_tup][0].extend(h[i_tup][0])
        moore_neighbs[j_tup][1].extend(h[i_tup][1])

    m=inf
    for v in moore_neighbs.values():
      m_cand,x_cand,y_cand=min_dist(*v)
      if m_cand<m:
        x,y,m=x_cand,y_cand,m_cand
      if not m:
        return (m,x,y) if ret_x_y else (m,y,x)
    return (m,x,y) if ret_x_y else (m,y,x)

  else:
    n_pts=len(X)
    if n_pts==1:
      return inf,None,None
    elif n_pts==2:
      return nsm((nparr(X[0])-nparr(X[1]))**2),X[0],X[1]
    X=nparr(X)
    pair0=gen.choice(X.shape[0],X.shape[0])
    pair1=(pair0+1+gen.choice(X.shape[0]-1,X.shape[0]))%X.shape[0]
    X_c0=X[pair0]
    X_c1=X[pair1]
    dists=nsm((X_c0-X_c1)**2,axis=1)
    m=inf
    for m_cand,x_cand,y_cand in zip(dists,X_c0,X_c1):
      if m_cand<m:
        m,x,y=m_cand,x_cand,y_cand
        if not m: #uh oh!
          return m,x,y

    h={}
    X_r=rnd(X/m).astype(int)
    for x,x_r in zip(X,X_r):
      x_r=tuple(x_r)
      if x_r in h:
        h[x_r].append(x)
      else:
        h[x_r]=[x]
    h_tups_arrs=[(t,nparr(t)) for t in h]
    for i in range(X.shape[1]):
      h_tups_arrs.sort(key=lambda x:x[0][i]) #stable sort so get nearby pts
    moore_neighbs={k:list(v) for k,v in h.items()}
    n_neighbs=len(h_tups_arrs)
    for i,(i_tup,i_arr) in enumerate(h_tups_arrs):#Check Moore nhoods
      for j in range(i+1,n_neighbs):
        j_tup,j_arr=h_tups_arrs[j]
        if nmx(abs(i_arr-j_arr))>1:
          break
        moore_neighbs[i_tup].extend(h[j_tup])
        moore_neighbs[j_tup].extend(h[i_tup])
    for v in moore_neighbs.values():
      m_cand,x_cand,y_cand=min_dist(v)
      if m_cand<m:
        m,x,y=m_cand,x_cand,y_cand
        if not m:
          return m,x,y
    return m,x,y
