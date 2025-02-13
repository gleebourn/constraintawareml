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
from numpy import inf,unique as npunique,array as nparr,min as nmn,\
                  max as nmx,sum as nsm,log10 as npl10,round as rnd
from numpy.random import default_rng #Only used for deterministic routines
from jax.numpy import array,vectorize,zeros,log,log10,flip,maximum,minimum,\
                      concat,exp,ones,linspace,array_split,reshape,corrcoef,\
                      concatenate,unique,cov,expand_dims,identity,diag,average,\
                      sum as jsm,max as jmx
from jax.numpy.linalg import svdvals
from jax.scipy.signal import convolve
from jax.nn import tanh,softmax
from jax import grad,value_and_grad,jit
from jax.random import uniform,normal,split,key,choice,binomial,permutation
from sklearn.utils.extmath import cartesian
from pandas import read_csv
from matplotlib.pyplot import imshow,legend,show,scatter,xlabel,ylabel,\
                              gca,plot,title,savefig,close
from matplotlib.patches import Patch
from matplotlib.cm import jet
from pandas import read_pickle,read_parquet,concat,get_dummies

def read_input_if_ready():
  return stdin.readline().lower() if stdin in select([stdin],[],[],0)[0] else ''

leg=lambda t=None,h=None,l='upper right':legend(fontsize='x-small',loc=l,
                                                handles=h,title=t)
class cyc:
  def __init__(self,n,x0=1):
    self.list=[x0]*n
    self.n=n
    self.max_accessed=1

  def __getitem__(self,k):
    if isinstance(k,slice):
      return [self.list[i%self.n] for i in range(k.start,k.stop)]
    return self.list[k%self.n]

  def __setitem__(self,k,v):
    self.list[k%self.n]=v
    self.max_accessed=min(k+1,self.n)

  def avg(self):
    return sum(self.list[:self.max_accessed])/self.max_accessed

def init_ensemble():
  ap=ArgumentParser()
  ap.add_argument('mode',default='all',
                  choices=['all','adaptive_lr','imbalances',
                           'unsw','gmm','mnist','rbd24'])
  ap.add_argument('-no_U_V',action='store_true')
  ap.add_argument('-epochs',action='store_true')
  ap.add_argument('-resnet',default=0,type=int)
  ap.add_argument('-nnpca',action='store_true')
  ap.add_argument('-single_layer_upd',action='store_true')
  ap.add_argument('-p_scale',default=.5,type=float)
  ap.add_argument('-scale_before_sm',action='store_true')
  ap.add_argument('-softmax_U_V',action='store_true')
  ap.add_argument('-window_avg',action='store_true')
  ap.add_argument('-n_gaussians',default=4,type=int)
  ap.add_argument('-loss',default='loss',choices=list(dlosses))
  ap.add_argument('-reporting_interval',default=100,type=int)
  ap.add_argument('-gmm_spread',default=.05,type=float)
  ap.add_argument('-gmm_scatter_samples',default=10000,type=int)
  ap.add_argument('-no_gmm_compensate_variances',action='store_true')
  ap.add_argument('-gmm_min_dist',default=4,type=float)
  ap.add_argument('-force_batch_cost',default=0.,type=float)
  ap.add_argument('-no_adam',action='store_true')
  ap.add_argument('-adaptive_threshold',action='store_true')
  ap.add_argument('-act',choices=['relu','tanh','softmax'],default='tanh')
  ap.add_argument('--seed',default=20255202,type=int)
  ap.add_argument('-n_splits_img',default=100,type=int)
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
  ap.add_argument('-orthonormalise',action='store_true')
  ap.add_argument('-saving_interval',default=1000,type=int)
  ap.add_argument('-lr_init_min',default=1e-4,type=float)
  ap.add_argument('-lr_init_max',default=1e-2,type=float)
  ap.add_argument('-lr_min',default=1e-5,type=float)
  ap.add_argument('-lr_max',default=1e-1,type=float)
  ap.add_argument('-lrs',default=[1e-2],type=float,nargs='+')
  ap.add_argument('-lr_update_interval',default=1000,type=int)
  ap.add_argument('-recent_memory_len',default=1000,type=int)
  ap.add_argument('-gamma1',default=.1,type=float) #1- beta1 in adam
  ap.add_argument('-gamma2',default=.001,type=float)
  ap.add_argument('-avg_rate',default=.1,type=float)#binom avg parameter
  ap.add_argument('-unsw_test',default='~/data/UNSW_NB15_testing-set.csv')
  ap.add_argument('-unsw_train',default='~/data/UNSW_NB15_training-set.csv')
  ap.add_argument('-model_inner_dims',default=[],type=int,nargs='+')
  ap.add_argument('-bs',default=0,type=int)
  ap.add_argument('-res',default=1000,type=int)
  ap.add_argument('-outf',default='thlay')
  ap.add_argument('-model_sigma_w',default=1.,type=float)
  ap.add_argument('-model_sigma_b',default=1.,type=float)
  ap.add_argument('-no_sqrt_normalise_w',action='store_true')
  ap.add_argument('-no_glorot_uniform',action='store_true')
  ap.add_argument('-glorot_normal',action='store_true')
  ap.add_argument('-model_resid',default=False,type=bool)
  ap.add_argument('-p',default=.1,type=float)
  ap.add_argument('-target_fp',default=.01,type=float)
  ap.add_argument('-target_fn',default=.01,type=float)
  ap.add_argument('-clock_avg_rate',default=.1,type=float) #Track timings
  ap.add_argument('-threshold_accuracy_tolerance',default=.1,type=float)
  ap.add_argument('-fpfn_ratios',default=[2,.5],nargs='+',type=float)
  ap.add_argument('-target_fns',default=[.1],nargs='+',type=float)
  ap.add_argument('-target_tolerance',default=.5,type=float)
  ap.add_argument('-no_stop_on_target',action='store_true')
  ap.add_argument('-iterate_minority',action='store_true')
  #Silly
  ap.add_argument('-lr_phase',default=0.,type=float)
  ap.add_argument('-lr_momentum',default=0.05,type=float)
  ap.add_argument('-lr_amplitude',default=0.,type=float)
  ap.add_argument('-x_max',default=10.,type=float)
  
  a=ap.parse_args()
  if not isdir('reports'):
    mkdir('reports')
  a.outf='reports/'++a.outf
  a.report_dir=a.outf+'_report'
  a.n_imb=len(a.imbalances)
  if a.resnet and a.loss not in ['resnet_cost','resnet_cost_layer',
                                 'coalescence_res_cost']:
    if a.single_layer_upd:
      a.loss='resnet_cost_layer'
    else:
      a.loss='resnet_cost'
  elif a.nnpca:
    a.loss='nn_pca_loss'
  elif a.loss=='distribution_flow_cost':
    a.w_init=1.
    a.target_increment=.9

  a.glorot_uniform=(not a.no_glorot_uniform) and (not a.glorot_normal)
  a.stop_on_target=not a.no_stop_on_target
  a.gmm_compensate_variances=not a.no_gmm_compensate_variances
  a.adam=not a.no_adam
  a.fpfn_ratios,a.target_fns,a.lrs=array(a.fpfn_ratios),array(a.target_fns),array(a.lrs)
  a.out_dir=a.outf+'_report'
  a.step=0
  if a.bs>1 and a.iterate_minority:
    print('Minority iterating unavailable since bs>1, continuing without')
    a.iterate_minority=False
  if len(a.lrs)==1:
    a.lr=a.lrs[0]
  if not a.bs:
    if a.mode=='mnist':
      a.bs=128
    else:
      a.bs=1
  if not a.model_inner_dims:
    if a.mode=='gmm':
      a.model_inner_dims=[32,16]
    elif a.mode==['mnist','rbd24']:
      a.model_inner_dims=[64,32,16]
    else:
      a.model_inner_dims=[64,32]
  return a

f_to_str=lambda x,p=True:f'{x:.3g}'.ljust(10) if p else f'{x:.3g}'

exp_to_str=lambda e:'exp_p'+f_to_str(e.p,p=False)+'fpt'+f_to_str(e.target_fp,p=False)+\
                    'fnt'+f_to_str(e.target_fn,p=False)+'lr'+f_to_str(e.lr,p=False)

fpfnp_lab=lambda e:'FP_t,FN_t,p='+f_to_str(e.target_fp,p=False)+','+\
                   f_to_str(e.target_fn,p=False)+','+f_to_str(e.p,p=False)

even_indices=lambda arr:[v for i,v in enumerate(arr) if not i%2]

def relu(x):return minimum(1,maximum(-1,x))

@jit
def f_unbatched(A,B,x,act=tanh):
  for a,b in zip(A,B):
    x=act(x@a+b)
  return x
f=vectorize(f_unbatched,excluded=[0,1],signature='(m)->(n)')

def resnet_unbatched(A,B,x,act=tanh):
  for a,b in zip(A,B):
    x+=act(a@x+b)
  return x # final layer: sum components, check + or -.
resnet=vectorize(resnet_unbatched,excluded=[0,1],signature='(m)->(m)')

#Default: force two directions of growth,values specified are logarithmic
def svd_cost(A,B,x,act=tanh,top_growth=array([2,1]),imp=resnet,
             eps=1e-8,dimension=4,vol_growth=0.):
  if dimension is None: dimension=x.shape[1]
  iden=identity(x_dim)
  sqs=jsm(x**2,axis=1)
  dists_init=sqs+expand_dims(sqs,-1)-2*x@x.T
  x=imp(A,B,x)
  sqs=jsm(x**2,axis=1)
  dists_final=sqs+expand_dims(sqs,-1)-2*x@x.T
  distortion=dists_final/(dists_init+iden)
  growth=log(svdvals(distortion)[:dimension])
  return (top_growth-growth[:len(top_growth)])**2+(jsm(growth)-vol_growth)**2

def metric_cost(A,B,x,y,act=tanh,imp=resnet,target_dists=None):
  if target_dists==None:
    target_dists=y^expand_dims(y,-1).T
  x=imp(A,B,x)
  sqs=jsm(x**2,axis=1)
  return jsm(((sqs+expand_dims(sqs,-1)-2*x@x.T)-target_dists)**2)

def coalescence_cost(A,B,x,y,U,V,act=tanh,tol=1e-4):
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
  for a,b in zip(A,B):
    x=act(x@a+b)
    iden=identity(bs)
    sqs=jsm(x**2,axis=1)
    dists=iden+tol+sqs+expand_dims(sqs,-1)-2*x@x.T
    ldists=log(dists)
    ret+=jsm((old_ldists-ldists)*target_growth/dists) #Don't worry so much about far pts
    old_ldists=ldists
  return (ret+y@log(tol+1-x)+ny@log(tol+1+x))[0]

def nn_cost_expansion(A,B,xp,xn,contraction=False,act=tanh,tol=1e-2,imp=f):
  ret=0.
  dists_init=jsm((xp-xn)**2,axis=1)
  xp,xn=imp(A,B,xp,act=act),imp(A,B,xn,act=act)
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

def coalescence_res_cost(A,B,x,y,U,V,act=tanh,tol=1e-2):
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
  for a,b in zip(A,B):
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

def distribution_flow_cost(A,B,x,y,U,V,w_init,act=tanh,tol=1e-8):
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
  for i,(a,b) in enumerate(zip(A,B)):
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

def resnet_cost(A,B,x,y,U,V,act=tanh,eps=1e-8): #Already vectorised
  c=0
  n=len(A)
  UmV=y*(U+V)-U #weight positives and negatives by importance
  for a,b in zip(A,B):
    dx=act(x@a.T+b)
    #sg_dx=dx.T*UmV
    sg_dx=dx.T*(2*y-1)
    #c-=log(eps+jsm(sg_dx.T@sg_dx)) #force + and - difference direction correlations
    #c-=log(eps+jsm(sg_dx.T@sg_dx))
    #c-=jsm(log(1+eps+sg_dx.T@sg_dx))
    #c-=jsm(sg_dx.T@sg_dx)
    x+=dx
  return c-jsm(UmV*x.T) # final layer: sum components, check + or -.

def resnet_cost_layer(A,B,i,c,d,x,y,U,V,act=tanh,eps=1e-8): #Already vectorised
  ret=0.
  UmV=y*(U+V)-U
  for j,(a,b) in enumerate(zip(A,B)):
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

def l1_soft(A,B,x,y,U,V,softness=.1,act=tanh):
  y_smooth=f(A,B,x,act=act)
  a_p,a_n=y_smooth[y],y_smooth[~y]
  cts_fp=jsm(1.+a_n)
  cts_fn=jsm(1.-a_p)
  return U*cts_fp+V*cts_fn

def l1(A,B,x,y,U,V,act=tanh):
  return l1_soft(A,B,x,y,U,V,0.,act)

def cross_entropy(A,B,x,y,U,V,act=tanh,eps=1e-8):
  y_smooth=f(A,B,x,act=act)
  a_p,a_n=y_smooth[y],y_smooth[~y] # 0<y''=(1+y')/2<1
  cts_fn=-jsm(log(eps+1.+a_p)) #y=1 => H(y,y')=-log(y'')=-log((1+y')/2)
  cts_fp=-jsm(log(eps+1.-a_n)) #y=0 => H(y,y')=-log(1-y'')=-log((1-y')/2)
  return U*cts_fp+V*cts_fn

def cross_entropy_soft(A,B,x,y,U,V,act=tanh,normalisation=False,softness=.1,eps=1e-8):
  y_smooth=f(A,B,x,act=act)
  a_p,a_n=y_smooth[y],y_smooth[~y] # 0<y''=(1+y')/2<1
  cts_fn=-(1-softness)*jsm(log(eps+1.+a_p))-softness*jsm(log(eps+1.-a_p))
  cts_fp=-(1-softness)*jsm(log(eps+1.-a_n))-softness*jsm(log(eps+1.+a_n))
  return U*cts_fp+V*cts_fn

def nn_pca_loss(w_c,b_c,w_e,b_e,x,x_targ,eps=1e-8):
  x_c=f(w_c,b_c,x)
  l=cov(x_c)
  x_c_vars=var(x_c,axis=0)
  #x_c_vars/=jsm(x_c_vars)
  l+=jsm(x_c_vars[1:]/(eps+x_c_vars[:-1]))

  x_p=f(w_e,b_e,x_c)
  l+=jsm((x_p-x_targ)**2)
  return l#log(jsm(w_c[-1]@w_c[-1].T)))

dmetric_cost=value_and_grad(metric_cost,argnums=[0,1])
dsvd_cost=value_and_grad(svd_cost,argnums=[0,1])
dl1=value_and_grad(l1,argnums=[0,1])
dresnet_cost=value_and_grad(resnet_cost,argnums=[0,1])
dresnet_cost_layer=value_and_grad(resnet_cost_layer,argnums=[3,4])
dcross_entropy=value_and_grad(cross_entropy,argnums=[0,1])
dcross_entropy_soft=value_and_grad(cross_entropy_soft,argnums=[0,1])
dl1_soft=value_and_grad(l1_soft,argnums=[0,1])
ddistribution_flow_cost=value_and_grad(distribution_flow_cost,argnums=[0,1])
dcoalescence_cost=value_and_grad(coalescence_cost,argnums=[0,1])
dcoalescence_res_cost=value_and_grad(coalescence_res_cost,argnums=[0,1])
#dnn_cost_contraction=value_and_grad(nn_cost_contraction,argnums=[0,1])
dnn_cost_expansion=value_and_grad(nn_cost_expansion,argnums=[0,1])
dlosses={'loss':dcross_entropy,'l1':dl1,'cross_entropy':dcross_entropy,
         'l1_soft':dl1_soft,'cross_entropy_soft':dcross_entropy_soft,
         'resnet_cost':dresnet_cost,'resnet_cost_layer':dresnet_cost_layer,
         'nn_pca_loss':nn_pca_loss,'distribution_flow_cost':ddistribution_flow_cost,
         'coalescence_cost':dcoalescence_cost,
         'coalescence_res_cost':dcoalescence_res_cost}

def init_layers(k,layer_dimensions,sigma_w=1.,sigma_b=1.,
                no_sqrt_normalise=False,resnet=False,glorot_uniform=False,
                glorot_normal=False,orthonormalise=False):
  k1,k2=split(k)
  wb=[]
  n_steps=len(layer_dimensions)-1
  w_k=split(k1,n_steps)
  b_k=split(k2,n_steps)
  a=[]
  b=[]
  for i,(k,l,d_i,d_o) in enumerate(zip(w_k,b_k,layer_dimensions,layer_dimensions[1:])):
    if resnet:
      a.append(zeros(shape=(d_i,d_o)))
      b.append(zeros(shape=d_o))
    elif glorot_uniform:
      a.append((2*(6/(d_i+d_o))**.5)*(uniform(shape=(d_i,d_o),key=k)-.5))
      b.append(zeros(shape=d_o))
    elif glorot_normal:
      a.append(((2/(d_i+d_o))**.5)*(normal(shape=(d_i,d_o),key=k)))
      b.append(zeros(shape=d_o))
    else:
      a.append(normal(shape=(d_i,d_o),key=k))
      b.append(normal(shape=d_o,key=l))
  #if orthonormalise:
  #  for r,d_i,d_o in zip(ret,layer_dimensions,layer_dimensions[1:]):
  #    if d_i>d_o:
  #      r[0]=svd(r[0],full_matrices=False)[0]
  #    else:
  #      r[0]=svd(r[0],full_matrices=False)[2]
  #for r in ret:
  #  r[i][0]*=sigma_w
  #  ret[i][1]*=sigma_b
  #if no_sqrt_normalise:
  #  return ret
  #for r,d_i in zip(ret,layer_dimensions[1:]):
  #  ret[i][0]/=d_i**.5
  return a,b

def sample_x(bs,key):
  return 2*a.x_max*uniform(shape=(bs,2),key=key)-a.x_max

def colour_rescale(fpfn):
  l=log(array(fpfn))-log(a.fpfn_min)
  l/=log(a.fpfn_max)-log(a.fpfn_min)
  return jet(l)

def mk_experiment(p,thresh,fpfn_ratio,target_fn,lr,a):
  e=SimpleNamespace()
  e.bs=a.bs
  e.fpfn_ratio=float(fpfn_ratio)
  e.target_tolerance=a.target_tolerance
  e.steps_to_target=False
  e.avg_rate=a.avg_rate
  e.dw_l2=e.w_l2=0
  e.w_model=[v.copy() for v in a.w_model_init[0]],[v.copy() for v in a.w_model_init[1]]
  e.history_len=a.history_len
  e.step=0

  e.adam_V=[u*0. for u in e.w_model[0]],[u*0. for u in e.w_model[1]]
  e.adam_M=0.

  e.lr=float(lr)
  e.p=float(p) #"imbalance"
  #e.p_its=int(1/p) #if repeating minority class iterations
  e.target_fp=target_fn*fpfn_ratio
  e.target_fn=target_fn
  e.recent_memory_len=int((1/a.avg_rate)*max(1,1/(e.bs*min(e.target_fp,e.target_fn))))

  e.FPs=cyc(e.recent_memory_len,e.target_fp)
  e.FNs=cyc(e.recent_memory_len,e.target_fn)
  e.loss_vals=cyc(e.recent_memory_len,1)#a.bs)
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
    (a.x_train,a.y_train),(a.x_test,a.y_test),(_,a.x_columns)=rbd24()
    a.p=sum(a.y_train)/len(a.y_train)
    a.p_test=sum(a.y_test)/len(a.y_test)
    a.imbalances=[a.p]
    a.in_dim=len(a.x_train[0])
    if a.epochs:
      a.epoch_num=1
      a.offset=0
      a.x_train,a.y_train=shuffle_xy(k1,a.x_train,a.y_train)

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

  elif a.mode=='all':
    a.target_sigma_w=.75
    a.target_sigma_b=2.
  
    a.w_target=init_layers(k1,a.target_shape,a.target_sigma_w,a.target_sigma_b)

  if a.resnet:
    a.model_shape=[a.in_dim]*a.resnet
    a.zersq=zeros((a.in_dim,a.in_dim))
    a.zerarr=zeros(a.in_dim)
  else:
    a.model_shape=[a.in_dim]+a.model_inner_dims+[1]

  a.w_model_init=init_layers(k2,a.model_shape,sigma_w=a.model_sigma_w,
                             sigma_b=a.model_sigma_b,resnet=a.resnet,
                             glorot_uniform=a.glorot_uniform,
                             no_sqrt_normalise=a.no_sqrt_normalise_w,
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

  if a.mode=='all':
    a.targets=list(zip(a.fpfn_ratios,a.target_fns))
    experiments=[mk_experiment(p,a.thresholds[p],fpfn_ratio,target_fn,a.lr,a)\
                 for p in a.imbalances for (fpfn_ratio,target_fn) in a.targets\
                 for lr in a.lrs]
  elif a.mode=='adaptive_lr':
    experiments=[mk_experiment(.1,a.thresholds[.1],1.,.01,lr,a) for\
                 lr in a.lrs]
  elif a.mode in ['unsw','gmm','mnist','rbd24']:
    experiments=[mk_experiment(float(p),0.,fpfn_ratio,float(p*target_fn),a.lr,a)\
                 for p in a.imbalances for fpfn_ratio in a.fpfn_ratios\
                 for target_fn in a.target_fns]
  elif a.mode in ['imbalances']:
    experiments=[mk_experiment(float(p),float(a.thresholds[float(p)]),fpfn_ratio,
                               float(p*target_fn),a.lr,a)\
                 for p in a.imbalances for fpfn_ratio in a.fpfn_ratios\
                 for target_fn in a.target_fns]

  if a.mode=='gmm':
    min_dist=0
    while min_dist<a.gmm_min_dist:
      a.means=2*a.x_max*uniform(k5,(2*a.n_gaussians,a.in_dim))-a.x_max
      min_dist=min([jsm((A-B)**2) for b,A in enumerate(a.means) for B in a.means[b+1:]])
    a.variances=2*a.x_max*uniform(k6,2*a.n_gaussians)*a.gmm_spread #hmm
  return experiments

def shuffle_xy(k,x,y):
  shuff=permutation(k,len(y)).__array__()
  x,y=x[shuff],y[shuff]
  return x,y

activations={'tanh':tanh,'relu':relu,'softmax':softmax}

def get_xy(a,imbs,bs,k):
  k1,k2,k3,k4=split(k,4)
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
      z=normal(k1,shape=(bs,a.in_dim))
      if a.gmm_compensate_variances:
        z/=(-2*log(p))**.5
      x=z*a.variances[mix,None]+a.means[mix]
      ret[float(p)]=x,y
  elif a.mode=='rbd24':
    if a.epochs: #epochs: sample randomly shuffled dataset without replacement
      next_offset=a.offset+a.bs
      if next_offset>len(a.y_train):
        a.epoch_num+=1
        a.x_train,a.y_train=shuffle_xy(k1,a.x_train,a.y_train)
        a.offset=0
        next_offset=a.bs
      ret={float(a.p):(a.x_train[a.offset:next_offset],
                       a.y_train[a.offset:next_offset])}
      a.offset=next_offset
    else: #sample without replacement
      batch_indices=choice(k1,len(a.y_train),shape=(a.bs,))
      ret={float(a.p):(a.x_train[batch_indices],
                      array(a.y_train[batch_indices]))}

  elif a.mode=='unsw':
    batch_indices=choice(k1,len(a.y_train),shape=(a.bs,))
    ret={float(p):(a.x_train[batch_indices],
                   array(a.y_train[batch_indices]==a.cats[p])) for p in a.imbalances}
  elif a.mode=='mnist': #force imbalance of mnist dataset
    n_pos=[int(binomial(k,a.bs,p)) for k,p in zip(split(k1,a.n_imb),a.imbalances)]
    x_pos=[choice(k,a.x_train_pos,shape=(np,)) for k,np in\
           zip(split(k2,a.n_imb),n_pos)]
    x_neg=[choice(k,a.x_train_neg,shape=(a.bs-np,)) for k,np in\
           zip(split(k3,a.n_imb),n_pos)]
    x_all=[(xn if not len(xp) else(xp if not len(xn) else concatenate([xp,xn]))) for\
           xp,xn in zip(x_pos,x_neg)]
    perms=[permutation(k,a.bs) for k in split(k4,a.n_imb)]

    ret={p:(x[perm],perm<np) for p,x,np,perm in zip(a.imbalances,x_all,n_pos,perms)}
  else:
    ret=dict()
    for p in imbs:
      x=sample_x(a.bs,k1)
      ret[float(p)]=x,f(a.w_target[0],a.w_target[1],x).flatten()

  return ret[ret_single] if ret_single else ret

def evaluate_fp_fn(e,y_p,y_t):
  e.FP=int(nsm(y_p&(~y_t)))/e.bs #Stop jax weirdness after ADC
  e.FN=int(nsm(y_t&(~y_p)))/e.bs
  e.cost=(e.FP/e.target_fp+e.FN/e.target_fn)
  e.FPs[e.step]=e.FP
  e.FNs[e.step]=e.FN
  e.trad_avg=False
  if e.trad_avg:
    e.fp=e.FPs.avg()
    e.fn=e.FNs.avg()
  else:
    e.fp_amnt=e.avg_rate*min(1,e.bs*e.fp)
    e.fn_amnt=e.avg_rate*min(1,e.bs*e.fn)
    e.fp*=(1-e.fp_amnt)
    e.fp+=e.fp_amnt*e.FP
    e.fn*=(1-e.fn_amnt)
    e.fn+=e.fn_amnt*e.FN

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
      e.history.dw=even_indices(e.history.dw)
      e.history.loss_vals=even_indices(e.history.loss_vals)
      e.history.l//=2
  if e.fp<e.target_tolerance*e.target_fp and\
     e.fn<e.target_tolerance*e.target_fn and\
     not e.steps_to_target:
       e.steps_to_target=e.step
    

def compute_U_V(fp,fn,target_fp,target_fn,p,sm=False,p_scale=.5,scale_before_sm=True):
  #e.p_empirical*=(1-min(e.fp_amnt,e.fn_amnt))
  #e.p_empirical+=(1-min(e.fp_amnt,e.fn_amnt))*jsm(e.y_t)/e.bs
  #U,V=log(1+fp/target_fp),log(1+fn/target_fn)
  #U=u/(u+v)
  #V=v/(u+v)
  #U,V=softmax(array([gamma1*fp/target_fp,gamma1*fn/target_fn]))
  #U,V=softmax(array([fp/target_fp,fn/target_fn]))
  #U,V=fp/target_fp,fn/target_fn
  if sm:
    if scale_before_sm:
      U,V=softmax(array([fp/target_fp,fn/(target_fn*p**p_scale)]))
    else:
      U,V=softmax(array([fp/target_fp,fn/target_fn]))
  else:
    U,V=fp/target_fp,fn/target_fn
  if not scale_before_sm:
    V/=p**p_scale #scale dfn with the imbalance
  nUV=U+V
  if nUV>0:
    U/=nUV
    V/=nUV
  return U,V

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

def update_weights(a,e,upd,start=None,end=None):
  offset=0 if start is None else start
  e.dw_l2=e.w_l2=0
  adva=e.adam_V[0][start:end]
  advb=e.adam_V[1][start:end]
  #wmoda=e.w_model[0][start:end]
  #wmodb=e.w_model[1][start:end]
  if a.adam:
    e.adam_M*=(1-a.gamma2)
    for i,(u,v) in enumerate(zip(*upd)):
      adva[i]*=(1-a.gamma1)
      advb[i]*=(1-a.gamma1)
      adva[i]+=a.gamma1*u
      advb[i]+=a.gamma1*v
      e.adam_M+=a.gamma2*(nsm(u**2)+nsm(v**2))
    #for k in upd:
    for i,(s,t) in enumerate(zip(adva,advb)):
      delta_u=e.lr*s/(e.adam_M**.5+1e-8)
      delta_v=e.lr*t/(e.adam_M**.5+1e-8)
      e.w_model[0][i+offset]-=delta_u
      e.w_model[1][i+offset]-=delta_v
      e.w_l2+=nsm(e.w_model[0][i+offset]**2)+\
              nsm(e.w_model[0][i+offset]**2)
      e.dw_l2+=nsm(delta_u**2)+nsm(delta_v**2)
  else:
    for i,(s,t) in enumerate(zip(upd[0],upd[1])):
      delta_u=e.lr*s
      delta_v=e.lr*t
      e.dw_l2+=nsm(delta_u**2)+nsm(delta_v**2)
      e.w_model[0][i+offset]-=delta_u
      e.w_model[1][i+offset]-=delta_v
      e.w_l2+=nsm(u**2)+nsm(v**2)

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

def plot_2d(experiments,fd_tex,a,act,line,k):
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
    y_p=concat([(f(e.w_model[0],e.w_model[1],
                   _x,act=act)>0).flatten() for _x in x_split])
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
      y_p_s=f(e.w_model[0],e.w_model[1],x_t,act=act).flatten()>0
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

get_cost=lambda e:e.history.cost
get_lr=lambda e:e.history.lr
get_dw=lambda e:e.history.dw
def plot_historical_statistics(experiments,fd_tex,a,smoothing=100):
  if fd_tex:
    print('\\subsection{Historical statistics}',file=fd_tex)
  for get_var,yl,desc in zip([get_cost,get_lr,get_dw],
                             ['log(1e-8+fp/target_fp+fn/target_fn)','log(lr)',
                              'log(dw)'],
                             ['Loss','Learning_rate','Change_in_weights']):
    for e in experiments:
      arr=[log10(a) for a in get_var(e)]
      if smoothing:
        ker=ones(smoothing)/smoothing
        arr=convolve(array(arr,dtype=float),ker,'same')
      if a.mode=='unsw':
        lab=a.cats[e.p]
      else:
        lab=fpfnp_lab(e)
      plot([log10(a) for a in arr],label=lab)
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
    plot(log10(smoothed_fp),log10(smoothed_fn),label=fpfnp_lab(e))
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

def report_progress(a,experiments,line,act,k):
  k1,k2=split(k)
  print('|'.join([t.ljust(10) for t in ['p','target_fp','target_fn',
                                        'lr','fp','fn','w','dw','U','V','complete']]))
  for e in experiments:
    print('|'.join([f_to_str(t) for t in [e.p,e.target_fp,e.target_fn,e.lr,
                                          e.fp,e.fn,e.w_l2,e.dw_l2,e.U,e.V]])+'|'+\
          (f_to_str(e.steps_to_target) if e.steps_to_target else 'no'))
  for e in experiments:
    print('Recent batches: FPs:',
          e.FPs[e.step-5:e.step],'FNs:',
          e.FNs[e.step-5:e.step])
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
    try:
      print(vars(a)[line.split('?')[0]])
    except KeyError:
      print('"',line.split('?')[0],'" not in a')
    return
  elif '*' in line:
    print('a variables:')
    [print(v) for v in vars(a).keys()]
    return

  fd_tex=False
  if 'x' in line:
    print('Bye!')
    exit()
  if 'e' in line and a.mode in ['mnist','rbd24']:
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
        print('fp_train_0,fn_train_0:',
              f_to_str(sum([nsm(f(e.w_model[0],e.w_model[1],
                                  x_train_neg[i:i+a.bs])>0) for\
                            i in range(0,len(x_train_neg),a.bs)])/len(a.x_train)),
              f_to_str(sum([nsm(f(e.w_model[0],e.w_model[1],
                                  x_train_pos[i:i+a.bs])<=0) for\
                            i in range(0,len(x_train_pos),a.bs)])/len(a.x_train)))
        print('fp_test_0,fn_test_0:',
              f_to_str(sum([nsm(f(e.w_model[0],e.w_model[1],
                                  x_test_neg[i:i+a.bs])>0) for\
                            i in range(0,len(x_test_neg),a.bs)])/len(a.x_test)),
              f_to_str(sum([nsm(f(e.w_model[0],e.w_model[1],
                                  x_test_pos[i:i+a.bs])<=0) for\
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
              f_to_str(nsm(f(e.w_model[0],e.w_model[1],
                             x_train_neg)>0)/len(a.x_train)),
              f_to_str(nsm(f(e.w_model[0],e.w_model[1],
                             x_train_pos)<=0)/len(a.x_train)))
        print('fp_test_0,fn_test_0:',
              f_to_str(nsm(f(e.w_model[0],e.w_model[1],
                             x_test_neg)>0)/len(a.x_test)),
              f_to_str(nsm(f(e.w_model[0],e.w_model[1],
                             x_test_pos)<=0)/len(a.x_test)))
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

            report_fns.append(f_to_str(e.p*nsm(f(e.w_model[0],e.w_model[1],
                                                 a.x_test_pos)<=0)/\
                                       (.1*len(a.x_test))))
            report_fps.append(f_to_str((1-e.p)*nsm(f(e.w_model[0],e.w_model[1],
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
  for k,v in a.time_avgs.items():
    print(k,log10(v))
    if fd_tex: print('\\texttt{'+k.replace('_','\\_')+'}&'+\
                     f_to_str(log10(v))+'\\\\\n',file=fd_tex)
  if fd_tex: print('\\end{tabular}',file=fd_tex)

  if a.in_dim==2 and 'c' in line:
    plot_2d(experiments,fd_tex,a,act,line,k1)

  if 'i' in line:
    model_desc='Model shape:\n'+('->'.join([str(l) for l in a.model_shape]))+'\n'
  
    if a.no_glorot_uniform:
      model_desc+='\n'.join(['- matrix weight variance:'+\
                             f_to_str(a.model_sigma_w,p=False),
                             '- bias variance:'+f_to_str(a.model_sigma_b,p=False),
                             '- residual initialisation:'+\
                             f_to_str(a.model_resid,p=False),
                             'sqrt variance correction:'+\
                             str(not a.no_sqrt_normalise_w)])
    else:
      model_desc+='\n- Glorot uniform initialisation'
    model_desc+='\n- batch size:'+str(a.bs)
    if a.mode=='gmm':
      model_desc+='\n- learning rate:'+str(a.lr)
    if a.iterate_minority:
      model_desc+='Iterating minority class 1/p times'
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

  if fd_tex:
    print('\\end{document}',file=fd_tex,flush=True)
    fd_tex.close()

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

def rbd24(preproc=True,split_test_train=True,rescale_log=True,
          raw_pickle_file=str(Path.home())+'/data/rbd24/rbd24.pkl',
          processed_pickle_file=str(Path.home())+'/data/rbd24/rbd24_proc.pkl'):
  if split_test_train and preproc and isfile(processed_pickle_file):
    print('Loading procesed pickle...')
    with open(processed_pickle_file,'rb') as fd:
      return load(fd)
  if isfile(raw_pickle_file):
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

  if preproc:
    df=preproc_rbd24(df,rescale_log=rescale_log)
  if not split_test_train:
    return df
  x=get_dummies(df.drop(['label','user_id','timestamp'],axis=1))
  x_cols=x.columns
  x=x.__array__().astype(float)
  y=df.label.__array__().astype(bool)
  l=len(y)
  split_point=int(l*.7)
  x_train,x_test=x[:split_point],x[split_point:]
  y_train,y_test=y[:split_point],y[split_point:]
  if split_test_train and preproc:
    print('Saving processed pickle...')
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
    if max([npl10(1+m) for m in maximums])>rescale_log:
      print('Note that rescale factor will not map values to be <=1')
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
