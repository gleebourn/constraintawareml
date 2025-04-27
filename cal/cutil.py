from argparse import ArgumentParser
from pickle import load,dump
from types import SimpleNamespace
from itertools import count
from csv import writer
from os.path import isdir,isfile
from os import mkdir,listdir,get_terminal_size,devnull
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from pathlib import Path
from select import select
from sys import stdin,stdout
from time import perf_counter
from itertools import accumulate
from json import dumps# as jump
from numpy import inf,unique as npunique,array as nparr,min as nmn,number,\
                  max as nmx,sum as nsm,log10 as npl10,round as rnd,geomspace
from numpy.random import default_rng #Only used for deterministic routines
from sklearn.preprocessing import StandardScaler
from jax.numpy import array,vectorize,zeros,log,log10,flip,maximum,minimum,pad,\
                      concat,exp,ones,linspace,array_split,reshape,corrcoef,eye,\
                      concatenate,unique,cov,expand_dims,identity,\
                      diag,average,triu_indices,sum as jsm,max as jmx
from jax.numpy.linalg import svdvals
from jax.scipy.signal import convolve
from jax.nn import tanh,softmax
from jax.random import uniform,normal,split,key,choice,binomial,permutation
from jax.tree import map as jma,reduce as jrd
from jax import grad,value_and_grad,jit,config,vmap
from jax.lax import scan,while_loop,switch
from jax.lax.linalg import svd
from sklearn.utils.extmath import cartesian
from sklearn.model_selection import train_test_split
from pandas import read_csv,concat
from matplotlib.pyplot import imshow,legend,show,scatter,xlabel,ylabel,\
                              gca,plot,title,savefig,close,rcParams,yscale
from matplotlib.patches import Patch
from matplotlib.cm import jet
rcParams['font.family'] = 'monospace'
from pandas import read_pickle,read_parquet,concat,get_dummies
from traceback import format_exc
from sklearn.ensemble import RandomForestRegressor
from csv import reader,writer,DictWriter
from imblearn.combine import SMOTETomek,SMOTEENN

def read_input_if_ready():
  return stdin.readline().lower() if stdin in select([stdin],[],[],0)[0] else ''

def resnet(w,x,act=tanh,first_layer_no_skip=True):
  if first_layer_no_skip:
    x=act(x@w[0][0]+w[1][0])
    A=w[0][1:]
    B=w[1][1:]
  for a,b in zip(*w):
    #x=pad_or_trunc(x,len(b))+act(x@a+b)
    x+=act(x@a+b)
  return jsm(x,axis=1) # final layer: sum components, check + or -.

def change_worst_j(states,consts,benches,shapes,n_to_change,k):
  ks=split(k,n_to_change)
  ranked=[x[0] for x in sorted(enumerate(benches),key=lambda x:-x[1]['div'])]
  worst=ranked[:n_to_change]
  best=ranked[n_to_change:]
  for i,j,l in zip(best,worst,ks):
    c=consts[i]
    l0,l=split(l)
    z=normal(l,2)
    consts[j]={'beta':c['beta'],'tfp':c['tfp'],'tfn':c['tfn'],'beta1':c['beta1'],'beta2':c['beta2'],
                'lr':c['lr']*exp(z[0]),'reg':c['reg']*exp(z[1]),'eps':1e-8}  
    states[j]={'w':init_layers(shapes[i],init_dist,k=l0),
               'm':init_layers(shapes[i],'zeros'),
               'v':init_layers(shapes[i],'ones')}

def plot_stopping_times(experiments,fd_tex,report_dir):
  for e in experiments: e.fpfn_target=float(e.fpfn_target)
  completed_experiments=[e for e in experiments if e.steps_to_target]
  try:
    fpfn_targets=list(set([e.fpfn_targets for e in experiments]))
    for rat in fpfn_targets:
      x=[log10(e.p) for e in completed_experiments if e.fpfn_target==rat]
      y=[log10(e.steps_to_target) for e in completed_experiments if\
         e.fpfn_target==rat]
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

def mod_desc(a):
  return\
  '- Model shape:\n'+('->'.join([str(l) for l in a.model_shape]))+\
  '\n- Activation: '+a.activation+'\n'+\
  '- Implementation: '+a.implementation+'\n'+\
  '- Model initialisation: '+str(a.initialisation)+'\n'\
  '- Matrix weight multiplier: '+str(a.mult_a)+'\n'\
  '- Sqrt variance correction:'+str(a.sqrt_normalise_a)+'\n'\
  '- Batch size:'+str(a.bs)+'\n'\
  '- learning rate(s):'+f_to_str(a.lrs)
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
