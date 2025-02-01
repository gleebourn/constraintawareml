from argparse import ArgumentParser
from pickle import load,dump
from types import SimpleNamespace
from csv import writer
from jax.numpy import array,vectorize,zeros,log,flip,maximum,minimum,concat,exp,ones,\
     sum as nsm,max as nmx
from jax.scipy.signal import convolve
from jax.nn import tanh,softmax
from jax import grad
from jax.random import uniform,normal,split,key,choice
from sklearn.utils.extmath import cartesian
from pandas import read_csv
from matplotlib.pyplot import imshow,legend,show,scatter,xlabel,ylabel,\
                              gca,plot,title,savefig,close
from matplotlib.patches import Patch
from matplotlib.cm import jet

leg=lambda t=None,h=None,l='upper right':legend(fontsize='x-small',loc=l,
                                                handles=h,title=t)

def init_ensemble():
  ap=ArgumentParser()
  ap.add_argument('mode',default='all',
                  choices=['single','all','adaptive_lr','imbalances','unsw','gmm'])
  ap.add_argument('-n_gaussians',default=4,type=int)
  ap.add_argument('-reporting_interval',default=10,type=int)
  ap.add_argument('-gmm_spread',default=.025,type=float)
  ap.add_argument('-gmm_scatter_samples',default=10000,type=int)
  ap.add_argument('-force_batch_cost',default=0.,type=float)
  ap.add_argument('-no_adam',action='store_true')
  ap.add_argument('-adaptive_threshold',action='store_true')
  ap.add_argument('-act',choices=['relu','tanh','softmax'],default='tanh')
  ap.add_argument('--seed',default=20255202,type=int)
  ap.add_argument('-n_splits_img',default=100,type=int)
  ap.add_argument('-imbalances',type=float,default=[.01,.02,.1],nargs='+')
  ap.add_argument('-in_dim',default=2,type=int)
  ap.add_argument('-unsw_cat_thresh',default=.1,type=int)
  ap.add_argument('-lr_resolution',default=16,type=int)
  ap.add_argument('-all_resolution',default=5,type=int)
  ap.add_argument('-max_history_len',default=16384,type=int)
  ap.add_argument('-weight_normalisation',default=0.,type=float)
  ap.add_argument('-orthonormalise',action='store_true')
  ap.add_argument('-saving_interval',default=100,type=int)
  ap.add_argument('-lr_init_min',default=1e-4,type=float)
  ap.add_argument('-lr_init_max',default=1e-2,type=float)
  ap.add_argument('-lr_min',default=1e-5,type=float)
  ap.add_argument('-lr_max',default=1e-1,type=float)
  ap.add_argument('-lrs',default=[1e-4],type=float,nargs='+')
  ap.add_argument('-lr_update_interval',default=1000,type=int)
  ap.add_argument('-fpfn_memory_len',default=1000,type=int)
  ap.add_argument('-gamma1',default=.1,type=float) #1- beta1 in adam
  ap.add_argument('-gamma2',default=.001,type=float)
  ap.add_argument('-avg_rate',default=.1,type=float)#binom avg parameter
  ap.add_argument('-unsw_test',default='~/data/UNSW_NB15_testing-set.csv')
  ap.add_argument('-unsw_train',default='~/data/UNSW_NB15_training-set.csv')
  ap.add_argument('-model_inner_dims',default=[32,16,8],type=int,nargs='+')
  ap.add_argument('-bs',default=1,type=int)
  ap.add_argument('-res',default=1000,type=int)
  ap.add_argument('-outf',default='thlay')
  ap.add_argument('-model_sigma_w',default=1.,type=float)#.3
  ap.add_argument('-model_sigma_b',default=1.,type=float)#0.
  ap.add_argument('-no_sqrt_normalise_w',action='store_true')
  ap.add_argument('-no_glorot_uniform',action='store_true')
  ap.add_argument('-model_resid',default=False,type=bool)
  ap.add_argument('-p',default=.1,type=float)
  ap.add_argument('-target_fp',default=.01,type=float)
  ap.add_argument('-target_fn',default=.01,type=float)
  ap.add_argument('-clock_avg_rate',default=.1,type=float) #Track timings
  ap.add_argument('-threshold_accuracy_tolerance',default=.01,type=float)
  ap.add_argument('-target_fps',default=[.2,.05],nargs='+',type=float)
  ap.add_argument('-target_fns',default=[.1],nargs='+',type=float)
  #Silly
  ap.add_argument('-lr_phase',default=0.,type=float)
  ap.add_argument('-lr_momentum',default=0.05,type=float)
  ap.add_argument('-lr_amplitude',default=0.,type=float)
  ap.add_argument('-x_max',default=10.,type=float)
  
  a=ap.parse_args()
  a.target_fps,a.target_fns,a.lrs=array(a.target_fps),array(a.target_fns),array(a.lrs)
  a.out_dir=a.outf+'_report'
  a.step=0
  if len(a.lrs)==1:
    a.lr=a.lrs[0]
  return a

f_to_str=lambda x,p=True:f'{x:.5g}'.ljust(10) if p else f'{x:.5g}'

exp_to_str=lambda e:'exp_p'+f_to_str(e.p,p=False)+'fpt'+f_to_str(e.target_fp,p=False)+\
                    'fnt'+f_to_str(e.target_fn,p=False)+'lr'+f_to_str(e.lr,p=False)

fpfnp_lab=lambda e:'FP_t,FN_t,p='+f_to_str(e.target_fp,p=False)+','+\
                   f_to_str(e.target_fn,p=False)+','+f_to_str(e.p,p=False)

even_indices=lambda arr:[v for i,v in enumerate(arr) if not i%2]

def relu(x):return minimum(1,maximum(-1,x))

def f_unbatched(w,x,act=tanh):
  i=0
  while ('b',i) in w:
    x=act(x.dot(w[('w',i)])+w[('b',i)])
    i+=1
  if act==softmax:
    x-=.5
  return x
f=vectorize(f_unbatched,excluded=[0],signature='(m)->(n)')

def loss(w,x,y,U,V,act=tanh,normalisation=False):
  y_smooth=f(w,x,act=act)
  #cts_fp=nsm(maximum(V-U,y_smooth)[~y])
  #cts_fn=nsm(maximum(U-V,-y_smooth)[y])
  #cts_fp=nsm(maximum(0,y_smooth)[~y])
  #cts_fn=nsm(maximum(0,-y_smooth)[y])
  #cts_fp=nsm((~y)*(y_smooth+.5))
  #cts_fn=nsm(y*(.5-y_smooth))
  cts_fp=nsm((y_smooth+1.)[~y])
  cts_fn=nsm((1.-y_smooth)[y])
  if normalisation:
    l2=sum([nsm(w[('w',i)]**2)*nf for\
            i,nf in enumerate(normalisation)])
  return U*cts_fp+V*cts_fn+l2

dloss=grad(loss)

def init_layers(k,sigma_w,sigma_b,layer_dimensions,no_sqrt_normalise=False,
                resid=False,glorot_uniform=False,orthonormalise=False):
  k1,k2=split(k)
  wb=[]
  n_steps=len(layer_dimensions)-1
  w_k=split(k1,n_steps)
  b_k=split(k2,n_steps)
  ret=dict()
  for i,(k,l,d_i,d_o) in enumerate(zip(w_k,b_k,layer_dimensions,layer_dimensions[1:])):
    if glorot_uniform:
      ret[('w',i)]=2*(6/(d_i+d_o))**.5*(uniform(shape=(d_i,d_o),key=k)-.5)
      ret[('b',i)]=zeros(shape=d_o)
    else:
      ret[('w',i)]=normal(shape=(d_i,d_o),key=k)
      ret[('b',i)]=normal(shape=d_o,key=l)
    if resid:
      ret[('w',i)]+=eye(*ret[('w',i)].shape)
  if glorot_uniform:
    return ret
  if orthonormalise:
    for i,(d_i,d_o) in enumerate(zip(layer_dimensions,layer_dimensions[1:])):
      if d_i>d_o:
        ret[('w',i)]=svd(ret[('w',i)],full_matrices=False)[0]
      else:
        ret[('w',i)]=svd(ret[('w',i)],full_matrices=False)[2]
  for i in range(len(layer_dimensions)-1):
    ret[('w',i)]*=sigma_w
    ret[('b',i)]*=sigma_b
  if no_sqrt_normalise:
    return ret
  for i,d_i in enumerate(layer_dimensions[1:]):
    ret[('w',i)]/=d_i**.5
  return ret

def sample_x(bs,key):
  return 2*a.x_max*uniform(shape=(bs,2),key=key)-a.x_max

def colour_rescale(fpfn):
  l=log(array(fpfn))-log(a.fpfn_min)
  l/=log(a.fpfn_max)-log(a.fpfn_min)
  return jet(l)

def mk_experiment(w_model_init,p,thresh,target_fp,target_fn,lr,fpfn_memory_len,
                  max_history_len,avg_rate):
  e=SimpleNamespace()
  e.avg_rate=avg_rate
  e.FPs=[0]*fpfn_memory_len
  e.FNs=[0]*fpfn_memory_len
  e.dw_l2=e.w_l2=0
  e.w_model=w_model_init.copy()
  e.max_history_len=max_history_len
  e.fpfn_memory_len=fpfn_memory_len
  e.step=0

  e.lr=float(lr)
  e.p=float(p) #"imbalance"
  e.target_fp=target_fp
  e.target_fn=target_fn
  e.fp=target_fp
  e.fn=target_fn#.25 #softer start if a priori assume doing well

  e.U=e.V=1
  e.thresh=thresh
  e.history=SimpleNamespace(FP=[],FN=[],lr=[],cost=[],dw=[],resolution=1,l=0)
  return e

def init_experiments(a,global_key):
  a.global_key=global_key
  k1,k2,k3,k4,k5,k6=split(global_key,6)
  a.time_avgs=dict()
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

  a.target_sigma_w=.75
  a.target_sigma_b=2.
  
  a.w_target=init_layers(k1,a.target_sigma_w,a.target_sigma_b,
                         a.target_shape)
  a.model_shape=[a.in_dim]+a.model_inner_dims+[1]

  a.w_model_init=init_layers(k2,a.model_sigma_w,a.model_sigma_b,a.model_shape,
                             resid=a.model_resid,
                             glorot_uniform=not a.no_glorot_uniform,
                             no_sqrt_normalise=a.no_sqrt_normalise_w,
                             orthonormalise=a.orthonormalise)
  a.normalisation_factors=[a.weight_normalisation/(nsm(a.w_model_init[('w',i)]**2)*\
                                                   a.w_model_init[('w',i)].size) for\
                           i in range(len(a.model_shape)-1)]

  if a.mode=='unsw':
    a.imbalances=(a.y_train.value_counts()+a.y_test.value_counts())/\
                  (len(df_train)+len(df_test))
    a.cats={float(p):s for p,s in zip(a.imbalances,a.y_train.value_counts().index)}
    a.imbalances=a.imbalances[a.imbalances>a.unsw_cat_thresh]
  
  if a.mode in ['unsw','gmm']:
    a.thresholds={float(p):0. for p in a.imbalances}
  else:
    print('Finding thresholds...')
    
    thresholding_sample_size=int(1/(a.threshold_accuracy_tolerance**2*\
                                    a.imbalance_min))
    x_thresholding=sample_x(thresholding_sample_size,k3)
    
    y_t_cts=f(a.w_target,x_thresholding).flatten()
    y_t_cts_sorted=y_t_cts.sort()
    a.thresholds={float(p):y_t_cts_sorted[-int(p*len(y_t_cts_sorted))]\
                  for p in a.imbalances}
    
    print('Imbalances and thresholds')
    for i,t in a.thresholds.items(): print(i,t)
  
  a.loop_master_key=k4
  a.step=0

  if a.mode=='all':
    a.targets=list(zip(a.target_fps,a.target_fns))
    experiments=[mk_experiment(a.w_model_init,p,a.thresholds[p],target_fp,target_fn,lr,
                               a.fpfn_memory_len,a.max_history_len,a.avg_rate)\
                 for p in a.imbalances for (target_fp,target_fn) in a.targets\
                 for lr in a.lrs]

  elif a.mode=='single':
    experiments=[mk_experiment(a.w_model_init,.1,a.thresholds[.1],.01,.01,
                               a.lr,a.fpfn_memory_len,a.max_history_len,a.avg_rate)]
  elif a.mode=='adaptive_lr':
    experiments=[mk_experiment(a.w_model_init,.1,a.thresholds[.1],.01,.01,
                               lr,a.fpfn_memory_len,a.max_history_len,a.avg_rate) for\
                 lr in a.lrs]
  elif a.mode in ['unsw','gmm']:
    experiments=[mk_experiment(a.w_model_init,float(p),0.,float(p*target_fp),
                               float(p*target_fn),a.lr,a.fpfn_memory_len,
                               a.max_history_len,a.avg_rate)\
                 for p in a.imbalances\
                 for target_fp in a.target_fps\
                 for target_fn in a.target_fns]
  elif a.mode in ['imbalances']:
    experiments=[mk_experiment(a.w_model_init,float(p),float(a.thresholds[float(p)]),
                               float(p*target_fp),float(p*target_fn),
                               a.lr,a.fpfn_memory_len,a.max_history_len,a.avg_rate)\
                 for p in a.imbalances\
                 for target_fp in a.target_fps\
                 for target_fn in a.target_fns]

  if a.mode=='gmm':
    a.means=2*a.x_max*uniform(k5,(2*a.n_gaussians,a.in_dim))-a.x_max
    a.variances=2*a.x_max*uniform(k6,2*a.n_gaussians)*a.gmm_spread #hmm

  return experiments

activations={'tanh':tanh,'relu':relu,'softmax':softmax}

def get_xy(a,imbs,bs,k):
  k1,k2=split(k)
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
      mix=choice(k1,2*a.n_gaussians,shape=(bs,),p=probs)
      y=mix>=a.n_gaussians
      z=normal(k2,shape=(bs,a.in_dim))
      x=z*a.variances[mix,None]+a.means[mix]
      ret[float(p)]=x,y

  elif a.mode=='unsw':
    batch_indices=choice(k1,len(a.y_train),shape=(a.bs,))
    ret={float(p):(a.x_train[batch_indices],
                   array(a.y_train[batch_indices]==a.cats[p])) for p in a.imbalances}
  else:
    ret=dict()
    for p in imbs:
      x=sample_x(a.bs,k1)
      ret[float(p)]=x,f(a.w_target,x).flatten()

  return ret[ret_single] if ret_single else ret

def evaluate_fp_fn(e,y_p,y_t,cost_multiplier):
  e.FP=int(nsm(y_p&(~y_t))) #Stop jax weirdness after ADC
  e.FN=int(nsm(y_t&(~y_p)))
  e.fp_cost=(e.FP/(e.bs*e.target_fp))*cost_multiplier
  e.fn_cost=(e.FN/(e.bs*e.target_fp))*cost_multiplier
  e.FPs[e.step%e.fpfn_memory_len]=e.FP
  e.FNs[e.step%e.fpfn_memory_len]=e.FN
  window_div=min(e.step,e.fpfn_memory_len)
  e.fp_window=sum(e.FPs)/window_div
  e.fn_window=sum(e.FNs)/window_div
  e.cost_window=(e.fp_window/e.target_fp+e.fn_window/e.target_fn)/e.bs

  return e.fp_cost<1 and e.fn_cost<1

def update_fp_fn(e):
  e.fp_amnt=e.avg_rate*min(1,e.bs*e.fp)
  e.fn_amnt=e.avg_rate*min(1,e.bs*e.fn)
  #e.fp_amnt=1.
  #e.fn_amnt=1.
  e.fp*=(1-e.fp_amnt)
  e.fp+=e.fp_amnt*e.FP/e.bs
  e.fn*=(1-e.fn_amnt)
  e.fn+=e.fn_amnt*e.FN/e.bs
  if not e.step%e.history.resolution:
    e.history.FP.append(e.FP)
    e.history.FN.append(e.FN)
    e.history.lr.append(e.lr)
    e.history.cost.append(e.cost_window)
    e.history.l+=1
    if e.history.l>e.max_history_len:
      e.history.resolution*=2
      e.history.FP=even_indices(e.history.FP)
      e.history.FN=even_indices(e.history.FN)
      e.history.lr=even_indices(e.history.lr)
      e.history.cost=even_indices(e.history.cost)
      e.history.dw=even_indices(e.history.dw)
      e.history.l//=2

def compute_U_V(fp,fn,target_fp,target_fn,p):
  #e.p_empirical*=(1-min(e.fp_amnt,e.fn_amnt))
  #e.p_empirical+=(1-min(e.fp_amnt,e.fn_amnt))*nsm(e.y_t)/e.bs
  #U,V=log(1+fp/target_fp),log(1+fn/target_fn)
  #U=u/(u+v)
  #V=v/(u+v)
  #U,V=softmax(array([gamma1*fp/target_fp,gamma1*fn/target_fn]))
  #U,V=softmax(array([fp/target_fp,fn/target_fn]))
  U,V=fp/target_fp,fn/target_fn
  V/=p #scale dfn with the imbalance
  nUV=U+V
  if nUV>0:
    U/=nUV
    V/=nUV
  return U,V

def update_lrs(a,experiments): 
  experiments=sorted(experiments,key=lambda x:x.cost_window)
  goodnesses=array([1/(1e-8+e.cost_window) for e in experiments])
  e_lr=v_lr=0.
  for e,g in zip(experiments,goodnesses):
    print('lr,un-normalised goodnesses=',e.lr,g)
    le_lr+=log(e.lr)/log(2)
    lv_lr+=(log(e.lr)/log(2))**2
  le_lr/=len(experiments)
  lv_lr/=len(experiments)
  lv_lr-=le_le**2
  print('E(log(lr)),V(log(lr))=',le_lr,lv_lr)
  goodnesses/=nsm(goodnesses)
  experiment_indices=array(range(len(experiments)))
  e=experiments[-1]
  parent=experiments[int(choice(emit_key(),experiment_indices,p=goodnesses))]
  e.lr=parent.lr
  w=lambda x,y,z:(x*z,y/z)
  if parent.lr>a.lr_max: rule=lambda x,y:(x,y*exp(-abs(normal(emit_key()))))
  elif parent.lr<a.lr_min: rule=lambda x,y:(x,y*exp(abs(normal(emit_key()))))
  else: rule=lambda x,y:w(x,y,exp(normal(emit_key())))
  parent.lr,e.lr=rule(parent.lr,e.lr)

  if uniform(emit_key())<e.cost_window/(1e-8+parent.cost_window)-1:
    print('Weight copying')
    e.w_model=parent.w_model.copy()
    e.adam_M=parent.adam_M.copy()
    e.adam_V=parent.adam_V.copy()
    e.fp=float(parent.fp)
    e.fn=float(parent.fn)
  a.lrs=array([e.lr for e in experiments])
  return experiments

def update_weights(a,e,upd):
  e.dw_l2=e.w_l2=0
  if a.no_adam:
    for k in upd:
      delta=e.lr*upd[k]
      e.dw_l2+=nsm(delta**2)
      e.w_model[k]-=delta
      e.w_l2+=nsm(e.w_model[k]**2)
  else:
    try:
      e.adam_M*=(1-a.gamma2)
      for k in upd:#Should apply to all bits simultaneously?
        e.adam_V[k]*=(1-a.gamma1)
        e.adam_V[k]+=a.gamma1*upd[k]
        e.adam_M+=a.gamma2*nsm(upd[k]**2)
    except AttributeError: #initialise adam weights
      e.adam_V=upd
      e.adam_M=sum([nsm(upd[k]**2) for k in upd])
    for k in e.adam_V:
      delta=e.lr*e.adam_V[k]/(e.adam_M**.5+1e-8)
      e.w_model[k]-=delta
      ch_l2=nsm(delta**2)
      e.dw_l2+=ch_l2
      weight_l2=nsm(e.w_model[k]**2)
      e.w_l2+=weight_l2

def report_progress(a,experiments,line,act):
  print('|'.join([t.ljust(10) for t in ['p','target_fp','target_fn',
                                        'lr','fp','fn','w','dw','U','V']]))
  for e in experiments:
    print('|'.join([f_to_str(t) for t in [e.p,e.target_fp,e.target_fn,e.lr,
                                          e.fp,e.fn,e.w_l2,e.dw_l2,e.U,e.V]]))
  for e in experiments:
    print('Recent outcomes:')
    print('FP by batch',e.FPs[a.step%a.fpfn_memory_len-5:a.step%a.fpfn_memory_len])
    print('FN by batch',e.FNs[a.step%a.fpfn_memory_len-5:a.step%a.fpfn_memory_len])

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
  fp_perf=[]
  fn_perf=[]
  if 'r' in line:
    line+='clis'
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
      n_fps=len(a.target_fps)
      row=['imbalance','target_fps']+['']*(n_fps-1)+['target_fn','fps']+\
          ['']*(n_fps-1)+['fns']+['']*(n_fps-1)
      if a.mode=='unsw':
        row=['attack_cat']+row
      w.writerow(row)
      if fd_tex:
        conf_fill='r'*n_fps
        ct='l' if a.mode=='unsw' else ''
        print('\\begin{tabular}{l'+ct+'|'+conf_fill+'|r|'+conf_fill+'|'+conf_fill+'}',
              file=fd_tex)
        print(' & '.join(row).replace('_',' ')+'\\\\',file=fd_tex)
        print('\\hline',file=fd_tex)
      n_imbalances=len(a.imbalances)
      for p in a.imbalances:
        tgt_fps=[]
        recent_fps=[]
        recent_fns=[]
        for e in [e for e in experiments if e.p==p]:
          tgt_fps.append(f_to_str(e.target_fp))
          fp_hist=e.history.FP[-int(10/e.p**2):]
          fn_hist=e.history.FN[-int(10/e.p**2):]
          recent_fps.append(f_to_str(sum(fp_hist)/(e.bs*len(fp_hist))))
          recent_fns.append(f_to_str(sum(fn_hist)/(e.bs*len(fn_hist))))
        row=[f_to_str(p)]+tgt_fps+[f_to_str(p/10)]+recent_fps+recent_fns
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
    print(k,log(v)/log(2))
    if fd_tex: print('\\texttt{'+k+'}&'+f_to_str(log(v)/log(2))+'\\\\\n',file=fd_tex)
  if fd_tex: print('\\end{tabular}',file=fd_tex)

  if a.in_dim==2 and 'c' in line:
    if fd_tex:
      print('\\subsection{2d visualisation of classifications}',file=fd_tex)
      print('Here a 2d region is learned.\\\\',file=fd_tex)
      print('\n\\begin{figure}[H]',file=fd_tex)
      print('\\centering',file=fd_tex)
    for e in experiments:
      if a.mode=='gmm':
        x_t,y_t=get_xy(a,e.p,a.gmm_scatter_samples,emit_key())
        col_mat=[[1.,1,1],[0,0,0]]#fp,fn,tp,tn
        labs=['Predict +','Predict -']
        x_0_max=nmx(x_t[:,0])
        x_0_min=-nmx(-x_t[:,0])
        x_1_max=nmx(x_t[:,1])
        x_1_min=-nmx(-x_t[:,1])
      else:
        x_0_max=x_1_max=a.x_max
        x_0_min=x_1_min=-a.x_max
      x=cartesian([linspace(x_0_min,x_0_max,num=a.res),
                   linspace(x_1_min,x_1_max,num=a.res)])
      x_split=array_split(x,a.n_splits_img)
      y_p=concat([(f(e.w_model,_x,act=act)>0).flatten() for _x in x_split])
      y_p=flip(y_p.reshape(a.res,a.res),axis=1) #?!?!?!

      cm=None
      if 'b' in line: #draw boundary
        cols=abs(convolve(y_p,array([[1,1,1],[1,-8,1],[1,1,1]]))).T
        cols/=-(nmx(cols)+nmx(-cols))
        cols+=nmx(-cols)
        cm='gray'
      else:
        if a.mode=='gmm':
          regions=array([y_p,~y_p]).T
        else:
          y_t=concat([f(a.w_target,_x)>e.thresh for _x in x_split])\
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
        img_name=exp_to_str(e)+'.png'
        savefig(a.report_dir+'/'+img_name,dpi=500)
        print('\\begin{subfigure}{.33\\textwidth}',file=fd_tex)
        print('\\centering',file=fd_tex)
        print('\\includegraphics[width=.9\\linewidth,scale=1]{'+img_name+'}',
              file=fd_tex)
        print('\\end{subfigure}%',file=fd_tex)
        print('\\hfill',file=fd_tex)
        close()
      else:
        show()

    if fd_tex:
      print('\\end{figure}',file=fd_tex)
    fp_perf.append(e.fp/e.target_fp)
    fn_perf.append(e.fn/e.target_fn)

  if 'i' in line:
    model_desc='Model shape:\n'+('->'.join([str(l) for l in a.model_shape]))+'\n'
  
    a.no_glorot_uniform=False
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
    if a.mode in ['single','gmm']:
      model_desc+='\n- learning rate:'+str(a.lr)
    print(model_desc)
    if fd_tex:
      print('\\subsection{Model parameters}',file=fd_tex)
      print('Here the batch size was set to '+str(a.bs)+'.\\\\',file=fd_tex)
      print('\\texttt{'+(model_desc.replace('\n','}\\\\\n\\texttt{'))+'}\\\\',
            file=fd_tex)
  if 's' in line and a.mode=='all':
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
  if 'l' in line:
    if fd_tex:
      print('\\subsection{Historical statistics}',file=fd_tex)
    get_cost=lambda e:e.history.cost
    get_lr=lambda e:e.history.lr
    get_dw=lambda e:e.history.dw
    for get_var,yl,desc in zip([get_cost,get_lr,get_dw],
                               ['log(1e-8+fp/target_fp+fn/target_fn)','log(lr)',
                                'log(dw)'],
                               ['Loss','Learning_rate','Change_in_weights']):
      for e in experiments:
        arr=get_var(e)
        if a.mode=='unsw':
          lab=a.cats[e.p]
        else:
          lab=fpfnp_lab(e)
        plot([log(a)/log(2) for a in arr],label=lab)
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
      plot(log(smoothed_fp)/log(2),log(smoothed_fn)/log(2),label=fpfnp_lab(e))
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

  if fd_tex:
    print('\\end{document}',file=fd_tex,flush=True)
    fd_tex.close()

def save_ensemble(a,experiments,global_key):
  with open(a.out_dir+'/ensemble.pkl','wb') as fd:
    a.global_key=global_key
    dump((a,experiments,global_key),fd)
