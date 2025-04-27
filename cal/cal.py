from itertools import count
from pathlib import Path
from time import perf_counter
from json import dump as jump
from numpy import array,geomspace
from numpy.random import default_rng #Only used for deterministic routines
from csv import reader,writer,DictWriter
from cal.kal import SKL
from cal.jal import NNPar
from cal.rs import Resampler,resamplers
from cal.dl import load_ds
from cal.ts import TimeStepper

class ModelEvaluation:
  def __init__(self,directory=None,ds='unsw',seed=1729,lab_cat=True,sds=False,
               categorical=True,out_f=None,reload_prev=None,vp_in_res=True,fp_in_res=True,
               fixed_params={'nn':dict(n_epochs=100,start_width=128,end_width=32,depth=2,bs=128,bias=.1,
                                       act='relu',init='glorot_normal',eps=1e-8,beta1=.9,beta2=.999),
                             'sk':{}},methods=['sk','nn'],
               varying_params={'sk':[{'ska':dict(max_depth=i,
                                                 **({'n_jobs':-1} if rg=='RandomForestRegressor' else {})),\
                                      'regressor':rg} for i in range(2,15,2) for rg in
                                     ('RandomForestRegressor','HistGradientBoostingRegressor')],
                               'nn':[{'lr_ad':la,'reg':rl*la,'lrfpfn':lf}for la in [.0001] for\
                                     rl in [1e-2] for lf in geomspace(.0002,.03,4)]}):
    if directory is None:
      directory='modeval_'+(ds if isinstance(ds,str) else 'cust')
    self.directory=Path(directory)
    if reload_prev is None:
      for n in count():
        dir_n=self.directory.joinpath(str(n))
        if not dir_n.exists(): break
    else:
      dir_n=self.directory.join_path(str(reload_prev))
      
    self.res_dir=dir_n
    if not self.res_dir.exists():
      self.res_dir.mkdir()
    self.done_fp=self.res_dir/'done.csv'
    self.done_fp=self.res_dir/'done.csv'
    self.res_fp=self.res_dir/'res.csv'
    self.header=['cat_lab','p','method','param_index','resampler','fp_target','fn_target',
                 'fp_train','fn_train','fp_test','fn_test']
    self.fp_fp=self.res_dir/'fixed_params.json'
    self.vp_fp=self.res_dir/'varying_params.csv'
    self.header=[]
    self.excluded_cols=[]
    if vp_in_res:
      self.excluded_cols+=['fixed_params']
    else:
      self.excluded_cols+=['param_index']
    if not fp_in_res:
      self.excluded_cols+=['fixed_params']
    self.logf=(self.res_dir/'modeval.log').open('w')
    self.parent_seed=seed
    self.rs_seeds={}
    self.rng=default_rng(seed)
    if isinstance(ds,str):
      self.X_trn,Y_trn,self.X_tst,Y_tst,self.sc=load_ds(ds,random_split=self.seed(),logf=self.logf,
                                                        lab_cat=lab_cat,single_dataset=sds)
    else:
      self.X_trn,Y_trn,self.X_tst,Y_tst,self.sc=ds
      ds='cust'
    self.ds_name=('lc_' if lab_cat else '')+(('sds_'+sds+'_') if sds else '')+ds
    if lab_cat:
      Y_labs=set(Y_trn)
      self.Y_trn={l:(Y_trn==l).to_numpy() for l in Y_labs}
      self.Y_tst={l:(Y_tst==l).to_numpy() for l in Y_labs}
    else:
      self.Y_trn={'all':Y_trn}
      self.Y_tst={'all':Y_tst}
    if not isinstance(self.X_trn,dict):
      self.X_trn={l:self.X_trn for l in self.Y_trn}
      self.X_tst={l:self.X_tst for l in self.Y_trn}

    self.p_trn={l:yt.mean() for l,yt in self.Y_trn.items()}
    self.p_tst={l:yt.mean() for l,yt in self.Y_tst.items()}
    self.regressors=[]
    self.thresholds={}
    self.res=[]
    self.rs=Resampler(self.X_trn,self.Y_trn,self.directory,self.ds_name,
                      self.seed(),p=self.p_trn,logf=self.logf)
    self.n_complete=0
    self.varying_params=varying_params
    self.fixed_params=fixed_params
    self.methods=[]
    self.resamplers=[]
    self.jobs=[]
  
  def log(self,*a):
    print(*a,file=self.logf,flush=True)

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
    self.tfps={l:array([t[0] for t in v]) for l,v in self.targets.items()}
    self.tfns={l:array([t[1] for t in v]) for l,v in self.targets.items()}
    self.tfpfns={l:array([t[2] for t in v]) for l,v in self.targets.items()}
    self.n_targets=n_targets

  def define_jobs(self,methods=['nn','sk'],resamplers=['']+list(resamplers),varying_params=None,
                  check_done=True,jobs=None):
    if jobs is None:
      if varying_params is None:
        varying_params=self.varying_params
      jobs=[(method,resampler,vp) for resampler in resamplers\
            for method in methods for vp in varying_params[method]]
      new_jobs=[j for j in jobs if not j in self.jobs]
    self.methods+=[j[0] for j in new_jobs if not j[0] in self.methods]
    self.resamplers+=[j[1] for j in new_jobs if not j[2] in self.resamplers]
    for m in methods:
      vp_existing=self.varying_params.pop(m,[])
      self.varying_params[m]=vp_existing+[job[2] for job in jobs if not job[2] in vp_existing]
    self.jobs+=new_jobs
    self.n_remaining=len(self.jobs)
    self.log('Scheduled',self.n_remaining,'jobs:')
    [self.log(*j) for j in self.jobs]
    if check_done and self.done_fp.exists():
      with self.done_fp.open('r') as fd:
        done_jobs=[r for r in reader(fd)]
        self.jobs=[j for j in self.jobs if not j in done_jobs]
        self.n_complete=len(done_jobs)
        self.log('Already completed:')
        [self.log(*dj) for dj in done_jobs]
        self.log('Still to complete:')
        [self.log(*j) for j in self.jobs]
        self.n_remaining=len(self.jobs)

  def run_jobs(self):
    while self.n_remaining:
      job=self.jobs.pop()
      m,rs,vp=job
      rs=rs or 'none'
      self.log('Initialising model...')
      self.log('method:',m)
      self.log('method index:',self.varying_params[m].index(vp))
      self.log('resampling:',rs)
      self.log('Globally fixed parameters:')
      [self.log(k,v) for k,v in self.fixed_params[m].items()]
      self.log('Locally fixed parameters:')
      [self.log(k,v) for k,v in vp.items()]
      self.init_model(job)
      self.log('Benchmarking...')
      self.benchmark_model(job)
      self.log('Updating results...')
      self.update_results(job)
      self.log('Removing job...')
      self.rm_job(job)
      self.log(self.n_complete,'experiments completed with',self.n_remaining,'to go...')

  def rm_job(self,job):
    self.regressors=[jr for jr in self.regressors if not jr[0]==job]
    self.res=[jr for jr in self.res if not jr[0]==job]
    self.n_complete+=1
    self.n_remaining-=1

  def init_model(self,job):
    method,resampler,pv=job
    pf=self.fixed_params[method]
    match method:
      case 'sk':
        r={l:SKL(pv['regressor'],self.tfpfns[l],pv['ska'],p,self.rs.get_p(l),
                 seed=self.seed()) for l,p in self.p_trn.items()}
      case 'nn':
        r={l:NNPar(self.seed(),p,self.tfps[l],self.tfns[l],p_resampled=self.rs.get_p(l),init=pf['init'],
                   lrfpfn=pv['lrfpfn'],reg=pv['reg'],start_width=pf['start_width'],end_width=pf['end_width'],
                   depth=pf['depth'],lr=pv['lr_ad'],beta1=pf['beta1'],beta2=pf['beta2'],eps=pf['eps'],
                   bias=pf['bias'],logf=self.logf) for l,p in self.p_trn.items()}
      case _:
        raise NotImplementedError('Method',method,'not found')
    self.regressors.append((job,r))

  def benchmark_model(self,job):
    method,resampler,_=job
    self.rs.set_resampler(resampler)
    regressor=next(r for j,r in self.regressors if j==job)
    ms=self.fixed_params[method]
    res={}
    for l,reg in regressor.items():
      self.log('Training for label',l,'...')
      r={'trn':{},'tst':{},'tgt':{targs[2]:dict(zip(('fp','fn','fpfn'),targs)) for targs in self.targets[l]}}
      reg.fit(*self.rs.get_resampled(l),X_raw=self.X_trn[l],Y_raw=self.Y_trn[l])

      for stage,X,Y in [('tst',self.X_tst[l],self.Y_tst[l]),('trn',self.X_trn[l],self.Y_trn[l])]:
        Yp=reg.predict(X)
        fps=(Yp&~Y).mean(axis=1)
        fns=((~Yp)&Y).mean(axis=1)
        for tfpfn,fp,fn in zip(self.tfpfns[l],fps,fns):
          r[stage][tfpfn]=dict(fp=fp,fn=fn,fpfn=fp/fn)
      res[l]=r
    self.res.append((job,res))
  
  def res_to_row(self,method,resampler,vp,l,tfpfn):
    res=next(r[l] for j,r in self.res if j==(method,resampler,vp))
    ret=dict(method=method,resampler=resampler,cat_lab=l,p=self.p_trn[l],
             **{'varying_params':vp,'param_index':self.varying_params[method].index(vp),
                'fixed_params':self.fixed_params[method]})
    ret.update({k+lab:res[s][tfpfn][k] for k in ('fp','fn') for (lab,s) in\
                (('_target','tgt'),('_train','trn'),('_test','tst'))})
    return ret

  def update_results(self,single=False):
    append_done=self.done_fp.exists()
    if single:
      newly_done=[single]
    elif append:
      with self.done_fp.open('r') as fd:
        previously_done=list(reader(fd))
        print(previously_done)
      newly_done=[r for r in self.res if not r in previously_done]
    else:
      newly_done=self.res

    append_res=self.res_fp.exists()
    with self.res_fp.open('a' if append_res else 'w') as fd:
      rows=[self.res_to_row(method,resampler,vp,l,tfpfn) for (method,resampler,vp) in newly_done for\
            l in self.p_trn for _,_,tfpfn in self.targets[l]]
      if not self.header:
        self.header=sorted(list(rows[0]),key=lambda s:s[::-1])
        [self.header.remove(c) for c in self.excluded_cols]
      w=DictWriter(fd,self.header,extrasaction='ignore')
      if not append_res:w.writeheader()
      w.writerows(rows)

    with self.done_fp.open('a' if append_done else 'w') as fd:
      w=writer(fd)
      [w.writerow(r) for r in newly_done]
    with self.fp_fp.open('w') as fd:
      jump({'fixed':self.fixed_params,'methods':self.methods,'resamplers':self.resamplers},fd)
    if not self.vp_fp.exists():
      with self.vp_fp.open('w') as fd:
        w=DictWriter(fd,['param_index','method']+\
                        sum([list(self.varying_params[m][0]) for m in self.methods],[]))
        w.writeheader()
        for m in self.methods:
          [w.writerow(dict(param_index=i,method=m,**vps)) for i,vps in enumerate(self.varying_params[m])]
