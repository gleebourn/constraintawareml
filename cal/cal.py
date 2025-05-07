from itertools import count,product
from pathlib import Path
from time import perf_counter
from json import dump as jump
from numpy import array,geomspace
from numpy.random import default_rng #Only used for deterministic routines
from csv import reader,writer,DictWriter
from cal.kal import SKL
from cal.rs import Resampler,resamplers
from cal.dl import load_ds
from cal.ts import TimeStepper
from pickle import dump

def dict_to_tup(d):
  return tuple(sorted(d.items()))

def tup_to_dict(t):
  return {k:v for k,v in t}

class ModelEvaluation:
  def __init__(self,directory=None,ds='unsw',seed=1729,lab_cat=True,sds=False,
               categorical=True,out_f=None,reload_prev=None,methods=['sk','nn'],
               params={'nn':[dict(lr_ad=la,reg=rl*la,lrfpfn=lf,bias=b,times=(10,20,40,80,160),start_width=sw,
                                  end_width=ew,depth=3,bs=128,act='relu',init='glorot_normal',eps=1e-8,
                                  beta1=.9,beta2=.999,layer_norm=layer_norm) for la,rl,lf,b,layer_norm,sw,ew in\
                             product([.001],[1e-2],[.005],[0.],[False],[128],[32])],
                       'sk':[dict(max_depth=i,regressor=rg,
                                  **({'n_jobs':-1} if rg=='RandomForestRegressor' else {}))\
                             for i,rg in product([14],['RandomForestRegressor'])]
                                                      #'HistGradientBoostingRegressor')),
                       }):
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
    self.smooth_preds_dir=self.res_dir/'smooth_preds'
    self.smooth_preds_dir.mkdir(parents=True)
    self.res_fp=self.res_dir/'res.csv'
    self.header=[]
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
    self.regressors={}
    self.benchmarked={}
    self.rs=Resampler(self.X_trn,self.Y_trn,self.directory,self.ds_name,
                      self.seed(),p=self.p_trn,logf=self.logf)
    self.n_complete=0
    self.params=params
    self.methods=[]
    self.resamplers=set()
    self.jobs=set()
  
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

  def define_jobs(self,methods=['nn','sk'],resamplers=list(resamplers)+[''],params=None):
    if params is None:
      params=self.params
    for m in methods:
      for pmd in params[m]:
        pmt=dict_to_tup(pmd)
        [self.jobs.add((m,pmt,rs)) for rs in resamplers]
    self.methods=[j[0] for j in self.jobs]
    self.params={m:[tup_to_dict(j[1]) for j in self.jobs if j[0]==m]}
    self.resamplers=list({j[2] for j in self.jobs})
    self.n_remaining=len(self.jobs)
    self.log(self.n_remaining,'jobs to perform:')
    [self.log(m,rs,'params:',tup_to_dict(pmt)) for m,pmt,rs in self.jobs]

  def run_jobs(self):
    for job in self.jobs:
      m,pmt,rs=job
      rs=rs or 'none'
      self.log('Initialising model...')
      self.log('method:',m)
      self.log('resampling:',rs)
      self.log('Parameters:')
      [self.log(k,v) for k,v in pmt]
      self.init_job(job)
      self.log('Benchmarking...')
      self.benchmark_job(job)
      self.log('Updating results...')
      self.update_results()
      self.log('Removing job...')
      self.rm_job(job)
      self.log(self.n_complete,'experiments completed with',self.n_remaining,'to go...')
    self.log('Completed all jobs!')
    return

  def rm_job(self,job):
    self.regressors.pop(job)
    self.benchmarked.pop(job)
    self.n_complete+=1
    self.n_remaining-=1

  def init_job(self,job):
    method,pmt,rs=job
    pm=tup_to_dict(pmt)
    match method:
      case 'sk':
        r={l:SKL(pm['regressor'],self.tfpfns[l],{k:v for k,v in pm.items() if k!='regressor'},
                 p,seed=self.seed()) for l,p in self.p_trn.items()}
      case 'nn':
        from cal.jal import NNPar
        r={l:NNPar(self.seed(),p,self.tfps[l],self.tfns[l],p_resampled=self.rs.get_p(l,rs),init=pm['init'],
                   lrfpfn=pm['lrfpfn'],reg=pm['reg'],start_width=pm['start_width'],end_width=pm['end_width'],
                   depth=pm['depth'],lr=pm['lr_ad'],beta1=pm['beta1'],beta2=pm['beta2'],eps=pm['eps'],
                   times=pm['times'],bias=pm['bias'],logf=self.logf,layer_norm=pm['layer_norm'],
                   adap_thresh=pm['adap_thresh'])\
           for l,p in self.p_trn.items()}
      case _:
        raise NotImplementedError('Method',method,'not found')
    self.regressors[job]=r

  def benchmark_job(self,job):
    method,_,resampler=job
    self.rs.set_resampler(resampler)
    regressor=self.regressors[job]
    self.benchmarked[job]={}
    for l,reg in regressor.items():
      times=reg.times
      self.log('Training for label',l,'...')
      reg.fit(*self.rs.get_resampled(l,resampler),X_raw=self.X_trn[l],Y_raw=self.Y_trn[l])
      self.log('...completed training for label',l,'...')
      r={t:{'trn':{},'tst':{},'tgt':{targs[2]:dict(zip(('fp','fn','fpfn'),targs))\
            for targs in self.targets[l]}} for t in times}

      for stage,X,Y in [('tst',self.X_tst[l],self.Y_tst[l]),('trn',self.X_trn[l],self.Y_trn[l])]:
        Yp=reg.predict(X)
        Yp_smooth=reg.predict_smooth(X)
        job_str=stage+'_'+str(job)+'_'+l
        h=str(hash(job_str))
        with (self.smooth_preds_dir/'hashtab.csv').open('a') as fd:
          fd.write(job_str+','+h+'\n')

        with (self.smooth_preds_dir/(h+'.pkl')).open('wb') as fd:
          dump((Y,Yp,Yp_smooth),fd)
        fps={t:(yp&~Y).mean(axis=1) for t,yp in Yp.items()}
        fns={t:((~yp)&Y).mean(axis=1) for t,yp in Yp.items()}
        for t in times:
          for tfpfn,fp,fn in zip(self.tfpfns[l],fps[t],fns[t]):
            r[t][stage][tfpfn]=dict(fp=fp,fn=fn,fpfn=fp/fn)
      for t in times:
        self.benchmarked[job][t]=self.benchmarked[job].pop(t,{})
        self.benchmarked[job][t][l]=r[t]
  
  def res_to_rows(self,job,l,tfpfn):
    method,pmt,resampler=job
    rows=[]
    for t,bm_t in self.benchmarked[job].items():
      res=bm_t[l]
      row=dict(method=method,resampler=resampler,cat_lab=l,p=self.p_trn[l],
               params=tup_to_dict(pmt),time=t)
      row.update({k+lab:res[s][tfpfn][k] for k in ('fp','fn') for (lab,s) in\
                  (('_target','tgt'),('_train','trn'),('_test','tst'))})
      rows.append(row)
    return rows

  def update_results(self,jobs=None):
    to_update=self.benchmarked if jobs is None else {j for j in self.benchmarked if j in jobs}

    append_res=self.res_fp.exists()
    with self.res_fp.open('a' if append_res else 'w') as fd:
      rows=sum([self.res_to_rows(job,l,tfpfn) for job in to_update for\
                l in self.p_trn for _,_,tfpfn in self.targets[l]],[])
      if not self.header:
        self.header=sorted(list(rows[0]),key=lambda s:s[::-1])
      w=DictWriter(fd,self.header,extrasaction='ignore')
      if not append_res:w.writeheader()
      w.writerows(rows)
