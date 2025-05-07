from itertools import count
from numpy import geomspace
from jax.numpy import array,zeros,array,exp,ones,eye,sum as jsm,maximum
from jax.nn import tanh,softmax,relu
from jax.random import uniform,normal,split,key,choice,permutation
from jax.tree import map as jma
from jax import grad,jit,vmap
from jax.lax import scan,while_loop
from jax.lax.linalg import svd
from jax.scipy.special import xlogy

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

def _forward(w,x,act,layer_norm=False):
  l=w[:-1]
  abl=w[-1]
  for ab in l:
    z=x@ab[0]
    if layer_norm:
      z=ab[2]*((z-z.mean(axis=-1))/(1e-8+z.var(axis=-1)**.5))
    x=act(z+ab[1])
  y=x@abl[0]+abl[1]
  return y

#forward=jit(_forward,static_argnames=['batchnorm','get_transform','ll_act','transpose','act'])
forward=jit(_forward,static_argnames=['act','layer_norm'])
vorward=jit(vmap(_forward,(0,None,None),0),static_argnames=['act'])

activations={'tanh':tanh,'softmax':softmax,'linear':jit(lambda x:x),
             'relul':jit(lambda x:maximum(x,.03*x)),'relu':relu}

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

def init_layers(layer_dimensions,initialisation,k=None,mult_a=1.,n_par=None,bias=0.,
                sqrt_normalise=False,orthonormalise=False,transpose=True,layer_norm=False):
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
  G=[]
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
    if layer_norm:
      G.append(ones(e_dims+(d_o,)))
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
  AB=[A,B]
  if layer_norm:
    AB.append(G)
  if transpose:
    return list(list(ab) for ab in zip(*AB))
  else:
    return AB

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

def _set_lw(consts,fp,fn):
  return exp(consts['lrfpfn']*(-fp/consts['tfp']+fn/consts['tfn']))
set_lw=jit(vmap(_set_lw,(0,0,0),0))#,None

def get_reshuffled(k,X_trn,Y_trn,n_batches,bs,last):
  X,Y=shuffle_xy(k,X_trn,Y_trn)
  return X[0:last].reshape(n_batches,bs,-1),Y[0:last].reshape(n_batches,bs)

get_reshuffled=jit(get_reshuffled,static_argnames=['n_batches','bs','last'])

def loss(w,x,y,lw,act):
  yp=forward(w,x,act)
  return jsm((lw*y+(1-y)/lw)*(1-2*y)*yp)

def ce(w,x,y,lw,act):
  yp=act(forward(w,x,act))
  return jsm(xlogy(lw*y,yp)+xlogy((1-y)/lw,1-yp))

dl=grad(loss)
def _upd(x,y,state,consts,act):
  return dict_adam_no_bias(dl(state['w'],x,y,state['lw'],act),state,consts)

upd=vmap(_upd,(None,None,0,0,None),0)

def _steps(states,consts,X,Y,act):
  return scan(lambda s,xy:(upd(xy[0],xy[1],s,consts,act),0),states,(X,Y))[0]
steps=jit(_steps,static_argnames=['act'])

def _y_smooth_fp_fn(w,X,Y,act):
  yps=forward(w,X,act)
  Yp_smooth=yps.flatten()
  Yp=Yp_smooth>0.
  Ypb=Yp_smooth>0
  fp,fn=(Ypb&~Y).mean(),(Y&~Ypb).mean()
  return Yp_smooth,fp,fn

y_smooth_fp_fn=jit(_y_smooth_fp_fn,static_argnames=['act'])

calc_fp_fn=lambda w,X,Y,act:y_smooth_fp_fn(w,X,Y,act)[1:]

def _calc_thresh(w,tfpfn,X,Y,act,tol=1e-1):
  Yp_smooth,fp,fn=y_smooth_fp_fn(w,X,Y,act)
  return search_thresh(Y,Yp_smooth,tfpfn,tol),fp,fn

calc_thresh=jit(vmap(_calc_thresh,(0,0,None,None,None),0),static_argnames=['act','tol'])

def _nn_epochs(ks,n_epochs,bs,X,Y,X_raw,Y_raw,consts,states,act,n_batches,last,
               logf=None,adap_thresh=True,layer_norm=False,start_epoch=1):
  tfps=consts['tfp']
  tfns=consts['tfn']
  lrfpfns=consts['lrfpfn']
  lrs=consts['lr']
  regs=consts['reg']
  tfpfns=tfps/tfns
  for epoch,k in enumerate(ks,start_epoch):
    X_b,Y_b=get_reshuffled(k,X,Y,n_batches,bs,last)
    states=steps(states,consts,X_b,Y_b,act)
    if adap_thresh:
      thresh,fp,fn=calc_thresh(states['w'],tfpfns,X_raw,Y_raw,act)
      states['w'][-1][1]-=thresh.reshape(-1,1)
    else:
      fp,fn=calc_fp_fn(states['w'],X_raw,Y_raw,act)
    states['lw']*=set_lw(consts,fp,fn)
    print('nn epoch',epoch,file=logf,flush=True)
  return states

#_nn_epochs=jit(_nn_epochs,static_argnames=['n_epochs','n_batches','bs','last','act'])

def nn_epochs(k,n_epochs,bs,X,Y,consts,states,X_raw=None,Y_raw=None,act='relu',
              adap_thresh=True,logf=None,layer_norm=False,start_epoch=1):
  if X_raw is None:
    X_raw,Y_raw=X,Y
  if isinstance(act,str):
    act=activations[act]
  n_batches=len(X)//bs
  last=n_batches*bs
  ks=split(k,n_epochs)
  return _nn_epochs(ks,n_epochs,bs,X,Y,X_raw,Y_raw,consts,states,act,n_batches,last,
                    logf=logf,layer_norm=layer_norm,start_epoch=start_epoch)

def vectorise_fl_ls(x,l):
  if isinstance(x,list):
    return array(x)
  elif isinstance(x,float):
    return ones(l)*x 
  return x

class NNPar:
  def __init__(self,seed,p,tfp,tfn,p_resampled=None,lrfpfn=.03,reg=1e-6,inner_dims=None,
               in_dim=None,lw=None,start_width=None,end_width=None,depth=None,bs=128,
               act='relu',init='glorot_normal',times=100,lr=1e-4,beta1=.9,beta2=.999,
               eps=1e-8,bias=.1,acc=.1,min_tfpfn=1,max_tfpfn=100,logf=None,n_par=None,
               adap_thresh=True,layer_norm=False):
    self.logf=logf
    self.p=p
    self.p_resampled=p if p_resampled is None else p_resampled
    if lw is None:
      lw=((1-self.p_resampled)/self.p_resampled)**.5
    self.inner_dims=list(geomspace(start_width,end_width,depth+1,dtype=int)) if\
                         inner_dims is None else inner_dims
    self.ke=KeyEmitter(seed)

    self.times=(times,) if isinstance(times,int) else times #epochs when weights will be saved
    self.init=init
    self.bias=bias
    #consts
    self.lrfpfn=lrfpfn
    self.reg=reg
    self.lr=lr
    self.beta1=beta1
    self.beta2=beta2
    self.eps=eps
    self.tfp=tfp
    self.tfn=tfn
    self.bs=bs
    #states
    self.lw=lw
    self.act=act
    self.adap_thresh=adap_thresh
    self.layer_norm=layer_norm

    self.states=None
    self.states_by_epoch={}
    self.consts=None
    self.n_par=n_par
  
  def get_states(self):
    return self.states

  def set_consts_states(self,in_dim):
    self.in_dim=in_dim
    self.mod_shape=[self.in_dim]+self.inner_dims+[1]
    if self.consts is None:
      consts=dict(lrfpfn=self.lrfpfn,tfp=self.tfp*(1-self.p_resampled)/(1-self.p),
                  tfn=self.tfn*self.p_resampled/self.p,beta1=self.beta1,beta2=self.beta2,
                  lr=self.lr,reg=self.reg,eps=self.eps)
      if self.n_par is None:
        self.n_par=max((len(x) if hasattr(x,'__len__') else 1 for x in consts.values()))
      self.consts={k:vectorise_fl_ls(v,self.n_par) for k,v in consts.items()}
    if self.states is None:
      self.states=dict(lw=vectorise_fl_ls(self.lw,self.n_par),
                       **init_layers_adam(self.mod_shape,self.init,k=self.ke.emit_key(),
                                     n_par=self.n_par,bias=self.bias))

  def fit(self,X,Y,X_raw=None,Y_raw=None):
    self.set_consts_states(len(X[0]))
    nl=0
    for n in sorted(self.times):
      self.states=nn_epochs(self.ke.emit_key(),n-nl,self.bs,X,Y,self.consts,
                            self.states,X_raw=X_raw,start_epoch=n+1,
                            Y_raw=Y_raw,act=self.act,logf=self.logf,
                            adap_thresh=self.adap_thresh)
      print('Saving state at time',n,file=self.logf,flush=True)
      self.states_by_epoch[n]=self.states
      nl=n

  def predict(self,X):
    return {t:v>0 for t,v in self.predict_smooth(X).items()}

  def predict_smooth(self,X):
    Yp={t:vorward(self.states_by_epoch[t]['w'],X,activations[self.act]) for t in self.times}
    return {t:y.reshape(y.shape[:-1]) for t,y in Yp.items()}
