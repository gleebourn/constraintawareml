from itertools import count
from numpy import geomspace
from jax.numpy import array,zeros,array,exp,ones,eye,sum as jsm,maximum
from jax.nn import tanh,softmax
from jax.random import uniform,normal,split,key,choice,permutation
from jax.tree import map as jma
from jax import grad,value_and_grad,jit,vmap
from jax.lax import scan,while_loop
from jax.lax.linalg import svd

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

def _nn_epochs(ks,n_epochs,bs,X,Y,X_raw,Y_raw,consts,states,act,n_batches,last,logf=None):
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
    print('nn epoch',epoch,file=logf)
  return states

#_nn_epochs=jit(_nn_epochs,static_argnames=['n_epochs','n_batches','bs','last','act'])

def nn_epochs(k,n_epochs,bs,X,Y,consts,states,X_raw=None,Y_raw=None,act='relu',logf=None):
  if X_raw is None:
    X_raw,Y_raw=X,Y
  if isinstance(act,str):
    act=activations[act]
  n_batches=len(X)//bs
  last=n_batches*bs
  ks=split(k,n_epochs)
  return _nn_epochs(ks,n_epochs,bs,X,Y,X_raw,Y_raw,consts,states,act,n_batches,last,logf=logf)

def vectorise_fl_ls(x,l):
  if isinstance(x,list):
    return array(x)
  elif isinstance(x,float):
    return ones(l)*x 
  return x

class NNPar:
  def __init__(self,seed,p,tfp,tfn,p_resampled=None,lrfpfn=.03,reg=1e-6,mod_shape=None,
               in_dim=None,bet=None,start_width=None,end_width=None,depth=None,bs=128,
               act='relu',init='glorot_normal',n_epochs=100,lr=1e-4,beta1=.9,beta2=.999,
               eps=1e-8,bias=.1,acc=.1,min_tfpfn=1,max_tfpfn=100,logf=None):
    self.logf=logf
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
                          X_raw=X_raw,Y_raw=Y_raw,act=self.act,logf=self.logf)

  def predict(self,X):
    Yp=vorward(self.states['w'],X,activations[self.act])
    return Yp.reshape(Yp.shape[:-1])>0
