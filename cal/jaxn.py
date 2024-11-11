import jax
jax.config.update('jax_traceback_filtering', 'off')

from jax.nn import sigmoid,relu
from jax.numpy import dot,vectorize,zeros,square,sqrt,array,sum as jsum
from jax.random import normal,key,randint,permutation
from jax import grad,jit
from jax.errors import JaxRuntimeError

#@jit #jitting this causes an iteration over 0d array error!!!!
def rand_batch(X,y,batch_size,key):
  indices=randint(key,batch_size,0,y.shape[0])
  return X[indices],y[indices]

@jit
def hlp_infer(x,params):
  A1=params['A1']
  b1=params['b1']
  A2=params['A2']
  b2=params['b2']
  assert len(x.shape)==1 and x.shape[0]==A1.shape[1],\
         'input shape of '+ str(x.shape)+' != '+\
         'required input shape of ('+str(A1.shape[1])+',)' 
  l1=sigmoid(dot(A1,x)+b1)
  return sigmoid(dot(A2,l1)+b2)

@jit
def tlp_infer(x,params):
  A0=params['A0']
  b0=params['b0']
  A1=params['A1']
  b1=params['b1']
  A2=params['A2']
  b2=params['b2']
  assert len(x.shape)==1 and x.shape[0]==A0.shape[1],\
         'input shape of '+ str(x.shape)+' != '+\
         'required input shape of ('+str(A1.shape[1])+',)' 
  l1=sigmoid(dot(A0,x)+b0)
  l2=sigmoid(dot(A1,l1)+b1)
  return sigmoid(dot(A2,l2)+b2)

#dflt_layer_dims=[128,64,64,32,32,16,16,16]
#dflt_layer_dims=[32,32,32,16,16,16,8,8,8]
dflt_layer_dims=[256,256,256,256,128,128,128,128,64,64,64,64]
#dflt_layer_dims=[256,256,256,256,256,256,256,256,128,128,128,128,64,64,64,64]
#dflt_layer_dims=[256,256,256,256,128,128,128,128,128,128,128,128,64,64,64,64,64,64,64,64]
#@jit
def mk_nlp_infer(layer_dims=dflt_layer_dims):
  layer_dims.append(1)
  l=len(layer_dims)
  def nlp_infer(x,params):
    for i in range(l):
      x=sigmoid(dot(params[('A',i)],x)+params[('b',i)])
    return x[0]
  return vectorize(nlp_infer,excluded=[1],signature='(n)->()')

def mk_nlp_init_params(layer_dims=dflt_layer_dims):
  layer_dims.append(1)
  def nlp_init_params(input_dim):
    ret=dict()
    for i,out_dim in zip(range(len(layer_dims)),layer_dims):
      ret[('A',i)]=zeros(shape=(out_dim,input_dim))
      ret[('b',i)]=zeros(shape=out_dim)
      input_dim=out_dim
    return ret
  return nlp_init_params

hlp_infer_v=vectorize(hlp_infer,excluded=[1],signature='(n)->()')
tlp_infer_v=vectorize(tlp_infer,excluded=[1],signature='(n)->()')

def tlp_init_params(input_dim,l_dim=128,l2_dim=64):
  return dict(A0=zeros(shape=(l_dim,input_dim)),
              b0=zeros(shape=l_dim),
              A1=zeros(shape=(l2_dim,l_dim)),
              b1=zeros(shape=l2_dim),
              A2=zeros(shape=l2_dim),
              b2=0.)

def hlp_init_params(input_dim,l_dim=64):
  return dict(A1=zeros(shape=(l_dim,input_dim)),
              b1=zeros(shape=l_dim),
              A2=zeros(shape=l_dim),
              b2=0.)

def gsum(y):
  try:
    return jsum(y)
  except JaxRuntimeError as e:
    return sum(y)

#Assume both y and y_pred are floats but y always 1. or 0.
def b_tp(y,y_pred):
  return gsum((y==1.)&(y_pred>.5))

def b_fp(y,y_pred):
  return gsum((y==0.)&(y_pred>.5))

def b_fn(y,y_pred):
  return gsum((y==1.)&(y_pred<=.5))

##Want a count that reflects binary count well
##Continuous function which looks like step function at .5
#K=8
#def coerce(x):
#  xp=x**K
#  return xp/(xp+(1-x)**K)
#
#coerce_v=vectorize(coerce)
#
#def c_fp(y,y_pred):
#  return gsum(coerce_v((1-y)*y_pred))
#
#def c_fn(y,y_pred):
#  return gsum(coerce_v(y*(1-y_pred)))

def c_tp(y,y_pred):
  return gsum(y*y_pred)

def c_fp(y,y_pred):
  return gsum((1-y)*y_pred)

def c_fn(y,y_pred):
  return gsum(y*(1-y_pred))

class bin_optimiser:

  def __init__(self,input_dim,init_params=tlp_init_params,lr=.001,seed=0,implementation=tlp_infer_v,
               beta1=.9,beta2=.999,eps=.00000001,kappa=.9,beta=1,nu=50):
    self.key=key(seed)
    self.lr=lr
    self.input_dim=input_dim
    self.init_params=init_params
    self.params=init_params(input_dim)
    self.randomise_params()
    self.implementation=implementation

    self.beta=beta
    self.kappa=kappa
    self.om_kappa=1-kappa
    self.nu=nu
    self.fp_weight=0#self.fn_weight=.5
    
    def _fp(X,y,params):
      return c_fp(y,implementation(X,params))

    def _fn(X,y,params):
      return c_fn(y,implementation(X,params))

    
    self._dfp=grad(_fp,argnums=2)
    self._dfn=grad(_fn,argnums=2)

    ##Adam params
    #Don't bother with the fairly pointless time dependent bias removal
    #Eg, for default beta2 beta2^t<.01 when t>4600 or so.
    self.beta1=beta1
    self.beta2=beta2
    self.gamma1=1-beta1
    self.gamma2=1-beta2
    self.m=self.init_params(self.input_dim)
    self.v=0
    self.eps=eps

  def b_tp(self,y,y_pred):
    return b_tp(y,y_pred)/len(y)

  def b_fp(self,y,y_pred):
    return b_fp(y,y_pred)/len(y)
                  
  def b_fn(self,y,y_pred):
    return b_fn(y,y_pred)/len(y)
                  
  def c_tp(self,y,y_pred):
    return c_tp(y,y_pred)/len(y)
                  
  def c_fp(self,y,y_pred):
    return c_fp(y,y_pred)/len(y)
                  
  def c_fn(self,y,y_pred):
    return c_fn(y,y_pred)/len(y)
                  
  def update_fp_w(self,y,y_pred):
    #Check which way the skew is.
    #We are aiming for fp/fn=beta**2.
    #u>0 -> too many fp
    #u<0 -> too many fn
    #u=self.c_fp(y,y_pred)/self.beta-self.c_fn(y,y_pred)*self.beta
    u=self.b_fp(y,y_pred)/self.beta-self.b_fn(y,y_pred)*self.beta
    #u=sigmoid(self.nu*u)
    self.fp_weight=self.kappa*self.fp_weight+u#self.om_kappa*u
    #self.fn_weight=1-self.fp_weight

  def dfp(self,X,y):
    return self._dfp(X,y,self.params)

  def dfn(self,X,y):
    return self._dfn(X,y,self.params)

  def adam_step(self,X,y):
    y_pred=self.infer(X)
    self.update_fp_w(y,y_pred)

    '''
    upd=self.dfp if self.fp_weight>0 else self.dfn
    #Only update one of two, if borderline do it randomly
    #upd=self.dfp if self.fp_weight>normal(self.key) else self.dfn
    upd=upd(X,y)
    self.v*=self.beta2
    for key in upd:
      self.m[key]*=self.beta1
      #upd=self.fp_weight*dfp[key]+self.fn_weight*dfn[key]
      self.m[key]+=self.gamma1*upd[key]
      self.v+=self.gamma2*jsum(square(upd[key]))
    '''
    updp,updn=self.dfp(X,y),self.dfn(X,y)
    self.v*=self.beta2
    U=sigmoid(self.fp_weight)
    V=1-U
    for key in updp:
      self.m[key]*=self.beta1
      upd=U*updp[key]/(self.eps+sqrt(jsum(square(updp[key]))))+V*updn[key]/(self.eps+sqrt(jsum(square(updn[key]))))
      self.m[key]+=self.gamma1*upd
      self.v+=self.gamma2*jsum(square(upd))

    mult=self.lr/(self.eps+sqrt(self.v))
    for key in updp:
      self.params[key]-=mult*self.m[key]
    #self.params['A2']=self.params['A2'].sort()

  def randomise_params(self,amount=1):
    for k in self.params:
      self.params[k]*=1-amount
      try:
        shape=self.params[k].shape
      except AttributeError as e:
        shape=()
      self.params[k]+=amount*normal(self.key,shape=shape)

  def infer(self,x):
    try:
      return self.implementation(x,self.params)
    except JaxRuntimeError as e:
      print('could not do vectorised inference... gonna be slow...')
      y_pred=array([self.infer(array([t])) for t in x])
      print('...done')
      return y_pred

  def run_epoch(self,X_all,y_all,batch_size=32,verbose=True):
    n_rows=len(y_all)
    n_batches=n_rows//batch_size #May miss 0<=t<32 rows
    perm=permutation(self.key,n_rows)

    X_all=X_all[perm]
    y_all=y_all[perm]

    report_interval=n_batches//10
    for i in range(n_batches):
      self.adam_step(X_all[i:i+batch_size],y_all[i:i+batch_size])
      if verbose and not((i%report_interval)and((i-10)%report_interval)):
        print('Epoch',100*(i/n_batches),'% complete... training performance:')
        self.bench(X_all,y_all)

  def bench(self,X,y):
    y_pred=self.infer(X)
    tp=self.b_tp(y,y_pred)
    fp=self.b_fp(y,y_pred)
    fn=self.b_fn(y,y_pred)
    cts_tp=self.c_tp(y,y_pred)
    cts_fp=self.c_fp(y,y_pred)
    cts_fn=self.c_fn(y,y_pred)
    bt2=self.beta**2
    print(f'[bin,cts] tp: [{tp:.5f},{cts_tp:.5f}]')
    print(f'[bin,cts] fp: [{fp:.5f},{cts_fp:.5f}]')
    print(f'[bin,cts] fn: [{fn:.5f},{cts_fn:.5f}]')
    print(f'[bin,cts,tgt] fp/fn: [{fp/fn:10.5f},{cts_fp/cts_fn:10.5f},{bt2:10.5f}]')
    print(f'bin pr:{tp/(tp+fp):.5f}')
    print(f'bin rc:{tp/(tp+fn):.5f}')
    print(f'fp wgt: {self.fp_weight:.5f}')

  def run_epochs(self,X_train,y_train,X_test=None,y_test=None,batch_size=32,n_epochs=100,verbose=2):
    for i in range(1,n_epochs+1):
      if verbose: print('Beginning epoch',i,'...')
      self.run_epoch(X_train,y_train,batch_size=batch_size,verbose=verbose==2)
      if verbose:
        print('...done!')
        print('Training performance:')
        self.bench(X_train,y_train)
        if not(X_test is None):
          print('Testing performance:')
          self.bench(X_test,y_test)

class learner:
  def __init__(self):
    pass

  def fd(self,x):
    return x

  def bk(self,x,y):
    return x

class vneuron(learner):
  def __init__(self,in_dim,out_dim,seed=0,activation=sigmoid):
    self.key=key(seed)
    self.activation=activation
    self.params=dict(A=normal(self.key,shape=(out_dim,in_dim)),
                     b=normal(self.key,shape=(out_dim,)))

  def fd(self,x):
    return self.activation(dot(self.params['A'],x)+self.params['b'])

  def bk(self,x,y):
    return x







