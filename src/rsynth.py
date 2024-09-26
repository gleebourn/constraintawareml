from numpy import ones, int32,prod,pi,cos,vectorize,array,sort
from numpy.random import default_rng


def iterate_small_prods(k,N):
  '''
  Returns an iterator that cycles over the positive ints
  x1...xk such that their product is less than N.
  '''
  ret=ones(k,dtype=int32)
  cur_prod=1
  while True:
    yield ret.copy()
    
    for i in range(k):
      next_prod=cur_prod+cur_prod//ret[i]

      if next_prod<=N:
        cur_prod=next_prod
        ret[i]+=1
        break
      else:
        cur_prod//=ret[i]
        ret[i]=1

      if i==k-1:
        return

def mk_brownian_sheet(dim,std,fidelity,seed=None):
  r=default_rng(seed=seed)
  num_terms=0
  for _ in iterate_small_prods(dim,1/fidelity): num_terms+=1
  z=r.normal(scale=std,size=num_terms)

  largest_allowed_product=1/fidelity

  def brownian_sheet(x):
    ret=0
    for term,samp in zip(iterate_small_prods(dim,largest_allowed_product),z):

      ret+=samp*prod((cos(2*pi*term*x)-1)/term)
    
    return ret*pi**(-dim)
  
  return brownian_sheet

def brownian_dataset(n_cols,n_rows,std,fidelity,seed=None):
  r=default_rng(seed=seed)
  B=mk_brownian_sheet(n_cols,std,fidelity)
  X=r.uniform(size=(n_rows,n_cols))
  #y=vectorize(B)(X)
  y=array([B(x) for x in X])
  return X,y

def brownian_region(n_cols,n_rows,std,fidelity,p,seed=None):
  X,y=brownian_dataset(n_cols,n_rows,std,fidelity,seed=seed)
  thresh=sort(y)[int(len(y)*p)]
  return X,y<thresh


def mk_synthetic_linear(n_cols,n_rows,er=.01,scheme='iidunif',dimy=10):
  if scheme=='iidunif':
    X_synthetic=(2*np.random.rand(n_rows,n_cols)-1)
    actual=(2*np.random.rand(n_cols)-1)
    noise=er*(2*np.random.rand(n_rows)-1)
    y_synthetic=2*(X_synthetic.dot(actual)+noise > 0)-1
    return X_synthetic,2*(y_synthetic>0)-1

  if scheme=='walkingmodel':
    X_synthetic=(2*np.random.rand(n_rows,n_cols)-1)
    y_synthetic=np.zeros(n_rows)
    actual=np.random.normal(size=n_cols)
    actual/=np.linalg.norm(actual)
    for i in range(n_rows):
      y_synthetic[i]=actual.dot(X_synthetic[i,:])
      actual+=er*np.random.normal(size=n_cols)
      actual/=np.linalg.norm(actual)
    return X_synthetic,2*(y_synthetic>0)-1

  if scheme=='ndwalkingmodel':
    X_synthetic=(2*np.random.rand(n_rows,n_cols)-1)
    y_synthetic=np.zeros((n_rows,dimy))
    actual=np.random.normal(size=(dimy,n_cols))
    actual/=np.linalg.norm(actual)
    for i in range(n_rows):
      y_synthetic[i]=actual.dot(X_synthetic[i,:])
      actual+=er*np.random.normal(size=(dimy,n_cols))
      actual/=np.linalg.norm(actual)
    return X_synthetic,y_synthetic
