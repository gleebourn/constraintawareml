from numpy.random import default_rng
from multiprocessing import Pool

class Job:
  def __init__(self,l,p=None,x=None,y=None,y_pred=None):
    self.learners_p=dict()
    self.learners_x=dict()
    self.learners_y=dict()
    self.learners_y_pred=dict()
    self.set_p(l,p)
    self.set_x(l,x)
    self.set_y(l,y)
    self.set_y_pred(l,y_pred)

  def set_p(self,l,p):
    self.learners_p[l]=p

  def set_x(self,l,x):
    self.learners_x[l]=x

  def set_y(self,l,y):
    self.learners_y[l]=y

  def set_y_pred(self,l,y_pred):
    self.learners_y_pred[l]=y_pred


  def p(self,l):
    return self.learners_p[l] if\
           l in self.learners_p else None

  def x(self,l):
    return self.learners_x[l] if\
           l in self.learners_x else None

  def y(self,l):
    return self.learners_y[l] if\
           l in self.learners_y else None

  def y_pred(self,l):
    return self.learners_y_pred[l] if\
           l in self.learners_y_pred else None


  def set_y_pred(self,l,y):
    self.learners_y_pred[l]=y

class Learner:
  def __init__(self):
    pass
  
  def init_p(self):
    return None

class ComposedLearner(Learner):
  def __init__(self,composees):

    self.composite_learners=[]

    for a in composees:
      if isinstance(a,ComposedLearner):
        self.composite_learners+=a.composite_learners
      else:
        self.composite_learners.append(a)

  def infer(self,j):
    #Set the x of the first composee
    j.set_x(self.composees[0],j.x(self))

    for a in self.composite_learners:
      y=a.infer(j)

    j.set_y(self,y)
    return y
  
  def request(self,j,update=True):
    j.set_focus(self)
    #Set the y of the last composee
    j.set_y(self.composees[-1],j.y())
    for a in reversed(self.composite_learners):
      y=a.request(y,j,update=update)
    return y
  
  def update(self,j):
    self.infer(j)
    self.request(j,update=True)

class FunctionalLearner(Learner):
  
  # Inspired by Spivak's work in https://arxiv.org/abs/1711.10455
  
  # Depending on the details though, if parallelised updates may not
  # be performed in a preictable order.  This may be fine if
  # they are somehow roughly independent of each other - eg, if each
  # update gives a p_new with p_new  still fairly close to p_pold.

  def __init__(self,inf,req,upd,p,seed=None):
    self._inf=inf
    self._req=req
    self._upd=upd

  def __add__(self,other):
    return ComposedLearner([self,other])

  # When we make an inference, we may or may not store
  # x, and we may in addition story y_pred if it is used
  # for the update and request methods.
  def infer(self,j):
    j.set_y_pred(self,self._inf(j.p(self),j.x(self)))
    return j.y_pred(self)

  def request(self,j,update=True): #x already saved
    req=self._req(j.p(self),j.x(self),j.y(self),y_pred=j.y_pred(self))
    if update:
      self.update(j)
    return req
  
  def update(self,j):
    y_pred=j.y_pred(self)
    if y_pred is None:
      j.set_p(self,self._upd(j.p(self),j.x(self),j.y(self)))
    else:
      j.set_p(self,self._upd(j.p(self),j.x(self),j.y(self),y_pred=y_pred))

class TensoredLearner(Learner):
  def __init__(self,tensees):
    self.tensored_learners=[]

    for a in tensees:
      if isinstance(a,TensoredLearner):
        self.tensored_learners+=a.tensored_learners
      else:
        self.tensored_learners.append(a)
  
  def infer(self,j):
    with Pool(len(self.tensored_learners)) as p:
      def subjob(l):
        k=Job(l)
        k.set_x(l,j.x(l))
        l.infer(k)
        j.set_y_pred(l,k.y_pred(l))

      p.map(subjob,self.tensored_learners)

  def request(self,j,update=True):
    with Pool(len(self.tensored_learners)) as p:
      def subjob(l):
        k=Job(l,j.p(l),j.x(l),j.y(l))
        l.request(k,update=update)
        j.set_y_pred(l,k.y_pred(l))
        return k.p(l)

      new_params=p.map(subjob,self.tensored_learners)
      if update:
        j.set_p(new_params)
