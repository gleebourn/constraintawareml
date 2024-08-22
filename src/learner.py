
class ComposedLearner:
  pass

class Learner:
  
  def __init__(self):
    self.depth=0
  def __add__(self,other):
    return ComposedLearner(self,other)

  def step(self,p,x,y):
    return (p,x,y)

  def infer(self,p,x):
    return self.step(p,x,0)[2]
  def request(self,p,x,y):
    return self.step(p,x,y)[1]
  def update(self,p,x,y):
    return self.step(p,x,y)[0]

class ComposedLearner(Learner):
  def __init__(self,args):
    super().__init__()
    self.atomic_learners[]
    for a in args:
      if isinstance(a,ComposerLearner):
        self.atomic_learners+=a.atomic_learners
      else:
        self.atomic_learners.append(a)
  def step(self,p,x,y):
    for i in range(len(p)-1):
      _,y,z=self.atomic_learners[i+1].step(p[i+1],self.atomic_learners[i](p[i],x,None),z)
      p,x,_=self.first.step(pq[0],x,y)
    return (p_out,q_out),x_out,z_out

