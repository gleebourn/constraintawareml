from queue import Queue

p_dict=dict() #Data passed downward
q_dict=dict() #Data passed upward
x_dict=dict() #Data passed forward
y_dict=dict() #Data passed backward

class state:
  def __init__(self,data=None,arr=False):
    self.data=data
    self.arr=arr

  def set(self,data):
    if not self.arr:
      self.__init__(data)
    else:
      [a.set(data(i)) for a,i in zip(self.arr,data)]
    
  def get(self):
    if not self.arr:
      return self.data
    else:
      return [s.get() for s in self.data]

class PL:
  def __init__(self,f,b,x_in=None,y_in=None,p_in=None,x_out=None,y_out=None,p_out=None):
    self._f=f
    self._b=b

    interface=[x_in,y_in,p_in,x_out,y_out,p_out]
    interface=[i if isinstance(i,state) else state(i) for i in interface]
    
    [self.x_in,self.y_in,self.p_in,self.x_out,self.y_out,self.p_out]=interface

    self.l=left_neighbour
    self.r=right_neighbour
  
  def f(self):
    self.x_out.data=self._f(self.p_in.data,self.x_in.data)
  
  def b(self):
    self.p_out.data,self.x_out.data=self._b(self.p_in.data,self.x_in.data,self.y_in.data)

  def forward(self,p,x):
    self.p_in.data=p
    self.x_in.data=x
    self.f()
    return self.x_out.data

  def backward(self,p,x,y):
    self.p_in.data=p
    self.x_in.data=x
    self.y_in.data=y
    self.b()
    return self.p_out.data,self.x_out.data

def set_p(pl,p):[pl.p_in.set(s) for pl,s in zip(pls,p)]

def serial_composition(pls):
  _pls=[]
  for pl in pls:
    try:
      _pls+=pl.arr
    except AttributeError:
      _pls.append(pl)
  pls=_pls
  p_ins=state(arr=[pl.p_in for pl in pls])
  p_outs=state(arr=[pl.p_out for pl in pls])

  p_in_state=state(arr=p_ins)
  p_out_state=state(arr=p_outs)

  for i in range(len(pls[:-1])):
    pls[i].y_in=pls[i+1].x_out
    pls[i+1].x_in=pls[i].y_out
  
  def _f(p,x): #Input disgarded as variables already correctly set
    [pl.f() for pl in pls]
    return pls[-1].x_out.get()
  
  def _b(p,x,y):
    [pl.f() for pl in pls[:-1]]
    [pl.b() for pl in reversed(pls)]
    return 

  ret=PL(_f,_b,pls[0].x_in,pls[-1].y_in,p_in_state,
         pls[-1].x_out,pls[0].y_out,p_out_state)
  ret.arr=pls
  return ret

