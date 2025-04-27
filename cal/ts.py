from time import perf_counter

def f_to_str(X,lj=None,prec=2,p=False,logf=None):
  if lj is None:
    lj=prec+6
  if not(isinstance(X,str)):
    try:
      X=float(X)
      X='{0:.{1}g}'.format(X,prec)
    except:
      pass
  if isinstance(X,str):
    if lj:
      X=X.ljust(lj)
  else:
    X=''.join((f_to_str(x,lj=lj,prec=prec) for x in X))
  if p:
    print(X,file=logf)
  return X

class TimeStepper:
  def __init__(self,clock_avg_rate=.01,time_avgs={},logf=None):
    self.clock_avg_rate=clock_avg_rate
    self.tl=perf_counter()
    self.time_avgs=time_avgs
    self.lab_len=max([0]+[len(lab) for lab in time_avgs])
    self.logf=logf
  def get_timestep(self,label=False,start_immed=False):
    t=perf_counter()
    if label:
      try:
        self.time_avgs[label]+=(1+self.clock_avg_rate)*\
                               self.clock_avg_rate*\
                               float(t-self.tl)
      except:
        print('New time waypoint',label,file=self.logf)
        self.time_avgs[label]=(1+self.clock_avg_rate)*\
                              float(t-self.tl) if\
                              start_immed else 1e-8
        self.lab_len=max(len(label),self.lab_len)
      self.time_avgs[label]*=(1-self.clock_avg_rate)
    self.tl=t
  def report(self,p=False,prec=3):
    tai=self.time_avgs.items()
    lj=max(self.lab_len,prec+5)+1
    #tsr='\n'.join([str(k)+':'+str(log10(v)) for k,v in tai])
    tsr=f_to_str(self.time_avgs,lj=lj,prec=prec)+'\n'+\
        f_to_str(self.time_avgs.values(),lj=lj,prec=prec)
    if p:
      print(tsr,file=self.logf)
    tsrx='\n'.join(['\\texttt{'+k.replace('_','\\_')+'}&'+\
                    f_to_str(log10(v))+'\\\\\n' for k,v in tai])
    tsrx+='\\end{tabular}'
    return tsr,tsrx
