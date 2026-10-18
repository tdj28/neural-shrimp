import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import numpy as np
from branch_trace import trace_branch
class Linear:
 n=1
 def residual(self,x,t):return -.5*x+t[0]
 def monodromy(self,x,t):return np.array([[.5]])
 def param_jacobian(self,x,t,which):return np.array([[1.]])
attempts=[];states=[]
r,stop,ys=trace_branch(Linear(),np.array([2.]),[1.,0.],steps=4,ds=.001,ds_max=.001,k=1,on_attempt=attempts.append,on_state=lambda i,s:states.append((i,s)))
assert len(r)==len(ys)==len(states)==4 and len(attempts)>=4
assert all(abs(s[0]-2*p['lambda'])<1e-10 and p['multipliers']==[[.5,0.]] for s,p in zip(ys,r))
class Failed(Linear):
 def residual(self,x,t):return np.array([1.])
a=[];r,stop,ys=trace_branch(Failed(),np.array([2.]),[1.,0.],max_newton=2,on_attempt=a.append)
assert not r and not ys and len(a)==3 and stop.startswith('no convergence')
print('2 actual small-family controls passed: full state/attempt persistence and failed-start preservation')
