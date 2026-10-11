import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from neural_shrimp.signed_flip_scout import evaluate
class Linear:
 n=1
 def cusp_step(self,x,h,o):return[(.2+.1*h+.2*o)*x[0]]
def test_full_section_derivative_and_direction_columns():
 z=np.array([.4,.7,1.]);h=.5;p=3;a=.2+.1*h+.2*z[1];v,J,dh,d=evaluate(Linear(),z,h,p)
 E=np.array([[a**p-1,p*a**(p-1)*.2*z[0],0],[0,p*a**(p-1)*.2*z[2],a**p+1],[0,0,z[2]]]);assert np.max(abs(J-E))<1e-13
 assert np.max(abs(dh-[p*a**(p-1)*.1*z[0],p*a**(p-1)*.1*z[2],0]))<1e-13
 assert len(d['orbit'])==p+1 and len(d['native_returns'])==p

def test_native_input_retained_before_failure():
 class Broken:
  n=1
  def cusp_step(self,*a):raise ValueError('failed phase')
 import pytest
 events=[]
 with pytest.raises(ValueError,match='failed phase'):evaluate(Broken(),[0,.5,1],.2,18,observe=lambda k,v:events.append((k,v)))
 assert [x[0]for x in events]==['native_input']and events[0][1]['phase']==0
