import sys,numpy as np,pytest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from neural_shrimp.fold_flip_scout import augmented,qualify
class Nonlinear:
 n=2
 def cusp_step(self,x,h,o):return[(.2+.01*h)*x[0]+.03*x[0]*x[1]+.02*o,(-.3+.02*o)*x[1]+.02*x[0]*x[0]+.01*h]
def test_all_native_augmented_columns_against_nonlinear_differences():
 y=np.array([.3,.2,.5,.6,.8,.6,-.6,.8]);v,J,d=augmented(Nonlinear(),y,3);eps=1e-5
 for j in range(8):
  e=np.eye(8)[j]*eps;diff=(augmented(Nonlinear(),y+e,3)[0]-augmented(Nonlinear(),y-e,3)[0])/(2*eps)
  assert max(abs(diff-J[:,j]))<1e-9
 assert len(d['orbit'])==len(d['q_phases'])==len(d['w_phases'])==4

def test_both_direction_inputs_retained_and_actual_calls():
 events=[];count=[];y=np.array([.3,.2,.5,.6,.8,.6,-.6,.8]);augmented(Nonlinear(),y,3,tick=lambda:count.append(1),observe=lambda k,r:events.append((k,r)))
 assert len(count)==6 and [v['direction']for k,v in events if k=='native_input']==['plus']*3+['minus']*3
 class Broken:
  n=2
  def cusp_step(self,*a):raise ValueError('phase failure')
 events=[]
 with pytest.raises(ValueError,match='phase failure'):augmented(Broken(),y,3,observe=lambda k,r:events.append((k,r)))
 assert len(events)==1 and events[0][0]=='native_input'

def test_spectrum_and_primitive_gates_do_not_relabel_unstable_or_degenerate():
 y=np.r_[np.zeros(3),1.,1.,1.,0.,0.,0.,1.,0.];orbit=np.zeros((19,3));orbit[1:18,0]=np.arange(1,18)*.1
 d=dict(monodromy=np.diag([1.,-1.,.5]),orbit=orbit);a=qualify(y,np.zeros(11),d);assert a['passed']
 d['monodromy']=np.diag([1.,-1.,1.1]);assert not qualify(y,np.zeros(11),d)['passed']
 d['monodromy']=np.diag([1.,-1.,1.]);assert not qualify(y,np.zeros(11),d)['gates']['simple']
 d['monodromy']=np.diag([1.,-1.,.5]);d['orbit'][9]=0;assert not qualify(y,np.zeros(11),d)['gates']['primitive']
