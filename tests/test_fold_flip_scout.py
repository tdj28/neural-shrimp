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

def test_damped_correct_retains_rejected_trials_and_original_guard():
 from neural_shrimp.fold_flip_scout import damped_correct
 events=[]
 def f(y):return np.array([y[0]*y[0]-1]),np.array([[2*y[0]]]),{}
 y,v,J,d,calls=damped_correct(f,[.9],observe=lambda k,r:events.append((k,r)))
 assert abs(y[0]-1)<1e-10 and calls==len([1 for k,r in events if k=='augmented_return'])
 trials=[r for k,r in events if k=='line_search'];assert not trials[0]['local_guard']and all(r['local_guard']and r['new_residual']<r['old_residual']for r in trials if r['passed'])
 with pytest.raises(RuntimeError,match='evaluation cap'):damped_correct(f,[.9],max_evaluations=1)

def test_non_descent_and_resource_stop_do_not_relax_or_retry():
 from neural_shrimp.fold_flip_scout import damped_correct
 events=[]
 with pytest.raises(ValueError,match='line search failed'):damped_correct(lambda y:(np.ones(1),np.eye(1),{}),[1.],observe=lambda k,r:events.append((k,r)))
 assert not any(r['passed']for k,r in events if k=='line_search')
 events=[]
 def stopped(y):raise RuntimeError('resource')
 with pytest.raises(RuntimeError,match='resource'):damped_correct(stopped,[1.],observe=lambda k,r:events.append((k,r)))
 assert [k for k,r in events]==['augmented_input']
