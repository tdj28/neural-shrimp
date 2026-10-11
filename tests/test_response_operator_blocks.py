import numpy as np
import pytest
from neural_shrimp.response_operator_blocks import Operator,ActionBudget,inner_cg,probe

def dense(maps,steps):
 p,n,_=maps.shape;a=maps*np.exp(.05);k=np.zeros((steps*n,(steps+1)*n))
 for t in range(steps):k[t*n:(t+1)*n,t*n:(t+1)*n]=-a[t%p];k[t*n:(t+1)*n,(t+1)*n:(t+2)*n]=np.eye(n)
 return k.T@k+.0001*np.eye((steps+1)*n)

def test_all_principal_blocks_keep_free_and_cut_edges():
 maps=np.array([[[.8,.2],[-.1,.6]],[[.5,-.3],[.2,.7]]]);steps=8;op=Operator(maps,steps,1000);h=dense(maps,steps)
 for first,last in [(0,4),(4,8),(8,9),(0,9)]:
  x=np.arange((last-first)*2)/10+.3
  np.testing.assert_allclose(op.apply(x,first,last),h[first*2:last*2,first*2:last*2]@x,rtol=2e-14,atol=2e-14)

def test_budget_exhaustion_preserves_terminal_reserve():
 op=Operator(np.ones((1,1,1))*.8,16,48)
 op.rhs()
 with pytest.raises(ActionBudget):op.apply(np.ones(17))
 op.apply(np.ones(17),terminal=True)
 assert sum(op.counts.values())==48

def test_last_singleton_block_requires_no_phase_actions():
 op=Operator(np.ones((1,1,1))*.8,16,100);x,k=inner_cg(op,np.array([2.]),16,17,'last')
 assert k==1 and sum(op.counts.values())==0
 np.testing.assert_allclose(x,[2/1.0001])

def test_zero_inner_rhs_and_nonfinite_fail_closed():
 op=Operator(np.ones((1,1,1))*.8,16,100)
 assert inner_cg(op,np.zeros(4),0,4,'zero')[1]==0
 with pytest.raises(ValueError):inner_cg(op,np.full(4,np.nan),0,4,'bad')

def test_flexible_probe_retains_accuracy_with_all_actions_counted():
 events=[];maps=np.array([[[.3]]]);r=probe(maps,96480,lambda k,v:events.append((k,v)))
 assert r['solve_qualified'] and r['true_normal_relative_residual']<=1e-8
 assert r['counts']['JVP']>0 and sum(r['counts'].values())<=96480
 h=dense(maps,16);imp=np.zeros(16);imp[8]=1;k=np.zeros((16,17))
 for t in range(16):k[t,t]=-.3*np.exp(.05);k[t,t+1]=1
 reference=np.linalg.solve(h,k.T@imp)
 np.testing.assert_allclose(r['readout_column'],reference[9:10],rtol=1e-6,atol=1e-8)
 assert events[-1][0]=='terminal'

def test_callback_failure_is_not_numerical_abstention():
 def fail(*_):raise OSError('durability')
 with pytest.raises(OSError):probe(np.array([[[.3]]]),96480,fail)


def test_launcher_accounts_final_manifest_and_log(tmp_path):
 import importlib.util
 from pathlib import Path
 p=Path(__file__).resolve().parents[1]/'tools/launch_response_operator_blocks.py'
 spec=importlib.util.spec_from_file_location('launcher',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 (tmp_path/'process.log').write_bytes(b'x'*(4*1024**2))
 assert not m.final_byte_gate(tmp_path,{'exit_code':0})['passed']


def test_profile_normalizes_early_convergence_to_full_target_cost():
 import importlib.util
 from pathlib import Path
 p=Path(__file__).resolve().parents[1]/'tools/explore_response_operator_blocks.py'
 spec=importlib.util.spec_from_file_location('driver',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 r=m.profile_projection(1.,1000,{'JVP':500,'VJP':500})
 assert r['multiplier']==1+(96480+173664)/1000 and r['projected_seconds']>270
 r=m.profile_projection(1.,1000,{'JVP':86832,'VJP':86832});assert r['multiplier']==3


def test_interrupted_atomic_action_has_durable_start():
 events=[]
 def emit(k,v):
  events.append((k,v))
  if k=='operator_started':raise OSError('injected after start')
 op=Operator(np.ones((1,1,1))*.8,16,1000,emit)
 with pytest.raises(OSError):op.apply(np.ones(17))
 assert events[0][0]=='operator_started' and events[0][1]['phase_actions_upper']==32 and sum(op.counts.values())==0
