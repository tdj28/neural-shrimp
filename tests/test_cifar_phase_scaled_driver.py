import sys,json,copy,hashlib
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
import explore_cifar_phase_scaled as W
import launch_cifar_phase_scaled as L

def test_complete_fixed_work_is_distinct_from_accepted_correction():
 caps=dict(action=0,j2vp=0,hvp=818,direct_gradient=40)
 row=dict(status='phase_scaled_pilot_complete',comparison_qualified=True,accepted_orbit=False,spectrum_requested=False,operation_caps=caps,operation_attempted=dict(action=0,j2vp=0,hvp=816,direct_gradient=14),operation_returned=dict(action=0,j2vp=0,hvp=816,direct_gradient=14),qualification=dict(passed=True),linear=dict(iterations=400,scaled_actions=401,extra_phase_actions=4,forcing_pass=False,usable_descent=True),trials=[],accepted_index=None,last_accepted_state='initial-theta.npy')
 assert L.scientific_complete(row)
 bad=copy.deepcopy(row);bad['trials']=[dict(accepted_local_step=True)];bad['accepted_index']=0;bad['last_accepted_state']='trial-0-theta.npy';assert not L.scientific_complete(bad)
 bad['linear']['forcing_pass']=True;assert L.scientific_complete(bad)
 bad['operation_attempted']['hvp']=819;assert not L.scientific_complete(bad)
 bad=copy.deepcopy(row);bad['accepted_orbit']=True;assert not L.scientific_complete(bad)

def test_measured_full_fixed_physical_cost_cap():
 events=[dict(kind='operation_returned',operation=k,seconds=v)for k,v in [('hvp',.02),('direct_gradient',.03)]]
 out=W.measured_gate(events);assert out['forecast_seconds']==pytest.approx(2*(818*.02+40*.03)+20)
 events[0]['seconds']=1
 with pytest.raises(ValueError):W.measured_gate(events)
 assert W.CAPS['hvp']==10+2*(400+2)+4
 assert W.CAPS['direct_gradient']==2+12+2*13

def test_complete_nested_immutable_diagnostic(tmp_path):
 root=tmp_path/'diag';(root/'run').mkdir(parents=True)
 (root/'run'/'outcome.json').write_text('{}')
 (root/'launcher-result.json').write_text(json.dumps(dict(exit_code=0,owned_live_pids=[],worker_pid_absent=True,launcher_pid=999999998,owned_group_pid=999999999)))
 def seal():
  inv={str(p.relative_to(root)):W.A.L.entry(p)for p in root.rglob('*')if p.is_file()and p.name!='manifest.json'}
  (root/'manifest.json').write_text(json.dumps(dict(files=inv)))
  for p in root.rglob('*'):
   if p.is_file():p.chmod(0o444)
  return W.A.L.entry(root/'manifest.json')['sha256']
 pin=seal();assert W.verify_diagnostic(root,pin,lambda:None)['exit_code']==0
 file=root/'run/outcome.json';file.chmod(0o644);file.write_text('{"mutated":true}');file.chmod(0o444)
 with pytest.raises(ValueError):W.verify_diagnostic(root,pin,lambda:None)
