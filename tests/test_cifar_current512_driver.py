import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import explore_cifar_current512 as W
import launch_cifar_current512 as L

def test_current_card_rejects_old_operator_and_missing_audit():
 c=W.read(ROOT/W.CARD);c['status']='reviewed';W.check_card(c)
 for key,value in [('state','state-018.npy'),('independent512',None),('old_basis_reused',256),('origin','state-022.npy')]:
  d=copy.deepcopy(c);d[key]=value
  with pytest.raises(ValueError):W.check_card(d)
 assert W.BASE==L.OUT and callable(W.E.P.fd_error)

def test_profile_covers_all_fresh_actions_and_retention():
 r=W.forecasts(.05,3.,dict(hvp=.04,direct_gradient=.02),8.,1024**2,1024**2)
 assert r['passed']and r['forecast_raw_bytes']>900*1024**2
 assert not W.forecasts(2.,3.,dict(hvp=.04,direct_gradient=.02),8.,1024**2,1024**2)['passed']
 with pytest.raises(ValueError):W.forecasts(float('nan'),3.,dict(hvp=.04,direct_gradient=.02),8.,1,1024**2)

def test_completed_negative_and_accepted_units_account_every_call():
 row=dict(status='current512_correction_complete',accepted_orbit=False,rigorous_certificate=False,old_basis_reused=0,profile=dict(passed=True),qualification=dict(passed=True),operation_caps=W.CAPS,operation_attempted=dict(action=0,j2vp=0,hvp=514,direct_gradient=5),operation_returned=dict(action=0,j2vp=0,hvp=514,direct_gradient=5),basis=dict(prefix_columns=1,total_columns=512,new_basis_actions=511,completed_fixed_extension=True),correction=dict(direction_applied=False,accepted_index=None,trials=[]),direction_applied=False)
 assert L.scientific_complete(row)
 row['operation_attempted']['direct_gradient']=row['operation_returned']['direct_gradient']=7
 row['correction']=dict(direction_applied=True,accepted_index=2,trials=[dict(within_trust=False),dict(within_trust=True),dict(within_trust=True)]);row['direction_applied']=True
 assert L.scientific_complete(row)
 for key,value in [('old_basis_reused',1),('direction_applied',False)]:
  d=copy.deepcopy(row);d[key]=value;assert not L.scientific_complete(d)
 row['operation_returned']['hvp']=513;assert not L.scientific_complete(row)

def test_native_input_retained_before_error_and_inclusive_cap(tmp_path):
 store=W.Store(tmp_path/'run',limit=8*1024**2);b=W.Budget(store.event)
 class Problem:
  def hvp(self,x,v):return b.call('hvp',lambda:(_ for _ in()).throw(RuntimeError('failed native')))
 with pytest.raises(RuntimeError):W.E.Native(Problem(),store,b).hvp(np.zeros(98666),np.ones(98666))
 events=[json.loads(x)for x in(tmp_path/'run/events.jsonl').read_text().splitlines()]
 assert [e['kind']for e in events]==['native_input','operation_started']and b.attempted['hvp']==1 and b.returned['hvp']==0
 for k in('state','direction'):assert (tmp_path/'run'/events[0][k]['file']).is_file()
 b.attempted['hvp']=514
 with pytest.raises(ValueError):b.call('hvp',lambda:1)

def test_archive_import_closure(tmp_path):
 W.install()
 for n in W.B.SOURCES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import explore_cifar_current512 as w;w.install();r=Path.cwd();a={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert a<=set(w.B.SOURCES),a-set(w.B.SOURCES)"
 z=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
 assert z.returncode==0,z.stderr
