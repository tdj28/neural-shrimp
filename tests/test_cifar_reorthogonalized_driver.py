import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import explore_cifar_reorthogonalized as W
import launch_cifar_reorthogonalized as L

def test_fixed_same_operator_card_rejects_last_state_and_changed_budget():
    c=W.read(ROOT/W.CARD);c['status']='reviewed';W.check_card(c)
    for key,value in [('operator',dict(state='state-022.npy',gradient=23,first_hvp=774,true_hvp=1030)),('operation_caps',dict(action=0,j2vp=0,hvp=258,direct_gradient=1))]:
        d=copy.deepcopy(c);d[key]=value
        with pytest.raises(ValueError):W.check_card(d)
    assert W.BASE==L.OUT and W.LIMITS['seconds']==300

def test_profile_covers_all_actions_max_width_and_retention():
    r=W.forecasts(.05,3.,dict(hvp=.04,direct_gradient=.02),8.,1024**2)
    assert r['passed']and r['forecast_raw_bytes']>400*1024**2 and r['forecast_seconds']>60
    assert not W.forecasts(1.,3.,dict(hvp=.04,direct_gradient=.02),8.,1024**2)['passed']
    with pytest.raises(ValueError):W.forecasts(float('nan'),3.,dict(hvp=.04,direct_gradient=.02),8.,1)

def test_scientific_completion_includes_negative_not_false_equal_budget():
    row=dict(status='reorthogonalized_comparison_complete',accepted_orbit=False,rigorous_certificate=False,direction_applied=False,profile=dict(passed=True),operation_caps=W.CAPS,operation_attempted=dict(action=0,j2vp=0,hvp=257,direct_gradient=1),operation_returned=dict(action=0,j2vp=0,hvp=257,direct_gradient=1),basis=dict(actions=256,equal256_action_count=True),comparison=dict(qualified=True),comparison_qualified=True)
    assert L.scientific_complete(row)
    bad=copy.deepcopy(row);bad['operation_returned']['hvp']=256;assert not L.scientific_complete(bad)
    bad=copy.deepcopy(row);bad['direction_applied']=True;assert not L.scientific_complete(bad)
    row['basis']=dict(actions=5,equal256_action_count=False);row['operation_attempted']['hvp']=row['operation_returned']['hvp']=6;row['comparison']['qualified']=row['comparison_qualified']=False;assert L.scientific_complete(row)

def test_native_attempt_cap_and_retention_before_failed_call(tmp_path):
    store=W.Store(tmp_path/'run',limit=8*1024**2);b=W.Budget(store.event)
    class Problem:
        def hvp(self,x,v):return b.call('hvp',lambda:(_ for _ in()).throw(RuntimeError('retained failure')))
    native=W.E.Native(Problem(),store,b)
    with pytest.raises(RuntimeError):native.hvp(np.zeros(98666),np.ones(98666))
    events=[json.loads(x)for x in(tmp_path/'run/events.jsonl').read_text().splitlines()]
    assert [x['kind']for x in events]==['native_input','operation_started']and b.attempted['hvp']==1 and b.returned['hvp']==0
    for field in('state','direction'):assert (tmp_path/'run'/events[0][field]['file']).is_file()
    b.attempted['hvp']=257
    with pytest.raises(ValueError):b.call('hvp',lambda:1)

def test_closing_reserve_retains_final_failure(tmp_path):
    s=W.Store(tmp_path/'run',limit=4096,reserve=2048);s.write('near-cap',b'a'*2000)
    with pytest.raises(ValueError):s.write('overflow',b'b'*100)
    s.finish(dict(status='failed'));assert (tmp_path/'run/result.json').is_file()and(tmp_path/'run/manifest.json').is_file()

def test_isolated_full_executed_archive_closure(tmp_path):
    W.install()
    for name in W.B.SOURCES:
        dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    code="""import sys
from pathlib import Path
import explore_cifar_reorthogonalized as W
W.install();root=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(root))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(root)}
assert loaded<=set(W.B.SOURCES),loaded-set(W.B.SOURCES)
print(len(loaded))"""
    env=dict(os.environ,PYTHONPATH=str(tmp_path/'src')+os.pathsep+str(tmp_path/'tools'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    r=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,text=True,capture_output=True);assert r.returncode==0,r.stderr;assert int(r.stdout)>15
