import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import explore_cifar_stationary_continuation as W
import launch_cifar_stationary_continuation as L


def completed(passed=False):
    counts=dict(action=0,j2vp=0,hvp=4,direct_gradient=7)
    station=dict(passed=passed,steps=[dict(linear_actions=1,trials=[],forcing=.8,slope=-.2)])
    return dict(status='stationary_continuation_complete',comparison_qualified=True,accepted_orbit=False,rigorous_certificate=False,
        qualification=dict(passed=True),operation_caps=W.CAPS,operation_attempted=counts,operation_returned=counts,
        stationary=station,mode=None,seed_qualified=False)


def test_complete_negative_and_positive_counts_are_fixed():
    r=completed();assert L.scientific_complete(r)
    bad=copy.deepcopy(r);bad['stationary']['passed']=True;assert not L.scientific_complete(bad)
    bad=copy.deepcopy(r);bad['operation_attempted']['hvp']=5;assert not L.scientific_complete(bad)
    r['stationary']['passed']=True;r['mode']=dict(passed=True);r['seed_qualified']=True
    r['operation_attempted']['hvp']=37;r['operation_returned']['hvp']=37
    assert L.scientific_complete(r)


def test_gate_counts_complete_maximum_and_io_budget():
    events=[dict(kind='operation_returned',operation=k,seconds=v)for k,v in [('hvp',.08492048806510866),('direct_gradient',.9516282950062305)]]
    g=W.measured_gate(events);assert 370<g['forecast_seconds']<400
    events[0]['seconds']=.2
    with pytest.raises(ValueError):W.measured_gate(events)
    assert W.BASE==L.OUT and W.CAPS['hvp']==4*(256+1)+2+33 and W.CAPS['direct_gradient']==7+4*13


def test_native_attempt_retains_state_direction_before_failure(tmp_path):
    from neural_shrimp.cifar_restart_diagnostic import Store
    store=Store(tmp_path/'run',limit=8*1024**2);budget=W.Budget(store.event)
    class Failed:
        def hvp(self,x,v):raise RuntimeError('synthetic fail')
    native=W.Native(Failed(),store,budget)
    with pytest.raises(RuntimeError):native.hvp(np.zeros(98666),np.ones(98666))
    row=json.loads((tmp_path/'run/events.jsonl').read_text().splitlines()[0]);assert row['kind']=='native_input'
    for field in ('state','direction'):assert (tmp_path/'run'/row[field]['file']).is_file()
    assert not list((tmp_path/'run').glob('*return*'))


def test_archive_isolated_import_closure(tmp_path):
    W.install()
    for name in W.B.SOURCES:
        dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    code="""import sys
from pathlib import Path
import explore_cifar_stationary_continuation as W
W.install();root=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(root))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(root)}
assert loaded<=set(W.B.SOURCES),loaded-set(W.B.SOURCES)
assert Path(W.__file__).resolve().is_relative_to(root)
print(len(loaded))"""
    env=dict(os.environ,PYTHONPATH=str(tmp_path/'src')+os.pathsep+str(tmp_path/'tools'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    run=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,text=True,capture_output=True)
    assert run.returncode==0,run.stderr
    assert int(run.stdout)>10


def test_literal_audited_accepted_parent_and_relaxed_global_gates():
    row=dict(source_commit=W.EXEC,status='stationary_saved_merit_complete',comparison_qualified=True,
      accepted_orbit=False,rigorous_certificate=False,stationarity_qualified=False,accepted_index=3,
      trials=[dict(accepted=v)for v in[False,False,False,True]],direction_gate=dict(passed=True,old_forcing_pass=False),
      operation_attempted=dict(action=0,j2vp=0,hvp=1,direct_gradient=5),operation_returned=dict(action=0,j2vp=0,hvp=1,direct_gradient=5))
    W.check_parent(row)
    for key,value in [('accepted_index',2),('stationarity_qualified',True),('source_commit','0'*40)]:
        bad=copy.deepcopy(row);bad[key]=value
        with pytest.raises(ValueError):W.check_parent(bad)
    assert W.CONTINUATION_SETTINGS['forcing_limit']==.9 and W.SETTINGS['forcing_limit']==.5
    assert W.AUDIT_PIN=='364f7147e20590d0299cf5174d3631d02c3f64434d0bfa7c0b4fd63a3f013297'


def test_actual_merit_trials_with_point_seven_five_are_complete():
    row=completed();s=row['stationary']['steps'][0];s.update(forcing=.75,slope=-.1,trials=[dict(within_trust=True)],accepted_index=0)
    row['operation_attempted']['direct_gradient']=row['operation_returned']['direct_gradient']=8
    assert L.scientific_complete(row)
    s['forcing']=.91;assert not L.scientific_complete(row)
