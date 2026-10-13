"""Tiny CPU controls only; no target inputs, model construction, or passages."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'codex/hostile-fold-crosscheck'
sys.path.insert(0,str(HERE))
import passage_worker as worker
import passage_runtime as runtime


def fold():
    return dict(passed=True,stage='qualified',certificate_passed=False,
        passage_trajectories_evaluated=False,solver_history_complete=True,return_map='G=F^2',
        active_parameter_units='log10T',alpha=.083,gates={'fold':True},
        x_c=np.array([0.,0.]),r=np.array([1.,0.]),l=np.array([1.,2.]),
        parameter_c=-.04,a=-1.,b=2.,U=.05,ghost_parameter_direction=-1)


def plan():
    return dict(schema='codex-hostile-passage-v1',envelope=runtime.CAPS.copy(),
        source_sha256={k:'a'*64 for k in runtime.SOURCES},fold_authority=runtime.AUTHORITY.copy(),
        design=worker.design(fold()),basis={},scope=runtime.SCOPE)


def row(u=-1.,U=1.,cap=5,pred=3):
    arrays=dict(x_c=np.zeros(2),r=np.array([1.,0.]),l=np.array([1.,2.]))
    r=dict(id='tiny',u_in=u,u_out=U,parameter=-.01,max_G=cap,quadratic_G=pred,
        arctan_G=2.8,asymptotic_G=3.5,initial_state=worker.digest(np.array([u,0.])))
    return r,arrays


def test_six_frozen_negative_offsets_and_forecasts():
    f=fold();d=worker.design(f)
    assert [r['entrance'] for r in d['rows']]==['full','midpoint']*3
    assert [r['epsilon'] for r in d['rows']]==[-1e-5,-1e-5,-3e-5,-3e-5,-1e-4,-1e-4]
    for r in d['rows']:
        assert r['parameter']==f['parameter_c']+r['epsilon']<f['parameter_c']
        assert r['max_G']==2*r['quadratic_G']+20
        assert r['asymptotic_G']>r['arctan_G']
    assert d['total_cap_G']==sum(r['max_G'] for r in d['rows'])
    assert d['criterion']['secondary'].startswith('report')


@pytest.mark.parametrize('key,value',[('passed',False),('a',1.),('b',-2.),('U',.06),
    ('ghost_parameter_direction',1),('return_map','G=F'),('active_parameter_units','T')])
def test_wrong_fold_or_gauge_refused(key,value):
    f=fold();f[key]=value
    with pytest.raises(ValueError):worker.design(f)


def test_nonnormal_left_projection_and_strict_exit_clock():
    r,a=row();calls=[];saved=[]
    # l=(1,2) and right r=(1,0): this vector is in ker(l), despite a nonzero first coordinate.
    transverse=np.array([-2.,1.])*.01
    def step(x,p):
        calls.append((x.copy(),p));return x+np.array([1.,0.])+transverse
    result=worker.run_row(step,r,a,save=saved.append)
    assert result['measured_G']==3 and result['measured_F']==6
    assert result['pre_crossing_u']==pytest.approx(1.)
    assert result['crossing_u']>1 and len(calls)==3
    assert saved[-1]['state_trace'].shape==(4,2)
    np.testing.assert_allclose(saved[-1]['transverse'][-1],3*np.linalg.norm(transverse))
    assert result['primary_pass'] is True


def test_exact_boundary_is_not_exit():
    r,a=row();saved=[]
    z=worker.run_row(lambda x,p:x+np.array([1.,0.]),r,a,save=saved.append)
    assert z['measured_G']==3 and saved[-1]['u'].tolist()==[-1.,0.,1.,2.]


def test_midpoint_zero_burn_in_and_cap_censor():
    r,a=row(u=0,cap=2,pred=3);saved=[]
    z=worker.run_row(lambda x,p:x+np.array([.5,0.]),r,a,save=saved.append)
    assert z['status']=='censored' and z['measured_G'] is None
    assert z['measured_F'] is None and z['recorded_G']==2 and z['primary_pass'] is False
    assert saved[-1]['u'].tolist()==[0.,.5,1.]


def test_interruption_keeps_complete_prefix_and_initial_state():
    r,a=row();saved=[];count=0
    def check():
        nonlocal count
        count+=1
        if count==3:raise TimeoutError('bounded')
    with pytest.raises(TimeoutError):worker.run_row(lambda x,p:x+np.array([.2,0.]),r,a,save=saved.append,check=check)
    assert saved[-1]['state_trace'].shape==(3,2) and saved[-1]['recorded_G']==2
    assert saved[0]['recorded_G']==0


@pytest.mark.parametrize('out',[np.array([1.,np.nan]),np.array([1.,2.],dtype=np.float32),np.zeros((2,1))])
def test_bad_full_state_refused(out):
    r,a=row();saved=[]
    with pytest.raises(ValueError,match='invalid full-state'):worker.run_row(lambda x,p:out,r,a,save=saved.append)
    assert len(saved)==1 and saved[0]['recorded_G']==0


def test_initial_mutation_and_modes_refused():
    r,a=row();r['initial_state']['sha256']='0'*64
    with pytest.raises(ValueError,match='initial'):worker.run_row(lambda x,p:x,r,a,save=lambda v:None)
    a['l'][0]=2
    with pytest.raises(ValueError,match='paired'):worker.vectors(a)


def test_scalar_quadratic_returns_exact_frozen_prediction():
    f=fold();d=worker.design(f)
    for r in d['rows']:
        z=worker.run_row(lambda x,p:np.array([x[0]+f['a']*(p-f['parameter_c'])+f['b']*x[0]**2,0.]),
            r,f,save=lambda v:None)
        assert z['measured_G']==r['quadratic_G'] and z['primary_pass']


def test_plan_strict_and_alias_refusals():
    p=plan();assert runtime.validate_plan(p) is p
    for mutation in [lambda p:p['envelope'].update(cpu_threads=True),
        lambda p:p['envelope'].update(timeout_seconds=1800.),
        lambda p:p['source_sha256'].pop(runtime.SOURCES[0]),
        lambda p:p['fold_authority'].update(receipt_sha256='0'*64),
        lambda p:p.update(scope='natural_reinjection'),lambda p:p.update(extra=1)]:
        q=copy.deepcopy(p);mutation(q)
        with pytest.raises(ValueError):runtime.validate_plan(q)


def test_preimport_failure_status_retained_and_never_overwrites(tmp_path,monkeypatch):
    import os
    owned=tmp_path/'owned';owned.mkdir();attempt=owned/'passage-test';attempt.mkdir()
    output=attempt/'run';planpath=tmp_path/'plan.json';planpath.write_text('{}')
    monkeypatch.setattr(runtime,'OWNED',owned)
    monkeypatch.setattr(runtime.os,'getpriority',lambda *a:10)
    monkeypatch.setattr(runtime,'available',lambda:32*1024**3)
    monkeypatch.setattr(runtime.resource,'getrlimit',lambda *a:(12*1024**3,12*1024**3))
    monkeypatch.setattr(runtime.resource,'setrlimit',lambda *a:None)
    for k,v in dict(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TASK7_PLAN_SHA256='0'*64).items():
        monkeypatch.setenv(k,v)
    monkeypatch.setattr(sys,'argv',['passage_runtime.py','--plan',str(planpath),'--output',str(output)])
    with pytest.raises(ValueError,match='external plan hash'):runtime.main()
    status=(output/'status.json').read_bytes();decoded=json.loads(status)
    assert decoded['status']=='passage_failed' and decoded['model_G_calls']==0
    assert decoded['computational_completed'] is False and decoded['certificate_passed'] is False
    with pytest.raises(FileExistsError):runtime.main()
    assert (output/'status.json').read_bytes()==status


def test_isolated_help_without_model_import():
    p=subprocess.run([sys.executable,'-I','-B',str(HERE/'passage_runtime.py'),'--help'],capture_output=True,text=True,timeout=10)
    assert p.returncode==0 and '--plan' in p.stdout
