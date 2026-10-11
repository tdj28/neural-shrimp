"""Development lifecycle tests without loading historical neural evidence."""
from pathlib import Path
import sys
import time
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_two_write_interaction as run


def synthetic():
    cid='synthetic--p00'
    contexts={cid:dict(setting='synthetic',index=0)}
    samples=np.zeros((64,6,3))
    samples[:,1,0]=1;samples[:,2,1]=1;samples[:,5,2]=1
    refs={run.task('reference',cid,m)['id']:{'samples':samples.copy()} for m in run.sensitivity.METHODS}
    finite={}
    for dose,a in enumerate(run.AMPLITUDES):
        c=run.sensitivity.coefficients(samples,a)
        for m in run.sensitivity.METHODS:
            for s1,s2 in run.scoring.SIGN_PAIRS:
                values=c['m']+s1*c['A']+s2*c['B']+s1*s2*c['C']
                finite[run.task('finite',cid,m,dose=dose,sign1=s1,sign2=s2)['id']]={'samples':values.reshape(64,3)}
    return contexts,refs,finite


def test_exact_closed_census_and_separate_heldout_metadata():
    tasks=run.census()
    assert {k:len(v) for k,v in tasks.items()}==dict(phases=4,references=32,differences=68,finite=384)
    ids=[t['id'] for rows in tasks.values() for t in rows]
    assert len(ids)==488 and len(set(ids))==488
    assert run.phases()==run.phases() and len(run.phases())==40
    assert len({r['fraction'] for r in run.phases()})==40
    assert all(int(t['case']['id'].rsplit('p',1)[1])<8 for k,rows in tasks.items() if k!='phases' for t in rows)
    assert sum(p['split']=='heldout_reserved' for p in run.phases())==32
    benchmark=run.benchmark_tasks(tasks['finite'])
    assert len(benchmark)==32
    assert sum(t['case']['id'].startswith('core--') for t in benchmark)==16
    assert all(t['dose']==2 and t['method']['id']=='primary' for t in benchmark)


def test_reference_seal_and_saved_analysis_recover_analytic_parity():
    contexts,refs,finite=synthetic()
    seals,checks=run.reference_seal(contexts,refs)
    assert all(c['qualified'] for c in checks)
    assert len(seals['synthetic--p00'])==42
    rows=run.analyze(seals,finite,contexts)
    assert len(rows)==12 and all(r['qualified'] for r in rows)
    for row in rows:
        if row['observation']!='state':continue
        expected=32*row['amplitude']**4/(32*row['amplitude']**4+1e-8)
        assert row['scores']['second']==pytest.approx(expected,abs=1e-12)
        assert row['oracle_r_squared']==pytest.approx(expected,abs=1e-12)
        assert row['scores']['first']==0 and row['scores']['ablated']==0
        assert row['prediction_error_norm']<1e-15


def test_independent_disagreement_is_retained_and_fails_qualification():
    contexts,refs,finite=synthetic();seals,_=run.reference_seal(contexts,refs)
    key=run.task('finite','synthetic--p00','independent',dose=2,sign1=1,sign2=1)['id']
    finite[key]['samples']+=.01
    rows=run.analyze(seals,finite,contexts)
    affected=[r for r in rows if r['amplitude']==.01]
    assert affected and all(not r['qualified'] for r in affected)
    assert all(r['state_error']>.009 for r in affected)


def test_difference_check_exact_polynomial_sensitivities():
    contexts,refs,_=synthetic();cid='synthetic--p00';fd={}
    for step,h in [(-1,0.),*enumerate(run.CONTRACT['finite_difference_steps'])]:
        pairs=[(0,0)] if step==-1 else [(1,0),(-1,0),(0,1),(0,-1),*run.scoring.SIGN_PAIRS]
        for a,b in pairs:
            fd[run.task('difference',cid,'primary',step=step,sign1=a,sign2=b)['id']]={
                'samples':np.tile([a*h,b*h,a*b*h*h],(64,1))}
    checks=run.difference_checks(contexts,refs,fd)
    assert len(checks)==2 and all(r['qualified'] for r in checks)
    assert max(max(r['errors'].values()) for r in checks)<1e-12


def test_owned_worker_roundtrip_and_no_target_files(tmp_path,monkeypatch):
    for config in run.sensitivity.METHODS.values():monkeypatch.setitem(config,'steps_per_period',16)
    context=dict(W=np.zeros((2,2)),period=1.,initial=np.array([.1,-.2]),direction=np.array([1.,0.]))
    t=run.task('reference','synthetic','primary')
    stage=tmp_path/'stage'
    rows=run.runtime.execute([t],{'synthetic':context},stage,{},workers=1,
        deadline=time.monotonic()+20,per_task=10,worker_fn=run.worker)
    assert rows[0]['status']=='complete'
    restored=run.packets(stage,[t])[t['id']]
    np.testing.assert_array_equal(restored['samples'][:,3:],0)
    assert run.read(stage/'cleanup.json')['all_original_identities_absent'] is True
    p=stage/'payloads'/t['id']/'packet.json'
    p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='identity'):run.packets(stage,[t])


def test_failed_worker_is_not_promoted(tmp_path):
    t=run.task('unknown','synthetic','primary')
    stage=tmp_path/'stage'
    context=dict(W=np.zeros((2,2)),period=1.,initial=np.ones(2),direction=np.ones(2))
    rows=run.runtime.execute([t],{'synthetic':context},stage,{},workers=1,
        deadline=time.monotonic()+20,per_task=10,worker_fn=run.worker)
    assert rows[0]['status']=='failed'
    with pytest.raises(ValueError,match='Unqualified task'):run.packets(stage,[t])
