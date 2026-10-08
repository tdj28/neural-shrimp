"""Synthetic fixed-state controls. No saved target input/model is evaluated."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from neural_shrimp import s1r_gap_diagnostic as core
from neural_shrimp.period2_branch_io import Store,load_record
ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def make_budget():
    events=[];return core.Budget(lambda k,v:events.append((k,v))),events


def test_analytic_two_step_parameter_chain_on_quadratic():
    # Synthetic quadratic gradient, no neural model or target finite differences.
    Q=np.array([[2.,.2],[.2,3.]])
    derivative=lambda t:(float(t@Q@t/2),Q@t,Q)
    theta=np.array([.3,-.5]);rates=np.array([.1,.03]);budget,events=make_budget()
    result=core.evaluate_state(theta,rates,1,derivative,budget,'toy')
    J=np.eye(2)-rates[:,None]*Q
    assert np.allclose(result['A'],J@J-np.eye(2),atol=1e-15)
    dp=np.diag([np.log(10.)*.1,0.])
    expected=-J@dp@Q@theta-dp@Q@J@theta
    assert np.allclose(result['b'],expected,atol=1e-15)
    assert budget.attempted==dict(maps=2,jacobians=2,svds=0)
    assert budget.returned==budget.attempted
    eps=1e-6
    def residual(h):
        D=np.diag([.1*10**h,.03]);K=np.eye(2)-D@Q
        return K@K@theta-theta
    assert np.allclose(result['b'],(residual(eps)-residual(-eps))/(2*eps),atol=1e-10)


def test_simple_fold_null_tangent_and_scaled_conversion():
    # R=(x0²-h,x1): at the fold A=diag(0,1), b=(-1,0).
    budget,events=make_budget();A=np.diag([0.,1.]);b=np.array([-1.,0.])
    g=core.geometry(dict(A=A,b=b),np.array([3.,7.]),np.array([1.,0.,0.]),budget,'fold')
    assert g['raw_A']['rank']['status']=='numerically_rank_deficient'
    assert g['raw_B']['rank']['status']=='full_row_rank_resolved'
    for name in ('raw_B','scaled_B'):
        t=g[name]['null_tangent'];assert t['status']=='local_null_tangent'
        assert np.allclose(t['tangent'],[1.,0.,0.])
    assert budget.attempted['svds']==4
    B=next(v for k,v in events if k=='fold_scaled_B_svd')
    assert B['Vh'].shape==(3,3)
    assert abs(g['raw_A']['smallest_left_parameter_coupling'])==1


def test_nonsingular_illconditioned_is_not_fold_and_scaling_is_explicit():
    A=np.diag([1e-7,1.]);b=np.array([1.,2.]);budget,events=make_budget()
    d=np.array([4.,9.]);secant=np.array([-1e7,-2.,1.])
    g=core.geometry(dict(A=A,b=b),d,secant,budget,'nonfold')
    assert g['raw_A']['rank']['status']=='full_row_rank_resolved'
    assert g['raw_A']['rank']['isolated_soft_direction']
    assert not any('fold' in str(row.get('status','')) for row in g.values())
    raw=g['raw_B']['null_tangent']['tangent'];scaled=g['scaled_B']['null_tangent']['tangent']
    assert np.allclose(raw,scaled,atol=1e-10)
    item=next(v for k,v in events if k=='nonfold_scaled_B_svd')
    expect=np.column_stack((A,b))*np.r_[d,core.TOL['h_scale']][None,:]/d[:,None]
    assert np.array_equal(item['matrix'],expect)


def test_ambiguous_rank_and_multidimensional_nullspace_abstain():
    for A,b in [(np.diag([5e-11,1.]),np.zeros(2)),(np.zeros((2,2)),np.zeros(2))]:
        budget,_=make_budget();g=core.geometry(dict(A=A,b=b),np.ones(2),np.array([1.,0.,0.]),budget,'ambiguous')
        assert g['raw_B']['null_tangent']['tangent'] is None
        assert g['scaled_B']['null_tangent']['tangent'] is None
    # Two simultaneously soft directions must not receive a simple-direction coupling label.
    summary=core.rank_summary(np.array([1.,1e-11,1e-12]))
    assert not summary['isolated_soft_direction']


def test_zero_residual_and_corrupted_step():
    A=np.eye(2);R=np.array([.01,.02]);theta=np.ones(2)
    good=core.saved_linear_check(A,R,theta,-R,R)
    bad=core.saved_linear_check(A,R,theta,R,R)
    assert good['verdict']=='linear_equation_resolved'
    assert bad['verdict']=='material_linear_equation_mismatch'
    zero=core.saved_linear_check(A,np.zeros(2),theta,np.zeros(2),np.zeros(2))
    assert zero['verdict']=='abstain_nearzero_residual' and zero['true_relative_forcing'] is None
    limited=core.saved_linear_check(A,R,theta,-R,R+1e-5)
    assert limited['verdict']=='arithmetic_resolution_limited'


def test_phase_exchange_secants_and_orientation_abstention():
    p=np.array([[1.,2.],[-1.,-2.]]);c=np.array([[-1.1,-2.1],[1.1,2.1]])
    s=core.carried_secants(p,c,1.,.9,1)
    assert np.array_equal(s[('parent',0)],s[('child',1)])
    assert np.allclose(s[('parent',0)],[.1,.1,-.1])
    budget,_=make_budget();g=core.geometry(dict(A=np.diag([0.,1.]),b=np.array([-1.,0.])),np.ones(2),np.array([0.,1.,0.]),budget,'orthogonal')
    assert g['raw_B']['null_tangent']['status']=='abstain_ambiguous_orientation'


def test_saved_thirteen_trials_and_corrupted_state():
    it=dict(call=473,iteration=3,state=np.array([1.,2.]),residual=np.array([.1,.2]),step=np.array([.01,-.01]))
    trials=[dict(call=473,iteration=3,halving=h,state=it['state']+2.**-h*it['step'],residual=it['residual']*2,accepted=False) for h in range(13)]
    rows=core.saved_trials(it,trials);assert len(rows)==13 and all(r['ratio']==2 for r in rows)
    trials[7]['state'][0]+=1e-8
    with pytest.raises(ValueError,match='step bytes'):core.saved_trials(it,trials)


def test_caps_interrupt_and_nonfinite_raw_retention(tmp_path,monkeypatch):
    budget,events=make_budget();budget.attempted['svds']=24
    with pytest.raises(ValueError,match='operation cap'):core.decompose(np.eye(2),budget,'extra')
    assert budget.attempted['svds']==24
    store=Store(tmp_path/'partial',7*1024**2);budget=core.Budget(store.emit)
    def bad(t):return float('nan'),np.array([np.nan,0.]),np.eye(2)
    with pytest.raises(ValueError,match='finite float64'):core.phase(np.ones(2),np.ones(2),1,bad,budget,'bad')
    lines=[json.loads(s) for s in (store.root/'journal.jsonl').read_text().splitlines()]
    kind,payload=load_record(store.root,lines[-1]['file'])
    assert kind=='bad_analytic_return' and np.isnan(payload['gradient'][0])
    monkeypatch.setattr(core,'STOP',False)
    def interrupt(t):
        core.request_stop(None,None);return 1.,np.ones(2),np.eye(2)
    core.phase(np.ones(2),np.ones(2),1,interrupt,budget,'returned')
    assert budget.returned['maps']==1
    with pytest.raises(InterruptedError):budget.check()
    monkeypatch.setattr(core,'STOP',False)


def test_svd_result_saved_before_validation(monkeypatch):
    budget,events=make_budget()
    monkeypatch.setattr(np.linalg,'svd',lambda *a,**k:(np.eye(2),np.array([1.,np.nan]),np.eye(3)))
    with pytest.raises(ValueError,match='finite float64'):core.decompose(np.zeros((2,3)),budget,'invalid')
    assert events[-1][0]=='invalid_svd' and np.isnan(events[-1][1]['singular_values'][1])


def test_source_guard_before_output(tmp_path,monkeypatch):
    driver=module('diagnose_s1r_lower_gap')
    monkeypatch.setattr(driver,'sources',lambda:(_ for _ in ()).throw(ValueError('dirty')))
    with pytest.raises(ValueError,match='dirty'):driver.run(tmp_path/'out','a'*40)
    assert not (tmp_path/'out').exists()


def test_corrupted_saved_jacobian_rejects_qualification():
    driver=module('diagnose_s1r_lower_gap')
    pair=np.array([[1.,0.],[-1.,0.]])
    slot=dict(state=pair[0],native_phase=0,node=dict(pair=pair,spectra={'phases':[{'matrix':np.eye(2)}]}))
    assembly=dict(map1=pair[1],map2=pair[0],M=np.eye(2))
    assert driver.agreement(assembly,slot)['qualified_endpoint']
    assembly['M'][0,0]+=.01
    assert not driver.agreement(assembly,slot)['qualified_endpoint']


def test_launcher_source_contract_and_inherited_pipe(monkeypatch,tmp_path):
    launcher=module('launch_s1r_lower_gap')
    command=launcher.command(tmp_path/'out','a'*40)
    assert command[3]=='20s' and launcher.SUPERVISOR_SECONDS==25 and launcher.STORE_BYTES==7*1024**2
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    monkeypatch.setattr(launcher,'committed_sources',lambda:{'synthetic':'bound'})
    monkeypatch.setattr(launcher,'command',lambda out,review:[sys.executable,'-c','import os,time;pid=os.fork();time.sleep(5) if pid==0 else os._exit(0)'])
    monkeypatch.setattr(launcher.os,'getpriority',lambda *a:10)
    monkeypatch.setattr(launcher,'SUPERVISOR_SECONDS',.2)
    receipt=launcher.launch(tmp_path/'out',tmp_path/'launcher',head,head)
    assert receipt['failure']=='supervisor_timeout_or_inherited_pipe' and not receipt['successful']
    assert (tmp_path/'launcher'/'receipt.json').exists()


def test_six_slot_synthetic_pipeline_caps_storage_and_scope(tmp_path,monkeypatch):
    driver=module('diagnose_s1r_lower_gap')
    monkeypatch.setattr(driver,'sources',lambda:{'synthetic':'bound'})
    monkeypatch.setattr(driver.os,'getpriority',lambda *a:10)
    n=72;theta=np.arange(1,n+1,dtype=float)/100;pair=np.stack([theta,-theta])
    epsilon=.01;matrix=np.diag((-1+2*epsilon*theta**2)**2)
    node=dict(pair=pair,spectra={'phases':[{'matrix':matrix},{'matrix':matrix}]})
    slots=[]
    for prefix in ('predecessor','last'):
        for phase in (0,1):
            slots.append(dict(role=prefix+'_native_phase_'+str(phase),state=pair[phase],params=np.array([0.,0.]),
                rates=np.ones(2),native_phase=phase,node=node,secant=np.r_[np.ones(n),-.0005]))
    for role in ('nearest_failure_seed','nearest_failure_state'):
        slots.append(dict(role=role,state=theta,params=np.array([0.,0.]),rates=np.ones(2),native_phase=None,node=node,secant=np.r_[np.ones(n),-.0005]))
    it=dict(call=473,iteration=3,state=theta,residual=np.zeros(n),step=np.ones(n)*.01)
    trials=[dict(call=473,iteration=3,halving=h,state=theta+2.**-h*it['step'],residual=np.ones(n)*.1,accepted=False) for h in range(13)]
    data=dict(plan=json.loads((ROOT/driver.PLAN).read_text()),identity={},model={},x=np.ones((72,8)),y=np.ones(72),slots=slots,iteration=it,trials=trials,scale=np.ones(n),phase_binding={})
    monkeypatch.setattr(driver,'load_inputs',lambda:data)
    def polynomial(t,*args):
        loss=float(t@t-epsilon*np.sum(t**4)/4+epsilon*np.sum(theta**2*t**2)/2)
        return loss,2*t-epsilon*t*(t*t-theta*theta),np.diag(2-epsilon*(3*t*t-theta*theta))
    monkeypatch.setattr(driver,'loss_gradient_hessian',polynomial)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result=driver.run(tmp_path/'synthetic',head)
    assert result['status']=='diagnostic_complete' and len(result['completed_roles'])==6
    assert result['operation_attempted']==result['operation_returned']==dict(maps=12,jacobians=12,svds=24)
    assert not result['accepted_orbit'] and not result['applied_correction'] and not result['fold_verified']
    assert result['saved_linear_equation']['verdict']=='abstain_nearzero_residual'
    assert sum(p.stat().st_size for p in (tmp_path/'synthetic').iterdir())<7*1024**2
    events=[json.loads(line) for line in (tmp_path/'synthetic'/'journal.jsonl').read_text().splitlines()]
    records=[load_record(tmp_path/'synthetic',e['file']) for e in events]
    summaries=[p for k,p in records if k=='nearest_failure_state_complete']
    assert summaries[0]['tangent_scope']=='local residual level-set tangent at nonclosed state'
    B=[p for k,p in records if k=='last_native_phase_0_raw_B_svd'][0]
    assert B['Vh'].shape==(73,73)


def test_assembled_nonfinite_matrices_retained_before_rejection(monkeypatch):
    budget,events=make_budget()
    monkeypatch.setattr(core,'phase',lambda theta,*a:dict(image=theta.copy(),jacobian=np.eye(2)*1e200,parameter=np.zeros(2)))
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises(ValueError,match='finite float64'):
            core.evaluate_state(np.ones(2),np.ones(2),1,None,budget,'overflow')
    assert events[-1][0]=='overflow_assembly'
    assert not np.isfinite(events[-1][1]['M']).all()
