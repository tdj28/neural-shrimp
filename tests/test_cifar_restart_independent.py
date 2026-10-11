"""Independent source controls. Small synthetic operators/models only; no target data."""
import time
import numpy as np
import pytest
from neural_shrimp import cifar_linear_diagnostic as core
from neural_shrimp.cifar_restart_diagnostic import Budget,CountedProblemMixin,Prefix,Store


def test_real_scipy_one_cycle_counts_and_first_fifty_match():
    # Size exceeds restart200; 6 decades prevents this diagonal fixture trivially converging.
    diagonal=np.geomspace(1e-3,1e3,220);residual=np.ones(220)
    action=lambda v:diagonal*v
    old=core.solve_direction(action,residual,restart=50,cycles=4,max_matvec=205)
    events=[];control=Prefix(old[3]['estimated_relative_residual_history'],events.append)
    step,ap,e,new=core.solve_direction(action,residual,restart=200,cycles=1,max_matvec=202,explicit_zero_start=True,event=control)
    control.complete()
    assert old[3]['inner_iterations']==new['inner_iterations']==200
    assert old[3]['matvec_calls']==205 and new['matvec_calls']==202 and new['cycles']==1
    np.testing.assert_allclose(new['estimated_relative_residual_history'][:50],old[3]['estimated_relative_residual_history'][:50],rtol=1e-10,atol=1e-12)
    np.testing.assert_array_equal(ap,diagonal*step);np.testing.assert_array_equal(e,ap+residual)
    assert new['true_forcing_ratio']==pytest.approx(np.linalg.norm(e)/np.linalg.norm(residual),rel=1e-14)


def test_explicit_zero_guess_and_false_strict_convergence_stay_separate(monkeypatch):
    observed={}
    def fake(operator,b,**kwargs):
        observed.update(kwargs);kwargs['callback'](0.)
        return .9*b,0
    monkeypatch.setattr(core,'gmres',fake)
    r=np.array([1.,2.]);theta=np.zeros(2)
    step,ap,e,linear=core.solve_direction(lambda v:v,r,explicit_zero_start=True)
    np.testing.assert_array_equal(observed['x0'],np.zeros(2));assert observed['callback_type']=='pr_norm'
    assert linear['gmres_reported_converged'] and not linear['requested_tolerance_met']
    assert linear['forcing_test_pass'] and linear['descent_direction']
    saved=[];rows=core.evaluate_trials(theta,r,step,linear,lambda x,i:dict(residual=x+r,relative_separation=.1),save_state=lambda i,v:saved.append(v.copy()))
    assert rows[0]['inexact_step_candidate'] and all(not row['accepted_orbit'] and not row['applied_to_continuation'] for row in rows)
    np.testing.assert_array_equal(theta,np.zeros(2))
    for alpha,v in zip((1.,.5,.25,.125),saved):np.testing.assert_array_equal(v,alpha*step)
    with pytest.raises(ValueError):core.solve_direction(lambda v:v,r,explicit_zero_start=1)


def test_prefix_failing_callback_is_retained_and_short_prefix_rejected():
    baseline=np.linspace(.95,.85,200);events=[];prefix=Prefix(baseline,events.append)
    for i in range(49):prefix(dict(kind='gmres_inner',iteration=i+1,estimated_relative_residual=baseline[i]))
    with pytest.raises(ValueError):prefix.complete()
    with pytest.raises(ValueError):prefix(dict(kind='gmres_inner',iteration=50,estimated_relative_residual=baseline[49]+1e-6))
    assert len(prefix.values)==50 and events[-1]['iteration']==50


def test_direct_grad_counter_excludes_hvp_internal_grad_work():
    import torch
    from neural_shrimp.cifar import MLP,Problem
    class Counted(CountedProblemMixin,Problem):pass
    x=np.array([[.1,.2],[-.3,.4],[.7,-.5]]);y=np.array([[.1,.3],[-.2,.4],[.5,-.1]])
    problem=Counted(MLP(2,2,2),x,y,.01);problem.budget=Budget(lambda row:None)
    theta=torch.from_numpy(np.linspace(-.2,.3,12));vector=torch.from_numpy(np.linspace(.4,-.1,12))
    problem.grad(theta);one=problem.update(theta,.1);problem.update(one,.1)
    problem.hvp(theta,vector);problem.jvp_two_step(theta,vector,.1,theta_next=one)
    problem.budget.call('action',lambda:problem.jvp_two_step(theta,vector,.1,theta_next=one)-vector)
    expected={'direct_gradient':3,'hvp':5,'j2vp':2,'action':1}
    assert problem.budget.attempted==problem.budget.returned==expected


def test_failed_hvp_restores_depth_and_attempt_return_distinction():
    class Base:
        def grad(self,t):return t+1
        def hvp(self,t,v):self.grad(t);raise RuntimeError('synthetic failure')
    class Counted(CountedProblemMixin,Base):pass
    problem=Counted();events=[];problem.budget=Budget(events.append)
    with pytest.raises(RuntimeError):problem.hvp(np.ones(2),np.ones(2))
    assert problem._hvp_depth==0 and problem.budget.attempted['hvp']==1 and problem.budget.returned['hvp']==0
    assert problem.budget.attempted['direct_gradient']==0
    problem.grad(np.ones(2));assert problem.budget.attempted['direct_gradient']==problem.budget.returned['direct_gradient']==1


def test_returned_direction_survives_final_action_failure():
    saved=[];calls=[0]
    def action(v):
        calls[0]+=1
        if calls[0]==3:raise RuntimeError('independent final action failure')
        return np.array([1.,2.])*v
    with pytest.raises(RuntimeError):core.solve_direction(action,np.ones(2),restart=1,cycles=1,max_matvec=3,explicit_zero_start=True,save_direction=lambda p,info:saved.append((p.copy(),info)))
    assert len(saved)==1 and np.isfinite(saved[0][0]).all()

def test_stop_after_operation_return_preserves_return_before_next_boundary(monkeypatch):
    from neural_shrimp import cifar_restart_diagnostic as fixed
    monkeypatch.setattr(fixed, 'STOP_REQUESTED', False)
    events=[]; budget=fixed.Budget(events.append)
    def returned_operation():
        fixed.request_stop(None, None)
        return np.array([3., 4.])
    value=budget.call('action', returned_operation)
    np.testing.assert_array_equal(value, [3., 4.])
    assert budget.attempted['action']==budget.returned['action']==1
    assert events[-1]['kind']=='operation_returned'
    with pytest.raises(InterruptedError): budget.check()
    with pytest.raises(InterruptedError): budget.call('action', lambda: pytest.fail('sticky stop ignored'))
    assert budget.attempted['action']==1


def test_byte_exact_gate_distinguishes_signed_zero_dtype_and_shape():
    from neural_shrimp.cifar_restart_diagnostic import byte_equal
    x=np.array([0., 1.], dtype=np.float64)
    assert byte_equal(x, x.copy())
    assert not byte_equal(x, np.array([-0., 1.]))
    assert not byte_equal(x, x.astype(np.float32))
    assert not byte_equal(x, x.reshape(1,2))
