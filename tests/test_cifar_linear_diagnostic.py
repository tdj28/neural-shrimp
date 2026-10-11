import importlib.util
from pathlib import Path
import time

import numpy as np
import pytest

from neural_shrimp import cifar_linear_diagnostic as diagnostic


def test_unsuccessful_gmres_can_supply_only_a_merit_candidate():
    A=np.diag([1.,2.]);R=np.array([-1.,-1.]);theta=np.zeros(2)
    step,applied,true,linear=diagnostic.solve_direction(lambda v:A@v,R,restart=1,cycles=1,max_matvec=3)
    assert linear['gmres_info']==1 and not linear['requested_tolerance_met']
    np.testing.assert_allclose(true,A@step+R,atol=1e-15)
    assert linear['true_forcing_ratio']==pytest.approx(1/np.sqrt(10))
    assert linear['forcing_test_pass'] and linear['descent_direction']
    assert linear['inner_iterations']==1 and linear['matvec_calls']==3
    saved=[]
    rows=diagnostic.evaluate_trials(theta,R,step,linear,
        lambda x,i:dict(residual=A@x+R,relative_separation=.1),
        save_state=lambda i,x:saved.append(x.copy()))
    assert len(rows)==4 and rows[0]['inexact_step_candidate'] and rows[0]['merit_ratio']==pytest.approx(.1)
    assert not any(row['accepted_orbit'] or row['applied_to_continuation'] or row['closure_screen'] for row in rows)
    for alpha,x in zip(diagnostic.ALPHAS,saved):np.testing.assert_equal(x,theta+alpha*step)


def test_true_residual_rejects_spurious_solver_success(monkeypatch):
    def dishonest_solver(operator,b,**kwargs):
        kwargs['callback'](0.)
        return np.zeros_like(b),0
    monkeypatch.setattr(diagnostic,'gmres',dishonest_solver)
    R=np.array([1.,2.]);step,applied,true,linear=diagnostic.solve_direction(lambda v:v,R)
    assert linear['gmres_reported_converged'] and linear['estimated_relative_residual_history']==[0.]
    assert linear['true_forcing_ratio']==1 and not linear['forcing_test_pass'] and not linear['requested_tolerance_met']
    rows=diagnostic.evaluate_trials(np.zeros(2),R,step,linear,lambda x,i:dict(residual=R,relative_separation=.1))
    assert not any(row['inexact_step_candidate'] for row in rows)


def test_nonlinear_trials_backtrack_without_updating_the_base():
    theta=np.array([.5]);R=theta**3-1;A=3*theta[0]**2
    step,_,_,linear=diagnostic.solve_direction(lambda v:A*v,R,restart=1,cycles=1,max_matvec=3)
    rows=diagnostic.evaluate_trials(theta,R,step,linear,lambda x,i:dict(residual=x**3-1,relative_separation=.1))
    assert not rows[0]['armijo_pass'] and rows[1]['armijo_pass'] and rows[1]['inexact_step_candidate']
    assert not any(row['accepted_orbit'] for row in rows)
    np.testing.assert_equal(theta,[.5])


def test_absolute_merit_blocks_normalization_only_improvement():
    theta=np.zeros(1);R=np.array([-1.])
    step,_,_,linear=diagnostic.solve_direction(lambda v:1e-10*v,R,restart=1,cycles=1,max_matvec=3)
    rows=diagnostic.evaluate_trials(theta,R,step,linear,lambda x,i:dict(residual=np.array([1.1]),relative_separation=.1))
    assert rows[0]['closure_screen'] and not rows[0]['merit_decreased']
    assert not any(row['inexact_step_candidate'] or row['accepted_orbit'] for row in rows)


def test_fixed_point_and_nonfinite_trials_are_not_orbits():
    theta=np.zeros(1);R=np.array([-1.]);step,_,_,linear=diagnostic.solve_direction(lambda v:v,R,restart=1,cycles=1,max_matvec=3)
    rows=diagnostic.evaluate_trials(theta,R,step,linear,
        lambda x,i:dict(residual=np.zeros(1) if i==0 else np.array([np.nan]),relative_separation=0.))
    assert rows[0]['closure_screen'] and not rows[0]['primitive_screen'] and not rows[0]['accepted_orbit']
    assert all(not row['finite'] and not row['inexact_step_candidate'] for row in rows[1:])


def test_budget_deadline_and_finite_value_controls(monkeypatch):
    calls=[]
    def action(v):calls.append(1);return np.diag([1.,2.])@v
    with pytest.raises(TimeoutError):diagnostic.solve_direction(action,np.ones(2),deadline=time.monotonic()-1)
    assert not calls
    with pytest.raises(RuntimeError,match='matvec budget'):diagnostic.solve_direction(action,np.ones(2),restart=2,cycles=1,max_matvec=1)
    assert len(calls)==1
    for bad in (np.array([np.inf]),np.array([np.nan]),np.array([1j])):
        with pytest.raises(ValueError):diagnostic.solve_direction(lambda v:v,bad)
    with pytest.raises(ValueError):diagnostic.solve_direction(lambda v:np.full_like(v,np.nan),np.ones(1))
    monkeypatch.setattr(diagnostic,'gmres',lambda operator,b,**kw:(np.full_like(b,np.nan),1))
    with pytest.raises(ValueError):diagnostic.solve_direction(lambda v:v,np.ones(1))


@pytest.mark.parametrize('kwargs',[{'restart':0},{'cycles':True},{'max_matvec':1.5},{'rtol':np.nan},{'forcing_limit':1.},{'forcing_limit':-.1}])
def test_invalid_solver_controls_fail_before_calls(kwargs):
    with pytest.raises(ValueError):diagnostic.solve_direction(lambda v:pytest.fail('action reached'),np.ones(2),**kwargs)


def runner():
    path=Path(__file__).resolve().parents[1]/'tools/diagnose_cifar_linear_solve.py'
    spec=importlib.util.spec_from_file_location('cifar_diagnostic_runner_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_frozen_input_manifest_and_source_fail_closed(tmp_path,monkeypatch):
    module=runner();folder=tmp_path/'input';folder.mkdir();(folder/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest digest'):module.verify_folder(folder,'0'*64)
    monkeypatch.setattr(module,'source_identity',lambda:(_ for _ in ()).throw(ValueError('uncommitted source')))
    output=tmp_path/'output'
    with pytest.raises(ValueError,match='uncommitted'):module.run(output)
    assert not output.exists()
    with pytest.raises(module.Interrupted):module.stop(15,None)


def test_driver_parity_and_retained_byte_budget(tmp_path,monkeypatch):
    module=runner();assert module.parity(np.array([1.]),np.array([1.]))['passed']
    for actual in (np.array([np.nan]),np.array([2.])):
        with pytest.raises(ValueError):module.parity(actual,np.array([1.]))
    monkeypatch.setitem(module.SETTINGS,'new_file_bytes_cap',100)
    with pytest.raises(RuntimeError,match='byte budget'):module.save_array(tmp_path,'oversize',np.ones(20))
    assert not list(tmp_path.iterdir())



def test_returned_direction_saved_before_true_residual_budget_failure():
    saved=[]
    with pytest.raises(RuntimeError,match='matvec budget'):
        diagnostic.solve_direction(lambda v:np.diag([1.,2.])@v,np.array([-1.,-1.]),
            restart=1,cycles=1,max_matvec=2,save_direction=lambda p,info:saved.append((p.copy(),info)))
    assert len(saved)==1 and saved[0][1]==1 and np.isfinite(saved[0][0]).all()



def test_review_identity_is_required_before_output(tmp_path,monkeypatch):
    module=runner();monkeypatch.setattr(module,'source_identity',lambda:{})
    for value in (None,'not-a-commit','0'*39,'g'*40):
        with pytest.raises(ValueError,match='review commit'):module.run(tmp_path/'absent',value)
    assert not (tmp_path/'absent').exists()
