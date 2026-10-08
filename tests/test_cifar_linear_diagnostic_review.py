"""Independent synthetic review controls; never load CIFAR target artifacts."""
import numpy as np
from neural_shrimp.cifar_linear_diagnostic import solve_direction,evaluate_trials


def test_existing_full_restart_budget_has_no_hidden_fifth_cycle():
    # Ill-conditioned synthetic diagonal operator exercises all four cycles.
    diagonal=np.geomspace(1.,1e8,80)
    calls=[]
    def action(v):
        calls.append(1)
        return diagonal*v
    r=np.ones(80)
    p,applied,e,result=solve_direction(action,r)
    assert result['inner_iterations']==200
    assert result['gmres_info']==4
    assert result['matvec_calls']==len(calls)==205
    np.testing.assert_allclose(e,diagonal*p+r,rtol=0,atol=1e-15)
    assert result['true_forcing_ratio']==np.linalg.norm(e)/np.linalg.norm(r)
    assert result['merit_directional_derivative']==r@applied


def test_forcing_implies_descent_but_actual_armijo_is_still_required():
    a=np.diag([1.,2.]);r=np.array([-1.,-1.])
    p,applied,e,result=solve_direction(lambda v:a@v,r,restart=1,cycles=1,max_matvec=3)
    q=np.linalg.norm(e)/np.linalg.norm(r)
    assert q<=.5 and r@applied<=-(1-q)*(r@r)+1e-14
    rows=evaluate_trials(np.zeros(2),r,p,result,
        lambda theta,index:dict(residual=2*r,relative_separation=.1))
    assert all(not row['armijo_pass'] and not row['inexact_step_candidate'] for row in rows)
    assert all(not row['accepted_orbit'] and not row['applied_to_continuation'] for row in rows)
