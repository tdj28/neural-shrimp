"""Prospective numerical amendment: exact scope and analytic pulse checks."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import refine_two_write_interaction as r


def test_exact_failed_history_group_and_stricter_configs():
    tasks=r.tasks()
    assert len(tasks)==8 and len({t['id'] for t in tasks})==8
    assert {t['case']['id'] for t in tasks}=={'core--p21'}
    assert {t['dose'] for t in tasks}=={2}
    assert {(t['sign1'],t['sign2']) for t in tasks}=={(1,1),(1,-1),(-1,1),(-1,-1)}
    for method,cfg in r.METHODS.items():
        original=r.h.sensitivity.METHODS[method]
        assert cfg['method']==original['method']
        assert cfg['rtol']<original['rtol'] and cfg['atol']<original['atol']
        assert cfg['steps_per_period']==16*original['steps_per_period']


@pytest.mark.parametrize('method',['primary','independent'])
def test_strict_pulse_solver_matches_exact_linear_flow(method,monkeypatch):
    monkeypatch.setitem(r.METHODS[method],'steps_per_period',16)
    lam=np.array([-.2,-.7]);x=np.array([.4,-.3]);q=np.array([1.,2.]);times=np.array([1.,1.5,2.])
    out=r.strict_finite(lambda t,x:lam*x,x,q,1.,times,.1,-.2,method)
    exact=(x+.1*q)*np.exp(times[:,None]*lam)-.2*q*np.exp((times[:,None]-1)*lam)
    np.testing.assert_allclose(out['samples'],exact,atol=3e-13,rtol=3e-13)
    np.testing.assert_allclose(out['event_after'],out['event_before']-.2*q,atol=1e-15)


def test_solver_failure_is_explicit(monkeypatch):
    class Failure:
        success=False
    monkeypatch.setattr(r,'solve_ivp',lambda *a,**k:Failure())
    with pytest.raises(RuntimeError,match='first segment'):
        r.strict_finite(None,np.ones(2),np.ones(2),1.,np.array([1.,2.]),.1,.1,'primary')
