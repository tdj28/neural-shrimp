import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from baby_shrimp_memory_total import (nested_hessian_rows, normalized_eta_step,
                                      finite_difference_hessian_rows, observed_orders)


def test_nested_forward_ad_matches_independent_analytic_hessian():
    import torch
    point = torch.tensor([[[.3, -.4]]], dtype=torch.float64)
    eta = torch.tensor([[[.7, .2]]], dtype=torch.float64)
    directions = torch.tensor([[[.4, -.2]], [[-.3, .8]], [[.9, .7]]], dtype=torch.float64)

    def f(x):
        return torch.stack([x[..., 0]**3+x[..., 0]*x[..., 1],
                            torch.sin(x[..., 1])+x[..., 0]**2], dim=-1)

    def jvp_rows(x, v):
        return torch.func.jvp(f, (x.repeat(len(v), 1, 1),), (v,))[1]

    actual = nested_hessian_rows(jvp_rows, point, eta, directions)
    v0, v1 = directions[..., 0], directions[..., 1]
    expected = torch.stack([6*.3*.7*v0+.7*v1+.2*v0,
                            2*.7*v0-torch.sin(point[..., 1])*.2*v1], dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
    errors = []
    for relative in (1e-2, 3e-3, 1e-3):
        step = normalized_eta_step(point.numpy(), eta.numpy(), relative)
        fd = finite_difference_hessian_rows(jvp_rows, point, eta, directions, step)
        errors.append(float((fd-actual).norm()/actual.norm()))
    assert all(1.9 < order < 2.1 for order in observed_orders(errors, (1e-2, 3e-3, 1e-3)))


def test_eta_step_has_declared_relative_state_displacement():
    state = np.array([3., 4.])
    eta = np.array([0., 2.5])
    step = normalized_eta_step(state, eta, 1e-3)
    assert step == pytest.approx(.002)
    assert np.linalg.norm(step*eta)/np.linalg.norm(state) == pytest.approx(1e-3)
    assert normalized_eta_step(state, np.zeros(2), 1e-3) is None
    with pytest.raises(ValueError):
        normalized_eta_step(np.zeros(2), eta, 1e-3)
    with pytest.raises(ValueError):
        normalized_eta_step(state, eta, -1e-3)


def test_unsupported_nested_derivative_fails_without_fallback():
    import torch
    point = torch.ones((1, 1, 2), dtype=torch.float64)
    def unsupported(x, v):
        raise NotImplementedError("second forward derivative unavailable")
    with pytest.raises(NotImplementedError, match="unavailable"):
        nested_hessian_rows(unsupported, point, point, point)


def test_observed_orders_do_not_invent_a_zero_error_convergence_rate():
    assert observed_orders([0., 0., 0.]) == [None, None]
    with pytest.raises(ValueError):
        observed_orders([1., np.nan, .01])
