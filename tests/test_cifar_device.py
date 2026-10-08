"""Tiny dense oracles for the opt-in device bridge; no scientific data runs."""
import numpy as np
import pytest
import torch
from neural_shrimp.cifar import MLP, Problem, as_tensor
from neural_shrimp.cifar_device import DeviceProblem, HostTwoStepOperator, parity_metrics


def test_cpu_device_derivatives_and_complex_bridge_match_dense_oracle():
    rng = np.random.default_rng(71)
    model = MLP(d=2, width=2, k=2)
    x, y = rng.normal(size=(12, 2)), rng.normal(size=(12, 2))
    reference = Problem(model, x, y, .03)
    problem = DeviceProblem(model, x, y, .03)
    theta, vector = as_tensor(model.init(28207)), as_tensor(rng.normal(size=model.n_parameters))
    dense_hessian = torch.func.hessian(reference.loss)(theta)
    dense_j2 = torch.func.jacfwd(lambda t: reference.update(reference.update(t, .2), .2))(theta).numpy()
    torch.testing.assert_close(problem.grad(theta), reference.grad(theta), rtol=0, atol=0)
    torch.testing.assert_close(problem.hvp(theta, vector), dense_hessian @ vector, rtol=1e-12, atol=1e-13)
    bridge = HostTwoStepOperator(problem, theta, .2)
    np.testing.assert_allclose(bridge.matvec(vector.numpy()), dense_j2 @ vector.numpy(), rtol=1e-12, atol=1e-13)
    complex_vector = vector.numpy() + 1j*rng.normal(size=model.n_parameters)
    np.testing.assert_allclose(bridge.matvec(complex_vector), dense_j2 @ complex_vector, rtol=1e-12, atol=1e-13)
    assert len(bridge.measurements) == 3
    assert all(m['input_bytes'] == vector.numel()*8 and m['total_seconds'] >= m['device_j2vp_seconds'] >= 0
               for m in bridge.measurements)
    newton = HostTwoStepOperator(problem, theta, .2, subtract_identity=True)
    np.testing.assert_allclose(newton.matvec(vector.numpy()), (dense_j2-np.eye(theta.numel())) @ vector.numpy(),
                               rtol=1e-12, atol=1e-13)
    assert problem.x.device.type == 'cpu' and problem.x.dtype == torch.float64


def test_parity_is_coordinatewise_and_fails_nonfinite():
    expected = np.array([0., 1., 1e6])
    assert parity_metrics(expected+1e-12, expected)['passed']
    assert not parity_metrics(expected+1e-6, expected)['passed']
    assert not parity_metrics(np.array([np.nan, 1., 1e6]), expected)['passed']
    with pytest.raises(ValueError):
        parity_metrics(expected[:1], expected)
