import numpy as np
import pytest
import torch
from neural_shrimp.tiny_lm import TinyTransformer, LMProblem, lanczos_extremes, ar_diagnostics


def small():
    model = TinyTransformer(vocabulary=5, context=3, width=2, expansion=2)
    theta = model.initialize(dtype=torch.float64)
    tokens = torch.tensor([[0, 1, 2], [2, 3, 4]])
    targets = torch.tensor([[1, 2, 3], [3, 4, 0]])
    return model, LMProblem(model, tokens, targets), theta


def test_causal_mask_and_tied_parameter_accounting():
    model, problem, theta = small()
    changed = problem.tokens.clone()
    changed[:, -1] = 0
    torch.testing.assert_close(model.forward(theta, changed)[:, :-1], model.forward(theta, problem.tokens)[:, :-1])
    assert list(model.shapes).count('embedding') == 1
    assert not any('output' in name for name in model.shapes)
    assert sum(x.numel() for x in model.unflatten(theta).values()) == model.n_parameters


def test_matrix_free_update_jvp_against_dense_and_finite_difference():
    _, problem, theta = small()
    vector = torch.arange(1, len(theta)+1, dtype=theta.dtype)
    vector /= vector.norm()
    eta = .03
    dense = torch.func.jacrev(lambda p: problem.update(p, eta))(theta)
    actual = problem.update_jvp(theta, vector, eta)
    torch.testing.assert_close(actual, dense@vector, rtol=1e-9, atol=1e-10)
    for h in (1e-5, 5e-6):
        fd = (problem.update(theta+h*vector, eta)-problem.update(theta-h*vector, eta))/(2*h)
        torch.testing.assert_close(actual, fd, rtol=2e-4, atol=2e-5)
    second_dense = torch.func.jacrev(lambda p: problem.update(problem.update(p, eta), eta))(theta)
    torch.testing.assert_close(problem.two_step_jvp(theta, vector, eta), second_dense@vector, rtol=1e-8, atol=1e-9)


def test_lanczos_reports_correct_dense_symmetric_extremes():
    matrix = torch.tensor([[3., .2, 0], [.2, -2., .1], [0, .1, 1.]], dtype=torch.float64)
    result = lanczos_extremes(lambda v: matrix@v, torch.tensor([1., 2., 3.], dtype=torch.float64), steps=3)
    expected = torch.linalg.eigvalsh(matrix)
    assert abs(result['maximum']['value']-float(expected[-1])) < 1e-12
    assert abs(result['minimum']['value']-float(expected[0])) < 1e-12
    assert result['maximum']['absolute_residual'] < 1e-12


def test_reserved_seeds_rejected_and_ar_controls():
    with pytest.raises(ValueError, match='reserved'):
        TinyTransformer(5).initialize(seed=29000)
    assert ar_diagnostics(np.ones(100))['outcome'] == 'nearly_constant'
    out = ar_diagnostics(np.sin(.3*np.arange(200)))
    assert out['ar2']['normalized_test_mse'] < out['ar1']['normalized_test_mse']
