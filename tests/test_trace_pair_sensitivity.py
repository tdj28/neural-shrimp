import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import trace_pair_sensitivity as sensitivity


def leading_trace(matrix):
    values = np.linalg.eigvals(matrix)
    return values[np.argsort(-abs(values))[:2]].real.sum()


def test_nonnormal_pair_sensitivity_matches_resolved_finite_difference():
    change = np.array([[1., 3., -.5], [0., 1., 2.], [0., 0., 1.]])
    matrix = change @ np.diag([.31, -.24, .0002]) @ np.linalg.inv(change)
    derivative = np.array([[.3, .2, -.1], [.2, -.4, .1], [-.1, .2, .5]])
    result = sensitivity.trace_pair_sensitivity(matrix, lambda v: derivative @ v)
    assert result["passed"]
    np.testing.assert_allclose(result["left"].T @ result["right"], np.eye(2), atol=1e-13)
    errors = []
    for step in [1e-3, 3e-4, 1e-4]:
        fd = (leading_trace(matrix + step * derivative) -
              leading_trace(matrix - step * derivative)) / (2 * step)
        errors.append(abs(fd - result["trace_derivative"]))
    assert errors[-1] < errors[0] / 50
    assert errors[-1] < 1e-5


def test_pair_trace_sensitivity_is_similarity_invariant():
    matrix = np.array([[.31, 1.1, .7], [0., -.24, .9], [0., 0., .0002]])
    derivative = np.array([[.2, .1, -.3], [.4, .3, -.1], [.2, -.1, .5]])
    change = np.array([[1., .2, -.3], [.1, 1., .2], [.3, -.2, 1.]])
    inverse = np.linalg.inv(change)
    original = sensitivity.trace_pair_sensitivity(matrix, lambda v: derivative @ v)
    transformed = sensitivity.trace_pair_sensitivity(
        change @ matrix @ inverse, lambda v: change @ derivative @ inverse @ v)
    assert transformed["trace_derivative"] == pytest.approx(original["trace_derivative"], rel=1e-11)


def test_memory_derivative_matches_multistep_product_and_cyclic_spectrum():
    steps = [np.array([[.3, .4], [.05, -.2]]),
             np.array([[.7, -.1], [.2, .4]]),
             np.array([[-.2, .3], [.1, .6]])]
    directions = np.array([[.4, -.2], [1., .3]])

    def product(beta, matrices):
        result = np.eye(2)
        for matrix in matrices:
            result = ((1-beta) * matrix + beta * np.eye(2)) @ result
        return result

    direct_traces = []
    for shift in range(len(steps)):
        ordered = steps[shift:] + steps[:shift]
        callbacks = [lambda v, matrix=matrix: matrix @ v for matrix in ordered]
        mv, explicit = sensitivity.memory_tangent_derivative(callbacks, directions)
        np.testing.assert_allclose(mv, product(0, ordered) @ directions, atol=1e-15)
        step = 1e-5
        fd = ((product(step, ordered) - product(-step, ordered)) @ directions / (2 * step))
        np.testing.assert_allclose(explicit, fd, rtol=1e-8, atol=1e-10)
        _, derivative = sensitivity.memory_tangent_derivative(callbacks, np.eye(2))
        direct_traces.append(np.trace(derivative))
    np.testing.assert_allclose(direct_traces, direct_traces[0], atol=1e-14)


@pytest.mark.parametrize("matrix,options", [
    (np.diag([.2, -.1, .01]), {}),
    (np.array([[0., -.2], [.2, 0.]]), {}),
    (np.diag([.2, .2]), {}),
    (np.array([[.2, 1e5], [0., -.1]]), {"condition_limit": 100.}),
    (np.array([[np.nan, 0.], [0., -.1]]), {}),
])
def test_invalid_or_unqualified_pair_fails_loudly(matrix, options):
    with pytest.raises(ValueError):
        sensitivity.isolated_leading_pair(matrix, **options)


def test_tangent_callback_failure_is_not_a_zero_derivative():
    with pytest.raises(ValueError):
        sensitivity.memory_tangent_derivative([], np.eye(2))
    with pytest.raises(ValueError):
        sensitivity.memory_tangent_derivative([lambda v: np.zeros((2, 2))], np.eye(2))
    with pytest.raises(ValueError):
        sensitivity.trace_pair_sensitivity(np.diag([.2, -.1]), lambda v: np.full_like(v, np.nan))
