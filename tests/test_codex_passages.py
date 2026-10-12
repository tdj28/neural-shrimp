"""Synthetic passage counts and consumed-reference comparison, no model load."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "codex/crosscheck/passages.py"
SPEC = importlib.util.spec_from_file_location("independent_passage_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QuadraticGroupMap:
    def __init__(self, constant_increment=None):
        self.prefixes, self.calls = [], []
        self.constant_increment = constant_increment

    def prefix_cache(self, alpha):
        self.prefixes.append(alpha)
        return alpha

    def period_map(self, state, alpha, log_temperature, period, cache=None):
        assert state.shape == (3, 2) and state.dtype == torch.float64
        assert period == 2 and cache == alpha
        self.calls.append(alpha)
        result = state.clone()
        drift = self.constant_increment if self.constant_increment is not None else (alpha - .5) + 2 * state[:, 0] ** 2
        result[:, 0] = state[:, 0] + drift
        result[:, 1] = .5 * state[:, 1]
        return result


def run(model, **kwargs):
    return MODULE.run_controlled_passages(model, np.zeros(2), np.array([-1., 0.]),
                                          np.array([-1., 0.]), .25, .5, -.115,
                                          -1., -2., **kwargs)


def test_all_eighteen_rows_match_quadratic_prediction_and_group_callbacks():
    model = QuadraticGroupMap()
    saved = []
    eps = (.005, .01, .02, .03, .04, .05)
    result = run(model, epsilons=eps, on_group=lambda i, value: saved.append((i, value["G_evaluations"])))
    assert len(result["rows"]) == 18 and result["all_exits_observed"]
    assert len(model.prefixes) == 6 and len(saved) == 6
    assert result["batch_model_steps"] == 2 * len(model.calls)
    assert result["single_state_equivalent_steps"] == 6 * len(model.calls)
    assert [row["eps"] for row in result["rows"]] == [value for value in eps for _ in range(3)]
    for column, row in enumerate(result["rows"]):
        assert row["measured_G"] == row["predicted_discrete_G"]
        count = row["measured_G"]
        assert np.all(result["u"][:count - 1, column] <= .25)
        assert result["u"][count - 1, column] > .25
        assert np.isnan(result["u"][count:, column]).all()
    assert result["fixed_inputs"]["orientation"] == -1
    expected = np.random.default_rng(0).normal(size=2)
    expected[0] = 0
    expected /= np.linalg.norm(expected)
    np.testing.assert_array_equal(result["fixed_inputs"]["transverse"], expected)


def test_strict_exit_not_nonstrict_and_reference_comparison_ignores_late_tails():
    original = np.array([[-.25 + g * .125, g * .125, -.25 + g * .125] for g in range(1, 6)])
    rows = [dict(eps=.01, entrance=kind, measured_G=count) for kind, count in zip(MODULE.ENTRANCES, [5, 3, 5], strict=True)]
    original[3:, 1] = [1e9, -1e9]
    result = run(QuadraticGroupMap(constant_increment=.125), epsilons=(.01,), reference_u=original, reference_rows=rows)
    assert [row["measured_G"] for row in result["rows"]] == [5, 3, 5]
    for row, n in zip(result["reference_comparison"], [4, 2, 4], strict=True):
        assert row["count_difference"] == 0 and row["shared_pre_exit_G"] == n
        assert row["maximum_absolute_u_error"] == 0
    old = original.copy()
    old[0, 0] += 1e-6
    compared = MODULE.compare_reference(result["u"], result["rows"], old, .25, rows)
    assert compared[0]["maximum_absolute_u_error"] == pytest.approx(1e-6)


def test_censoring_never_substitutes_iteration_cap_and_bad_reference_fails():
    result = run(QuadraticGroupMap(constant_increment=.125), epsilons=(.01,), max_g=2)
    assert not result["all_exits_observed"]
    assert all(row["measured_G"] is None and row["censored"] for row in result["rows"])
    assert all(row["recorded_G"] == 2 for row in result["rows"])
    bad = [dict(eps=.01, entrance=kind, measured_G=1) for kind in MODULE.ENTRANCES]
    with pytest.raises(ValueError, match="first-strict-exit"):
        MODULE.compare_reference(result["u"], result["rows"], np.zeros((2, 3)), .25, bad)


def test_entrances_preserve_saved_scale_and_input_arrays():
    xc, r, l = np.array([.2, -.7]), np.array([1., 0.]), np.array([1., .3])
    saved = [value.copy() for value in [xc, r, l]]
    result = MODULE.fixed_entrances(xc, r, l, .03, 2., 3.)
    for value, original in zip([xc, r, l], saved, strict=True):
        np.testing.assert_array_equal(value, original)
    assert abs(result["left"] @ result["transverse"]) < 1e-14
    np.testing.assert_allclose((result["states"] - xc) @ l, [-.03, 0, -.03], rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="already satisfy"):
        MODULE.fixed_entrances(xc, 2*r, l, .03, 2., 3.)
    with pytest.raises(ValueError, match="same-sign"):
        MODULE.fixed_entrances(xc, r, l, .03, -2., 3.)
