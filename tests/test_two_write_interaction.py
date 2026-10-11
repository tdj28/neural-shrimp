"""Independent analytic controls only; no neural evidence is loaded."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import two_write_interaction as interaction


def a(values):
    return np.array(values, dtype=np.float64)


def test_factorial_sign_order_and_reconstruction():
    means = a([[10, 2], [6, -4], [2, 6], [-2, 4]])
    c = interaction.factorial_components(means)
    np.testing.assert_array_equal(c["m"], [4, 2])
    np.testing.assert_array_equal(c["A"], [4, -3])
    np.testing.assert_array_equal(c["B"], [2, 2])
    np.testing.assert_array_equal(c["C"], [0, 1])
    np.testing.assert_array_equal(interaction.reference_means(**c), means)


def test_additive_negative_control_despite_threshold_xor_accuracy():
    # Explicit XOR counterexample: an affine threshold can reach 75%, while
    # optimal affine population regression still has R-squared zero.
    means = a([[2], [0], [0], [-2]])
    truth = a([1, -1, -1, 1])
    assert np.mean(np.where(means[:, 0] > 1, 1, -1) == truth) == 0.75
    result = interaction.reference_observer(a([1]), a([1]), a([0]), 0.1)
    np.testing.assert_array_equal(result["w"], [0])
    assert interaction.parity_risk(result["w"], a([0]), means, 0.1)["r_squared"] == 0
    # Any nonzero linear weights introduce error rather than exploiting parity.
    assert interaction.parity_risk(a([0.3]), a([0]), means, 0.1)["r_squared"] < 0


def test_orthogonal_injected_parity_has_known_recoverability_and_ablation():
    A, B, C, m = a([2, 0, 0]), a([0, 3, 0]), a([0, 0, 0.4]), a([1, -2, 5])
    sigma = 0.3
    out = interaction.reference_observer(A, B, C, sigma)
    np.testing.assert_allclose(out["w"], [0, 0, 1.6], atol=1e-14)
    means = interaction.reference_means(m, A, B, C)
    score = interaction.parity_risk(out["w"], m, means, sigma)
    assert score["r_squared"] == pytest.approx(0.16 / 0.25)
    zero = interaction.reference_observer(A, B, C * 0, sigma)["w"]
    assert interaction.parity_risk(zero, m, means, sigma)["r_squared"] == 0


def test_svd_weights_match_independent_dense_normal_equation():
    rng = np.random.default_rng(4925)
    A, B, C, m = rng.normal(size=(4, 9))
    sigma = 0.4
    covariance = sum(np.outer(v, v) for v in (A, B, C)) + sigma**2 * np.eye(9)
    expected = np.linalg.solve(covariance, C)
    out = interaction.reference_observer(A, B, C, sigma)
    np.testing.assert_allclose(out["w"], expected, atol=2e-14, rtol=2e-13)
    assert np.linalg.norm(out["scaled_equation_residual"]) < 1e-14
    score = interaction.parity_risk(out["w"], m, interaction.reference_means(m, A, B, C), sigma)
    assert score["r_squared"] == pytest.approx(float(C @ expected), abs=2e-14)


def test_risk_preserves_wrong_center_penalty_and_negative_scores():
    means = a([[3], [1], [1], [3]])
    # Predictor exactly knows noiseless parity at center 2. Noise adds 0.25.
    assert interaction.parity_risk(a([1]), a([2]), means, 0.5)["r_squared"] == 0.75
    # Keeping an incorrect reference center 0 costs four more units of risk.
    assert interaction.parity_risk(a([1]), a([0]), means, 0.5)["r_squared"] == -3.25


def test_rank_deficient_templates_scalar_closed_form():
    A, B, C = a([2]), a([3]), a([4])
    w = interaction.reference_observer(A, B, C, 0.5)["w"]
    np.testing.assert_allclose(w, [4 / 29.25])


def test_secondary_nonlinear_observation_can_create_cross_term():
    # Additive state has no interaction; a static quadratic observation has 2AB.
    state = interaction.reference_means(a([0.7]), a([0.2]), a([0.3]), a([0]))
    assert interaction.factorial_components(state)["C"][0] == pytest.approx(0, abs=1e-16)
    assert interaction.factorial_components(state**2)["C"][0] == pytest.approx(0.12)
    assert abs(interaction.factorial_components(np.tanh(state))["C"][0]) > 0.01


@pytest.mark.parametrize("scale", [1e-150, 1e-4, 1e4, 1e150])
def test_common_coordinate_unit_change_preserves_risk(scale):
    A, B, C, m = a([1, 2]), a([3, 1]), a([0.4, -0.7]), a([1, -4])
    sigma = 0.3
    w = interaction.reference_observer(A, B, C, sigma)["w"]
    baseline = interaction.parity_risk(w, m, interaction.reference_means(m, A, B, C), sigma)
    scaled = interaction.reference_observer(A * scale, B * scale, C * scale, sigma * scale)["w"]
    means = interaction.reference_means(m * scale, A * scale, B * scale, C * scale)
    actual = interaction.parity_risk(scaled, m * scale, means, sigma * scale)
    assert actual["r_squared"] == pytest.approx(baseline["r_squared"], abs=1e-13)


@pytest.mark.parametrize("sigma", [0, -1, True, np.bool_(True), np.nan, np.inf, "0.1"])
def test_invalid_noise_rejected(sigma):
    with pytest.raises(ValueError, match="sigma"):
        interaction.reference_observer(a([1]), a([1]), a([1]), sigma)
    with pytest.raises(ValueError, match="sigma"):
        interaction.parity_risk(a([1]), a([0]), a([[1], [2], [3], [4]]), sigma)


@pytest.mark.parametrize("bad", [[1.], np.ones(1, dtype=np.float32), a([np.nan]), a([]), a([[1.]])])
def test_invalid_template_rejected(bad):
    with pytest.raises(ValueError):
        interaction.reference_observer(bad, a([1]), a([1]), 0.1)


def test_incorrect_census_and_unrepresentable_noise_rejected():
    with pytest.raises(ValueError, match="four rows"):
        interaction.factorial_components(np.ones((3, 2)))
    with pytest.raises(ValueError, match="dimensions"):
        interaction.parity_risk(a([1]), a([0]), np.ones((4, 2)), 0.1)
    with pytest.raises(ValueError, match="variance"):
        interaction.reference_observer(a([1]), a([1]), a([1]), 1e-200)


def test_risk_overflow_is_failure_not_infinite_scientific_score():
    with pytest.raises(ValueError, match="representable"):
        interaction.parity_risk(a([1e300]), a([0]), np.ones((4, 1)), 1.0)
