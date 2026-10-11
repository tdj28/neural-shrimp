"""Synthetic-only mathematical tests; no target files, trajectories or fits."""

import builtins
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import two_write_memory as tw


def arr(value):
    return np.array(value, dtype=np.float64)


def cdf(value):
    return 0.5 * math.erfc(-value / math.sqrt(2))


def test_covariance_observer_suppresses_nuisance_and_orients_first_signal():
    A, B = arr([1, 1]), arr([10, 0])
    out = tw.linear_observer(A, B, 1.0)
    expected = arr([1 / 101, 1])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(out["h"], expected, rtol=1e-14, atol=1e-16)
    assert out["first_projection"] > 0
    assert np.linalg.norm(out["h"]) == pytest.approx(1)
    assert out["covariance_condition"] == pytest.approx(101)
    assert out["scaled_equation_residual_norm"] < 1e-14


def test_observer_zero_nuisance_negative_signal_and_common_unit_change():
    A, B = arr([-2, 3]), arr([0.5, 0.7])
    h = tw.linear_observer(A, B, 0.2)["h"]
    np.testing.assert_allclose(tw.linear_observer(A * 7, B * 7, 1.4)["h"], h)
    np.testing.assert_allclose(tw.linear_observer(A, -B, 0.2)["h"], h)
    np.testing.assert_allclose(tw.linear_observer(-A, B, 0.2)["h"], -h)
    np.testing.assert_allclose(tw.linear_observer(A, np.zeros(2), 0.2)["h"], A / np.linalg.norm(A))


def test_zero_first_signal_is_explicitly_undefined_without_fallback():
    with pytest.raises(ValueError, match="undefined"):
        tw.linear_observer(np.zeros(2), arr([1, 2]), 0.1)


@pytest.mark.parametrize("bad", [arr([np.nan, 0]), arr([np.inf, 1]), np.ones(2, dtype=np.float32), [1., 2.], np.ones((1, 2)), np.array([], dtype=np.float64)])
def test_observer_rejects_invalid_arrays(bad):
    with pytest.raises(ValueError):
        tw.linear_observer(bad, arr([1, 2]), 0.1)


@pytest.mark.parametrize("sigma", [0, -1, True, np.bool_(True), np.nan, np.inf, "0.1"])
def test_noise_must_be_positive_finite_real(sigma):
    with pytest.raises(ValueError):
        tw.linear_observer(arr([1]), arr([2]), sigma)
    with pytest.raises(ValueError):
        tw.balanced_accuracy(arr([1]), np.zeros((2, 3, 1)), sigma)


def test_scaled_covariance_handles_tiny_common_physical_units():
    h = tw.linear_observer(arr([2e-200]), arr([1e-200]), 1e-200)["h"]
    np.testing.assert_array_equal(h, arr([1]))


def test_unrepresentable_covariance_does_not_return_an_arbitrary_observer():
    with pytest.raises(ValueError):
        tw.linear_observer(arr([1, 2]), arr([1, 1]), 1e-200)


def test_unrepresentable_condition_is_a_controlled_failure():
    with pytest.raises(ValueError, match="condition"):
        tw.linear_observer(arr([1]), arr([1]), 1e-160)


def test_six_mean_census_and_known_gaussian_accuracy():
    means = tw.linear_means(arr([0.8]), arr([0.3]))
    np.testing.assert_allclose(means[:, :, 0], arr([[-1.1, -0.8, -0.5], [0.5, 0.8, 1.1]]))
    out = tw.balanced_accuracy(arr([1]), means, 0.4)
    assert out["first_signs"] == (-1, 1)
    assert out["second_signs"] == (-1, 0, 1)
    assert out["without_second"] == pytest.approx(cdf(2))
    assert out["with_second"] == pytest.approx((cdf(2.75) + cdf(1.25)) / 2)
    assert out["interference_loss"] == pytest.approx(out["without_second"] - out["with_second"])
    assert out["probability_correct"].shape == (2, 3)


def test_observer_scale_invariance_and_reversed_orientation():
    means = tw.linear_means(arr([0.8, -0.3]), arr([0.2, 0.5]))
    h = arr([1, -2])
    original = tw.balanced_accuracy(h, means, 0.3)
    scaled = tw.balanced_accuracy(11 * h, means, 0.3)
    reversed_score = tw.balanced_accuracy(-h, means, 0.3)
    np.testing.assert_allclose(scaled["probability_correct"], original["probability_correct"])
    np.testing.assert_allclose(reversed_score["probability_correct"], 1 - original["probability_correct"])


def test_exact_zero_signal_and_no_nuisance_limits():
    zero = tw.balanced_accuracy(arr([1]), tw.linear_means(arr([0]), arr([0])), 1)
    assert zero["with_second"] == zero["without_second"] == 0.5
    no_nuisance = tw.balanced_accuracy(arr([1]), tw.linear_means(arr([1]), arr([0])), 1)
    assert no_nuisance["interference_loss"] == 0
    assert no_nuisance["with_second"] == pytest.approx(cdf(1))


def test_nonlinear_means_are_scored_without_imposing_antisymmetry():
    means = arr([[[-2.], [0.], [0.4]], [[0.3], [1.], [3.]]])
    out = tw.balanced_accuracy(arr([1]), means, 1)
    expected = arr([[cdf(2), 0.5, cdf(-0.4)], [cdf(0.3), cdf(1), cdf(3)]])
    np.testing.assert_allclose(out["probability_correct"], expected)


def test_no_first_write_control_remains_chance_with_arbitrary_second_signal():
    result = tw.no_first_write_control(arr([1, -2]), arr([[8, 9], [-1, 3], [0.2, -0.7]]), 0.4)
    np.testing.assert_allclose(result["balanced_accuracy_by_second"], 0.5, atol=1e-16)
    assert result["with_second"] == result["without_second"] == 0.5


def test_fixed_observer_contrasts_do_not_refit_or_reorient_arms():
    h = arr([1])
    identity = tw.linear_means(arr([1]), arr([0]))
    arms = {"transverse": identity.copy(), "timing": -identity}
    result = tw.fixed_observer_contrasts(h, identity, arms, 0.2)
    assert result["interventions"]["timing"]["score"]["with_second"] < 1e-6
    assert result["interventions"]["timing"]["identity_minus_arm_with_second"] > 0.999
    assert result["interventions"]["transverse"]["identity_minus_arm_with_second"] == 0
    np.testing.assert_array_equal(result["interventions"]["timing"]["score"]["h"], h)


@pytest.mark.parametrize("bad", [np.zeros((2, 2, 1)), np.zeros((3, 2, 1)), np.zeros((2, 3, 2)), np.full((2, 3, 1), np.nan), np.zeros((2, 3, 1), dtype=np.float32)])
def test_gaussian_mixture_requires_complete_typed_census(bad):
    with pytest.raises(ValueError):
        tw.balanced_accuracy(arr([1]), bad, 0.1)


def test_zero_observer_and_unrepresentable_standardized_mean_reject():
    with pytest.raises(ValueError):
        tw.balanced_accuracy(arr([0]), np.zeros((2, 3, 1)), 1)
    with pytest.raises(ValueError):
        tw.balanced_accuracy(arr([1]), tw.linear_means(arr([1]), arr([0])), 1e-320)


def test_four_level_scalar_counterexample_does_not_claim_linear_chance_is_erasure():
    # One scalar mode: means -4,-2,+2,+4; first-bit classes alternate.
    baseline = tw.periodic_single_mode(arr([3]), rho=1 / 3)
    h = tw.linear_observer(baseline["A"], baseline["B"], 0.1)["h"]
    linear = tw.balanced_accuracy(h, baseline["means"], 0.1)
    assert linear["with_second"] == 0.5
    # A fixed nonlinear interval rule, NOT a claimed exact Bayes oracle:
    # predict + on (-3,0) union (3,infinity). Four levels are recoverable.
    correct = []
    for first in (-1, 1):
        for second in (-1, 1):
            mu = first + 3 * second
            p_plus = cdf((0 - mu) / 0.1) - cdf((-3 - mu) / 0.1) + cdf((mu - 3) / 0.1)
            correct.append(p_plus if first == 1 else 1 - p_plus)
    assert math.fsum(correct) / 4 > 0.999999


def test_oblique_components_amplify_without_orthogonal_energy_identity():
    out = tw.oblique_components(arr([1, 0]), arr([1, 10]), arr([0, 1]))
    np.testing.assert_array_equal(out["phase"], arr([10, 0]))
    np.testing.assert_array_equal(out["transverse"], arr([-10, 1]))
    np.testing.assert_array_equal(out["partition_residual"], np.zeros(2))
    assert out["phase_amplification"] == 10
    assert out["transverse_amplification"] > 10
    assert out["phase_norm"] ** 2 + out["transverse_norm"] ** 2 != out["displacement_norm"] ** 2
    assert out["normalization_residual"] == 0
    assert out["transverse_linear_time_leakage"] == 0


def test_normalization_and_nonidempotence_are_not_repaired():
    out = tw.oblique_components(arr([1, 0]), arr([1.01, 2]), arr([1, 3]))
    assert out["normalization"] == 1.01
    assert out["normalization_residual"] == pytest.approx(0.01)
    assert np.max(np.abs(out["Q_idempotence_residual"])) > 0
    assert np.max(np.abs(out["gradient_P"])) > 0
    assert out["transverse_linear_time_leakage"] != 0
    np.testing.assert_array_equal(out["Q"], arr([[1.01, 2], [0, 0]]))


def test_zero_displacement_and_zero_flow_have_explicit_diagnostics():
    out = tw.oblique_components(arr([0, 0]), arr([1, 2]), arr([0, 0]))
    assert out["normalization_residual"] == -1
    assert out["phase_amplification"] is None
    assert out["transverse_amplification"] is None
    np.testing.assert_array_equal(out["P"], np.eye(2))


def test_intervention_retains_requested_versus_realized_float_addition():
    reference, state = arr([1e16, 0]), arr([1e16, 1])
    result = tw.intervene(reference, state, arr([1, 0]), arr([1, 0.5]), "timing")
    np.testing.assert_array_equal(result["requested_displacement"], arr([0.5, 0]))
    np.testing.assert_array_equal(result["realized_displacement"], arr([0, 0]))
    np.testing.assert_array_equal(result["realization_error"], arr([-0.5, 0]))
    assert result["replacement_norm"] == 1


@pytest.mark.parametrize("mode", tw.INTERVENTIONS)
def test_replacement_semantics_and_no_mutation(mode):
    reference, state = arr([1, 2]), arr([2, 3])
    original_reference, original_state = reference.copy(), state.copy()
    result = tw.intervene(reference, state, arr([1, 0]), arr([1, 2]), mode)
    expected = {"identity": arr([2, 3]), "timing": arr([4, 2]), "transverse": arr([-1, 3])}[mode]
    np.testing.assert_array_equal(result["state"], expected)
    np.testing.assert_array_equal(reference, original_reference)
    np.testing.assert_array_equal(state, original_state)


def test_noncommuting_composition_resets_first_before_adding_second():
    before, after = arr([[1, 2], [0, 1]]), arr([[2, 0], [1, 1]])
    H = arr([[1, 0], [0, 0]])
    first, second = arr([1, 2]), arr([3, 4])
    result = tw.compose_two_write(before, after, H, first, second)
    expected = after @ (H @ (before @ first) + second)
    wrong = after @ H @ (before @ first + second)
    np.testing.assert_array_equal(result["total"], expected)
    assert not np.array_equal(expected, wrong)
    np.testing.assert_array_equal(result["second_response"], after @ second)
    np.testing.assert_array_equal(result["composition_residual"], np.zeros(2))


def test_batched_clock_composition_and_zero_first_control():
    before, H = np.eye(2), arr([[1, 3], [0, 0]])
    after = np.stack((np.eye(2), arr([[1, 2], [3, 4]])))
    result = tw.compose_two_write(before, after, H, np.zeros(2), arr([1, -1]))
    assert result["total"].shape == (2, 2)
    np.testing.assert_array_equal(result["first_response"], np.zeros((2, 2)))
    np.testing.assert_array_equal(result["total"], result["second_response"])


@pytest.mark.parametrize("field,value", [
    ("flow", arr([1, 2, 3])), ("gradient", arr([np.inf, 0])),
    ("gradient", np.ones(2, dtype=np.float32)), ("displacement", arr([[1, 2]])),
])
def test_oblique_inputs_are_validated_without_implicit_conversion(field, value):
    data = dict(flow=arr([1, 0]), gradient=arr([1, 1]), displacement=arr([0, 1]))
    data[field] = value
    with pytest.raises(ValueError):
        tw.oblique_components(**data)


@pytest.mark.parametrize("field,value", [
    ("phi_before", np.eye(3)), ("phi_after", np.zeros((2, 2, 3))),
    ("H", np.eye(2, dtype=np.float32)), ("second", arr([np.nan, 0])),
])
def test_composition_rejects_wrong_shapes_dtypes_and_nonfinite(field, value):
    data = dict(phi_before=np.eye(2), phi_after=np.eye(2), H=np.eye(2), first=arr([1, 0]), second=arr([0, 1]))
    data[field] = value
    with pytest.raises(ValueError):
        tw.compose_two_write(**data)


def test_observer_and_prediction_outputs_do_not_alias_inputs():
    A, B = arr([1, 2]), arr([2, 1])
    original_A, original_B = A.copy(), B.copy()
    baseline = tw.periodic_single_mode(B, 0.5)
    baseline["B"][:] = 0
    observer = tw.linear_observer(A, B, 0.5)
    observer["h"][:] = 0
    np.testing.assert_array_equal(A, original_A)
    np.testing.assert_array_equal(B, original_B)


@pytest.mark.parametrize("arms", [{}, {"identity": np.zeros((2, 3, 1))}, {1: np.zeros((2, 3, 1))}])
def test_ambiguous_or_empty_contrast_census_rejects(arms):
    with pytest.raises(ValueError):
        tw.fixed_observer_contrasts(arr([1]), np.zeros((2, 3, 1)), arms, 0.1)


def test_unknown_intervention_and_complex_multiplier_reject():
    with pytest.raises(ValueError):
        tw.intervene(arr([0]), arr([1]), arr([1]), arr([1]), "orthogonal")
    with pytest.raises(ValueError):
        tw.periodic_single_mode(arr([1]), 0.5 + 0.1j)


def test_declared_exponential_mode_and_signed_periodic_baseline():
    result = tw.one_mode_response(arr([0, 2]), 2, math.log(2) / 2, 3, 4, retention=0.5)
    np.testing.assert_allclose(result["A"], arr([0.75, 0.375]))
    np.testing.assert_allclose(result["B"], arr([4, 2]))
    periodic = tw.periodic_single_mode(arr([2, 3]), -0.4)
    np.testing.assert_allclose(periodic["A"], arr([-0.8, -1.2]))
    assert periodic["rho"] == -0.4
    erased = tw.periodic_single_mode(arr([2, 3]), 0.4, retention=0)
    np.testing.assert_array_equal(erased["A"], np.zeros(2))


@pytest.mark.parametrize("change", [{"elapsed_after": arr([-1])}, {"spacing": 0}, {"decay_rate": -1}, {"first_gain": True}, {"second_gain": np.inf}, {"retention": np.nan}])
def test_one_mode_invalid_parameters_reject(change):
    kwargs = dict(elapsed_after=arr([0, 1]), spacing=1, decay_rate=1, first_gain=1, second_gain=1, retention=1)
    kwargs.update(change)
    with pytest.raises(ValueError):
        tw.one_mode_response(**kwargs)


def test_pure_calls_have_no_file_random_solver_or_fitting_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected data access, random sampling or fitting")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(np, "save", forbidden)
    monkeypatch.setattr(np, "polyfit", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    A, B = arr([1, 2]), arr([2, 0])
    h = tw.linear_observer(A, B, 0.5)["h"]
    means = tw.linear_means(A, B)
    tw.balanced_accuracy(h, means, 0.5)
    tw.no_first_write_control(h, np.zeros((3, 2)), 0.5)
    tw.fixed_observer_contrasts(h, means, {"timing": means}, 0.5)
    tw.intervene(np.zeros(2), A, arr([1, 0]), arr([1, 1]), "transverse")
    tw.compose_two_write(np.eye(2), np.eye(2), np.eye(2), A, B)
    tw.one_mode_response(arr([0, 1]), 1, 0.5, 1, 2)
    tw.periodic_single_mode(B, 0.7)


def test_identity_preserves_state_when_reference_roundtrip_would_round_it_away():
    reference = arr([1e16, 0.])
    state = arr([1., 2.])
    assert not np.array_equal(reference + (state-reference), state)
    result = tw.intervene(reference, state, arr([1., 0.]), arr([1., 0.]), "identity")
    assert np.array_equal(result["state"], state)
    assert np.array_equal(result["realized_replacement"], np.zeros(2))
