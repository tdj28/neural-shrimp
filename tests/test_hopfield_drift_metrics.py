"""Synthetic waveform checks only; no neural experiment trajectories."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hopfield_drift_metrics import periodic_interpolate, waveform_errors


def reference(n=2048):
    phase = np.arange(n) / n
    return np.column_stack((.8 * np.sin(2 * np.pi * phase),
                            .6 * np.cos(2 * np.pi * phase),
                            .4 * np.sin(4 * np.pi * phase + .3)))


def test_phase_offset_is_removed_but_synchronous_error_is_retained():
    ref = reference()
    times = np.arange(0., 6., .01)
    candidate = periodic_interpolate(ref, times + .237)
    result = waveform_errors(ref, times, candidate, 1.)
    assert result["phase_offset_cycles"] == pytest.approx(.237, abs=1e-7)
    assert result["aligned_rms"]["aggregate"] < 1e-7
    assert result["synchronous_rms"]["aggregate"] > .4
    assert all(w["aligned_rms"]["aggregate"] < 1e-7 for w in result["windows"])


def test_absolute_time_epoch_is_not_reset_and_seam_phase_is_refined():
    ref = reference()
    times = np.arange(2.37, 5.37, .01)
    candidate = periodic_interpolate(ref, times - .0017)
    result = waveform_errors(ref, times, candidate, 1.)
    assert result["phase_offset_cycles"] == pytest.approx(-.0017, abs=1e-7)
    assert result["aligned_rms"]["aggregate"] < 1e-7


def test_amplitude_error_cannot_be_removed_by_phase_alignment():
    ref = reference()
    times = np.arange(0., 6., .01)
    candidate = .5 * periodic_interpolate(ref, times)
    result = waveform_errors(ref, times, candidate, 1.)
    assert result["aligned_rms"]["aggregate"] == pytest.approx(
        np.sqrt(np.mean(candidate**2)), rel=1e-6)
    # Linear interpolation slightly modulates amplitude within a reference bin.
    assert abs(result["phase_offset_cycles"]) < .05 / len(ref)


def test_wrong_frequency_remains_visible_in_global_and_window_errors():
    ref = reference()
    times = np.arange(0., 20., .01)
    candidate = periodic_interpolate(ref, times * 1.025)
    result = waveform_errors(ref, times, candidate, 1.)
    assert result["aligned_rms"]["aggregate"] > .25
    window_rms = [w["aligned_rms"]["aggregate"] for w in result["windows"]]
    assert max(window_rms) > 3 * min(window_rms)


@pytest.mark.parametrize("component", [0, 1, 2])
def test_every_activation_component_contributes(component):
    ref = np.zeros((32, 3))
    times = np.arange(20.) / 10.
    candidate = np.zeros((20, 3))
    candidate[:, component] = .6
    result = waveform_errors(ref, times, candidate, 1.)
    assert result["aligned_rms"]["aggregate"] == pytest.approx(.6 / np.sqrt(3))
    assert result["aligned_rms"]["per_component"][component] == pytest.approx(.6)
    assert sum(result["aligned_rms"]["per_component"]) == pytest.approx(.6)


def test_aliasing_can_make_zero_sample_error_and_is_explicitly_flagged():
    phase = np.arange(512) / 512
    ref = np.column_stack((np.sin(2 * np.pi * 8 * phase),
                           np.zeros(512), np.zeros(512)))
    times = np.arange(32.) / 8.
    result = waveform_errors(ref, times, np.zeros((32, 3)), 1.)
    assert result["synchronous_rms"]["aggregate"] < 1e-12
    sampling = result["sampling"]
    assert not sampling["candidate_cadence_resolves_reference_spectrum"]
    assert sampling["unresolved_reference_power_fraction_per_component"][0] > .999
    dense_times = np.arange(256.) / 64.
    dense = waveform_errors(ref, dense_times, periodic_interpolate(ref, dense_times), 1.)
    assert dense["sampling"]["candidate_cadence_resolves_reference_spectrum"]


def test_periodic_interpolation_wraps_and_does_not_repeat_endpoint():
    ref = np.array([[0., 0.], [1., 1.], [0., 0.], [-1., -1.]])
    out = periodic_interpolate(ref, [0., 1., -.125, .875, 2.25])
    np.testing.assert_allclose(out, [[0., 0.], [0., 0.], [-.5, -.5], [-.5, -.5], [1., 1.]])


def test_rms_is_bounded_and_invalid_input_is_rejected():
    ref = np.ones((32, 3))
    times = np.arange(20.) / 10.
    candidate = -np.ones((20, 3))
    assert waveform_errors(ref, times, candidate, 1.)["aligned_rms"]["aggregate"] == 2.
    with pytest.raises(ValueError, match="Activation"):
        waveform_errors(ref, times, candidate * 2., 1.)
    with pytest.raises(ValueError, match="strictly increasing"):
        waveform_errors(ref, times[::-1], candidate, 1.)
    with pytest.raises(ValueError, match="positive"):
        waveform_errors(ref, times, candidate, 0.)
    with pytest.raises(ValueError, match="phase_grid"):
        waveform_errors(ref, times, candidate, 1., phase_grid=7)
    with pytest.raises(ValueError, match="Expected"):
        waveform_errors(ref[:, :2], times, candidate, 1.)
