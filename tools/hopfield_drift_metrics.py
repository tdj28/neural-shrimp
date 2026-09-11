"""Fixed-period, phase-aware activation errors; no trajectory integration.

The reference is one uniformly sampled cycle with its endpoint excluded. Its
first sample defines phase zero at time zero. Candidate times use that same
epoch; they are NOT silently shifted to the candidate's first sample. The
aligned error permits one constant phase offset for the entire recording. It
never fits frequency, aligns windows independently, or uses nearest-orbit
distance. Consequently frequency drift remains an error.

Linear interpolation and the finite phase search are numerical approximations,
not exact orbit distances. A cadence diagnostic checks whether the *sampled*
reference contains appreciable Fourier power at or above the candidate's
Nyquist frequency; it cannot rule out aliasing already present in the reference.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def periodic_interpolate(reference, phases):
    """Interpolate endpoint-excluded samples at phases in cycle units."""
    reference = np.asarray(reference, dtype=float)
    phases = np.asarray(phases, dtype=float)
    if (reference.ndim != 2 or reference.shape[0] < 2
            or not np.isfinite(reference).all() or not np.isfinite(phases).all()):
        raise ValueError("A finite (samples, components) reference and finite phases are required")
    coordinate = np.remainder(phases, 1.0) * len(reference)
    left = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - left
    return ((1.0 - fraction[..., None]) * reference[left % len(reference)]
            + fraction[..., None] * reference[(left + 1) % len(reference)])


def _rms(errors):
    per_component = np.sqrt(np.mean(errors * errors, axis=0))
    aggregate = float(np.sqrt(np.mean(errors * errors)))
    return {"aggregate": min(2.0, aggregate),
            "per_component": np.minimum(2.0, per_component).tolist()}


def _sampling_diagnostic(reference, times, period):
    # The largest sampling gap is conservative if candidate times are uneven.
    gaps = np.diff(times)
    nyquist_harmonic = period / (2.0 * np.max(gaps))
    power = np.abs(np.fft.rfft(reference - reference.mean(axis=0), axis=0)) ** 2
    weights = np.full(len(power), 2.0)
    weights[0] = 0.0
    if len(reference) % 2 == 0:
        weights[-1] = 1.0
    power *= weights[:, None]
    unresolved = np.arange(len(power)) >= nyquist_harmonic * (1.0 - 1e-12)
    total = power.sum(axis=0)
    fraction = np.divide(power[unresolved].sum(axis=0), total,
                         out=np.zeros(reference.shape[1]), where=total > 1e-24)
    return {
        "reference_samples_per_cycle": len(reference),
        "candidate_minimum_samples_per_cycle": float(period / np.max(gaps)),
        "candidate_gap_min": float(np.min(gaps)),
        "candidate_gap_max": float(np.max(gaps)),
        "unresolved_reference_power_fraction_per_component": fraction.tolist(),
        "candidate_cadence_resolves_reference_spectrum": bool(np.max(fraction) <= 1e-6),
        "power_fraction_tolerance": 1e-6,
        "limitation": "Checks sampled reference spectrum only; does not exclude reference aliasing",
    }


def waveform_errors(reference_activations, candidate_times, candidate_activations,
                    period, *, phase_grid=128, window_cycles=1.0):
    """Return bounded activation RMS with synchronous and global phase alignment.

    Inputs contain all three activation components, each in [-1, 1]. A 128-bin
    phase scan (by default) is followed by bounded scalar refinement within one
    bin of its best point, with a deterministic smallest-index tie break. This
    is not a guarantee of the global optimum for arbitrarily sharp waveforms;
    the chosen phase may also be non-unique for symmetric/repeated waveforms.
    The better of the coarse and refined objectives is retained. Window errors
    reuse this SAME fitted phase, so they do not hide accumulated timing drift.
    Samples receive equal weight; use uniform candidate times for a time RMS.
    """
    reference = np.asarray(reference_activations, dtype=float)
    times = np.asarray(candidate_times, dtype=float)
    candidate = np.asarray(candidate_activations, dtype=float)
    if (reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 8
            or times.ndim != 1 or len(times) < 2
            or candidate.shape != (len(times), 3)):
        raise ValueError("Expected reference (N>=8, 3), times (M>=2,), and activations (M, 3)")
    if (not np.isfinite(reference).all() or not np.isfinite(times).all()
            or not np.isfinite(candidate).all() or not np.all(np.diff(times) > 0)):
        raise ValueError("Finite arrays and strictly increasing times are required")
    if np.max(np.abs(reference)) > 1.0 + 1e-12 or np.max(np.abs(candidate)) > 1.0 + 1e-12:
        raise ValueError("Activation components must lie in [-1, 1]")
    if not np.isfinite(period) or period <= 0 or not np.isfinite(window_cycles) or window_cycles <= 0:
        raise ValueError("Period and window_cycles must be positive and finite")
    if isinstance(phase_grid, bool) or not isinstance(phase_grid, (int, np.integer)) or phase_grid < 8:
        raise ValueError("phase_grid must be an integer >= 8")
    reference = np.clip(reference, -1.0, 1.0)
    candidate = np.clip(candidate, -1.0, 1.0)
    phases = times / float(period)

    def objective(offset):
        error = candidate - periodic_interpolate(reference, phases + offset)
        return float(np.mean(error * error))

    coarse = np.asarray([objective(k / phase_grid) for k in range(phase_grid)])
    best_index = int(np.argmin(coarse))
    best_phase = best_index / phase_grid
    refined = minimize_scalar(objective, method="bounded",
                              bounds=(best_phase - 1.0 / phase_grid,
                                      best_phase + 1.0 / phase_grid),
                              options={"xatol": 1e-10, "maxiter": 64})
    chosen = (float(refined.x) if refined.success and np.isfinite(refined.fun)
              and refined.fun < coarse[best_index] else best_phase)
    chosen = float((chosen + 0.5) % 1.0 - 0.5)
    synchronous_error = candidate - periodic_interpolate(reference, phases)
    aligned_error = candidate - periodic_interpolate(reference, phases + chosen)

    duration = float(window_cycles * period)
    count = max(1, int(np.ceil((times[-1] - times[0]) / duration)))
    window_index = np.minimum(np.floor((times - times[0]) / duration).astype(int), count - 1)
    windows = []
    for index in range(count):
        mask = window_index == index
        if not np.any(mask):
            continue
        windows.append({"index": index, "first_sample_time": float(times[mask][0]),
                        "last_sample_time": float(times[mask][-1]),
                        "sample_count": int(mask.sum()),
                        "synchronous_rms": _rms(synchronous_error[mask]),
                        "aligned_rms": _rms(aligned_error[mask])})
    return {
        "period": float(period), "phase_offset_cycles": chosen,
        "synchronous_rms": _rms(synchronous_error),
        "aligned_rms": _rms(aligned_error),
        "windows": windows, "window_cycles": float(window_cycles),
        "phase_search": {"coarse_bins": int(phase_grid),
                         "coarse_best_rms": float(np.sqrt(coarse[best_index])),
                         "refinement_success": bool(refined.success),
                         "refinement_evaluations": int(refined.nfev),
                         "method": "coarse global scan, bounded refinement around best bin"},
        "sampling": _sampling_diagnostic(reference, times, float(period)),
    }
