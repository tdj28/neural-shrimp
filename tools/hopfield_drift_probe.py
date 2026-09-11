"""Short native traces and independent local parameter sensitivity probes.

No atlas is consulted. Parameter derivatives are in the native (alpha, gamma)
coordinates. Initial states are held fixed as parameters change. A sensitivity
Gram matrix projects away ONE constant time shift over the whole recording;
it does not remove frequency changes or independently rephase samples/windows.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

from hopfield_diagnostics import hopfield_rhs, hopfield_jacobian

ROOT = Path(__file__).resolve().parents[1]


class ProbeFailure(RuntimeError):
    """A numerical failure with raw payload that callers must retain."""
    def __init__(self, message, payload):
        super().__init__(message)
        self.payload = payload


def build_trace(directory):
    """Build separate trace library and return its ctypes function and receipt."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / ("hopfield_drift.dylib" if sys.platform == "darwin" else "hopfield_drift.so")
    compiler = os.environ.get("CXX", "c++")
    flags = ["-O3", "-std=c++17", "-fPIC", "-dynamiclib" if sys.platform == "darwin" else "-shared"]
    source = ROOT / "native/hopfield_drift.cpp"
    subprocess.run([compiler, *flags, str(source), "-o", str(library)], check=True)
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    info = {"compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            "flags": flags, "library_sha256": digest(library),
            "source_sha256": digest(source),
            "included_model_sha256": digest(ROOT / "native/hopfield.cpp")}
    dll = ctypes.CDLL(str(library))
    fn = dll.hopfield_trace
    pointer = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [ctypes.c_double, ctypes.c_double, pointer, ctypes.c_double,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int, pointer, pointer, pointer]
    fn.restype = ctypes.c_int
    return fn, info


def _integer_ratio(numerator, denominator, name):
    ratio = numerator / denominator
    if not np.isfinite(ratio) or ratio > np.iinfo(np.int32).max - 1:
        raise ValueError(f"{name} exceeds the native step range")
    count = round(ratio)
    if not np.isclose(ratio, count, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} must be an integer")
    return int(count)


def integrate_trace(fn, alpha, gamma, initial, dt, duration=300.0,
                    sample_dt=0.1, *, failure_path=None):
    """Return float64 (times, states) at every requested sample and both endpoints.

    A failed native call raises ProbeFailure carrying the entire partial trace,
    last attempted state, progress counters and status. If failure_path is
    supplied, the same payload is saved before raising.
    """
    initial = np.asarray(initial, dtype=np.float64).copy()
    if initial.shape != (3,) or not np.isfinite(initial).all():
        raise ValueError("initial must contain three finite state components")
    if not np.isfinite([alpha, gamma, dt, duration, sample_dt]).all() or dt <= 0 or duration <= 0 or sample_dt <= 0:
        raise ValueError("Finite parameters and positive integration times are required")
    steps = _integer_ratio(duration, dt, "duration/dt")
    stride = _integer_ratio(sample_dt, dt, "sample_dt/dt")
    if steps < 1 or stride < 1 or steps % stride:
        raise ValueError("duration must contain an integer number of positive sampling strides")
    count = steps // stride + 1
    times = np.arange(count, dtype=np.float64) * (stride * float(dt))
    # Avoid a one-ulp overshoot of a caller's exact requested t_span endpoint.
    times[-1] = float(duration)
    states = np.full((count, 3), np.nan, dtype=np.float64)
    final = np.full(3, np.nan, dtype=np.float64)
    progress = np.full(2, np.nan, dtype=np.float64)
    ptr = lambda value: value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    status = fn(float(alpha), float(gamma), ptr(initial), float(dt), steps,
                stride, count, ptr(states), ptr(final), ptr(progress))
    valid = (status == count and np.isfinite(states).all()
             and np.isfinite(final).all() and np.array_equal(progress, [steps, count])
             and np.array_equal(states[0], initial) and np.array_equal(states[-1], final))
    if not valid:
        payload = dict(alpha=alpha, gamma=gamma, initial=initial, dt=dt,
                       duration=duration, sample_dt=sample_dt, times=times, states=states,
                       final=final, progress=progress, native_status=int(status))
        if failure_path is not None:
            np.savez_compressed(failure_path, **payload)
        raise ProbeFailure(f"Native trace failed or returned inconsistent output: {status}", payload)
    return times, states


def phase_projected_gram(activation_sensitivity, activation_velocity):
    """Project flattened parameter columns off a single flattened time-shift vector.

    S has shape (samples, 3, 2), v has shape (samples, 3). The returned Gram is
    [S.T S - outer(S.T v, S.T v)/(v.T v)]/(3*samples). It is computed from
    projected columns to avoid cancellation. If v is zero, nothing is removed.
    Eigenvectors are unit length and have their first numerically nonzero entry
    positive. A degeneracy flag warns that the minimum direction is non-unique.
    """
    sensitivity = np.asarray(activation_sensitivity, dtype=np.float64)
    velocity = np.asarray(activation_velocity, dtype=np.float64)
    if (sensitivity.ndim != 3 or sensitivity.shape[1:] != (3, 2)
            or len(sensitivity) == 0 or velocity.shape != sensitivity.shape[:2]
            or not np.isfinite(sensitivity).all() or not np.isfinite(velocity).all()):
        raise ValueError("Expected finite sensitivity (N,3,2) and velocity (N,3)")
    columns = sensitivity.reshape(-1, 2)
    phase = velocity.ravel()
    norm_squared = float(phase @ phase)
    coefficients = (phase @ columns) / norm_squared if norm_squared > 0.0 else np.zeros(2)
    projected = columns - phase[:, None] * coefficients[None, :]
    gram = projected.T @ projected / len(phase)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    minimum = eigenvectors[:, 0].copy()
    nonzero = np.flatnonzero(np.abs(minimum) > 1e-14)
    if minimum[nonzero[0]] < 0:
        minimum *= -1
    unique = bool(eigenvalues[1] - eigenvalues[0] > 1e-10 * max(abs(eigenvalues[1]), 1e-300))
    return {"gram": gram, "unprojected_gram": columns.T @ columns / len(phase),
            "eigenvalues": eigenvalues, "minimum_eigenvector": minimum,
            "minimum_eigenvector_unique": unique,
            "phase_coefficients": coefficients,
            "phase_velocity_norm_squared": norm_squared,
            "sample_count": len(sensitivity),
            "parameter_order": ["alpha", "gamma"]}


def local_parameter_sensitivity(alpha, gamma, initial, times):
    """Integrate x and its 3x2 fixed-initial-state parameter derivatives using DOP853."""
    initial = np.asarray(initial, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if (initial.shape != (3,) or not np.isfinite(initial).all()
            or not np.isfinite([alpha, gamma]).all()
            or times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] < 0 or times[-1] <= 0 or not np.all(np.diff(times) > 0)):
        raise ValueError("Finite parameters, initial (3,), and increasing nonnegative times required")

    def augmented(t, packed):
        state = packed[:3]
        sensitivity = packed[3:].reshape(3, 2)
        activation = np.tanh(state)
        forcing = np.zeros((3, 2))
        forcing[0, 0] = activation[2]
        forcing[2, 1] = activation[0]
        derivative = hopfield_jacobian(t, state, alpha, gamma) @ sensitivity + forcing
        return np.concatenate((hopfield_rhs(t, state, alpha, gamma), derivative.ravel()))

    started = time.perf_counter()
    solution = solve_ivp(augmented, (0.0, float(times[-1])), np.r_[initial, np.zeros(6)],
                         method="DOP853", t_eval=times, rtol=1e-10, atol=1e-12, max_step=.1)
    elapsed = time.perf_counter() - started
    if not solution.success or solution.y.shape != (9, len(times)) or not np.isfinite(solution.y).all():
        raise ProbeFailure("Parameter sensitivity integration failed: " + solution.message,
                           {"alpha": alpha, "gamma": gamma, "initial": initial,
                            "requested_times": times, "returned_times": solution.t,
                            "augmented_states": solution.y, "nfev": solution.nfev,
                            "wall_seconds": elapsed})
    states = solution.y[:3].T
    state_sensitivity = solution.y[3:].T.reshape(-1, 3, 2)
    activation_derivative = 1.0 - np.tanh(states) ** 2
    activation_sensitivity = activation_derivative[:, :, None] * state_sensitivity
    state_velocity = np.asarray([hopfield_rhs(t, x, alpha, gamma) for t, x in zip(times, states)])
    activation_velocity = activation_derivative * state_velocity
    return {"times": times.copy(), "states": states,
            "state_sensitivity": state_sensitivity,
            "activation_sensitivity": activation_sensitivity,
            "activation_velocity": activation_velocity,
            **phase_projected_gram(activation_sensitivity, activation_velocity),
            "wall_seconds": elapsed, "nfev": int(solution.nfev),
            "solver": {"method": "DOP853", "rtol": 1e-10, "atol": 1e-12, "max_step": .1}}
