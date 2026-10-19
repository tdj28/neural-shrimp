"""Capture the five seed-check images without another model pass.

This is a source-preparation helper, not a caller or execution admission. It
copies the loop and distance expression of SoftOrbit.return_distances from
bea0bff82e16d190d64569fcd56fdaabcfa4540b:tools/explore_llm_soft_loop.py:490-495.
The current producer method has the same body. No producer module, tensor
backend, model, file or process is imported or accessed here.

The supplied cell.f must be the separately source-authenticated one-step map.
Duck-typed metadata checks below do not authenticate a tensor implementation,
model, effective arithmetic precision, map, backend or original execution.
The caller remains responsible for the existing closure/primitive gates,
source/input/asset pins, exclusive output, partial-output preservation and
separate runtime admission. Returned distances admit no scientific claim.
"""
import math


def _validate_state(value):
    """Require the narrow declared CPU binary64 window and finite entries."""
    shape = tuple(value.shape)
    if shape != (1, 1, 896) or any(type(size) is not int for size in shape):
        raise ValueError('seed state shape must be exactly (1, 1, 896)')
    if str(value.dtype) != 'torch.float64' or value.device.type != 'cpu':
        raise ValueError('declared CPU torch.float64 seed state required')
    finite = value.isfinite().all().item()
    if type(finite) is not bool or not finite:
        raise ValueError('finite seed state required')


def capture_seed_phases(cell, X, on_phase, *, period=5):
    """Return five original relative distances and emit six independent copies.

    ``on_phase(index, snapshot)`` receives X0 before any map call, then each
    newly computed X1 through X5 immediately after its validation. Each snapshot
    is ``state.detach().clone()``: callback mutation cannot change the live
    recurrence or another callback's snapshot. Indices 0..4 are the five seed
    phases; index 5 is the closing image, not a sixth distinct phase. Preserve
    the batch/window axes: every supplied snapshot has shape (1, 1, 896).

    A successful call makes exactly five cell.f calls, in the original order,
    and evaluates the original distance expression after those calls. State
    validation and copying add no map evaluations. Invalid states or callback
    exceptions propagate immediately and prevent subsequent map calls. Earlier
    callback evidence belongs to the caller and is neither removed nor marked
    complete. Distances are returned only after all six callbacks succeed.
    """
    if type(period) is not int or period != 5:
        raise ValueError('only the exact period-5 seed capture is supported')
    if not callable(on_phase) or not callable(getattr(cell, 'f', None)):
        raise ValueError('one-step map and phase callback must be callable')
    _validate_state(X)
    initial_norm = float(X.norm())
    if not math.isfinite(initial_norm) or initial_norm <= 0:
        raise ValueError('finite nonzero seed norm required')

    hist = [X]
    on_phase(0, X.detach().clone())
    for _ in range(period):
        following = cell.f(hist[-1])
        _validate_state(following)
        hist.append(following)
        on_phase(len(hist) - 1, following.detach().clone())
    distances = [float((hist[q] - X).norm() / X.norm())
                 for q in range(1, period + 1)]
    if not all(math.isfinite(value) for value in distances):
        raise ValueError('finite seed return distances required')
    return distances
