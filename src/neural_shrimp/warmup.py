"""Exploratory full-state warmup and intervention diagnostics.

A changing-rate trajectory is compared with the period-2 orbit of each
instantaneous frozen map. Loss spikes, local departure and recovery are
separate operational endpoints, not bifurcation classifications.
"""

import numpy as np
from scipy.optimize import brentq

from .criticality import critical_pair
from .gdmaps import GridMap
from .orbits import certify_orbit


def orbit_reference(net, x, y, h, o, seed):
    grid = GridMap(net, x, y, [10.0 ** h], [10.0 ** o])
    fn, jac = grid.cell_map(0)
    result = certify_orbit(fn, jac, seed, 2, candidate_tolerance=0.1)
    if not result["verified"]:
        raise ValueError(f"period-2 numerical refinement failed: {result['reason']}")
    orbit = np.asarray(result["orbit"])
    mu, q, _, _ = critical_pair(jac(orbit[1]) @ jac(orbit[0]))
    losses = net.loss(orbit, x, y)
    return {"h": float(h), "o": float(o), "orbit": orbit,
            "alpha": float(np.log(abs(mu))), "direction": q.real / np.linalg.norm(q.real),
            "loss_mean": float(np.mean(losses)), "loss_max": float(np.max(losses)),
            "closure_max": float(np.max(np.abs(fn(orbit[1]) - orbit[0]))),
            "primitive_separation": result["min_divisor_separation"],
            "branch_reference_applies": True}


def matched_references(net, x, y, row, targets):
    """Match log|mu|, not raw learning-rate offsets, across soft/hard points."""
    h0, o = row["log10_hidden_boundary"], row["log10_output"]
    seed = np.asarray(row["orbit_point"])
    references = []
    for target in targets:
        def mismatch(h):
            return orbit_reference(net, x, y, h, o, seed)["alpha"] - target
        h = brentq(mismatch, h0 - 0.03, h0 + 0.03, xtol=2e-12)
        ref = orbit_reference(net, x, y, h, o, seed)
        if abs(ref["alpha"] - target) > 2e-9:
            raise ValueError("matched multiplier tolerance failed")
        references.append(ref)
        seed = ref["orbit"][0]
    return references


def orbit_distance(states, orbit):
    """Phase-invariant Euclidean distance using every parameter."""
    return np.min(np.linalg.norm(np.asarray(states)[:, None] - orbit[None], axis=-1), axis=1)


def reduced_reference(initial, top, factor):
    """Global-rate baseline; its old-orbit distance is not a local-map reference."""
    if not 0 < factor < 1:
        raise ValueError("global reduction factor must be between zero and one")
    return {**initial, "h": top["h"] + np.log10(factor),
            "o": top["o"] + np.log10(factor), "alpha": float("nan"),
            "branch_reference_applies": False}


def rollout(net, x, y, initial, segments):
    """Integrate batched pairs, retaining all pair-end states and both losses.

    Each segment is (pair_count, per_cell_reference, phase). Reference rates
    are held fixed for both updates of each pair. Diverged cells remain NaN;
    zero buffers for inactive cells only prevent contamination of other cells.
    """
    theta = np.asarray(initial, dtype=np.float64).copy()
    cells, dimension = theta.shape
    total = sum(count for count, _, _ in segments)
    state = np.full((total, cells, dimension), np.nan)
    loss = np.full((total, cells, 2), np.nan)
    distance = np.full((total, cells), np.nan)
    alpha = np.full((total, cells), np.nan)
    ref_loss_max = np.empty((total, cells))
    ref_valid = np.empty((total, cells), dtype=bool)
    rates = np.empty((total, cells, 2))
    phase = np.empty(total, dtype=np.int16)
    alive = np.all(np.isfinite(theta), axis=1)
    index = 0
    for count, refs, label in segments:
        if len(refs) != cells or count < 0:
            raise ValueError("segment shape/count mismatch")
        grid = GridMap(net, x, y, [10 ** r["h"] for r in refs], [10 ** r["o"] for r in refs])
        for _ in range(count):
            theta[~alive] = 0
            with np.errstate(over="ignore", invalid="ignore"):
                middle, l0 = grid.step(theta, return_loss=True)
                next_theta, l1 = grid.step(middle, return_loss=True)
            valid = (alive & np.all(np.isfinite(next_theta), axis=1)
                     & np.all(np.isfinite(middle), axis=1)
                     & (np.max(np.abs(next_theta), axis=1) <= 1e6)
                     & np.isfinite(l0) & np.isfinite(l1))
            theta = next_theta
            theta[~valid] = np.nan
            alive = valid
            state[index] = theta
            loss[index, :, 0] = np.where(valid, l0, np.nan)
            loss[index, :, 1] = np.where(valid, l1, np.nan)
            for j, ref in enumerate(refs):
                distance[index, j] = orbit_distance(theta[j:j + 1], ref["orbit"])[0]
                alpha[index, j] = ref["alpha"]
                ref_loss_max[index, j] = ref["loss_max"]
                ref_valid[index, j] = ref["branch_reference_applies"]
                rates[index, j] = (10 ** ref["h"], 10 ** ref["o"])
            phase[index] = label
            index += 1
    return {"state": state, "loss": loss, "distance": distance, "alpha": alpha,
            "reference_loss_max": ref_loss_max, "reference_applies": ref_valid,
            "rates": rates, "phase": phase, "initial_state": np.asarray(initial)}


def summarize(trace, cell, stable_reference, mask=None, tail=1000):
    """Thresholds are operational; continuous values accompany every flag."""
    select = np.ones(len(trace["phase"]), dtype=bool) if mask is None else np.asarray(mask)
    losses = trace["loss"][select, cell]
    distances = trace["distance"][select, cell]
    valid_local = trace["reference_applies"][select, cell]
    reference = trace["reference_loss_max"][select, cell]
    finite = np.all(np.isfinite(losses), axis=1)
    local = finite & valid_local
    peak_ratio = float(np.max(np.max(losses[finite], axis=1) / reference[finite])) if finite.any() else None
    max_local = float(np.max(distances[local])) if local.any() else None
    terminal = trace["state"][-min(tail, len(trace["state"])):, cell]
    recovered_distance = orbit_distance(terminal, stable_reference["orbit"])
    terminal_loss = trace["loss"][-len(terminal):, cell].mean(axis=1)
    valid_tail = bool(np.all(np.isfinite(terminal_loss)))
    mean_tail = float(np.mean(terminal_loss)) if valid_tail else None
    distance_tail = float(np.median(recovered_distance)) if valid_tail else None
    return {"finite_selected_pairs": int(finite.sum()), "selected_pairs": int(select.sum()),
            "diverged_or_inherited_divergence": bool(not np.all(np.isfinite(trace["state"][:, cell]))),
            "peak_loss_ratio_to_frozen_orbit": peak_ratio,
            "loss_spike_above_25_percent": None if peak_ratio is None else peak_ratio > 1.25,
            "max_local_distance": max_local,
            "local_departure_above_0_5": None if max_local is None else max_local > 0.5,
            "terminal_mean_loss": mean_tail,
            "terminal_loss_ratio_to_initial_orbit_mean": None if mean_tail is None else mean_tail / stable_reference["loss_mean"],
            "loss_recovered_within_10_percent": None if mean_tail is None else mean_tail <= 1.1 * stable_reference["loss_mean"],
            "terminal_median_distance_to_initial_orbit": distance_tail,
            "returned_to_initial_orbit_within_0_01": None if distance_tail is None else distance_tail < 0.01}


def branch_access(trace, cell):
    """Descriptive accessibility checks; finite horizon cannot prove basin exclusion."""
    pre = np.flatnonzero(trace["phase"] == 0)
    crossing = np.flatnonzero(trace["alpha"][:, cell] >= 0)
    if not len(pre) or not len(crossing):
        raise ValueError("prehold and nonnegative multiplier stages are required")
    k = int(crossing[0])
    departure = np.flatnonzero(trace["distance"][k:, cell] > 0.5)
    pre_distance = float(trace["distance"][pre[-1], cell])
    crossing_distance = float(trace["distance"][k, cell])
    return {"distance_after_prehold": pre_distance,
            "distance_at_first_nonnegative_log_multiplier": crossing_distance,
            "near_reference_at_crossing_within_0_01": bool(np.isfinite(crossing_distance) and crossing_distance < 0.01),
            "first_departure_after_crossing_pairs": int(departure[0]) if len(departure) else None}
