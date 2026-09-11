"""Deterministic EXP-006 atlas selection and frozen, paired descriptive analysis.

There is no target integration in this module. Atlas distances and directions
use raw connection-weight Euclidean coordinates: ||delta W||_F equals
sqrt(delta alpha**2 + delta gamma**2). Rectangle-normalized coordinates are
used ONLY to spread anchor attempts across the already observed atlas.

Trials are paired within an anchor, orbit phase, radius and numerical step.
Numerical steps and orbit phases are not independent statistical units.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

SEED = 20260911
RADII = (0.00025, 0.001, 0.004, 0.016)
ARMS = ("outward", "inward", "tangent_plus", "tangent_minus",
        "local_plus", "local_minus", "random_plus", "random_minus",
        "shuffled_outward", "shuffled_inward", "atlas_margin")
COMPARATORS = {"local_symmetric": ("local_plus", "local_minus"),
               "random_symmetric": ("random_plus", "random_minus"),
               "shuffled_inward": ("shuffled_inward",)}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _grid_coordinates(grid):
    q = np.asarray(grid["period"])
    alpha = np.asarray(grid["alpha"], dtype=float)
    gamma = np.asarray(grid["gamma"], dtype=float)
    if q.ndim != 2 or min(q.shape) < 3:
        raise ValueError("A rectangular period atlas of at least 3 by 3 is required")
    if not np.issubdtype(q.dtype, np.number) or not np.isfinite(q).all() or np.any(q < 0) or np.any(q != np.floor(q)):
        raise ValueError("Period atlas labels must be finite nonnegative integers")
    if alpha.ndim == 1:
        alpha = np.broadcast_to(alpha, q.shape)
    if gamma.ndim == 1:
        gamma = np.broadcast_to(gamma[:, None], q.shape)
    if alpha.shape != q.shape or gamma.shape != q.shape:
        raise ValueError("Coordinate arrays do not match the period atlas")
    if (not np.isfinite(alpha).all() or not np.isfinite(gamma).all()
            or not np.all(alpha == alpha[0, :])
            or not np.all(gamma == gamma[:, :1])
            or not np.all(np.diff(alpha[0]) > 0)
            or not np.all(np.diff(gamma[:, 0]) > 0)):
        raise ValueError("Finite increasing rectangular coordinate axes are required")
    return q, alpha, gamma


def select_anchors(grid, core_index, *, count_per_stratum=4, seed=SEED):
    """Select fixed attempts without looking at any new dynamics.

    Eligible atlas cells belong to the core's 8-connected period-nine
    component. Their closest *observed non-nine* cell must be strictly nearer
    than every crop edge. No padded zeros or presumed exterior outside the
    rectangle are used. A nearest-neighbor distance is a pixel-atlas proxy,
    not a continued bifurcation boundary or guaranteed control margin.

    Sort by (raw margin, row, column), split ranks into three np.array_split
    thirds, and choose count_per_stratum from each. Within each third, a seeded
    first point is followed by maximin spacing in rectangle-width coordinates;
    exact ties use lexicographic row/column order. No target-dependent retries.
    """
    q, aa, gg = _grid_coordinates(grid)
    if (isinstance(count_per_stratum, bool)
            or not isinstance(count_per_stratum, (int, np.integer))
            or count_per_stratum < 1):
        raise ValueError("count_per_stratum must be a positive integer")
    core = tuple(core_index)
    if (len(core) != 2 or any(not isinstance(x, (int, np.integer)) for x in core)
            or not (0 <= core[0] < q.shape[0] and 0 <= core[1] < q.shape[1])
            or q[core] != 9):
        raise ValueError("The declared core must be a period-nine atlas cell")
    labels, _ = label(q == 9, structure=np.ones((3, 3), dtype=int))
    members = np.argwhere(labels == labels[core])
    exterior_indices = np.argwhere(q != 9)
    if not len(exterior_indices):
        raise ValueError("The atlas has no observed non-nine cell")
    exterior = np.column_stack((aa[q != 9], gg[q != 9]))
    tree = cKDTree(exterior)
    coordinates = np.column_stack((aa[tuple(members.T)], gg[tuple(members.T)]))
    distances, nearest = tree.query(coordinates)
    # cKDTree does not promise a lexicographic equal-distance tie break.
    # Resolve candidates near its minimum against the explicit coordinate set.
    for index, (point, distance) in enumerate(zip(coordinates, distances)):
        nearby = tree.query_ball_point(point, np.nextafter(distance, np.inf) + 1e-14)
        squared = np.sum((exterior[nearby] - point) ** 2, axis=1)
        minimum = float(np.min(squared))
        tolerance = max(1e-28, 32 * np.finfo(float).eps * minimum)
        tied = [int(k) for k, d in zip(nearby, squared) if d <= minimum + tolerance]
        nearest[index] = min(tied)  # argwhere/exterior are row-column ordered.
        distances[index] = np.linalg.norm(exterior[nearest[index]] - point)
    edge = np.minimum.reduce((coordinates[:, 0] - aa[0, 0],
                              aa[0, -1] - coordinates[:, 0],
                              coordinates[:, 1] - gg[0, 0],
                              gg[-1, 0] - coordinates[:, 1]))
    keep = np.flatnonzero(distances < edge)
    ranked = sorted(keep.tolist(), key=lambda k: (float(distances[k]),
                                                 int(members[k, 0]), int(members[k, 1])))
    strata = np.array_split(np.asarray(ranked, dtype=int), 3)
    if any(len(stratum) < count_per_stratum for stratum in strata):
        raise ValueError("Insufficient eligible atlas cells for all frozen strata; do not replace")
    rng = np.random.default_rng(seed)
    widths = np.asarray([aa[0, -1] - aa[0, 0], gg[-1, 0] - gg[0, 0]])
    anchors = []
    for group, stratum in zip(("lower_margin", "middle_margin", "upper_margin"), strata):
        pool = sorted(stratum.tolist(), key=lambda k: tuple(members[k]))
        chosen = [pool[int(rng.integers(len(pool)))]]
        while len(chosen) < count_per_stratum:
            available = [k for k in pool if k not in chosen]
            scores = [min(float(np.sum(((coordinates[k] - coordinates[j]) / widths) ** 2))
                          for j in chosen) for k in available]
            chosen.append(available[int(np.argmax(scores))])
        for k in chosen:
            row, column = (int(x) for x in members[k])
            outward = (exterior[nearest[k]] - coordinates[k]) / distances[k]
            item = {"anchor": f"anchor-{len(anchors):02d}", "group": group,
                    "row": row, "column": column,
                    "alpha": float(coordinates[k, 0]), "gamma": float(coordinates[k, 1]),
                    "atlas_period": int(q[row, column]), "raw_margin": float(distances[k]),
                    "crop_edge_lower_bound": float(edge[k]),
                    "nearest_non_nine_row": int(exterior_indices[nearest[k], 0]),
                    "nearest_non_nine_column": int(exterior_indices[nearest[k], 1]),
                    "outward": outward.tolist(), "inward": (-outward).tolist(),
                    "tangent": [-float(outward[1]), float(outward[0])]}
            if "states" in grid:
                item["captured_atlas_state"] = np.asarray(grid["states"])[row, column].tolist()
            anchors.append(item)
    for index, item in enumerate(anchors):
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        donor = anchors[(index - 1) % len(anchors)]
        item["random_unit"] = [float(np.cos(angle)), float(np.sin(angle))]
        item["shuffled_outward"] = donor["outward"].copy()
        item["shuffled_from"] = donor["anchor"]
    return {"seed": int(seed), "count_per_stratum": int(count_per_stratum),
            "anchor_attempts": len(anchors), "atlas_shape": list(q.shape),
            "core_index": [int(v) for v in core], "core_component_cells": len(members),
            "eligible_atlas_cells": len(keep), "stratum_sizes": [len(s) for s in strata],
            "metric": "raw Euclidean connection weights; equal radius equals equal Frobenius weight change",
            "spread_metric": "rectangle-width normalization, used only for anchor coverage",
            "crop_rule": "nearest observed non-nine cell distance strictly below nearest crop-edge distance",
            "sham_rule": "cyclic preceding anchor supplies outward normal; no identity assignments",
            "anchors": anchors}


def load_anchor_packet(evidence_directory, *, count_per_stratum=4, seed=SEED):
    """Read an immutable EXP-004 focus atlas and bind all consumed file bytes."""
    directory = Path(evidence_directory)
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete":
        raise ValueError("The source experiment must have a complete receipt")
    entries = {entry["path"]: entry for entry in receipt["artifacts"]}
    if len(entries) != len(receipt["artifacts"]):
        raise ValueError("Duplicate source receipt paths")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("experiment_id") != "EXP-004":
        raise ValueError("An EXP-004 source manifest is required")
    focus = next((g for g in manifest["grids"] if g["name"] == "focus"), None)
    if focus is None:
        raise ValueError("The source manifest has no focus grid")
    selected_path = directory / "selected.json"
    selected = json.loads(selected_path.read_text())
    files = sorted((directory / "focus").glob("row-*.npz"))
    if not files:
        raise ValueError("No focus atlas rows found")
    expected_names = [f"row-{i:03d}.npz" for i in range(focus["gamma"][2])]
    if [p.name for p in files] != expected_names:
        raise ValueError("Source focus row census does not match its manifest")
    consumed = [directory / "manifest.json", selected_path, *files]
    sources = [{"path": "receipt.json", "sha256": _sha(receipt_path),
                "bytes": receipt_path.stat().st_size}]
    for path in consumed:
        relative = path.relative_to(directory).as_posix()
        if relative not in entries or _sha(path) != entries[relative]["sha256"]:
            raise ValueError(f"Source receipt mismatch: {relative}")
        if "bytes" in entries[relative] and path.stat().st_size != entries[relative]["bytes"]:
            raise ValueError(f"Source byte count mismatch: {relative}")
        sources.append({"path": relative, "sha256": _sha(path), "bytes": path.stat().st_size})
    rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as saved:
            rows.append({key: saved[key].copy() for key in ("alpha", "gamma", "period", "states")})
    grid = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    if grid["period"].shape != (focus["gamma"][2], focus["alpha"][2]):
        raise ValueError("Source focus array shape does not match its manifest")
    if (not np.array_equal(grid["gamma"], np.linspace(*focus["gamma"]))
            or not np.array_equal(grid["alpha"], np.tile(np.linspace(*focus["alpha"]),
                                                        (focus["gamma"][2], 1)))):
        raise ValueError("Source focus coordinates do not match its manifest")
    core = selected["core"]
    packet = select_anchors(grid, (core["row"], core["column"]),
                            count_per_stratum=count_per_stratum, seed=seed)
    packet["source_experiment"] = "EXP-004"
    packet["source_files"] = sources
    packet["source_freeze_commit"] = receipt.get("source_commit", receipt.get("git_commit"))
    return packet


def atlas_margin_direction(grid, anchor, radius, *, angle_count=512):
    """Select an old-atlas margin direction without new trajectory outcomes.

    Score 512 uniformly spaced raw-weight unit directions by the nearest
    atlas pixel's distance to an observed non-nine pixel, only when the
    destination lies in the crop and that pixel belongs to the anchor's
    connected nine-peak component. Otherwise score zero. The score is a
    quantized atlas proxy, not the actual destination's dynamical margin.
    A smallest-angle tie break is fixed. If every score is zero, use the
    anchor's inward normal and retain the explicit fallback status.
    """
    q, aa, gg = _grid_coordinates(grid)
    if (not np.isfinite(radius) or radius <= 0
            or isinstance(angle_count, bool) or not isinstance(angle_count, (int, np.integer))
            or angle_count < 4):
        raise ValueError("A positive finite radius and integer angle_count >= 4 are required")
    row, column = int(anchor["row"]), int(anchor["column"])
    if not (0 <= row < q.shape[0] and 0 <= column < q.shape[1]) or q[row, column] != 9:
        raise ValueError("Anchor must identify a nine-peak atlas cell")
    point = np.asarray([anchor["alpha"], anchor["gamma"]], dtype=float)
    if not np.array_equal(point, [aa[row, column], gg[row, column]]):
        raise ValueError("Anchor coordinates must equal the declared atlas cell")
    labels, _ = label(q == 9, structure=np.ones((3, 3), dtype=int))
    exterior = np.column_stack((aa[q != 9], gg[q != 9]))
    if not len(exterior):
        raise ValueError("The atlas has no observed non-nine cell")
    angles = np.arange(angle_count) * (2 * np.pi / angle_count)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    destinations = point + radius * directions
    inside = ((destinations[:, 0] >= aa[0, 0]) & (destinations[:, 0] <= aa[0, -1])
              & (destinations[:, 1] >= gg[0, 0]) & (destinations[:, 1] <= gg[-1, 0]))
    # argmin on ascending axes fixes half-pixel ties at the lower index.
    columns = np.argmin(np.abs(destinations[:, 0, None] - aa[0]), axis=1)
    rows = np.argmin(np.abs(destinations[:, 1, None] - gg[:, 0]), axis=1)
    component = labels[rows, columns] == labels[row, column]
    centers = np.column_stack((aa[rows, columns], gg[rows, columns]))
    distances = cKDTree(exterior).query(centers)[0]
    scores = np.where(inside & component, distances, 0.0)
    fallback = not np.any(scores > 0)
    index = None if fallback else int(np.argmax(scores))
    direction = (np.asarray(anchor["inward"], dtype=float) if fallback else directions[index])
    if direction.shape != (2,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Fallback inward direction must have unit raw-weight norm")
    selected = point + radius * direction
    edge = float(min(selected[0] - aa[0, 0], aa[0, -1] - selected[0],
                     selected[1] - gg[0, 0], gg[-1, 0] - selected[1]))
    selected_score = None if fallback else float(scores[index])
    return {"direction": direction.tolist(), "radius": float(radius),
            "angle_count": int(angle_count), "selected_angle_index": index,
            "selected_score": selected_score, "destination": selected.tolist(),
            "fallback_inward": fallback,
            "selection_status": "all_scores_zero_inward_fallback" if fallback else "maximum_observed_pixel_margin",
            "selected_crop_edge_lower_bound": edge,
            "selected_score_crop_censored": bool(selected_score is not None and selected_score >= edge),
            "angle_scores": scores.tolist(), "angle_inside_crop": inside.tolist(),
            "angle_component_eligible": component.tolist(),
            "nearest_pixel_rows": rows.tolist(), "nearest_pixel_columns": columns.tolist(),
            "scope": "Fixed old-atlas nearest-pixel score; no target dynamics or exact boundary guarantee"}


def _trial_key(record):
    if isinstance(record["phase"], bool) or not isinstance(record["phase"], (int, np.integer)):
        raise ValueError("Trial phases must be integer indices")
    return (str(record["anchor"]), int(record["phase"]), float(record["radius"]),
            str(record["arm"]), float(record["dt"]))


def _error(record):
    if record.get("status") != "complete":
        return None
    try:
        value = float(record["wave_metrics"]["aligned_rms"]["aggregate"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) and 0 <= value <= 2 else None


def summarize_trials(records, anchor_ids, eligible_anchor_ids, *, phases=(0, 1, 2, 3),
                     radii=RADII, dt_values=(0.01, 0.005), bootstrap_samples=10000,
                     seed=SEED, zero_phases=None):
    """Compute paired descriptive effects without success-only deletion.

    Records use keys anchor, phase (integer index), radius, arm, dt, status,
    and wave_metrics['aligned_rms']['aggregate']. Arms are declared in ARMS.
    Zero controls use radius=0 and arm='zero'; zero_phases defaults to phases.
    Missing/nonfinite/failed required trials make the affected radius-step
    contrasts undefined, including when only a nondirect control is missing.
    A missing zero control makes every radius at that step undefined.

    The sign convention is comparator error minus atlas-margin-policy error, so
    positive means lower error for the geometry policy. Local/random policies
    average their two signs, rather than choosing a sign from target outcomes.
    Each anchor gets equal weight; its phases and four radii are averaged first.
    Percentile intervals resample anchors 10,000 times by default. These are
    exploratory fixed-anchor descriptive intervals, not population inference.
    """
    anchor_ids = tuple(str(a) for a in anchor_ids)
    eligible = tuple(str(a) for a in eligible_anchor_ids)
    phases, radii, steps = tuple(phases), tuple(float(r) for r in radii), tuple(float(d) for d in dt_values)
    zeros = phases if zero_phases is None else tuple(zero_phases)
    if (len(set(anchor_ids)) != len(anchor_ids) or len(set(eligible)) != len(eligible)
            or not set(eligible) <= set(anchor_ids)):
        raise ValueError("Unique declared anchor IDs and an eligible subset are required")
    if (len(set(phases)) != len(phases) or len(set(radii)) != len(radii)
            or len(set(steps)) != len(steps) or not set(zeros) <= set(phases)
            or any(not isinstance(p, (int, np.integer)) for p in phases)
            or any(r <= 0 or not np.isfinite(r) for r in radii)
            or any(d <= 0 or not np.isfinite(d) for d in steps)
            or not phases or not radii or not steps or bootstrap_samples < 1):
        raise ValueError("Invalid frozen census or bootstrap settings")
    expected = {(a, p, r, arm, dt) for a in eligible for p in phases
                for r in radii for arm in ARMS for dt in steps}
    expected |= {(a, p, 0.0, "zero", dt) for a in eligible for p in zeros for dt in steps}
    indexed = {}
    for record in records:
        key = _trial_key(record)
        if key not in expected:
            raise ValueError(f"Unplanned trial: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate trial: {key}")
        indexed[key] = record
    missing = sorted(expected - indexed.keys())
    invalid = sorted(k for k, r in indexed.items() if _error(r) is None)
    rng = np.random.default_rng(seed)
    draws = (rng.integers(0, len(eligible), size=(bootstrap_samples, len(eligible)))
             if eligible else None)

    def effect_summary(values):
        if values is None:
            return {"defined": False, "mean_benefit": None, "ci95": None,
                    "anchor_effects": None}
        array = np.asarray(values, dtype=float)
        samples = array[draws].mean(axis=1)
        return {"defined": True, "mean_benefit": float(array.mean()),
                "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
                "anchor_effects": {a: float(v) for a, v in zip(eligible, array)}}

    panels = []
    for dt in steps:
        radius_effects = {name: [] for name in COMPARATORS}
        zero_complete = all(k in indexed and _error(indexed[k]) is not None
                            for k in expected if k[4] == dt and k[2] == 0)
        per_radius = []
        for radius in radii:
            keys = {k for k in expected if k[4] == dt and k[2] == radius}
            complete = bool(eligible) and zero_complete and all(
                k in indexed and _error(indexed[k]) is not None for k in keys)
            effects, means = {}, {}
            if complete:
                means = {arm: float(np.mean([_error(indexed[(a, p, radius, arm, dt)])
                                             for a in eligible for p in phases])) for arm in ARMS}
            for name, comparator_arms in COMPARATORS.items():
                values = None
                if complete:
                    values = [float(np.mean([
                        np.mean([_error(indexed[(a, p, radius, arm, dt)]) for arm in comparator_arms])
                        - _error(indexed[(a, p, radius, "atlas_margin", dt)]) for p in phases]))
                              for a in eligible]
                radius_effects[name].append(values)
                effects[name] = effect_summary(values)
            per_radius.append({"radius": radius, "complete": complete,
                               "arm_mean_rmse": means if complete else None, "effects": effects})
        pooled = {}
        for name, all_values in radius_effects.items():
            values = (np.mean(np.asarray(all_values), axis=0).tolist()
                      if all(v is not None for v in all_values) else None)
            pooled[name] = effect_summary(values)
        panels.append({"dt": dt, "zero_controls_complete": zero_complete,
                       "per_radius": per_radius, "equal_radius_average": pooled})
    return {"anchor_attempts": len(anchor_ids), "eligible_anchor_count": len(eligible),
            "eligible_anchors": list(eligible),
            "ineligible_anchors": [a for a in anchor_ids if a not in set(eligible)],
            "anchor_coverage": len(eligible) / len(anchor_ids) if anchor_ids else None,
            "planned_eligible_trials": len(expected), "observed_trials": len(indexed),
            "status_counts": dict(sorted(Counter(str(r.get("status", "missing_status"))
                                                  for r in indexed.values()).items())),
            "missing_trials": [list(k) for k in missing], "invalid_trials": [list(k) for k in invalid],
            "complete": not missing and not invalid, "panels": panels,
            "bootstrap": {"seed": int(seed), "samples": int(bootstrap_samples),
                          "unit": "anchor", "method": "percentile; same resampled anchors across contrasts"},
            "estimand": "comparator aligned activation RMSE minus atlas-margin-policy RMSE; equal anchor, phase and radius weights",
            "scope": "Exploratory paired effects conditional on baseline-qualified fixed anchor attempts; steps are refinements, not replicates"}
