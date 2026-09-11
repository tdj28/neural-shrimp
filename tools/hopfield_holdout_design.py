"""New waveform-intervention anchors conditional on the already observed atlas.

No integration, baseline certification or target outcome is used here. The
EXP-004 map remains unchanged, including every non-nine label used for raw
weight distances. The EXP-006 starting locations define excluded halos; their
waveform scores do not enter this selection. This is not an unseen-atlas test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from hopfield_drift_design import _grid_coordinates, load_anchor_packet

SEED = 20260912
EXCLUSION_RADIUS = 0.04
PHASE_INDICES = (337, 1012, 1687, 2362)
REFERENCE_SAMPLES = 2700


def select_holdout_anchors(grid, core_index, development_anchors, *,
                           count_per_stratum=4, seed=SEED,
                           exclusion_radius=EXCLUSION_RADIUS):
    """Freeze margin-stratified attempts after a normalized development halo.

    The closed halo is normalized Euclidean distance <=0.04 using the full
    old focus widths, 0.024 for alpha and 0.41 for gamma. It excludes starting
    anchors and neighborhoods, not all destinations explored in EXP-006.
    Raw margins and directions still use Euclidean connection-weight units.

    After the old observed-boundary/crop test AND halo exclusion, sort by
    (margin,row,column), split ranks into thirds, and choose four per third.
    The first point in each third is seeded; subsequent choices maximize
    minimum distance to already chosen points in that third using normalized
    coordinates. Exact spacing ties use row/column order. Too few candidates
    raises an error: no smaller halo, replacement stratum or alternate seed.
    """
    q, aa, gg = _grid_coordinates(grid)
    if (isinstance(count_per_stratum, bool)
            or not isinstance(count_per_stratum, (int, np.integer))
            or count_per_stratum < 1):
        raise ValueError("A positive integer count_per_stratum is required")
    if not np.isfinite(exclusion_radius) or exclusion_radius <= 0:
        raise ValueError("A positive finite normalized exclusion radius is required")
    core = tuple(core_index)
    if (len(core) != 2 or any(not isinstance(v, (int, np.integer)) for v in core)
            or not (0 <= core[0] < q.shape[0] and 0 <= core[1] < q.shape[1])
            or q[core] != 9):
        raise ValueError("The core must be an existing nine-peak pixel")
    development = np.asarray([[a["alpha"], a["gamma"]] for a in development_anchors], dtype=float)
    if development.ndim != 2 or development.shape[1] != 2 or len(development) < 1 or not np.isfinite(development).all():
        raise ValueError("Finite development starting coordinates are required")
    if len(np.unique(development, axis=0)) != len(development):
        raise ValueError("Development starting coordinates must be unique")
    widths = np.asarray([aa[0, -1] - aa[0, 0], gg[-1, 0] - gg[0, 0]])
    labels, _ = label(q == 9, structure=np.ones((3, 3), dtype=int))
    members = np.argwhere(labels == labels[core])
    points = np.column_stack((aa[tuple(members.T)], gg[tuple(members.T)]))
    exterior_indices = np.argwhere(q != 9)
    exterior = np.column_stack((aa[q != 9], gg[q != 9]))
    if not len(exterior):
        raise ValueError("The atlas has no observed non-nine boundary pixel")
    tree = cKDTree(exterior)
    margins, nearest = tree.query(points)
    for i, (point, margin) in enumerate(zip(points, margins)):
        near = tree.query_ball_point(point, np.nextafter(margin, np.inf) + 1e-14)
        square = np.sum((exterior[near] - point) ** 2, axis=1)
        minimum = float(square.min())
        tolerance = max(1e-28, 32 * np.finfo(float).eps * minimum)
        nearest[i] = min(int(k) for k, distance in zip(near, square) if distance <= minimum + tolerance)
        margins[i] = np.linalg.norm(exterior[nearest[i]] - point)
    edges = np.minimum.reduce((points[:, 0] - aa[0, 0], aa[0, -1] - points[:, 0],
                               points[:, 1] - gg[0, 0], gg[-1, 0] - points[:, 1]))
    delta = (points[:, None, :] - development[None, :, :]) / widths
    development_distance = np.sqrt(np.sum(delta * delta, axis=2))
    closest_development = np.argmin(development_distance, axis=1)
    halo_distance = development_distance[np.arange(len(points)), closest_development]
    crop_eligible = margins < edges
    halo_eligible = halo_distance > exclusion_radius
    eligible = np.flatnonzero(crop_eligible & halo_eligible)
    ranking = sorted(eligible.tolist(), key=lambda i: (float(margins[i]), int(members[i, 0]), int(members[i, 1])))
    thirds = np.array_split(np.asarray(ranking, dtype=int), 3)
    if any(len(third) < count_per_stratum for third in thirds):
        raise ValueError("Insufficient candidates after the frozen halo; no fallback or replacement")
    rng = np.random.default_rng(seed)
    anchors = []
    for group, third in zip(("lower_margin", "middle_margin", "upper_margin"), thirds):
        pool = sorted(third.tolist(), key=lambda i: tuple(members[i]))
        chosen = [pool[int(rng.integers(len(pool)))]]
        while len(chosen) < count_per_stratum:
            available = [i for i in pool if i not in chosen]
            scores = [min(float(np.sum(((points[i] - points[j]) / widths) ** 2)) for j in chosen)
                      for i in available]
            chosen.append(available[int(np.argmax(scores))])
        for i in chosen:
            row, column = (int(v) for v in members[i])
            outward = (exterior[nearest[i]] - points[i]) / margins[i]
            donor = int(closest_development[i])
            item = {"anchor": f"holdout-{len(anchors):02d}", "group": group,
                    "row": row, "column": column, "alpha": float(points[i, 0]),
                    "gamma": float(points[i, 1]), "atlas_period": int(q[row, column]),
                    "raw_margin": float(margins[i]), "crop_edge_lower_bound": float(edges[i]),
                    "nearest_non_nine_row": int(exterior_indices[nearest[i], 0]),
                    "nearest_non_nine_column": int(exterior_indices[nearest[i], 1]),
                    "outward": outward.tolist(), "inward": (-outward).tolist(),
                    "tangent": [-float(outward[1]), float(outward[0])],
                    "minimum_development_distance": float(halo_distance[i]),
                    "nearest_development_index": donor,
                    "nearest_development_anchor": str(development_anchors[donor].get("id", development_anchors[donor].get("anchor", donor))),
                    "phase_indices": list(PHASE_INDICES), "reference_samples": REFERENCE_SAMPLES}
            if "states" in grid:
                item["captured_atlas_state"] = np.asarray(grid["states"])[row, column].tolist()
            anchors.append(item)
    return {"experiment_id": "EXP-007", "seed": int(seed),
            "anchor_attempts": len(anchors), "count_per_stratum": int(count_per_stratum),
            "core_index": [int(v) for v in core], "atlas_shape": list(q.shape),
            "metric": "raw Euclidean connection weights; equal radius equals equal Frobenius weight change",
            "halo_and_spread_metric": "Euclidean coordinates normalized by the full observed focus rectangle widths",
            "normalization_widths": widths.tolist(), "exclusion_radius": float(exclusion_radius),
            "exclusion_boundary": "exclude normalized distance <= radius from any EXP-006 starting anchor",
            "phase_indices": list(PHASE_INDICES), "reference_samples": REFERENCE_SAMPLES,
            "phase_epoch_rule": "index * certified period / 2700; approximate odd eighths, not exact eighths",
            "selection_audit": {"core_component_cells": len(members),
                "crop_eligible_cells": int(crop_eligible.sum()),
                "halo_excluded_component_cells": int((~halo_eligible).sum()),
                "halo_excluded_crop_eligible_cells": int((crop_eligible & ~halo_eligible).sum()),
                "eligible_cells": len(eligible), "stratum_sizes": [len(s) for s in thirds],
                "development_anchor_count": len(development),
                "selected_minimum_development_distance": min(a["minimum_development_distance"] for a in anchors)},
            "scope": "New starting settings, phases and scored interventions conditional on a previously seen atlas. Halo does not exclude every earlier intervention destination; not unseen-map or population generalization.",
            "anchors": anchors}


def load_holdout_packet(evidence_directory, development_manifest_path, *,
                        count_per_stratum=4, seed=SEED, exclusion_radius=EXCLUSION_RADIUS):
    """Validate/hash the old source and development manifest, then select attempts."""
    directory = Path(evidence_directory)
    development_path = Path(development_manifest_path)
    old = load_anchor_packet(directory)
    development = json.loads(development_path.read_text())
    if development.get("experiment_id") != "EXP-006" or len(development.get("anchors", [])) != 12:
        raise ValueError("The complete twelve-anchor EXP-006 development manifest is required")
    rows = []
    for path in sorted((directory / "focus").glob("row-*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            rows.append({key: saved[key].copy() for key in ("alpha", "gamma", "period", "states")})
    grid = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    packet = select_holdout_anchors(grid, old["core_index"], development["anchors"],
                                    count_per_stratum=count_per_stratum, seed=seed,
                                    exclusion_radius=exclusion_radius)
    packet.update(source_experiment="EXP-004", source_files=old["source_files"],
                  source_freeze_commit=old["source_freeze_commit"],
                  development_manifest={"path": str(development_path),
                      "sha256": hashlib.sha256(development_path.read_bytes()).hexdigest(),
                      "bytes": development_path.stat().st_size})
    return packet
