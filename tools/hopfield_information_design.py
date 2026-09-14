"""Outcome-free EXP-008 spatial blocks and old-atlas parameter features.

No trajectory, waveform metric, eligibility outcome or fitted predictor enters
this module. Distances for geometry use native connection-weight coordinates;
rectangle normalization is used only for spatial selection. A signed pixel
margin is not a continued bifurcation boundary or a state-space basin radius.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from hopfield_drift_design import _grid_coordinates, load_anchor_packet

PHASE_INDICES = (675, 1350)
REFERENCE_SAMPLES = 2700
EXCLUSION_RADIUS = .04
BLOCK_COUNT = 12
GEOMETRY_FEATURE_NAMES = (
    "anchor_raw_margin", "destination_signed_pixel_margin",
    "destination_in_core_component", "outward_projection",
    "destination_inside_crop", "destination_margin_crop_censored",
)


@dataclass(frozen=True)
class AtlasGeometry:
    alpha: np.ndarray
    gamma: np.ndarray
    period: np.ndarray
    component: np.ndarray
    signed_margin: np.ndarray
    exterior_indices: np.ndarray
    exterior: np.ndarray
    exterior_tree: cKDTree
    core_index: tuple


def build_atlas_geometry(grid, core_index):
    """Cache nearest-opposite-class pixel distances in raw weight units.

    The positive class is the core's 8-connected nine-peak component; another
    disconnected nine-peak component is outside this particular component.
    Both classes consist only of observed pixels. No crop padding is added.
    """
    q, aa, gg = _grid_coordinates(grid)
    core = tuple(core_index)
    if (len(core) != 2 or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in core)
            or not (0 <= core[0] < q.shape[0] and 0 <= core[1] < q.shape[1])
            or q[core] != 9):
        raise ValueError("The core must be an existing nine-peak pixel")
    components, _ = label(q == 9, structure=np.ones((3, 3), dtype=int))
    inside = components == components[core]
    outside_indices = np.argwhere(~inside)
    if len(outside_indices) == 0:
        raise ValueError("An observed pixel outside the core component is required")
    points = np.column_stack((aa.ravel(), gg.ravel()))
    exterior = np.column_stack((aa[~inside], gg[~inside]))
    interior = np.column_stack((aa[inside], gg[inside]))
    exterior_tree = cKDTree(exterior)
    outside_distance = exterior_tree.query(points)[0].reshape(q.shape)
    inside_distance = cKDTree(interior).query(points)[0].reshape(q.shape)
    signed = np.where(inside, outside_distance, -inside_distance)
    arrays = [aa[0].copy(), gg[:, 0].copy(), q.copy(), inside.copy(),
              signed, outside_indices, exterior]
    for value in arrays:
        value.setflags(write=False)
    return AtlasGeometry(*arrays, exterior_tree, tuple(int(x) for x in core))


def _nearest_exterior(context, point):
    distance, _ = context.exterior_tree.query(point)
    nearby = context.exterior_tree.query_ball_point(point, np.nextafter(distance, np.inf) + 1e-14)
    squared = np.sum((context.exterior[nearby] - point) ** 2, axis=1)
    minimum = float(squared.min())
    tolerance = max(1e-28, 32 * np.finfo(float).eps * minimum)
    index = min(int(i) for i, d in zip(nearby, squared) if d <= minimum + tolerance)
    delta = context.exterior[index] - point
    return float(np.linalg.norm(delta)), index, delta / np.linalg.norm(delta)


def _edge(context, points):
    points = np.asarray(points)
    return np.minimum.reduce((points[..., 0] - context.alpha[0], context.alpha[-1] - points[..., 0],
                              points[..., 1] - context.gamma[0], context.gamma[-1] - points[..., 1]))


def _anchor_point(context, anchor):
    row, column = anchor["row"], anchor["column"]
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in (row, column))
            or not (0 <= row < len(context.gamma) and 0 <= column < len(context.alpha))
            or not context.component[row, column]):
        raise ValueError("An anchor must identify a pixel in the core component")
    point = np.array([anchor["alpha"], anchor["gamma"]], dtype=float)
    if not np.array_equal(point, [context.alpha[column], context.gamma[row]]):
        raise ValueError("Anchor coordinates must equal its declared atlas pixel")
    return point


def geometry_features(context, anchor, direction, radius):
    """Return six fixed features and explicit coverage/censoring diagnostics.

    Destination membership and signed margin use its nearest observed pixel,
    with exact half-cell ties at the lower axis index. The magnitude is that
    pixel's raw distance to an observed pixel of the opposite component class.
    Outside the crop, membership and distance are unavailable: numeric zeros
    are paired with inside_crop=0 and margin_crop_censored=1. An in-crop margin
    reaching the destination's nearest crop edge is flagged as censored, since
    unobserved geometry could be nearer. Censored values are never certificates.
    """
    point = _anchor_point(context, anchor)
    direction = np.asarray(direction, dtype=float)
    if (direction.shape != (2,) or not np.isfinite(direction).all()
            or not np.isclose(np.linalg.norm(direction), 1., rtol=0, atol=1e-12)):
        raise ValueError("A finite native-unit direction is required")
    if isinstance(radius, (bool, np.bool_)) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("A positive finite native-weight radius is required")
    margin, nearest, outward = _nearest_exterior(context, point)
    destination = point + float(radius) * direction
    crop_edge = float(_edge(context, destination))
    inside_crop = crop_edge >= 0.
    column = int(np.argmin(abs(context.alpha - destination[0])))
    row = int(np.argmin(abs(context.gamma - destination[1])))
    signed = float(context.signed_margin[row, column]) if inside_crop else None
    member = bool(context.component[row, column]) if inside_crop else None
    censored = not inside_crop or abs(signed) >= crop_edge
    values = [margin, signed if signed is not None else 0., float(bool(member)),
              float(np.clip(np.dot(direction, outward), -1., 1.)), float(inside_crop), float(censored)]
    return {"feature_names": list(GEOMETRY_FEATURE_NAMES), "values": values,
            "diagnostics": {"destination": destination.tolist(), "nearest_pixel_row": row,
                "nearest_pixel_column": column, "destination_inside_crop": inside_crop,
                "destination_in_core_component": member, "destination_signed_pixel_margin": signed,
                "destination_crop_edge_lower_bound": crop_edge, "margin_crop_censored": bool(censored),
                "anchor_nearest_non_core_pixel": context.exterior_indices[nearest].tolist(),
                "anchor_outward": outward.tolist(),
                "scope": "Observed old-atlas parameter geometry only; no target outcomes or exact boundary guarantee"}}


def _partition(points, members, indices, count, path="", splits=None):
    """Recursive weighted-median cuts with complete coordinate ties intact."""
    if len(indices) < count:
        raise ValueError("Insufficient block members; no fallback or replacement")
    if count == 1:
        return [(path, np.asarray(indices, dtype=int))]
    left_count = count // 2
    right_count = count - left_count
    # Prefer the widest normalized axis; alpha breaks equal-span ties. If
    # it cannot furnish both requested numbers of leaves, try the other axis.
    spans = np.ptp(points[indices], axis=0)
    for axis in sorted(range(2), key=lambda i: (-float(spans[i]), i)):
        ordered = sorted(indices, key=lambda i: (float(points[i, axis]), int(members[i, 0]), int(members[i, 1])))
        cuts = [n for n in range(left_count, len(ordered) - right_count + 1)
                if points[ordered[n-1], axis] < points[ordered[n], axis]]
        if not cuts:
            continue
        cut = min(cuts, key=lambda n: (abs(n * count - len(ordered) * left_count), n))
        threshold = float((points[ordered[cut-1], axis] + points[ordered[cut], axis]) / 2)
        splits.append({"path": path, "axis": ["normalized_alpha", "normalized_gamma"][axis],
                       "threshold": threshold, "left_rule": "coordinate <= threshold",
                       "left_cells": cut, "right_cells": len(ordered)-cut,
                       "left_leaves": left_count, "right_leaves": right_count})
        return (_partition(points, members, ordered[:cut], left_count, path+"L", splits)
                + _partition(points, members, ordered[cut:], right_count, path+"R", splits))
    raise ValueError("Spatial coordinates cannot provide all blocks; no fallback or replacement")


def _anchor_record(context, row, column, identity):
    point = np.array([context.alpha[column], context.gamma[row]])
    margin, nearest, outward = _nearest_exterior(context, point)
    return {"id": identity, "anchor": identity, "row": int(row), "column": int(column),
            "alpha": float(point[0]), "gamma": float(point[1]), "atlas_period": int(context.period[row, column]),
            "raw_margin": margin, "crop_edge_lower_bound": float(_edge(context, point)),
            "nearest_non_core_row": int(context.exterior_indices[nearest, 0]),
            "nearest_non_core_column": int(context.exterior_indices[nearest, 1]),
            "outward": outward.tolist(), "inward": (-outward).tolist(),
            "tangent": [-float(outward[1]), float(outward[0])],
            "phase_indices": list(PHASE_INDICES), "reference_samples": REFERENCE_SAMPLES}


def select_information_anchors(grid, core_index, prior_anchors, *, block_count=BLOCK_COUNT,
                               exclusion_radius=EXCLUSION_RADIUS):
    """Partition all eligible old pixels into blocks, then select one per block.

    Eligibility is core membership, observed opposite-class margin strictly
    below the nearest crop-edge distance, and normalized distance > .04 from
    every EXP006/007 origin. Split the widest normalized coordinate recursively
    near its median, weighted by the requested number of descendant leaves.
    Equal-coordinate pixels stay together. Each leaf chooses the pixel closest
    to its normalized centroid, with row/column tie breaking. No outcome-based
    replacement or reduction of blocks/halos is allowed. Holes from the original
    component and prior halos remain; compact spatial partitioning does not
    imply statistical independence or topological connectedness of the pixels.
    """
    if isinstance(block_count, (bool, np.bool_)) or not isinstance(block_count, (int, np.integer)) or block_count < 1:
        raise ValueError("A positive integer block count is required")
    if isinstance(exclusion_radius, (bool, np.bool_)) or not np.isfinite(exclusion_radius) or exclusion_radius <= 0:
        raise ValueError("A positive finite exclusion radius is required")
    context = build_atlas_geometry(grid, core_index)
    prior = np.asarray([[a["alpha"], a["gamma"]] for a in prior_anchors], dtype=float)
    if prior.ndim != 2 or prior.shape[1] != 2 or len(prior) < 1 or not np.isfinite(prior).all():
        raise ValueError("Finite prior origin coordinates are required")
    if len(np.unique(prior, axis=0)) != len(prior):
        raise ValueError("Prior origin coordinates must be unique")
    widths = np.array([np.ptp(context.alpha), np.ptp(context.gamma)])
    origin = np.array([context.alpha[0], context.gamma[0]])
    members = np.argwhere(context.component)
    points = np.column_stack((context.alpha[members[:, 1]], context.gamma[members[:, 0]]))
    margins = context.signed_margin[tuple(members.T)]
    edge = _edge(context, points)
    prior_distances = np.linalg.norm((points[:, None, :] - prior[None, :, :]) / widths, axis=2)
    nearest_prior = np.argmin(prior_distances, axis=1)
    minimum_prior_distance = prior_distances[np.arange(len(points)), nearest_prior]
    crop_eligible = margins < edge
    halo_eligible = minimum_prior_distance > exclusion_radius
    eligible = np.flatnonzero(crop_eligible & halo_eligible)
    normalized = (points-origin)/widths
    splits = []
    leaves = _partition(normalized, members, eligible.tolist(), int(block_count), splits=splits)
    anchors, blocks = [], []
    for number, (path, indices) in enumerate(leaves):
        centroid = np.mean(normalized[indices], axis=0)
        chosen = min(indices, key=lambda i: (float(np.sum((normalized[i]-centroid)**2)),
                                            int(members[i, 0]), int(members[i, 1])))
        row, column = members[chosen]
        block_id = f"test-block-{number:02d}"
        a = _anchor_record(context, int(row), int(column), f"information-{number:02d}")
        a.update(block_id=block_id, group=block_id,
                 minimum_development_distance=float(minimum_prior_distance[chosen]),
                 nearest_development_index=int(nearest_prior[chosen]))
        anchors.append(a)
        blocks.append({"id": block_id, "split_path": path, "member_count": len(indices),
                       "members": sorted(members[indices].tolist()),
                       "normalized_centroid": centroid.tolist(),
                       "normalized_bounds": [normalized[indices].min(axis=0).tolist(), normalized[indices].max(axis=0).tolist()],
                       "representative_row": int(row), "representative_column": int(column), "anchor_id": a["id"]})
    pair = np.linalg.norm((np.array([[a["alpha"], a["gamma"]] for a in anchors])[:, None]
                          - np.array([[a["alpha"], a["gamma"]] for a in anchors])[None, :]) / widths, axis=2)
    np.fill_diagonal(pair, np.inf)
    return {"experiment_id": "EXP-008", "anchors": anchors, "blocks": blocks,
            "core_index": list(context.core_index), "atlas_shape": list(context.period.shape),
            "anchor_attempts": len(anchors), "block_count": int(block_count),
            "phase_indices": list(PHASE_INDICES), "reference_samples": REFERENCE_SAMPLES,
            "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
            "normalization_origin": origin.tolist(), "normalization_widths": widths.tolist(),
            "exclusion_radius": float(exclusion_radius), "partition_splits": splits,
            "selection_audit": {"core_component_cells": len(members), "crop_eligible_cells": int(crop_eligible.sum()),
                "halo_excluded_component_cells": int((~halo_eligible).sum()),
                "halo_excluded_crop_eligible_cells": int((crop_eligible & ~halo_eligible).sum()),
                "eligible_cells": len(eligible), "prior_origins": len(prior),
                "block_sizes": [b["member_count"] for b in blocks],
                "selected_minimum_development_distance": min(a["minimum_development_distance"] for a in anchors),
                "selected_minimum_pairwise_distance": float(pair.min()) if len(anchors)>1 else None},
            "scope": "New blocked starting settings conditional on a previously seen atlas; halos exclude prior origins, not all prior destinations. Blocks are descriptive resampling units, not independent populations.",
            "partition_rule": "Recursive widest-normalized-axis weighted median; coordinate ties intact; alpha wins axis ties; lower cut wins split ties; centroid-nearest representative with row/column ties; no replacements"}


def load_information_packet(evidence_directory, exp006_manifest_path, exp007_manifest_path):
    """Bind the immutable atlas and both prior anchor manifests, without scores."""
    directory = Path(evidence_directory)
    old = load_anchor_packet(directory)
    previous, manifests = [], []
    for expected, name in (("EXP-006", exp006_manifest_path), ("EXP-007", exp007_manifest_path)):
        path = Path(name)
        manifest = json.loads(path.read_text())
        if manifest.get("experiment_id") != expected or len(manifest.get("anchors", [])) != 12:
            raise ValueError(f"The complete twelve-anchor {expected} manifest is required")
        previous.append(manifest["anchors"])
        manifests.append({"experiment_id": expected, "path": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    rows = []
    for path in sorted((directory/"focus").glob("row-*.npz")):
        with np.load(path, allow_pickle=False) as raw:
            rows.append({key: raw[key].copy() for key in ("alpha", "gamma", "period")})
    grid = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    packet = select_information_anchors(grid, old["core_index"], previous[0]+previous[1])
    context = build_atlas_geometry(grid, old["core_index"])
    development = []
    for index, prior in enumerate(previous[1]):
        _anchor_point(context, prior)
        identity = str(prior.get("id", prior.get("anchor", f"holdout-{index:02d}")))
        a = _anchor_record(context, prior["row"], prior["column"], identity)
        a.update(block_id=f"development-{index:02d}", group="development", source_experiment="EXP-007")
        development.append(a)
    packet.update(development_anchors=development, source_experiment="EXP-004",
                  source_files=old["source_files"], source_freeze_commit=old["source_freeze_commit"],
                  prior_manifests=manifests)
    return packet
