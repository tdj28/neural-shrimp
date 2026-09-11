"""Synthetic holdout-selector checks; no target trajectories or certificates."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hopfield_drift_design as previous
import hopfield_holdout_design as holdout


def fixture():
    alpha = np.linspace(-.05, .05, 41)
    gamma = np.linspace(-.5, .5, 31)
    period = np.zeros((31, 41), dtype=int)
    period[4:27, 8:30] = 9
    period[1:3, 1:4] = 9
    grid = dict(alpha=np.tile(alpha, (31, 1)), gamma=gamma, period=period,
                states=np.zeros((31, 41, 3)))
    development = previous.select_anchors(grid, (15, 20))["anchors"]
    return grid, development


def test_new_attempts_are_deterministic_outside_every_development_halo():
    grid, development = fixture()
    snapshot = {k: v.copy() for k, v in grid.items()}
    packet = holdout.select_holdout_anchors(grid, (15, 20), development)
    assert packet == holdout.select_holdout_anchors(grid, (15, 20), development)
    assert packet["seed"] == 20260912 and packet["anchor_attempts"] == 12
    assert len({(a["row"], a["column"]) for a in packet["anchors"]}) == 12
    assert [a["group"] for a in packet["anchors"]] == (
        ["lower_margin"] * 4 + ["middle_margin"] * 4 + ["upper_margin"] * 4)
    old = np.asarray([[a["alpha"], a["gamma"]] for a in development])
    widths = np.asarray(packet["normalization_widths"])
    outside = np.argwhere(grid["period"] != 9)
    outside_coordinates = np.asarray([[grid["alpha"][i, j], grid["gamma"][i]] for i, j in outside])
    for anchor in packet["anchors"]:
        point = np.asarray([anchor["alpha"], anchor["gamma"]])
        distances = np.linalg.norm((old - point) / widths, axis=1)
        assert distances.min() > .04
        assert anchor["minimum_development_distance"] == pytest.approx(distances.min())
        assert anchor["nearest_development_index"] == int(np.argmin(distances))
        assert anchor["raw_margin"] == pytest.approx(np.linalg.norm(outside_coordinates-point, axis=1).min())
        assert anchor["raw_margin"] < anchor["crop_edge_lower_bound"]
        assert 4 <= anchor["row"] < 27
        for direction in ("outward", "inward", "tangent"):
            assert np.linalg.norm(anchor[direction]) == pytest.approx(1., abs=1e-14)
    assert packet["selection_audit"]["selected_minimum_development_distance"] > .04
    for key in grid:
        np.testing.assert_array_equal(grid[key], snapshot[key])


def test_selection_ignores_all_development_outcomes_and_never_relabels_the_atlas():
    grid, development = fixture()
    original = holdout.select_holdout_anchors(grid, (15, 20), development)
    altered = copy.deepcopy(development)
    for index, a in enumerate(altered):
        a.update(waveform_error=1000 * index, eligible=False, certificate={"verified": False})
    assert original == holdout.select_holdout_anchors(grid, (15, 20), altered)
    before = previous.select_anchors(grid, (15, 20))
    holdout.select_holdout_anchors(grid, (15, 20), development, exclusion_radius=.12)
    assert before == previous.select_anchors(grid, (15, 20))


def test_closed_halo_boundary_and_census_are_exact():
    axis = np.linspace(-1., 1., 17)
    period = np.zeros((17, 17), dtype=int)
    period[3:14, 3:14] = 9
    grid = dict(alpha=axis, gamma=axis, period=period)
    development = [{"anchor": "center", "alpha": 0., "gamma": 0.}]
    packet = holdout.select_holdout_anchors(grid, (8, 8), development,
                                          count_per_stratum=1, exclusion_radius=.25)
    centers = np.argwhere(period == 9)
    distances = np.linalg.norm(np.column_stack([axis[centers[:, 1]], axis[centers[:, 0]]]) / 2, axis=1)
    assert np.any(distances == .25)
    assert packet["selection_audit"]["halo_excluded_component_cells"] == int(np.sum(distances <= .25))
    assert all(a["minimum_development_distance"] > .25 for a in packet["anchors"])
    assert sum(packet["selection_audit"]["stratum_sizes"]) == packet["selection_audit"]["eligible_cells"]


def test_phases_are_new_exact_sample_indices_and_not_claimed_exact_eighths():
    grid, development = fixture()
    packet = holdout.select_holdout_anchors(grid, (15, 20), development)
    assert packet["phase_indices"] == [337, 1012, 1687, 2362]
    assert packet["reference_samples"] == 2700
    assert not set(packet["phase_indices"]) & {0, 675, 1350, 2025}
    np.testing.assert_array_equal(np.diff(packet["phase_indices"]), [675] * 3)
    assert not np.array_equal(np.array(packet["phase_indices"]) / 2700, [.125, .375, .625, .875])
    assert "not exact eighths" in packet["phase_epoch_rule"]


def test_insufficient_candidates_and_invalid_development_fail_without_fallback():
    grid, development = fixture()
    with pytest.raises(ValueError, match="no fallback"):
        holdout.select_holdout_anchors(grid, (15, 20), development, exclusion_radius=10.)
    with pytest.raises(ValueError, match="no fallback"):
        holdout.select_holdout_anchors(grid, (15, 20), development, count_per_stratum=1000)
    with pytest.raises(ValueError, match="unique"):
        holdout.select_holdout_anchors(grid, (15, 20), development + [development[0]])
    with pytest.raises(ValueError, match="Finite development"):
        holdout.select_holdout_anchors(grid, (15, 20), [])


def test_source_packet_binds_prior_data_and_development_manifest(tmp_path):
    grid, development = fixture()
    evidence = tmp_path / "EXP004"
    (evidence / "focus").mkdir(parents=True)
    files = []
    for i in range(len(grid["gamma"])):
        path = evidence / "focus" / f"row-{i:03d}.npz"
        np.savez(path, alpha=grid["alpha"][i], gamma=grid["gamma"][i],
                 period=grid["period"][i], states=grid["states"][i])
        files.append(path)
    (evidence / "manifest.json").write_text(json.dumps({"experiment_id": "EXP-004", "grids": [
        {"name": "focus", "alpha": [-.05, .05, 41], "gamma": [-.5, .5, 31]}]}))
    (evidence / "selected.json").write_text(json.dumps({"core": {"row": 15, "column": 20}}))
    files += [evidence / "manifest.json", evidence / "selected.json"]
    (evidence / "receipt.json").write_text(json.dumps({"status": "complete", "source_commit": "a"*40,
        "artifacts": [{"path": str(p.relative_to(evidence)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                       "bytes": p.stat().st_size} for p in files]}))
    manifest = tmp_path / "EXP006-plan.json"
    manifest.write_text(json.dumps({"experiment_id": "EXP-006", "anchors": development}))
    packet = holdout.load_holdout_packet(evidence, manifest)
    assert packet["source_experiment"] == "EXP-004"
    assert packet["source_freeze_commit"] == "a"*40
    assert packet["development_manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert len(packet["source_files"]) == len(files) + 1
    manifest.write_text(json.dumps({"experiment_id": "EXP-006", "anchors": development[:-1]}))
    with pytest.raises(ValueError, match="complete twelve-anchor"):
        holdout.load_holdout_packet(evidence, manifest)
