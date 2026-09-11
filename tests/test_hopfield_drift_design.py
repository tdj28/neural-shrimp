"""Synthetic atlas/analysis audits. No new neural trajectories are integrated."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import hopfield_drift_design as design


def atlas():
    alpha = np.linspace(-0.05, 0.05, 41)
    gamma = np.linspace(-0.5, 0.5, 31)
    periods = np.zeros((31, 41), dtype=int)
    periods[4:27, 8:30] = 9
    periods[1:3, 1:4] = 9  # A disconnected island is ineligible.
    return {"alpha": np.tile(alpha, (len(gamma), 1)), "gamma": gamma,
            "period": periods, "states": np.zeros((*periods.shape, 3))}


def test_selection_is_deterministic_paired_and_raw_metric():
    grid = atlas()
    packet = design.select_anchors(grid, (15, 20))
    assert packet == design.select_anchors(grid, (15, 20))
    assert packet["anchor_attempts"] == 12
    assert packet["stratum_sizes"] == list(np.array_split(
        np.arange(packet["eligible_atlas_cells"]), 3)[i].size for i in range(3))
    assert len({(a["row"], a["column"]) for a in packet["anchors"]}) == 12
    for index, anchor in enumerate(packet["anchors"]):
        assert 4 <= anchor["row"] < 27
        assert anchor["raw_margin"] < anchor["crop_edge_lower_bound"]
        for key in ("outward", "inward", "tangent", "random_unit", "shuffled_outward"):
            assert np.linalg.norm(anchor[key]) == pytest.approx(1.0, abs=1e-14)
        assert np.dot(anchor["outward"], anchor["tangent"]) == pytest.approx(0.0, abs=1e-14)
        destination = np.array([anchor["alpha"], anchor["gamma"]]) + np.array(anchor["outward"]) * anchor["raw_margin"]
        i, j = anchor["nearest_non_nine_row"], anchor["nearest_non_nine_column"]
        np.testing.assert_allclose(destination, [grid["alpha"][i, j], grid["gamma"][i]], atol=1e-15)
        assert grid["period"][i, j] != 9
        assert anchor["shuffled_from"] == packet["anchors"][(index - 1) % 12]["anchor"]
        assert anchor["shuffled_from"] != anchor["anchor"]
    assert [a["group"] for a in packet["anchors"]] == (
        ["lower_margin"] * 4 + ["middle_margin"] * 4 + ["upper_margin"] * 4)


def test_selection_retains_crop_rule_and_rejects_missing_source_boundary():
    grid = atlas()
    grid["period"][4:27, :30] = 9
    packet = design.select_anchors(grid, (15, 20))
    assert all(a["raw_margin"] < a["crop_edge_lower_bound"] for a in packet["anchors"])
    grid["period"][:] = 9
    with pytest.raises(ValueError, match="no observed"):
        design.select_anchors(grid, (15, 20))
    grid = atlas()
    with pytest.raises(ValueError, match="period-nine"):
        design.select_anchors(grid, (0, 0))
    with pytest.raises(ValueError, match="Insufficient"):
        design.select_anchors(grid, (15, 20), count_per_stratum=1000)


def test_nearest_boundary_ties_are_lexicographic():
    axis = np.arange(-10., 11.)
    periods = np.zeros((21, 21), dtype=int)
    periods[5:16, 5:16] = 9
    grid = {"alpha": axis, "gamma": axis, "period": periods}
    packet = design.select_anchors(grid, (10, 10), count_per_stratum=1)
    for anchor in packet["anchors"]:
        candidates = np.argwhere(periods != 9)
        center = np.array([anchor["row"], anchor["column"]])
        squared = np.sum((candidates - center) ** 2, axis=1)
        expected = candidates[np.argmin(squared)]
        assert [anchor["nearest_non_nine_row"], anchor["nearest_non_nine_column"]] == expected.tolist()


def test_margin_policy_uses_old_pixel_scores_and_explicit_fallback():
    grid = atlas()
    anchor = design.select_anchors(grid, (15, 20))["anchors"][0]
    result = design.atlas_margin_direction(grid, anchor, .001)
    assert result == design.atlas_margin_direction(grid, anchor, .001)
    assert result["angle_count"] == 512
    assert not result["fallback_inward"]
    assert result["selected_angle_index"] == int(np.argmax(result["angle_scores"]))
    assert result["selected_score"] == max(result["angle_scores"])
    assert np.linalg.norm(result["direction"]) == pytest.approx(1.)
    assert np.linalg.norm(np.asarray(result["destination"]) - [anchor["alpha"], anchor["gamma"]]) == pytest.approx(.001)
    scores = np.asarray(result["angle_scores"])
    assert np.all(scores[~np.asarray(result["angle_inside_crop"])] == 0)
    assert np.all(scores[~np.asarray(result["angle_component_eligible"])] == 0)
    fallback = design.atlas_margin_direction(grid, anchor, 100.)
    assert fallback["fallback_inward"]
    assert fallback["selected_angle_index"] is None
    assert fallback["direction"] == anchor["inward"]
    assert fallback["selection_status"] == "all_scores_zero_inward_fallback"
    assert not any(fallback["angle_scores"])


def test_source_packet_binds_consumed_bytes(tmp_path):
    grid = atlas()
    (tmp_path / "focus").mkdir()
    paths = []
    for i in range(len(grid["gamma"])):
        path = tmp_path / "focus" / f"row-{i:03d}.npz"
        np.savez(path, alpha=grid["alpha"][i], gamma=grid["gamma"][i],
                 period=grid["period"][i], states=grid["states"][i])
        paths.append(path)
    (tmp_path / "selected.json").write_text(json.dumps({"core": {"row": 15, "column": 20}}))
    (tmp_path / "manifest.json").write_text(json.dumps({"experiment_id": "EXP-004", "grids": [
        {"name": "focus", "alpha": [-.05, .05, 41], "gamma": [-.5, .5, 31]}]}))
    paths += [tmp_path / "manifest.json", tmp_path / "selected.json"]
    receipt = {"source_commit": "f" * 40, "status": "complete",
               "artifacts": [{"path": p.relative_to(tmp_path).as_posix(),
                              "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                              "bytes": p.stat().st_size} for p in paths]}
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    packet = design.load_anchor_packet(tmp_path)
    assert packet["source_freeze_commit"] == "f" * 40
    assert len(packet["source_files"]) == len(paths) + 1
    (tmp_path / "selected.json").write_text(json.dumps({"core": {"row": 15, "column": 19}}))
    with pytest.raises(ValueError, match="receipt mismatch"):
        design.load_anchor_packet(tmp_path)
    (tmp_path / "focus/row-030.npz").unlink()
    with pytest.raises(ValueError, match="row census"):
        design.load_anchor_packet(tmp_path)


def records():
    result = []
    for anchor in ("a", "b"):
        for dt in (.01, .005):
            for phase in (0, 1):
                for radius in (.001, .004):
                    for arm in design.ARMS:
                        value = {"atlas_margin": .2, "local_plus": .4, "local_minus": .8,
                                 "random_plus": .5, "random_minus": .7,
                                 "shuffled_inward": .9}.get(arm, 1.)
                        value += .1 if anchor == "b" and arm == "atlas_margin" else 0.
                        result.append({"anchor": anchor, "phase": phase, "radius": radius,
                                       "arm": arm, "dt": dt, "status": "complete",
                                       "wave_metrics": {"aligned_rms": {"aggregate": value}}})
                result.append({"anchor": anchor, "phase": phase, "radius": 0., "arm": "zero",
                               "dt": dt, "status": "complete",
                               "wave_metrics": {"aligned_rms": {"aggregate": 0.}}})
    return result


def summary(rows):
    return design.summarize_trials(rows, ["a", "b", "failed-baseline"], ["a", "b"],
                                   phases=(0, 1), radii=(.001, .004), bootstrap_samples=1000)


def test_paired_analysis_averages_signs_then_phases_radii_and_anchors():
    report = summary(records())
    assert report == summary(list(reversed(records())))
    assert report["complete"]
    assert report["anchor_coverage"] == pytest.approx(2 / 3)
    assert report["ineligible_anchors"] == ["failed-baseline"]
    assert report["planned_eligible_trials"] == 2 * 2 * (2 * 11 + 1) * 2
    for panel in report["panels"]:
        local = panel["equal_radius_average"]["local_symmetric"]
        assert local["defined"]
        assert local["anchor_effects"] == pytest.approx({"a": .4, "b": .3})
        assert local["mean_benefit"] == pytest.approx(.35)
        assert local["ci95"] == pytest.approx([.3, .4])
        assert panel["equal_radius_average"]["random_symmetric"]["mean_benefit"] == pytest.approx(.35)


def test_missing_trial_never_becomes_success_only_average():
    rows = records()
    missing = rows.pop(0)  # An outward diagnostic also invalidates the panel.
    report = summary(rows)
    assert not report["complete"]
    assert len(report["missing_trials"]) == 1
    coarse, fine = report["panels"]
    assert coarse["dt"] == missing["dt"]
    assert not coarse["per_radius"][0]["complete"]
    assert coarse["per_radius"][1]["complete"]
    assert not coarse["equal_radius_average"]["local_symmetric"]["defined"]
    assert fine["equal_radius_average"]["local_symmetric"]["defined"]


def test_failed_nonfinite_and_zero_control_trials_are_retained():
    rows = records()
    for row in rows:
        if row["arm"] == "zero" and row["dt"] == .005:
            row["status"] = "native_failure"
            break
    report = summary(rows)
    assert report["status_counts"]["native_failure"] == 1
    assert len(report["invalid_trials"]) == 1
    assert not report["panels"][1]["zero_controls_complete"]
    assert all(not p["complete"] for p in report["panels"][1]["per_radius"])
    for value in (float("nan"), -0.1, 2.1):
        rows = records()
        rows[0]["wave_metrics"]["aligned_rms"]["aggregate"] = value
        assert len(summary(rows)["invalid_trials"]) == 1


def test_duplicate_or_unplanned_trials_rejected_and_no_eligible_case_is_explicit():
    rows = records()
    with pytest.raises(ValueError, match="Duplicate"):
        summary(rows + [copy.deepcopy(rows[0])])
    rows[0]["anchor"] = "failed-baseline"
    with pytest.raises(ValueError, match="Unplanned"):
        summary(rows)
    rows = records()
    rows[0]["phase"] = .1
    with pytest.raises(ValueError, match="integer indices"):
        summary(rows)
    report = design.summarize_trials([], ["a", "b"], [])
    assert report["anchor_coverage"] == 0.
    assert report["planned_eligible_trials"] == 0
    assert not report["panels"][0]["equal_radius_average"]["local_symmetric"]["defined"]
