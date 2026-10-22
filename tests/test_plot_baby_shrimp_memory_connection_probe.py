"""Receipt-only plot validation with synthetic audited points; no rendering."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from test_baby_shrimp_memory_connection_probe_finish import evidence, write_json
import baby_shrimp_memory_connection_probe as driver
import baby_shrimp_memory_connection_probe_finish as finish
import plot_baby_shrimp_memory_connection_probe as plotter


def audited_receipt(*, partial=False):
    result, protocol, raw, history = evidence()
    if partial:
        result["complete"] = False
        result["roots"] = history[0]
    measurements = finish.recompute_measurements(result, protocol, raw)
    plot_data = measurements.pop("plot_data")
    # The finisher publishes phase numbers in qualified-point summaries, not
    # the raw point.json phase dictionaries used by the synthetic evaluator.
    qualified = [{**copy.deepcopy(point), "phases": [phase["phase"] for phase in point["phases"]]}
                 for point in raw.values()]
    return dict(id="BSC4-direct-connection-offline-audit", complete=result["complete"],
                qualified_points=qualified, measurements=measurements,
                plot_data=plot_data), protocol["plan"]


@pytest.mark.parametrize("partial", [False, True])
def test_audited_observations_cross_link_without_raw_arrays_or_rendering(partial):
    receipt, plan = audited_receipt(partial=partial)
    points = plotter.validate_plot_data(receipt, plan)
    assert points == {row["index"]: row for row in receipt["qualified_points"]}
    assert len(receipt["plot_data"]["roots"]) == (1 if partial else 6)
    if partial:
        assert receipt["plot_data"]["roots"][0]["cyclic_point_index"] is None


@pytest.mark.parametrize("change", ["unknown index", "coordinate", "beta", "boolean index", "duplicate index"])
def test_plotted_observations_require_the_correct_qualified_point(change):
    receipt, plan = audited_receipt()
    sample = receipt["plot_data"]["candidate_tilt"]["samples"][0]
    point = next(row for row in receipt["qualified_points"] if row["index"] == sample["point_index"])
    if change == "unknown index":
        sample["point_index"] = 100000
    elif change == "coordinate":
        point["coordinate"][0] += .001
    elif change == "beta":
        point["beta"] += .001
    elif change == "boolean index":
        point["index"] = True
    else:
        receipt["qualified_points"].append(copy.deepcopy(point))
    with pytest.raises(ValueError):
        plotter.validate_plot_data(receipt, plan)


@pytest.mark.parametrize("qualified", [False, None, 1, "true", "missing"])
def test_point_qualification_must_be_explicit_boolean_true(qualified):
    receipt, plan = audited_receipt()
    point = receipt["qualified_points"][0]
    if qualified == "missing":
        del point["qualified"]
    else:
        point["qualified"] = qualified
    with pytest.raises(ValueError, match="qualified"):
        plotter.validate_plot_data(receipt, plan)


@pytest.mark.parametrize("change", ["radius difference", "direction", "nonunit direction", "sample order", "center link"])
def test_candidate_tilt_values_and_geometry_cannot_drift_from_the_audit(change):
    receipt, plan = audited_receipt()
    tilt = receipt["plot_data"]["candidate_tilt"]
    if change == "radius difference":
        tilt["samples"][0]["radius_difference"] += 1e-8
    elif change == "direction":
        tilt["direction"] = [0., 1.]
    elif change == "nonunit direction":
        tilt["direction"] = [2., 0.]
    elif change == "sample order":
        tilt["samples"].reverse()
    else:
        receipt["measurements"]["candidate_tilt"]["center_point_index"] += 1
    with pytest.raises(ValueError):
        plotter.validate_plot_data(receipt, plan)


@pytest.mark.parametrize("change", ["sample order", "bracket count", "direction", "label", "case order"])
def test_axis_sample_order_sign_counts_and_direction_are_checked(change):
    receipt, plan = audited_receipt()
    axis = receipt["plot_data"]["axes"][0]["axes"][0]
    if change == "sample order":
        axis["samples"].reverse()
    elif change == "bracket count":
        receipt["measurements"]["axes"][0]["negative_axis_brackets"] += 1
    elif change == "direction":
        axis["direction"] = [0., 1.]
    elif change == "label":
        axis["label"] = "positive curvature"
    else:
        receipt["plot_data"]["axes"].reverse()
    with pytest.raises(ValueError):
        plotter.validate_plot_data(receipt, plan)


@pytest.mark.parametrize("change", ["primary residual", "cyclic residual", "basis", "link", "coordinate", "beta", "status"])
def test_root_residual_basis_and_measurement_links_are_checked(change):
    receipt, plan = audited_receipt()
    root = receipt["plot_data"]["roots"][0]
    if change in ("primary residual", "cyclic residual"):
        key = "primary_point_index" if change == "primary residual" else "cyclic_point_index"
        point = next(row for row in receipt["qualified_points"] if row["index"] == root[key])
        point["tau"] = 2e-10
    elif change == "basis":
        root["slice_basis"] = np.eye(2).tolist()
    elif change == "link":
        root["primary_point_index"] += 1
    elif change == "coordinate":
        root["center"][0] += .001
    elif change == "beta":
        root["beta"] = .006
    else:
        root["status"] = "invented stronger verification"
    with pytest.raises(ValueError):
        plotter.validate_plot_data(receipt, plan)


@pytest.mark.parametrize("phases", [None, [], [0], [0, 6], [0, 12, 6], [0, 6, 12, 18], "missing"])
def test_cyclic_root_label_requires_all_three_phases_in_their_saved_order(phases):
    receipt, plan = audited_receipt()
    index = receipt["plot_data"]["roots"][0]["cyclic_point_index"]
    point = next(row for row in receipt["qualified_points"] if row["index"] == index)
    if phases == "missing":
        del point["phases"]
    else:
        point["phases"] = phases
    with pytest.raises(ValueError, match="cyclic|phase"):
        plotter.validate_plot_data(receipt, plan)


def test_empty_audited_payload_does_not_produce_a_figure():
    receipt, plan = audited_receipt()
    receipt["plot_data"].update(candidate_tilt=None, axes=[], roots=[])
    receipt["measurements"].update(candidate_tilt={"status": "untested"}, axes=[], roots=[])
    with pytest.raises(ValueError, match="no audited observations"):
        plotter.validate_plot_data(receipt, plan)


def bound_receipt(tmp_path, monkeypatch):
    receipt, plan = audited_receipt()
    plan.update(model_revision="f"*40, start=[.073, -.15], coordinate_scales=[2.5e-6, 7.5e-6],
                anchor_directory="synthetic-anchor", anchor_hashes={"point.json": "1"*64},
                intervention="synthetic memory intervention")
    execution, analysis = "a"*40, "b"*40
    plan_path = "docs/probe-plan.json"
    analysis_paths = {plotter.FINISHER, "tools/baby_shrimp_memory_finish.py",
                      "tools/baby_shrimp_memory_orbit_alignment.py",
                      "tools/baby_shrimp_memory_determinant_gradient.py"}
    blobs = {(execution, path): ("synthetic execution source "+path).encode() for path in driver.SOURCES}
    blobs[(execution, plotter.DRIVER)] = ("SOURCES = "+repr(driver.SOURCES)+"\n").encode()
    blobs.update({(analysis, path): ("synthetic analysis source "+path).encode() for path in analysis_paths})
    blobs[(execution, plan_path)] = json.dumps(plan).encode()
    receipt.update(execution_source_commit=execution, analysis_commit=analysis,
                   execution_source_sha256={path: hashlib.sha256(blobs[(execution, path)]).hexdigest()
                                            for path in driver.SOURCES},
                   analysis_source_sha256={path: hashlib.sha256(blobs[(analysis, path)]).hexdigest()
                                           for path in analysis_paths},
                   plan_path=plan_path, plan_sha256=hashlib.sha256(blobs[(execution, plan_path)]).hexdigest(),
                   scientific_metadata={key: plan[key] for key in plotter.METADATA})
    calls = []
    def git_bytes(commit, path):
        calls.append((commit, path))
        if (commit, path) not in blobs:
            raise ValueError("unexpected frozen Git input")
        return blobs[(commit, path)]
    monkeypatch.setattr(plotter, "git_bytes", git_bytes)
    path = tmp_path/"audited-receipt.json"
    write_json(path, receipt)
    return path, receipt, plan, blobs, calls


def test_receipt_loader_binds_the_auditor_driver_literal_dependencies_and_plan(tmp_path, monkeypatch):
    path, receipt, _, blobs, calls = bound_receipt(tmp_path, monkeypatch)
    loaded, points = plotter.load_receipt(path, plotter.digest(path))
    assert loaded == receipt
    assert len(points) == len(receipt["qualified_points"])
    assert set(calls) == set(blobs)
    assert len(receipt["execution_source_sha256"]) == 16
    assert len(receipt["analysis_source_sha256"]) == 4


@pytest.mark.parametrize("change", ["wrong digest", "file changed"])
def test_loader_refuses_receipt_hash_mismatch_before_reading_git(tmp_path, monkeypatch, change):
    path, receipt, _, _, calls = bound_receipt(tmp_path, monkeypatch)
    expected = plotter.digest(path)
    if change == "wrong digest":
        expected = "0"*64
    else:
        receipt["complete"] = False
        write_json(path, receipt)
    with pytest.raises(ValueError, match="receipt hash"):
        plotter.load_receipt(path, expected)
    assert calls == []


@pytest.mark.parametrize("path", [plotter.FINISHER, "tools/baby_shrimp_memory_finish.py",
                                 "tools/baby_shrimp_memory_orbit_alignment.py",
                                 "tools/baby_shrimp_memory_determinant_gradient.py"])
def test_loader_requires_each_independent_auditor_source(tmp_path, monkeypatch, path):
    file, receipt, _, _, _ = bound_receipt(tmp_path, monkeypatch)
    del receipt["analysis_source_sha256"][path]
    write_json(file, receipt)
    with pytest.raises(ValueError, match="source manifest"):
        plotter.load_receipt(file, plotter.digest(file))


@pytest.mark.parametrize("path", driver.SOURCES)
def test_loader_requires_every_dependency_in_the_frozen_driver_literal(tmp_path, monkeypatch, path):
    file, receipt, _, _, _ = bound_receipt(tmp_path, monkeypatch)
    del receipt["execution_source_sha256"][path]
    write_json(file, receipt)
    with pytest.raises(ValueError, match="source manifest"):
        plotter.load_receipt(file, plotter.digest(file))


@pytest.mark.parametrize("change", ["analysis hash", "execution hash", "plan hash", "metadata",
                                    "plan metadata", "plan period", "receipt family", "completion type"])
def test_loader_rejects_source_plan_and_scientific_identity_mismatches(tmp_path, monkeypatch, change):
    path, receipt, plan, blobs, _ = bound_receipt(tmp_path, monkeypatch)
    if change == "analysis hash":
        receipt["analysis_source_sha256"][plotter.FINISHER] = "f"*64
    elif change == "execution hash":
        receipt["execution_source_sha256"][plotter.DRIVER] = "f"*64
    elif change == "plan hash":
        receipt["plan_sha256"] = "f"*64
    elif change == "metadata":
        receipt["scientific_metadata"]["model_revision"] = "e"*40
    elif change in ("plan metadata", "plan period"):
        if change == "plan metadata":
            plan["model_revision"] = "e"*40
        else:
            plan["period"] = 6
            receipt["scientific_metadata"]["period"] = 6
        key = (receipt["execution_source_commit"], receipt["plan_path"])
        blobs[key] = json.dumps(plan).encode()
        receipt["plan_sha256"] = hashlib.sha256(blobs[key]).hexdigest()
    elif change == "receipt family":
        receipt["id"] = "different audit"
    else:
        receipt["complete"] = 1
    write_json(path, receipt)
    with pytest.raises(ValueError):
        plotter.load_receipt(path, plotter.digest(path))


@pytest.mark.parametrize("source", ["SOURCES = []\n", "SOURCES = list()\n",
                                    "SOURCES = ['tools/baby_shrimp_memory_connection_probe.py']\nSOURCES = []\n"])
def test_loader_requires_one_literal_frozen_dependency_declaration(tmp_path, monkeypatch, source):
    path, receipt, _, blobs, _ = bound_receipt(tmp_path, monkeypatch)
    key = (receipt["execution_source_commit"], plotter.DRIVER)
    blobs[key] = source.encode()
    receipt["execution_source_sha256"][plotter.DRIVER] = hashlib.sha256(blobs[key]).hexdigest()
    write_json(path, receipt)
    with pytest.raises(ValueError):
        plotter.load_receipt(path, plotter.digest(path))
