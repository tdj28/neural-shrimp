"""Data-preserving summary layout and strict-loader integration, synthetic only."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
sys.path.insert(0, str(Path(__file__).parent))
import plot_baby_shrimp_mixed_junction_summary as summary
from test_plot_baby_shrimp_mixed_junction import fixture_receipt, bound_fixture


def display(mode="complete"):
    receipt, plan, raw = fixture_receipt(mode)
    data = summary.original.validate_plot_data(receipt, plan, raw, allow_partial=mode != "complete")
    return receipt, data


@pytest.mark.parametrize("mobile", [False, True])
def test_actual_values_errors_phases_and_unjoined_targets_preserved(mobile):
    import matplotlib.pyplot as plt
    receipt, data = display(); before = copy.deepcopy(data)
    fig = summary.build_summary(receipt, data, mobile=mobile, synthetic=True)
    feature, trace, table = fig.axes
    for row in data["rows"]:
        i = row["case_index"]
        artist = next(line for line in feature.lines if line.get_gid() == f"feature-case-{i}")
        np.testing.assert_array_equal(artist.get_xdata(), [row["beta_offset"]*1e9])
        np.testing.assert_array_equal(artist.get_ydata(), [row["feature"]*1e7])
        assert artist.get_linestyle() == "None"
        for j, point in enumerate(row["actual_traces"]):
            artist = next(c for c in trace.collections if c.get_gid() == f"trace-case-{i}-sample-{j}")
            assert artist.get_offsets()[0, 1] == point["tau"]*1e12
    segments = [segment for artist in feature.collections for segment in artist.get_segments()]
    assert len(segments) == 3
    for segment, row in zip(segments, data["rows"]):
        np.testing.assert_allclose(segment[:, 1], np.array([row["feature"]-row["feature_error"],
                                                         row["feature"]+row["feature_error"]])*1e7)
    assert len(trace.collections) == 12 and data == before
    assert all(len(line.get_xdata()) == 2 for line in trace.lines)  # only the horizontal zero guide
    plt.close(fig)


@pytest.mark.parametrize("mode", ["empty", "partial", "failed", "unfinished_stencil"])
def test_partial_plot_retains_all_statuses_without_zero_substitutes(mode):
    import matplotlib.pyplot as plt
    receipt, data = display(mode)
    fig = summary.build_summary(receipt, data, synthetic=True)
    table = fig.axes[2].tables[0]
    displayed = [cell.get_text().get_text().replace("\n", " ") for cell in table.get_celld().values()]
    assert "not attempted" in displayed
    assert "Unmeasured" in displayed
    if mode == "failed":
        assert "unresolved precision" in displayed and "Not admitted" in displayed
        marker = next(line for line in fig.axes[0].lines if line.get_gid() == "feature-case-0")
        assert marker.get_marker() == "D" and marker.get_markerfacecolor() == "white"
    measured = [line for line in fig.axes[0].lines if line.get_gid()]
    assert len(measured) == sum(r["feature"] is not None for r in data["rows"])
    plt.close(fig)


def test_dependency_must_match_its_committed_source(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "ROOT", tmp_path)
    blobs = {}
    for source in summary.SOURCES:
        p = tmp_path/source; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(source)
        blobs[source] = p.read_bytes()
    monkeypatch.setattr(summary.subprocess, "check_output", lambda *a, **kw: "a"*40)
    monkeypatch.setattr(summary.original, "git_bytes", lambda commit, source: blobs[source])
    commit, hashes = summary.frozen_sources()
    assert commit == "a"*40 and set(hashes) == set(summary.SOURCES)
    (tmp_path/summary.SOURCES[1]).write_text("changed validator")
    with pytest.raises(ValueError, match="validation dependency"):
        summary.frozen_sources()


def run_cli(path, sha, out, monkeypatch, *, partial=False):
    monkeypatch.setattr(summary, "frozen_sources", lambda: ("a"*40, {p: "f"*64 for p in summary.SOURCES}))
    monkeypatch.setattr(summary, "render", lambda receipt, data, output: (output/"synthetic.png").write_bytes(b"QA placeholder"))
    args = ["--receipt", str(path), "--receipt-sha256", sha, "--out", str(out)]
    if partial: args.append("--allow-partial")
    summary.main(args)


def test_cli_uses_original_strict_loader_and_preserves_inventory(tmp_path, monkeypatch):
    receipt, run, path, save, _ = bound_fixture(tmp_path, monkeypatch)
    out = tmp_path/"summary"
    run_cli(path, save(), out, monkeypatch)
    saved = json.loads((out/"receipt.json").read_text())
    expected = summary.original.load_receipt(path, summary.original.digest(path))[1]
    assert saved["displayed_cases"] == expected["rows"]
    assert saved["point_admissions"] == receipt["point_admissions"]
    assert saved["scope"] == summary.original.SCOPE
    assert set(saved["source_hashes"]) == set(summary.SOURCES)


@pytest.mark.parametrize("corruption", ["hash", "raw", "plan", "lifecycle"])
def test_cli_refuses_corruption_before_output_creation(tmp_path, monkeypatch, corruption):
    receipt, run, path, save, _ = bound_fixture(tmp_path, monkeypatch)
    if corruption == "raw": (run/"point-0000"/"point.json").write_text("{}")
    if corruption == "plan": receipt["plan_sha256"] = "0"*64
    if corruption == "lifecycle": receipt["lifecycle"]["python_finalization"] = False
    sha = save()
    if corruption == "hash": sha = "0"*64
    out = tmp_path/"summary"
    with pytest.raises(ValueError): run_cli(path, sha, out, monkeypatch)
    assert not out.exists()


def test_cli_partial_requires_explicit_flag_and_retains_failure(tmp_path, monkeypatch):
    receipt, run, path, save, _ = bound_fixture(tmp_path, monkeypatch, "failed")
    out = tmp_path/"summary"
    with pytest.raises(ValueError, match="allow-partial"):
        run_cli(path, save(), out, monkeypatch)
    assert not out.exists()
    run_cli(path, save(), out, monkeypatch, partial=True)
    saved = json.loads((out/"receipt.json").read_text())
    assert saved["terminal_failure"] == receipt["terminal_failure"]
    assert [r["status"] for r in saved["displayed_cases"]] == ["unresolved precision", "not attempted", "not attempted"]
    assert saved["partial_explicitly_allowed"] and not saved["scientific_complete"]


def test_existing_output_and_raw_tree_refused(tmp_path, monkeypatch):
    _, run, path, save, _ = bound_fixture(tmp_path, monkeypatch)
    for out in (run, run/"new-figure", tmp_path):
        with pytest.raises(ValueError, match="new output"):
            run_cli(path, save(), out, monkeypatch)
