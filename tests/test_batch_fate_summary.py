"""Model-free regression gates for BF's finite-horizon recurrence summary."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import batch_fate_summary as bfs  # noqa: E402


def write_run(root, name, starts=4, layouts="2,4,1", steps=600, caught=False):
    """Complete no-recurrence fixture with independently identifiable rows."""
    out = root / "explore" / name
    (out / "chunks").mkdir(parents=True)
    x = np.arange(starts * 3, dtype=np.float64).reshape(starts, 3) + 1
    directions = np.ones_like(x) / np.sqrt(3)
    args = dict(model="toy", layer=12, vectors="toy", concept="philosophy", alpha=.07248,
                log_t=-.148, starts=starts, spread=.05, seed=100, steps=steps, layouts=layouts,
                max_period=64, tol=1e-9, record_every=100, dtype="float64", device="cuda", threads=8,
                out=name)
    arrays = {"starts": x, "directions": directions}
    summary, labels = {}, []
    for size in [int(v) for v in layouts.split(",")]:
        label = f"b{size}"
        labels.append(label)
        rec = np.array([x + .01 * t for t in range(0, steps + 1, 100)])
        raw = {"cap": np.full(starts, -1, dtype=np.int64), "per": np.zeros(starts, dtype=np.int64),
               "rec": rec, "mind": np.full(starts, .1), "final": x + .01 * steps}
        if caught:
            raw["cap"][:] = 3
            raw["per"][:] = 2
            raw["mind"][:] = np.nan
            raw["rec"][1:] = np.nan
        tokens = {str(i): [1, 2] for i in range(starts)} if caught else {}
        for key, val in raw.items():
            arrays[f"{label}_{key}"] = val
        arrays[f"{label}_cyc_tok"] = np.array([json.dumps(tokens)])
        if caught:
            for i in range(starts):
                arrays[f"{label}_cyc_state_{i}"] = np.array([x[i], x[i] + .1])
        for start in range(0, starts, size):
            chunk = {key: val[:, start:start + size] if key == "rec" else val[start:start + size]
                     for key, val in raw.items()}
            keys = list(range(start, start + size)) if caught else []
            chunk.update(t_end=np.int64(3 if caught else steps),
                         cyc_tok=np.array([json.dumps({str(i): tokens[str(i)] for i in keys})]),
                         cyc_keys=np.array(keys, dtype=np.int64))
            for i in keys:
                chunk[f"cyc_state_{i}"] = arrays[f"{label}_cyc_state_{i}"]
            np.savez_compressed(out / "chunks" / f"{label}-{start:05d}.npz", **chunk)
        summary[label] = {"captured": starts if caught else 0, "not_captured": 0 if caught else starts, "seconds": 1.0}
    np.savez_compressed(out / "fate.npz", **arrays)
    record = {"args": args, "labels": labels, "summary": summary,
              "starts_sha256": hashlib.sha256(x.tobytes()).hexdigest()}
    (out / "fate.json").write_text(json.dumps(record))
    return out


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(bfs, "DATA", tmp_path)
    write_run(tmp_path, "main")
    write_run(tmp_path, "repeat4", layouts="4", steps=500)
    write_run(tmp_path, "repeat1", starts=2, layouts="1", steps=500)
    return tmp_path


def loaded_controls():
    return [(name, bfs.load(name)) for name in ("repeat4", "repeat1")]


def test_complete_fixtures_and_exact_controls_pass(runs):
    _, rec, z, lays = bfs.load("main")
    controls = bfs.validate_controls(rec, z, lays, loaded_controls())
    assert len(controls) == 2
    assert all(c["starts_sha256_match"] and c["recorded_states_bitwise_equal"] for c in controls)
    assert lays["b4"]["gates"]["rows_checked"] == 4
    assert lays["b4"]["gates"]["detector_residuals"] == "not_saved"


def test_missing_controls_fail_closed(runs):
    _, rec, z, lays = bfs.load("main")
    with pytest.raises(ValueError, match="missing BF"):
        bfs.validate_controls(rec, z, lays, [])


@pytest.mark.parametrize("defect", ["horizon", "settings", "start", "state", "detector", "hash", "duplicate"])
def test_invalid_controls_cannot_pass_from_a_matching_prefix(runs, defect):
    _, rec, z, lays = bfs.load("main")
    controls = loaded_controls()
    _, (_, c, a, ls) = controls[0]
    if defect == "horizon":
        c["args"]["steps"] = 400
    elif defect == "settings":
        c["args"]["tol"] = 1e-6
    elif defect == "start":
        a["starts"][-1, -1] += 1
    elif defect == "state":
        ls["b4"]["rec"][-1, -1, -1] += 1e-8
    elif defect == "detector":
        ls["b4"]["cap"][0] = 400
    elif defect == "hash":
        c["starts_sha256"] = c["starts_sha256"][:16] + "0" * 48
    else:
        controls.append(controls[0])
    with pytest.raises(ValueError):
        bfs.validate_controls(rec, z, lays, controls)


def test_missing_chunks_refused_even_when_aggregate_exists(runs):
    (runs / "explore/main/chunks/b1-00003.npz").unlink()
    with pytest.raises(ValueError, match="chunk set"):
        bfs.load("main")


def mutate_npz(path, edit):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    edit(a)
    np.savez_compressed(path, **a)


@pytest.mark.parametrize("key", ["b4_final", "b4_rec"])
def test_nonfinite_rows_refused(runs, key):
    mutate_npz(runs / "explore/main/fate.npz", lambda a: a[key].flat.__setitem__(-1, np.inf))
    with pytest.raises(ValueError):
        bfs.load("main")


def test_chunk_aggregate_disagreement_refused(runs):
    mutate_npz(runs / "explore/main/chunks/b4-00000.npz",
               lambda a: a["final"].flat.__setitem__(-1, 1000))
    with pytest.raises(ValueError, match="chunk final mismatch"):
        bfs.load("main")


def test_uncaptured_short_horizon_refused(runs):
    mutate_npz(runs / "explore/main/fate.npz", lambda a: a["b4_rec"].__setitem__(-1, np.nan))
    with pytest.raises(ValueError, match="incomplete horizon"):
        bfs.load("main")


def test_orbit_grouping_checks_all_phases_not_one_close_point():
    a = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    b = a.copy()
    b[1] = [5., 7.]  # Shares two points, but is not the same candidate cycle.
    c = np.roll(a, 1, axis=0)
    d = a[::-1]  # Same unordered set but incompatible phase order.
    reps, ids = bfs.orbit_ids({"b1": {"cyc": {0: a, 1: b, 2: c, 3: d}}})
    assert len(reps) == 3
    assert ids[("b1", 0)] == ids[("b1", 2)]
    assert ids[("b1", 0)] != ids[("b1", 1)] != ids[("b1", 3)]


def test_summary_exposes_limits_hashes_and_refuses_overwrite(runs, monkeypatch):
    output = runs / "receipt.json"
    monkeypatch.setattr(sys, "argv", ["batch_fate_summary.py", "--run", "main", "--control", "repeat4",
                                     "--control", "repeat1", "--out", str(output)])
    bfs.main()
    receipt = json.loads(output.read_text())
    assert receipt["schema"] == "batch-fate-recurrence-candidates-v2"
    assert "not verified chaotic" in receipt["candidate_semantics"]
    assert receipt["metadata_gaps"] and receipt["artifacts"]
    assert all(a["bytes"] > 0 and len(a["sha256"]) == 64 for a in receipt["artifacts"])
    assert receipt["per_layout"]["b4"]["not_captured"] == 4
    with pytest.raises(FileExistsError):
        bfs.main()


def test_early_candidate_termination_and_blank_tail_are_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(bfs, "DATA", tmp_path)
    write_run(tmp_path, "main", caught=True)
    write_run(tmp_path, "repeat4", layouts="4", steps=500, caught=True)
    write_run(tmp_path, "repeat1", starts=2, layouts="1", steps=500, caught=True)
    _, rec, z, lays = bfs.load("main")
    bfs.validate_controls(rec, z, lays, loaded_controls())
    assert len(lays["b4"]["cyc"]) == 4
    assert np.isnan(lays["b1"]["rec"][1:]).all()


def test_union_counts_overlaps_once():
    a = np.array([10, -1, 20, 30, -1])
    b = np.array([-1, 40, 50, -1, -1])
    c = np.array([-1, -1, -1, 60, -1])
    result = bfs.start_census({lab: {"cap": cap} for lab, cap in zip(("a", "b", "c"), (a, b, c))}, 5)
    assert result["detected_in_any_layout"] == 4
    assert result["undetected_in_all_layouts"] == 1
    assert result["detected_in_all_layouts"] == 0
    assert result["detection_step_range"] == [10, 60]


def test_saved_returns_compare_whole_periods_only():
    # q=8 and cadence100 mean adjacent records have different phase;
    # records200steps apart have the same phase.
    recorded = np.array([[[1., 0.]], [[2., 0.]], [[1.1, 0.]], [[2.2, 0.]], [[1.21, 0.]]])
    result = bfs.recorded_return_checks({"b4": {"cap": np.array([1]), "per": np.array([8]), "rec": recorded}}, 100)
    row = result["per_layout"]["b4"][0]
    assert row["post_detection_record_pairs"] == 2
    assert row["max_relative_distance_whole_periods_apart"] == pytest.approx(.1)
    assert result["largest_checked_distance"] == pytest.approx(.1)
    assert "intervening steps" in result["scope"]


def test_fourteen_detections_with_two_overlaps_mean_twelve_starts():
    layouts = {}
    for lab, rows in {"b32": [0, 1, 2, 3, 4], "b64": [4, 5, 6, 7, 8], "b1": [8, 9, 10, 11]}.items():
        cap = np.full(64, -1)
        cap[rows] = 1001
        layouts[lab] = {"cap": cap}
    result = bfs.start_census(layouts, 64)
    assert result["detected_in_any_layout"] == 12
    assert result["undetected_in_all_layouts"] == 52
    assert result["detected_in_all_layouts"] == 0
