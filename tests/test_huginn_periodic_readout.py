import hashlib
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import huginn_periodic_readout as m


def plan_fixture(tmp_path):
    states = np.arange(72 * 16 * 2 * 3, dtype=np.float32).reshape(72, 16, 2, 3)
    cells = np.repeat([[1.25, -.03]], 16, axis=0)
    paths = []
    for offset, eps in enumerate((.001, .0003)):
        path = tmp_path / f"tail-0032-{offset}.npz"
        np.savez(path, states=states, cells=cells, eps=eps)
        paths.append(str(path))
    direction = tmp_path / "direction.npy"
    np.save(direction, np.ones(3, dtype=np.float64))
    return {"model_revision": "a" * 40, "prompt": "fixed", "prompt_ids": [1, 2], "seed": 0,
            "selected_cell_index": 45, "selected_local_index": 13, "lambda": 1.25, "steering": -.03,
            "tail_paths": paths, "tail_sha256": [m.sha256(p) for p in paths],
            "steering_direction_path": str(direction), "steering_direction_sha256": m.sha256(direction),
            "tail_start_update": 3001, "tail_end_update": 3072, "extension_steps": 256,
            "chunk_steps": 128, "threads": 1, "max_seconds": 900, "memory_fraction": .8}


def test_relaxed_map_has_no_readout_feedback_and_preserves_expression():
    state = np.array([[1.234567, -2.345678]], dtype=np.float32)
    lam = np.float32(1.2532258)
    calls = []

    def core(x):
        calls.append(x.copy())
        return .2 * x + .7

    actual = m.relaxed_step(state, core, lam)
    expected = (1 - lam) * state + lam * (.2 * state + .7)
    assert np.array_equal(actual, expected)
    assert len(calls) == 1 and np.array_equal(calls[0], state)


def test_period_two_is_distinct_from_constant_top_token():
    initial = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    tracker = m.RecurrenceTracker(initial)
    first = tracker.observe(-initial, 1)
    second = tracker.observe(initial, 2)
    assert first["q2_relative"] is None
    assert second["q2_relative"] == [0., 0.]
    assert second["q1_relative"] == [2., 2.]
    # A many-to-one readout can favor the same token in distinct states.
    logits = np.array([[4., 1., -1.], [5., 0., -1.]], dtype=np.float32)
    stats = m.probability_summary(logits, 2)
    assert stats["top_ids"][:, 0].tolist() == [0, 0]
    assert not np.array_equal(stats["probabilities"][0], stats["probabilities"][1])


def test_horizons_do_not_confuse_warm_and_fresh_iterations():
    assert m.horizon_indices(1, 3072) == [3073, 1]
    assert m.horizon_indices(8192, 3072) == [11264, 8192]
    with pytest.raises(ValueError):
        m.horizon_indices(0, 3072)


def test_probability_stats_are_shift_invariant_and_finite():
    logits = np.array([[1000., 1001., -1000.]])
    a = m.probability_summary(logits, 3)
    b = m.probability_summary(logits - 1000, 3)
    assert np.allclose(a["probabilities"], b["probabilities"])
    assert np.isfinite(a["entropy_nats"]).all()
    check = m.compare_logits(logits, logits)
    assert check == {"max_abs": 0., "argmax_equal": True, "max_total_variation": 0.}
    with pytest.raises(ValueError):
        m.probability_summary([[float("nan"), 1]], 1)


def test_inputs_verified_at_both_offsets_and_precise_cell(tmp_path):
    plan = plan_fixture(tmp_path)
    m.validate_plan(plan)
    saved, direction, records = m.load_inputs(plan)
    assert saved.shape == (72, 2, 2, 3)
    assert np.array_equal(saved[:, 0], saved[:, 1])
    assert direction.dtype == np.float64
    assert len(records) == 2
    plan["lambda"] += 1e-9
    with pytest.raises(ValueError, match="parameter values"):
        m.load_inputs(plan)


def test_hash_failure_never_loads_unbound_evidence(tmp_path):
    plan = plan_fixture(tmp_path)
    plan["tail_sha256"][1] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        m.load_inputs(plan)


def test_different_offset_base_trajectories_fail_explicitly(tmp_path):
    plan = plan_fixture(tmp_path)
    path = Path(plan["tail_paths"][1])
    with np.load(path) as data:
        arrays = dict(data)
    arrays["states"][0, 13, 0, 0] += 1
    np.savez(path, **arrays)
    plan["tail_sha256"][1] = m.sha256(path)
    with pytest.raises(ValueError, match="identical saved base"):
        m.load_inputs(plan)


@pytest.mark.parametrize("key,value", [("max_seconds", 901), ("threads", 5),
                                      ("memory_fraction", .9), ("chunk_steps", 256),
                                      ("extension_steps", 16384)])
def test_resource_limits_are_enforced(tmp_path, key, value):
    plan = plan_fixture(tmp_path)
    plan[key] = value
    with pytest.raises(ValueError):
        m.validate_plan(plan)


def test_checkpoint_hashes_and_no_overwrite(tmp_path):
    writer = m.EvidenceWriter(tmp_path / "attempt")
    arrays = {"states": np.ones((2, 2, 3), dtype=np.float32),
              "logits": np.arange(16, dtype=np.float32).reshape(2, 2, 4)}
    receipt = writer.checkpoint("chunk", arrays, {"complete": True})
    assert receipt["sha256"] == m.sha256(writer.root / "chunk.npz")
    with np.load(writer.root / "chunk.npz") as saved:
        assert np.array_equal(saved["states"], arrays["states"])
    with pytest.raises(FileExistsError):
        writer.checkpoint("chunk", arrays, {})
    writer.manifest("manifest.json", True)
    manifest = json.loads((writer.root / "manifest.json").read_text())
    assert manifest["complete"]
    for filename, item in manifest["files"].items():
        path = writer.root / filename
        assert item["sha256"] == m.sha256(path)
        assert item["bytes"] == path.stat().st_size


def test_interrupted_atomic_write_preserves_temporary_file(tmp_path):
    path = tmp_path / "checkpoint"

    def fail(handle):
        handle.write(b"partial evidence")
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError):
        m._atomic_file(path, fail)
    assert not path.exists()
    partial = list(tmp_path.glob("checkpoint.partial-*"))
    assert len(partial) == 1
    assert partial[0].read_bytes() == b"partial evidence"


def test_catchable_failure_saves_unfinished_shard(tmp_path):
    writer = m.EvidenceWriter(tmp_path / "attempt")
    buffers = {"states": [], "logits": [], "rows": []}
    with pytest.raises(m.DeadlineExceeded):
        with m.preserve_completed_buffer(writer, buffers):
            buffers["states"].append(np.ones((2, 2, 3), dtype=np.float32))
            buffers["logits"].append(np.ones((2, 4), dtype=np.float32))
            buffers["rows"].append({"continuation_step": 129, "internal_updates": [3201, 129]})
            raise m.DeadlineExceeded("fixture")
    path = writer.root / "interrupted-buffer-00129-00129.npz"
    assert path.exists()
    with np.load(path) as data:
        assert data["states"].shape == (1, 2, 2, 3)
        assert data["internal_updates"].tolist() == [[3201, 129]]


def test_alarm_is_restored_and_enforced():
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(m.DeadlineExceeded):
        with m.hard_deadline(.02):
            time.sleep(.1)
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    assert signal.getsignal(signal.SIGALRM) == previous


def test_main_failure_keeps_prior_checkpoints_and_partial_manifest(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path)
    plan_path = tmp_path / "input-plan.json"
    plan_path.write_text(json.dumps(plan))
    destination = tmp_path / "attempt"

    def fail(plan, plan_path, source, writer, started):
        writer.checkpoint("finished", {"states": np.ones((2, 2), dtype=np.float32)}, {})
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(m, "execute", fail)
    monkeypatch.setattr(sys, "argv", ["tool", "--plan", str(plan_path),
                                     "--source-commit", "a" * 40, "--out", str(destination)])
    with pytest.raises(RuntimeError, match="fixture failure"):
        m.main()
    assert (destination / "finished.npz").exists()
    assert (destination / "failure.json").exists()
    assert not (destination / "manifest.json").exists()
    manifest = json.loads((destination / "partial-manifest.json").read_text())
    assert not manifest["complete"]
    assert manifest["files"]["finished.npz"]["sha256"] == m.sha256(destination / "finished.npz")


def test_wrong_readout_gate_refuses_argmax_mismatch():
    comparison = m.compare_logits([[1., 0.]], [[0., 1.]])
    with pytest.raises(ValueError, match="gate failed"):
        m._require_comparison(comparison, 2, 1, "fixture")
