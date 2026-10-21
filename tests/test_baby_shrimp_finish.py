"""Adversarial raw-output gates: missing cells and failed runs must not make a map."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
SPEC = importlib.util.spec_from_file_location("baby_shrimp_finish", Path(__file__).resolve().parents[1] /
                                             "tools/baby_shrimp_finish.py")
finish = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finish)


def sample():
    plan = json.loads((finish.ROOT / "docs/baby-shrimp-zoom/plan.json").read_text())
    plan.update(resolution=[2, 1024])
    band = {"name": "synthetic", "resolution": [2, 1024], "rows": [0, 2],
            "log10_temperature": plan["log10_temperature"]}
    shape = tuple(band["resolution"])
    arrays = {"alpha": np.linspace(*plan["alpha"], 1024),
              "log10_temperature": np.linspace(*plan["log10_temperature"], 2),
              "lyapunov": np.full(shape, -.01), "entropy": np.ones(shape),
              "period": np.zeros(shape, dtype=int), "text_period": np.zeros(shape, dtype=int),
              "family": np.full(shape, -1, dtype=int), "text_family": np.full(shape, -1, dtype=int)}
    args = {k: copy.deepcopy(plan[k]) for k in ("model", "layer", "vectors", "concept", "prompt", "opener",
            "window", "dtype", "weights", "transient", "samples", "record", "max_period", "tol", "eps", "chaos_level")}
    args.update(command="scan", device="cuda", threads=8, chunk=2048, resolution=band["resolution"],
                alpha=plan["alpha"], log_temperature=band["log10_temperature"], out="llm-soft-loop/pod64/bands/synthetic")
    info = {"args": args, "cells": 2048, "seconds": 30, "period_counts": {"0": 2048},
            "lyapunov_positive": 0, "families": [], "texts": []}
    return plan, band, arrays, info


def test_all_cells_and_metadata_pass():
    p, b, a, i = sample()
    assert finish.gate_arrays(a, i, p, b)["cells"] == 2048


@pytest.mark.parametrize("defect", ["nonfinite", "missing_row", "wrong_axis", "period_33", "wrong_summary",
                                   "short_horizon", "float32", "wrong_batch", "alternative_start", "unknown_family"])
def test_invalid_band_is_rejected(defect):
    p, b, a, i = sample()
    if defect == "nonfinite": a["lyapunov"][0, 0] = np.nan
    elif defect == "missing_row": a["entropy"] = a["entropy"][:1]
    elif defect == "wrong_axis": a["alpha"][0] += 1e-6
    elif defect == "period_33": a["period"][0, 0] = 33
    elif defect == "wrong_summary": i["period_counts"] = {"0": 2047}
    elif defect == "short_horizon": i["args"]["samples"] = 100
    elif defect == "float32": i["args"]["dtype"] = "float32"
    elif defect == "wrong_batch": i["args"]["chunk"] = 1024
    elif defect == "alternative_start": i["args"]["start_vector"] = "different.npy"
    elif defect == "unknown_family": a["family"][0, 0] = 7
    with pytest.raises(ValueError):
        finish.gate_arrays(a, i, p, b)


def fixture_archive(tmp_path, monkeypatch, numerical_status=0, checkpoint_done=2048):
    monkeypatch.setattr(finish, "DATA", tmp_path)
    monkeypatch.setattr(finish.subprocess, "check_output", lambda *a, **kw: b"synthetic source")
    plan, band, arrays, info = sample()
    folder = tmp_path / "explore/llm-soft-loop/pod64/bands/synthetic"
    folder.mkdir(parents=True)
    np.savez(folder / "scan.npz", **arrays)
    (folder / "scan.json").write_text(json.dumps(info))
    partial = {k: (v.ravel() if v.ndim > 1 else v) for k, v in arrays.items()}
    np.savez(folder / "scan-partial.npz", done=checkpoint_done, **partial)
    progress = [{"chunk_start": 0, "phase": phase, "step": step, "seconds": 1}
                for phase in ("transient", "samples") for step in (100, 200, 300)]
    (folder / "progress.jsonl").write_text("\n".join(map(json.dumps, progress)))
    logs = tmp_path / "explore/llm-soft-loop/pod64/logs"
    logs.mkdir()
    (logs / "hardware.txt").write_text("synthetic hardware")
    (logs / "versions.txt").write_text("synthetic versions")
    (logs / "scanband-synthetic.log").write_text('{"done": 2048, "of": 2048}\n')
    worker = tmp_path / "worker"
    worker.mkdir()
    (worker / "runner-stdout.txt").write_text(f"end scanband-synthetic status {numerical_status}\n")
    archive = worker / "results.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "explore", arcname="explore")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = {"hash_verified": True, "runner_returncode": 0, "local_sha256": sha, "remote_sha256": sha,
              "jobset": "scanband", "commit": "a" * 40, "pod_id": "synthetic", "gpu_type": "synthetic",
              "wall_seconds": 40}
    (worker / "result.json").write_text(json.dumps(result))
    return plan, band, worker


def test_archive_and_checkpoint_must_agree(tmp_path, monkeypatch):
    p, b, w = fixture_archive(tmp_path, monkeypatch)
    assert finish.gate_band(p, b, w)["gates"]["passed"]


def test_successful_shell_cannot_hide_failed_numerical_command(tmp_path, monkeypatch):
    p, b, w = fixture_archive(tmp_path, monkeypatch, numerical_status=124)
    with pytest.raises(ValueError, match="numerical command"):
        finish.gate_band(p, b, w)


def test_final_file_cannot_hide_incomplete_checkpoint(tmp_path, monkeypatch):
    p, b, w = fixture_archive(tmp_path, monkeypatch, checkpoint_done=1024)
    with pytest.raises(ValueError, match="checkpoint incomplete"):
        finish.gate_band(p, b, w)


def test_valid_archive_cannot_hide_changed_local_data(tmp_path, monkeypatch):
    p, b, w = fixture_archive(tmp_path, monkeypatch)
    (tmp_path / "explore/llm-soft-loop/pod64/bands/synthetic/progress.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="local/raw mismatch"):
        finish.gate_band(p, b, w)
