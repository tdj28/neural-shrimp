"""Tests of the gates of tools/lf3_tk_receipt.py on a small synthetic run folder (no model, no retrieved data)."""
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lf3_tk_receipt as receipt  # noqa: E402

FILES = {"explore/llm-soft-loop/pod64/topk/soft-k2/scan.json": b'{"args": {"model": "toy", "top_k": 2}, "cells": 4,'
                                                               b' "prefix_tokens": 3, "seconds": 1.5}',
         "explore/llm-soft-loop/pod64/topk/soft-k2/scan.npz": b"array bytes",
         "explore/llm-soft-loop/pod64/logs/pod.log": b"a shared log that later runs append to"}


def make_run(tmp_path, monkeypatch, record_changes=None, local_changes=None):
    data, folder = tmp_path / "data", tmp_path / "data" / "runpod" / "soft64" / "stamp"
    folder.mkdir(parents=True)
    with tarfile.open(folder / "results.tgz", "w:gz") as tar:
        for name, content in FILES.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    for name, content in {**FILES, **(local_changes or {})}.items():
        if content is None:
            continue
        (data / name).parent.mkdir(parents=True, exist_ok=True)
        (data / name).write_bytes(content)
    digest = hashlib.sha256((folder / "results.tgz").read_bytes()).hexdigest()
    record = {"pod_id": "pod", "name": "run", "gpu_type": "gpu", "jobset": "topk", "commit": "c" * 40,
              "runner_returncode": 0, "wall_seconds": 10, "remote_sha256": digest, "local_sha256": digest,
              "hash_verified": True, "termination_verified": True, "termination_direct_absent": True,
              "termination_inventory_absent": True, "started": 1793172609.3, "rate": 123.45,
              "upload": {"code_sha256": "d" * 64, "code_bytes": 5, "jobset_args": "", "data_bytes": 7}}
    record.update(record_changes or {})
    (folder / "result.json").write_text(json.dumps(record))
    monkeypatch.setattr(receipt.ela, "DATA", data)
    return folder


def test_binds_every_file_and_leaves_out_the_rate(tmp_path, monkeypatch):
    out, files = receipt.run_record(make_run(tmp_path, monkeypatch))
    assert sorted(files) == sorted(n for n in FILES if "/logs/" not in n)
    assert out["files_equal_to_archive"] == 2 and out["shared_log_files_skipped"] == 1
    assert out["started_utc"] == "2026-10-28T07:30:09Z"
    assert "rate" not in out and "123.45" not in json.dumps(out)
    name = "explore/llm-soft-loop/pod64/topk/soft-k2/scan.npz"
    assert files[name] == {"sha256": hashlib.sha256(FILES[name]).hexdigest(), "bytes": len(FILES[name])}


def test_shared_logs_may_differ(tmp_path, monkeypatch):
    folder = make_run(tmp_path, monkeypatch,
                      local_changes={"explore/llm-soft-loop/pod64/logs/pod.log": b"longer after a later run"})
    assert receipt.run_record(folder)[0]["files_equal_to_archive"] == 2


def test_refuses_a_changed_or_missing_local_file(tmp_path, monkeypatch):
    name = "explore/llm-soft-loop/pod64/topk/soft-k2/scan.npz"
    with pytest.raises(SystemExit, match="differs between"):
        receipt.run_record(make_run(tmp_path / "a", monkeypatch, local_changes={name: b"other bytes"}))
    with pytest.raises(SystemExit, match="not in the data folder"):
        receipt.run_record(make_run(tmp_path / "b", monkeypatch, local_changes={name: None}))


def test_refuses_a_wrong_archive_hash_or_an_unfinished_run(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="differs from the record"):
        receipt.run_record(make_run(tmp_path / "a", monkeypatch, {"remote_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="verified, completed run"):
        receipt.run_record(make_run(tmp_path / "b", monkeypatch, {"runner_returncode": 1}))
    with pytest.raises(SystemExit, match="verified, completed run"):
        receipt.run_record(make_run(tmp_path / "c", monkeypatch, {"termination_verified": False}))
    folder = make_run(tmp_path / "d", monkeypatch)
    record = json.loads((folder / "result.json").read_text())
    del record["commit"]
    (folder / "result.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="lacks"):
        receipt.run_record(folder)


def test_settings_are_read_from_each_scan(tmp_path, monkeypatch):
    _out, files = receipt.run_record(make_run(tmp_path, monkeypatch))
    scans, separation = receipt.settings(files)
    assert scans == {"topk/soft-k2": {"model": "toy", "top_k": 2, "cells": 4, "prefix_tokens": 3, "seconds": 1.5}}
    assert separation == {}
