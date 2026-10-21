"""Exercise V3 admission and installation using six tiny, hash-bound synthetic archives."""
import importlib.util
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("morph_fill", Path(__file__).resolve().parents[1] / "tools/morph_fill.py")
mf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mf)


def arrays():
    return {"alpha": np.array([-0.12, 0.12]), "log10_temperature": np.array([-0.45, 0.3]),
            "lyapunov": np.full((2, 2), -0.1), "entropy": np.ones((2, 2)),
            "period": np.ones((2, 2), dtype=int), "family": np.zeros((2, 2), dtype=int)}


def repack(run):
    """Update transport evidence after an intentional synthetic producer defect."""
    with tarfile.open(run / "results.tgz", "w:gz") as archive:
        for path in sorted((run / "extracted").rglob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(run / "extracted")))
    result = json.loads((run / "result.json").read_text())
    result.update(local_sha256=mf.sha256(run / "results.tgz"), remote_sha256=mf.sha256(run / "results.tgz"))
    (run / "result.json").write_text(json.dumps(result))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    data = tmp_path / "data"
    morph = data / "explore/llm-soft-loop/pod64/morph"
    morph.mkdir(parents=True)
    monkeypatch.setattr(mf, "DATA", data)
    monkeypatch.setattr(mf, "MORPH", morph)
    monkeypatch.setattr(mf, "SHAPE", (2, 2))
    runs = []
    for init, frames in mf.CHAINS.items():
        args = {"max_period": 24, "resolution": [2, 2], "concept": f"morph-{init:03d}",
                "init_states": None, "out": "old"}
        reference = morph / f"frame-{init:03d}"
        reference.mkdir()
        (reference / "scan.json").write_text(json.dumps({"args": args}))
        np.savez(reference / "scan.npz", **arrays())
        run = data / "runpod/soft64" / f"synthetic-{init}"
        src = run / "extracted/explore/llm-soft-loop/pod64/morph"
        for k in frames:
            dst = src / f"frame-{k:03d}"
            dst.mkdir(parents=True)
            current = dict(args, concept=f"morph-{k:03d}", out="new",
                           init_states=(f"llm-soft-loop/morph-fill-init/frame-{init:03d}" if k == frames[0]
                                        else f"llm-soft-loop/pod64/morph/frame-{k - 2:03d}"))
            summary = {"args": current, "cells": 4, "seconds": 1, "lyapunov_positive": 0,
                       "families": [{"family": 0, "token_ids": [1]}]}
            (dst / "scan.json").write_text(json.dumps(summary))
            (dst / "scan.png").write_bytes(b"synthetic image")
            np.savez(dst / "scan.npz", **arrays())
            np.savez(dst / "scan-partial.npz", **arrays(), done=4)
        result = {"pod_id": f"synthetic-{init}", "gpu_type": "synthetic", "commit": "a" * 40,
                  "jobset": "morphfill", "runner_returncode": 0, "hash_verified": True, "extract_clash": True,
                  "upload": {"jobset_args": f"llm-soft-loop/morph-fill-init/frame-{init:03d} {frames[0]} {frames[-1]} 2",
                             "data_sha256": "b" * 64}}
        (run / "result.json").write_text(json.dumps(result))
        repack(run)
        runs.append(run)
    return runs


def invoke(runs, tmp_path, install=False):
    output = tmp_path / "receipt.json"
    code = mf.main([*[str(run) for run in runs], "--out", str(output), *(["--install"] if install else [])])
    return code, json.loads(output.read_text())


def assert_no_install():
    assert not (mf.MORPH / "EXTRACTION-NOTES.txt").exists()
    assert all(not (mf.MORPH / f"frame-{k:03d}" / "scan.npz").exists() for k in mf.EXPECTED_FRAMES)


def test_complete_campaign_installs_all_bound_frames(campaign, tmp_path):
    code, receipt = invoke(campaign, tmp_path, install=True)
    assert code == 0 and receipt["summary"]["installed"] == 17
    for row in receipt["frames"]:
        assert row["grid_equal"] and row["entropy_finite"]
        for name, digest in row["installed"].items():
            assert mf.sha256(mf.MORPH / f"frame-{row['frame']:03d}" / name) == digest


@pytest.mark.parametrize("field,value", [("hash_verified", False), ("hash_verified", "true"),
                                        ("local_sha256", "0" * 64), ("remote_sha256", "0" * 64),
                                        ("runner_returncode", 124)])
def test_transport_rejection_precedes_frame_read(campaign, tmp_path, monkeypatch, field, value):
    for run in campaign:
        path = run / "result.json"
        result = json.loads(path.read_text())
        result[field] = value
        path.write_text(json.dumps(result))
    calls = []
    monkeypatch.setattr(mf, "check_frame", lambda *args: calls.append(args))
    code, receipt = invoke(campaign, tmp_path, install=True)
    assert code == 1 and not calls
    assert len(receipt["frames"]) == 17 and receipt["errors"]
    assert_no_install()


@pytest.mark.parametrize("change", ["missing_run", "duplicate_run", "missing_frame", "missing_png", "missing_partial",
                                   "extracted_tamper", "destination_collision", "missing_receipt"])
def test_campaign_preflight_never_partially_installs(campaign, tmp_path, change):
    run = campaign[-1]
    frame = run / "extracted/explore/llm-soft-loop/pod64/morph/frame-238"
    if change == "missing_run":
        campaign.pop()
    elif change == "duplicate_run":
        campaign.append(campaign[0])
    elif change in {"missing_frame", "missing_png", "missing_partial"}:
        name = {"missing_frame": "scan.npz", "missing_png": "scan.png", "missing_partial": "scan-partial.npz"}[change]
        (frame / name).unlink()
        repack(run)
    elif change == "extracted_tamper":
        (frame / "scan.json").write_text("tampered")
    elif change == "destination_collision":
        destination = mf.MORPH / "frame-238"
        destination.mkdir()
        (destination / "scan.png").write_bytes(b"immutable original")
    elif change == "missing_receipt":
        (run / "result.json").unlink()
    code, receipt = invoke(campaign, tmp_path, install=True)
    assert code == 1 and receipt["summary"]["installed"] == 0 and receipt["errors"]
    assert_no_install()
    if change == "destination_collision":
        assert (mf.MORPH / "frame-238/scan.png").read_bytes() == b"immutable original"


@pytest.mark.parametrize("field,value", [("period", np.ones((1, 2), dtype=int)),
                                        ("entropy", np.ones((2, 1))), ("family", np.zeros(4, dtype=int)),
                                        ("entropy", np.full((2, 2), np.nan)),
                                        ("period", np.full((2, 2), 1.5)),
                                        ("family", np.full((2, 2), 99)),
                                        ("alpha", np.array([-0.11, 0.12]))])
def test_bad_numerical_row_rejects_entire_install(campaign, tmp_path, field, value):
    run = campaign[-1]
    frame = run / "extracted/explore/llm-soft-loop/pod64/morph/frame-238"
    changed = arrays()
    changed[field] = value
    np.savez(frame / "scan.npz", **changed)
    repack(run)
    code, receipt = invoke(campaign, tmp_path, install=True)
    assert code == 1
    assert receipt["summary"]["checks_pass"] == 16
    assert not next(row for row in receipt["frames"] if row["frame"] == 238)["checks_pass"]
    assert_no_install()


def test_read_only_missing_set_fails_and_receipt_cannot_be_overwritten(campaign, tmp_path):
    code, receipt = invoke(campaign[:-1], tmp_path)
    assert code == 1 and receipt["coverage"]["missing"] == [236, 238]
    original = (tmp_path / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        invoke(campaign, tmp_path, install=True)
    assert (tmp_path / "receipt.json").read_bytes() == original
    assert_no_install()


def test_read_only_success_records_every_source_hash(campaign, tmp_path):
    code, receipt = invoke(campaign, tmp_path)
    assert code == 0 and receipt["summary"]["installed"] == 0
    assert len(receipt["frames"]) == 17
    for row in receipt["frames"]:
        assert set(row["source_sha256"]) == {*mf.FRAME_FILES, "scan-partial.npz"}
    assert_no_install()


@pytest.mark.parametrize("matches", [True, False])
def test_original_partial_reproduction_is_checked_and_preserved(campaign, tmp_path, matches):
    destination = mf.MORPH / "frame-016"
    destination.mkdir()
    old = arrays()
    for name in ("lyapunov", "period", "entropy", "family"):
        old[name] = old[name].ravel()
    if not matches:
        old["period"][:] = 3
    partial = destination / "scan-partial.npz"
    np.savez(partial, **old, done=4, family_keys=np.array("[[1]]"))
    original_hash = mf.sha256(partial)
    code, receipt = invoke(campaign, tmp_path, install=True)
    row = next(row for row in receipt["frames"] if row["frame"] == 16)
    assert code == (0 if matches else 1)
    assert row["reproduction"]["reproduced"] is matches
    assert row["reproduction"]["periodic_agreement_threshold"] == 0.9
    assert mf.sha256(partial) == original_hash
    if not matches:
        assert_no_install()


def test_duplicate_archive_member_is_rejected(campaign, tmp_path):
    run = campaign[-1]
    extracted = run / "extracted"
    with tarfile.open(run / "results.tgz", "w:gz") as archive:
        for path in sorted(extracted.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(extracted)))
        path = extracted / "explore/llm-soft-loop/pod64/morph/frame-238/scan.json"
        archive.add(path, arcname=str(path.relative_to(extracted)))
    result = json.loads((run / "result.json").read_text())
    result.update(local_sha256=mf.sha256(run / "results.tgz"), remote_sha256=mf.sha256(run / "results.tgz"))
    (run / "result.json").write_text(json.dumps(result))
    code, receipt = invoke(campaign, tmp_path, install=True)
    assert code == 1 and "duplicate archive member" in " ".join(receipt["errors"])
    assert_no_install()
