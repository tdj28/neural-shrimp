"""Receipt for Git of runs LF3 (LFM3 soft-token maps) and TK (top-k loops of Qwen2.5-0.5B), 28 Oct 2026.

The retrieved arrays live outside Git (../neural-shrimp-data). This tool binds them to the repository:

1. Each run's record (the orchestrator's result.json: pod, source commit, archive hash, return code, verified
   termination) is read, taking only the fields named in RUN_FIELDS. No price or rate is read or written.
2. Each retrieved archive (results.tgz) is hashed again and must equal the hash in its record.
3. Every file of the archive is compared, by SHA-256, with its unpacked copy in the data folder. The shared pod logs
   (pod64/logs) are skipped, because later runs append to them. Any other difference stops the tool.
4. Both summaries are recomputed from the arrays with tools/lf3_tk_summary.py and must equal, byte for byte, the
   summaries stored beside the arrays on 28 Oct 2026. They are then copied to the output folder.
5. The settings recorded with each scan and each separation run are collected, with the pinned model revisions.

Writes OUT/lfm3-summary.json, OUT/topk-summary.json and OUT/receipt.json (default OUT: docs/lf3-tk).

Usage: python tools/lf3_tk_receipt.py [OUT_DIR]
"""
import hashlib
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import lf3_tk_summary as summary  # noqa: E402

RUNS = {"LF3": "20261028T073007Z", "TK": "20261028T073008Z"}          # folders of the orchestrator, runpod/soft64
RUN_FIELDS = ("pod_id", "name", "gpu_type", "jobset", "commit", "runner_returncode", "wall_seconds", "remote_sha256",
              "local_sha256", "hash_verified", "termination_verified", "termination_direct_absent",
              "termination_inventory_absent")
UPLOAD_FIELDS = ("code_sha256", "code_bytes", "jobset_args")
SCAN_FIELDS = ("model", "layer", "concept", "dtype", "weights", "prompt", "opener", "window", "log_temperature",
               "alpha", "resolution", "transient", "samples", "max_period", "tol", "eps", "chaos_level", "top_k",
               "sample", "seed")
SEPARATION_FIELDS = ("model", "layer", "concept", "dtype", "top_k", "sample", "seed", "cells", "deltas", "repeats",
                     "transient", "steps", "threshold")
SHARED_LOGS = "explore/llm-soft-loop/pod64/logs/"
STORED = {"lfm3": "explore/llm-soft-loop/pod64/lfm3/summary.json",
          "topk": "explore/llm-soft-loop/pod64/topk/summary.json"}
MODELS = ("lfm3-1.7b-base", "lfm3-1.7b", "qwen0.5b")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def run_record(folder):
    """The whitelisted fields of one orchestrator record, the archive's hash, and its files against the data folder."""
    record = json.loads((folder / "result.json").read_text())
    missing = [k for k in RUN_FIELDS if k not in record]
    if missing:
        raise SystemExit(f"{folder}: the record lacks {missing}")
    out = {k: record[k] for k in RUN_FIELDS}
    out["upload"] = {k: record["upload"][k] for k in UPLOAD_FIELDS}
    out["started_utc"] = datetime.fromtimestamp(record["started"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not (out["hash_verified"] and out["termination_verified"] and out["runner_returncode"] == 0):
        raise SystemExit(f"{folder}: the record does not show a verified, completed run: {out}")
    archive = folder / "results.tgz"
    again = sha256(archive)
    if again != out["local_sha256"] or again != out["remote_sha256"]:
        raise SystemExit(f"{archive}: hash {again} differs from the record")
    out["archive_bytes"] = archive.stat().st_size
    files, skipped = {}, 0
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if member.name.startswith(SHARED_LOGS):
                skipped += 1
                continue
            digest = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
            local = ela.DATA / member.name
            if not local.exists():
                raise SystemExit(f"{member.name} is in {archive} and not in the data folder")
            if sha256(local) != digest:
                raise SystemExit(f"{member.name} differs between {archive} and the data folder")
            files[member.name] = {"sha256": digest, "bytes": member.size}
    out["files_equal_to_archive"] = len(files)
    out["shared_log_files_skipped"] = skipped
    return out, files


def settings(files):
    """What each scan.json and divergence.json says about its own run."""
    scans, separation = {}, {}
    for name in sorted(files):
        path = ela.DATA / name
        key = str(Path(name).parent.relative_to("explore/llm-soft-loop/pod64"))
        if path.name == "scan.json":
            record = json.loads(path.read_text())
            args = record["args"]
            scans[key] = {**{k: args[k] for k in SCAN_FIELDS if k in args}, "cells": record["cells"],
                          "prefix_tokens": record["prefix_tokens"], "seconds": record["seconds"]}
        elif path.name == "divergence.json":
            record = json.loads(path.read_text())
            separation[key] = {**{k: record["args"][k] for k in SEPARATION_FIELDS if k in record["args"]},
                               "state_norm": record["state_norm"], "threshold_log10": record["threshold_log10"]}
    return scans, separation


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "lf3-tk"
    out.mkdir(parents=True, exist_ok=True)
    runs, files = {}, {}
    for run, stamp in RUNS.items():
        runs[run], got = run_record(ela.DATA / "runpod" / "soft64" / stamp)
        clash = set(files) & set(got)
        if clash:
            raise SystemExit(f"files in both archives: {sorted(clash)[:5]}")
        files.update(got)
    scans, separation = settings(files)
    summaries = {}
    for key, compute in (("lfm3", summary.lf3), ("topk", summary.tk)):
        stored = (ela.DATA / STORED[key]).read_bytes()
        again = json.dumps(compute(), indent=1).encode()
        if again != stored:
            raise SystemExit(f"the {key} summary recomputed from the arrays differs from {STORED[key]}")
        if "missing" in again.decode():
            raise SystemExit(f"the {key} summary reports a missing input")
        target = out / f"{key}-summary.json"
        target.write_bytes(stored)
        summaries[key] = {"file": str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else target.name,
                          "sha256": hashlib.sha256(stored).hexdigest(), "bytes": len(stored),
                          "stored_with_the_arrays_as": STORED[key], "recomputed_equal": True}
    tool = Path(__file__).resolve()
    receipt = {
        "runs": "LF3 (LFM3 soft-token maps) and TK (top-k loops of Qwen2.5-0.5B-Instruct), 28 October 2026; "
                "exploratory",
        "tool": str(tool.relative_to(ROOT)), "tool_sha256": sha256(tool), "tool_bytes": tool.stat().st_size,
        "summary_tool": "tools/lf3_tk_summary.py", "summary_tool_sha256": sha256(ROOT / "tools" / "lf3_tk_summary.py"),
        "analysis_commit": git("rev-parse", "HEAD"),
        "analysis_worktree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_root": "../neural-shrimp-data",
        "run_records": {run: f"runpod/soft64/{stamp}/result.json" for run, stamp in RUNS.items()},
        "pods": runs,
        "model_revisions": {m: {"repository": ela.MODELS[m], "revision": ela.REVISIONS[ela.MODELS[m]]} for m in MODELS},
        "labels": {"chaotic": "exponent above 1e-3 and no detected period up to the scan's maximum",
                   "unresolved": "no detected period and no exponent above 1e-3",
                   "irregular": "top-k only: chaotic by the rule above, and the twin crossed a top-k set boundary "
                                "at least once in the measured steps",
                   "twin_offsets": "one (1e-6) in every scan"},
        "summaries": summaries,
        "scan_settings": scans,
        "separation_settings": separation,
        "files": files,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: receipt[k] for k in ("analysis_commit", "analysis_worktree_dirty", "created_utc", "pods",
                                              "model_revisions", "summaries")}, indent=1))
    print(f"{len(files)} files bound; {len(scans)} scans; {len(separation)} separation runs")


if __name__ == "__main__":
    main()
