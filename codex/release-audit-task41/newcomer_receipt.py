"""Read-only receipt for the single Task 41 newcomer run; no model calls."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("/data2/neural-shrimp")
SCRATCH = Path("/data2/neural-shrimp-data/scratch/audit-codex-1020b")
CLONE = SCRATCH / "newcomer/neural-shrimp"
DATA = SCRATCH / "newcomer/neural-shrimp-data"
OUTPUT = DATA / "explore/llm-soft-loop/example"
TAG = "1314528118ac1300c2cffd34fb60cfbfc67d2c82"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEST = ROOT / "codex/release-audit-task41/newcomer-evidence.json"


def pin(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if identity(before) != identity(after):
        raise ValueError("file changed while hashing")
    result = {"path": str(path), "bytes": before.st_size, "sha256": h.hexdigest()}
    if path.is_symlink():
        result["literal_link"] = os.readlink(path)
    return result


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(CLONE), *args], text=True, timeout=30
    ).strip()


def timed_step(name):
    path = SCRATCH / "logs" / f"{name}-time.txt"
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key in {"exit_code", "max_rss_kib"}:
                values[key] = int(value)
            elif key == "elapsed_seconds":
                values[key] = float(value)
    values["receipt"] = pin(path)
    values["log"] = pin(SCRATCH / "logs" / f"{name}.log")
    start = SCRATCH / "logs" / f"{name}-start.txt"
    if start.exists():
        values["start_utc"] = start.read_text().strip()
    return values


def main():
    import numpy as np

    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("offline required")
    if DEST.exists():
        raise FileExistsError(DEST)
    start = time.monotonic()
    result = {
        "schema": "task41-newcomer-receipt-v1",
        "script": pin(__file__),
        "clone": str(CLONE),
        "data": str(DATA),
        "expected_tag_commit": TAG,
        "head": git("rev-parse", "HEAD"),
        "peeled_tag": git("rev-parse", "arxiv-v1^{}"),
        "tracked_diff": git("diff", "--name-only", "HEAD"),
        "model_revision": REVISION,
        "steps": {name: timed_step(name) for name in
                  ("clone", "sync", "scan", "test-sync", "tests")},
        "files": [],
        "model_files": [],
        "limitations": [
            "A fresh clone and virtual environment on an existing host, not a new operating system.",
            "README scientific options retained; explicit --threads 6 added for assignment's resource limit.",
            "Dev extra added only for the selected tests, because README extras omit pytest.",
            "Finite-horizon scan labels are smoke-test outputs, not certified periodic or chaotic attractors.",
            "Tests are eight selected existing tests, not the full suite or a new model audit.",
        ],
    }
    if result["head"] != TAG or result["peeled_tag"] != TAG or result["tracked_diff"]:
        raise ValueError("clone source changed or wrong tag")
    summary = json.loads((OUTPUT / "scan.json").read_text())
    if summary["cells"] != 64:
        raise ValueError("wrong scan size")
    args = summary["args"]
    expected = {
        "device": "cpu", "dtype": "float64", "resolution": [8, 8],
        "alpha": [0.0625, 0.0775], "log_temperature": [-0.17, -0.125],
        "out": "llm-soft-loop/example", "threads": 6,
    }
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f"unexpected scan option {key}")
    result["scan"] = {
        "args": args, "cells": summary["cells"],
        "reported_seconds": summary["seconds"],
        "period_counts": summary["period_counts"],
        "lyapunov_positive": summary["lyapunov_positive"],
        "lyapunov_range": summary["lyapunov_range"],
        "arrays": {},
    }
    with np.load(OUTPUT / "scan.npz", allow_pickle=False) as arrays:
        for key in arrays.files:
            arr = arrays[key]
            record = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            if np.issubdtype(arr.dtype, np.number):
                record["finite"] = bool(np.isfinite(arr).all())
                if not record["finite"]:
                    raise ValueError("nonfinite scan array")
            result["scan"]["arrays"][key] = record
        for key in ("lyapunov", "period", "entropy"):
            if arrays[key].shape != (8, 8):
                raise ValueError("unexpected scan array shape")
    for path in sorted(OUTPUT.iterdir()):
        if not path.is_file() or path.name.startswith(".env"):
            raise ValueError("unexpected output member")
        result["files"].append(pin(path))
    model = DATA / "models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / REVISION
    for name in ("config.json", "tokenizer_config.json", "model.safetensors",
                 "tokenizer.json", "generation_config.json", "vocab.json", "merges.txt"):
        result["model_files"].append(pin(model / name))
    copied = DATA / "explore/llm-phase-map/vectors-l12"
    result["steering_files"] = []
    for path in sorted(copied.iterdir()):
        if path.is_file() and not path.name.startswith(".env"):
            original = CLONE / "docs/steering-vectors/vectors-l12" / path.name
            check = {"copied": pin(path), "source": pin(original)}
            check["bytes_equal"] = check["copied"]["sha256"] == check["source"]["sha256"]
            if not check["bytes_equal"]:
                raise ValueError("steering copy differs")
            result["steering_files"].append(check)
    testlog = (SCRATCH / "logs/tests.log").read_text()
    result["test_summary_lines"] = [
        line for line in testlog.splitlines()
        if " passed" in line or " failed" in line or " skipped" in line
    ]
    result["all_step_exits_zero"] = all(
        step.get("exit_code") == 0 for step in result["steps"].values()
    )
    result["elapsed_seconds"] = time.monotonic() - start
    with DEST.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(DEST), "all_step_exits_zero": result["all_step_exits_zero"]}))


if __name__ == "__main__":
    main()
