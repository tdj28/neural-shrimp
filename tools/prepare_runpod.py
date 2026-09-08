#!/usr/bin/env python3
"""Offline source packaging, quotation arithmetic, and a bounded workload runner.

This module never contacts RunPod and cannot provision or terminate resources.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tarfile
import time

ALLOWED_ROOTS = {"src", "neural_shrimp", "tools", "experiments", "configs", "schemas"}
ALLOWED_FILES = {"pyproject.toml", "requirements.txt", "requirements.lock", "uv.lock", "README.md"}
SECRET_NAMES = re.compile(r"(^\.env($|\.)|\.pem$|\.key$|^id_(rsa|ed25519)$|credential|secret|token)", re.I)
SECRET_CONTENT = re.compile(rb"-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}")
FORBIDDEN_PARTS = {".git", ".venv", "venv", "__pycache__", ".cache", "node_modules"}
RECEIPT = "runpod-bundle.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)


def checked_path(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or str(path) != name or any(p in FORBIDDEN_PARTS or SECRET_NAMES.search(p) for p in path.parts)):
        raise ValueError(f"Unsafe or credential-like archive path: {name!r}")
    return name


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def bundle(repo: Path, manifest: str, extra_inputs: list[str], output: Path) -> dict:
    repo = repo.resolve()
    if git(repo, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("Source worktree must be clean, including untracked files")
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected a full Git SHA-1 commit")
    requested = {checked_path(manifest), *(checked_path(p) for p in extra_inputs)}
    records = git(repo, "ls-tree", "-r", "-z", "HEAD").split(b"\0")
    files: dict[str, tuple[bytes, int]] = {}
    for record in records:
        if not record:
            continue
        meta, raw_name = record.split(b"\t", 1)
        mode, kind, object_id = meta.decode().split()
        name = raw_name.decode("utf-8")
        if PurePosixPath(name).parts[0] not in ALLOWED_ROOTS and name not in ALLOWED_FILES and name not in requested:
            continue
        checked_path(name)
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"Only regular committed files are allowed: {name}")
        data = git(repo, "cat-file", "blob", object_id)
        if SECRET_CONTENT.search(data):
            raise ValueError(f"Possible embedded credential detected: {name}")
        files[name] = (data, 0o755 if mode == "100755" else 0o644)
    if missing := requested - files.keys():
        raise ValueError(f"Inputs must be committed regular files: {sorted(missing)}")
    if not {"tools/prepare_runpod.py", "tools/runpod_entrypoint.sh"}.issubset(files):
        raise ValueError("Both RunPod tools must be committed before packaging")
    receipt = {
        "schema_version": 1, "source_commit": commit, "source_clean": True,
        "manifest": manifest, "inputs": sorted(requested),
        "files": {name: {"sha256": sha256(data), "bytes": len(data), "mode": mode}
                  for name, (data, mode) in sorted(files.items())},
        "runtime_scope": "CPU NumPy/SciPy reference; runtime dependencies are not vendored",
    }
    receipt_bytes = json_bytes(receipt)
    # Exclusive creation avoids replacing a frozen payload. Stable metadata makes
    # the tar bytes reproducible from the same commit and input selection.
    with output.open("xb") as handle, tarfile.open(fileobj=handle, mode="w") as archive:
        for name, (data, mode) in sorted({**files, RECEIPT: (receipt_bytes, 0o644)}.items()):
            info = tarfile.TarInfo(f"payload/{name}")
            info.size, info.mode, info.mtime = len(data), mode, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    return {"archive": str(output.resolve()), "archive_sha256": sha256(output.read_bytes()),
            "receipt_sha256": sha256(receipt_bytes), "source_commit": commit,
            "manifest_sha256": sha256(files[manifest][0]),
            "file_count": len(files)}


def verify_bundle(root: Path, expected_commit: str, expected_receipt_sha256: str) -> dict:
    root = root.resolve()
    receipt_path = root / RECEIPT
    if receipt_path.is_symlink():
        raise ValueError("Receipt cannot be a symlink")
    raw = receipt_path.read_bytes()
    if sha256(raw) != expected_receipt_sha256:
        raise ValueError("Bundle receipt hash mismatch")
    receipt = json.loads(raw)
    if receipt["source_commit"] != expected_commit or not receipt["source_clean"]:
        raise ValueError("Source commit/cleanliness mismatch")
    names = {checked_path(name) for name in receipt["files"]}
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in unpacked bundle: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != names | {RECEIPT}:
        raise ValueError("Unpacked bundle contains missing or extra files")
    for name in names:
        data = (root / name).read_bytes()
        record = receipt["files"][name]
        if len(data) != record["bytes"] or sha256(data) != record["sha256"]:
            raise ValueError(f"File integrity mismatch: {name}")
    if receipt["manifest"] not in names or not set(receipt["inputs"]).issubset(names):
        raise ValueError("Missing frozen manifest/input")
    return receipt


def positive(value: float, label: str, zero: bool = False) -> float:
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{label} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def estimate(args: argparse.Namespace) -> dict:
    for key in ("unit_seconds", "units", "hourly_usd", "slowdown_factor", "hard_wall_seconds", "hard_budget_usd", "no_progress_seconds"):
        positive(getattr(args, key), key)
    for key in ("setup_seconds", "retrieval_seconds", "storage_transfer_usd"):
        positive(getattr(args, key), key, zero=True)
    if args.slowdown_factor < 1:
        raise ValueError("Slow-case factor must be at least 1")
    if args.units != int(args.units):
        raise ValueError("Work-unit count must be an integer")
    if args.retrieval_seconds <= 0:
        raise ValueError("Reserve positive time for artifact retrieval and termination")
    for key in ("quote_source", "quote_observed_at", "measurement_ref", "measurement_host"):
        if not getattr(args, key).strip():
            raise ValueError(f"Missing {key}")
    dt.datetime.fromisoformat(args.quote_observed_at.replace("Z", "+00:00"))
    base = args.setup_seconds + args.unit_seconds * args.units + args.retrieval_seconds
    slow = args.setup_seconds + args.unit_seconds * args.units * args.slowdown_factor + args.retrieval_seconds
    cost = lambda seconds: seconds / 3600 * args.hourly_usd + args.storage_transfer_usd
    if slow > args.hard_wall_seconds:
        raise ValueError("Slow-case lifetime exceeds hard wall time; rescope before approval")
    if cost(args.hard_wall_seconds) > args.hard_budget_usd:
        raise ValueError("Worst permitted lifetime exceeds hard spend ceiling")
    if args.no_progress_seconds < args.unit_seconds * args.slowdown_factor:
        raise ValueError("No-progress timeout is shorter than a slow work unit")
    return {"schema_version": 1, "inputs": {key: value for key, value in vars(args).items()
             if key not in {"command", "output"}}, "base_seconds": base, "slow_seconds": slow,
            "base_cost_usd": cost(base), "slow_cost_usd": cost(slow),
            "hard_lifetime_cost_usd": cost(args.hard_wall_seconds),
            "approval": "NOT APPROVED: offline arithmetic only",
            "limitations": "A local CPU timing is not remote throughput qualification. No CUDA implementation or speedup is assumed."}


def validate_ownership(value: dict, task_id: str, environment_pod_id: str | None) -> None:
    for key in ("task_id", "pod_id", "preexisting_resource_ids", "create_response_sha256",
                "billed_start_epoch", "controller_watchdog_ref", "runtime_image_digest"):
        if key not in value:
            raise ValueError(f"Ownership receipt missing {key}")
    if (not isinstance(value["preexisting_resource_ids"], list)
            or not all(isinstance(item, str) and item for item in value["preexisting_resource_ids"])):
        raise ValueError("Pre-existing resource inventory must be a list of provider IDs")
    if value["task_id"] != task_id or not isinstance(value["pod_id"], str) or not value["pod_id"]:
        raise ValueError("Ownership receipt belongs to a different task or has no pod ID")
    if value["pod_id"] in value["preexisting_resource_ids"]:
        raise ValueError("Task may not operate on a pre-existing resource")
    if not environment_pod_id or value["pod_id"] != environment_pod_id:
        raise ValueError("RUNPOD_POD_ID does not match the provider ownership receipt")
    if not re.fullmatch(r"[0-9a-f]{64}", value["create_response_sha256"]):
        raise ValueError("A SHA256-bound provider creation response is required")
    if not re.search(r"@sha256:[0-9a-f]{64}$", value["runtime_image_digest"]):
        raise ValueError("Pin the actual runtime image by digest")
    if not value["controller_watchdog_ref"]:
        raise ValueError("An independently running controller watchdog must be recorded")
    positive(float(value["billed_start_epoch"]), "billed_start_epoch")
    if value["billed_start_epoch"] > time.time() + 5:
        raise ValueError("Billing start cannot be in the future")


def artifact_ledger(root: Path) -> dict:
    records = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Output contains a symlink; retain for manual review")
        if path.is_file():
            digest = hashlib.sha256()
            count = 0
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    count += len(chunk)
                    digest.update(chunk)
            records[path.relative_to(root).as_posix()] = {"bytes": count, "sha256": digest.hexdigest()}
    return records


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_workload(args: argparse.Namespace) -> dict:
    root = args.bundle_root.resolve()
    receipt = verify_bundle(root, args.expected_commit, args.expected_receipt_sha256)
    cost_raw = args.cost_report.read_bytes()
    ownership_raw = args.ownership_receipt.read_bytes()
    if sha256(cost_raw) != args.expected_cost_sha256:
        raise ValueError("Cost-report hash mismatch")
    if sha256(ownership_raw) != args.expected_ownership_sha256:
        raise ValueError("Ownership-receipt hash mismatch")
    cost = json.loads(cost_raw)
    # Recompute every bound rather than trusting derived JSON fields.
    validated = estimate(argparse.Namespace(**cost["inputs"]))
    if any(cost.get(k) != validated[k] for k in ("base_seconds", "slow_seconds", "hard_lifetime_cost_usd")):
        raise ValueError("Cost report arithmetic mismatch")
    ownership = json.loads(ownership_raw)
    validate_ownership(ownership, args.task_id, os.environ.get("RUNPOD_POD_ID"))
    if not args.approval_ref.strip():
        raise ValueError("Explicit human approval reference for this costed run is required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.attempt_id):
        raise ValueError("Invalid attempt ID")
    base = args.output_base.resolve()
    if base == root or root in base.parents:
        raise ValueError("Outputs must be outside the frozen source bundle")
    bounds = cost["inputs"]
    deadline = ownership["billed_start_epoch"] + bounds["hard_wall_seconds"] - bounds["retrieval_seconds"] - 5
    if deadline <= time.time():
        raise ValueError("No compute time remains before retrieval/termination reserve")
    base.mkdir(parents=True, exist_ok=True)
    attempt = base / args.attempt_id
    attempt.mkdir()  # An attempt can never be reused, even after failure.
    output = attempt / "results"
    command = [sys.executable, "-B", "-m", "neural_shrimp.cli", "run", "--manifest",
               str(root / receipt["manifest"]), "--output", str(output)]
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = str(root / "src") if (root / "src").exists() else str(root)
    environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    # The source archive requires no credential. Do not inherit typical provider,
    # repository, model-hub or cloud credentials into the scientific process.
    for key in list(environment):
        if any(marker in key.upper() for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "PRIVATE_KEY", "AWS_ACCESS", "AWS_SESSION")):
            environment.pop(key)
    run_receipt = {"schema_version": 1, "task_id": args.task_id, "attempt_id": args.attempt_id,
                   "source_commit": args.expected_commit, "bundle_receipt_sha256": args.expected_receipt_sha256,
                   "cost_report_sha256": sha256(cost_raw), "ownership_receipt_sha256": sha256(ownership_raw),
                   "pod_id": ownership["pod_id"], "approval_ref": args.approval_ref,
                   "command": command, "started_epoch": time.time(), "runtime": "CPU reference",
                   "runtime_image_digest": ownership["runtime_image_digest"],
                   "compute_deadline_epoch": deadline, "python": sys.version}
    write_new(attempt / "launch.json", json_bytes(run_receipt))
    reason = "process_exit"
    process = None
    last_progress = time.monotonic()
    previous_signature = None
    old_term_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise InterruptedError(f"Workload runner received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (attempt / "stdout.log").open("xb") as stdout, (attempt / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=root, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
            while process.poll() is None:
                if time.time() >= deadline:
                    reason = "hard_compute_deadline"
                    stop_process(process)
                    break
                # Progress means changed scientific artifacts, not a chatty log.
                signature = tuple(sorted((str(p.relative_to(output)), p.stat().st_size, p.stat().st_mtime_ns)
                                         for p in output.rglob("*") if p.is_file())) if output.exists() else ()
                if signature != previous_signature:
                    last_progress, previous_signature = time.monotonic(), signature
                if time.monotonic() - last_progress >= bounds["no_progress_seconds"]:
                    reason = "no_progress_timeout"
                    stop_process(process)
                    break
                time.sleep(0.2)
    except (InterruptedError, KeyboardInterrupt):
        reason = "runner_interrupted"
    finally:
        signal.signal(signal.SIGTERM, old_term_handler)
        if process is not None:
            stop_process(process)
        result = {**run_receipt, "ended_epoch": time.time(), "exit_code": process.returncode if process else None,
                  "reason": reason, "status": "process_exit_zero" if process and process.returncode == 0 and reason == "process_exit" else "failed_or_incomplete",
                  "pod_terminated": False,
                  "next_action": "Controller: retrieve, verify SHA256 ledger, terminate owned pod, and verify absence immediately."}
        write_new(attempt / "execution.json", json_bytes(result))
        write_new(attempt / "SHA256.json", json_bytes({"files": artifact_ledger(attempt)}))
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bundle", help="Export exact clean commit and selected frozen inputs; no network")
    b.add_argument("--repo", type=Path, default=Path.cwd())
    b.add_argument("--manifest", required=True)
    b.add_argument("--input", action="append", default=[])
    b.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify", help="Verify an already-unpacked source payload")
    v.add_argument("--bundle-root", type=Path, required=True)
    v.add_argument("--expected-commit", required=True)
    v.add_argument("--expected-receipt-sha256", required=True)
    e = sub.add_parser("estimate", help="Offline arithmetic from measured time and a supplied current quote")
    for name in ("unit-seconds", "units", "hourly-usd", "setup-seconds", "retrieval-seconds", "storage-transfer-usd", "slowdown-factor", "hard-wall-seconds", "hard-budget-usd", "no-progress-seconds"):
        e.add_argument(f"--{name}", type=float, required=True)
    for name in ("quote-source", "quote-observed-at", "measurement-ref", "measurement-host"):
        e.add_argument(f"--{name}", required=True)
    e.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("run", help="Bound only the workload on an already-approved owned pod")
    for name in ("bundle-root", "output-base", "cost-report", "ownership-receipt"):
        r.add_argument(f"--{name}", type=Path, required=True)
    for name in ("expected-commit", "expected-receipt-sha256", "expected-cost-sha256", "expected-ownership-sha256", "task-id", "attempt-id", "approval-ref"):
        r.add_argument(f"--{name}", required=True)
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "bundle":
            result = bundle(args.repo, args.manifest, args.input, args.output)
        elif args.command == "verify":
            result = verify_bundle(args.bundle_root, args.expected_commit, args.expected_receipt_sha256)
        elif args.command == "estimate":
            result = estimate(args)
            write_new(args.output, json_bytes(result))
        else:
            result = run_workload(args)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if args.command != "run" or result["status"] == "process_exit_zero" else 1
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
