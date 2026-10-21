"""Collect and audit the already-launched BSC2, BSZ2 and GBZ1 runs.

Read-only ownership checks; no new pods, retries, spending or parameter changes.
The separate run controllers/watchdogs own retrieval and verified termination.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import runpod_scan_group as group

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"
RUNS = DATA / "runpod/baby-shrimp-zoom"
AUDITS = DATA / "audits/baby-shrimp-zoom"
SPECS = {
    "roots": ("bsc2-20261021", ["neural-shrimp-bsc2"], "4df71ac702c099d43be55239e0942aa4ab8052ed"),
    "junction": ("bsz2-20261021", ["neural-shrimp-bsz2-junction"], "7ee72bb291fed959e720e83451ea9e34d0788e57"),
    "grandchild": ("gbz1-20261021", [f"neural-shrimp-gbz1-c{i:02d}" for i in range(4)], "e0c73d86b299a5c418d7d4e1fa2713fd9031001f"),
}
SOURCES = ["tools/baby_shrimp_rescan_collect.py", "tools/baby_shrimp_rescan_finish.py",
           "tools/baby_shrimp_crossing_finish.py", "tools/baby_shrimp_transect_finish.py", "tools/baby_shrimp_finish.py"]


def ready_workers(root, names, config, ledger):
    owned = group.owned(config, ledger)
    result = []
    for name in names:
        entries = [p for p in owned.values() if p["name"] == name]
        if len(entries) > 1:
            raise ValueError("ambiguous owned worker")
        if not entries or not entries[0]["terminated"]:
            return None
        files = list((root / "workers" / name).glob("*/result.json"))
        if len(files) != 1:
            raise ValueError("terminated worker lacks unique retrieval receipt")
        row = json.loads(files[0].read_text())
        if not (row["pod_id"] == entries[0]["pod_id"] and row["commit"] == config["source_commit"]
                and row["hash_verified"] and row["runner_returncode"] == 0):
            raise ValueError("retrieval/source gate failed")
        result.append(files[0].parent)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=15600)
    args = ap.parse_args()
    if not 0 < args.timeout <= 16200:
        ap.error("collection must be bounded to4.5hours")
    args.out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {}
    for source in SOURCES:
        actual = hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
        blob = subprocess.check_output(["git", "show", commit+":"+source], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != actual:
            raise ValueError("commit collection/audit source first")
        hashes[source] = actual
    configs = {}
    for name, (path, names, source) in SPECS.items():
        configs[name] = json.loads((RUNS / path / "config.json").read_text())
        if configs[name]["names"] != names or configs[name]["source_commit"] != source:
            raise ValueError("existing group identity differs")
    status = {"source_commit": commit, "source_sha256": hashes,
              "started_utc": datetime.now(timezone.utc).isoformat(), "completed": {}, "failures": {}}
    env = {**os.environ, **{k:"8" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    destinations = {"roots": AUDITS / "20261021-trace-slices-audit-01",
                    "junction": AUDITS / "20261021-junction-audit-01",
                    "grandchild": AUDITS / "20261021-grandchild-audit-01"}
    deadline = time.monotonic()+args.timeout
    try:
        while len(status["completed"])+len(status["failures"]) < len(SPECS):
            if time.monotonic() > deadline:
                raise TimeoutError("bounded collection expired; owned-pod guards remain independent")
            for name, (path, names, source) in SPECS.items():
                if name in status["completed"] or name in status["failures"]:
                    continue
                if name == "junction" and "roots" not in status["completed"] and "roots" not in status["failures"]:
                    continue
                try:
                    workers = ready_workers(RUNS / path, names, configs[name], group.ledger())
                    if workers is None:
                        continue
                    for file, sha in hashes.items():
                        if hashlib.sha256((ROOT/file).read_bytes()).hexdigest() != sha:
                            raise ValueError("audit source changed; retain this collection and start a new attempt")
                    if name == "roots":
                        command = [sys.executable, str(ROOT/"tools/baby_shrimp_crossing_finish.py"), "--worker", str(workers[0])]
                    else:
                        plan = "junction-scan-plan.json" if name == "junction" else "grandchild-scan-plan.json"
                        command = [sys.executable, str(ROOT/"tools/baby_shrimp_rescan_finish.py"), "--plan",
                                   str(ROOT/"docs/baby-shrimp-zoom"/plan), "--workers", *map(str, workers)]
                        if name == "junction" and "roots" in status["completed"]:
                            command += ["--curves", str(destinations["roots"]/"receipt.json")]
                    command += ["--out", str(destinations[name])]
                    with (args.out/f"{name}.log").open("x") as log:
                        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=900, check=True)
                    receipt = destinations[name]/"receipt.json"
                    status["completed"][name] = {"time": datetime.now(timezone.utc).isoformat(), "receipt": str(receipt),
                                                "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                                                "visual_inspection": "still required before user delivery"}
                    print(json.dumps({"audit_complete": name}), flush=True)
                except Exception as error:
                    status["failures"][name] = {"time": datetime.now(timezone.utc).isoformat(),
                                               "type": type(error).__name__, "error": str(error)}
                    print(json.dumps({"audit_failed": name, "error": str(error)}), flush=True)
            status["last_check_utc"] = datetime.now(timezone.utc).isoformat()
            group.save(args.out/"status.json", status)
            if len(status["completed"])+len(status["failures"]) < len(SPECS):
                time.sleep(30)
        if status["failures"]:
            raise ValueError("some audits failed; preserved all outputs and logs")
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        group.save(args.out/"status.json", status)


if __name__ == "__main__":
    main()
