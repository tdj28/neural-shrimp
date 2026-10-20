"""Root-owned Task40 local CPU admission, bounded child, and immutable receipt."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT = Path("/data2/neural-shrimp")
OUT = ROOT / "codex/arxiv-gonogo2-task40"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cpu", type=int, required=True)
    ap.add_argument("--wall", type=int, required=True)
    ap.add_argument("--gib", type=int, default=4)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    command = a.command[1:] if a.command[0] == "--" else a.command
    if command[:3] != ["uv", "run", "--no-sync"] or command[3] != "python":
        raise ValueError("Only the declared uv Python lane")
    script = Path(command[4]).resolve()
    if script.parent != OUT:
        raise ValueError("Only Task40 script execution")
    if "/" in a.name or not a.name:
        raise ValueError("Invalid receipt name")
    paths = {k: OUT / f"{a.name}.{k}" for k in ("receipt.json", "stdout.log", "stderr.log")}
    if any(p.exists() for p in paths.values()):
        raise ValueError("Attempt name already used")
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    def limits():
        os.nice(10)
        resource.setrlimit(resource.RLIMIT_AS, (a.gib * 1024**3,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (a.cpu,) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2,) * 2)
    rec = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(), "argv": command,
        "limits": {"cpu_seconds": a.cpu, "parent_wall_seconds": a.wall, "address_space_GiB": a.gib,
                   "file_bytes": 256*1024**2, "nice_increment": 10, "numerical_threads": 1},
        "environment": {k: env[k] for k in ("HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                          "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "timeout": False,
    }
    with paths["receipt.json"].open("x") as receipt:
        with paths["stdout.log"].open("xb") as stdout, paths["stderr.log"].open("xb") as stderr:
            start = time.monotonic()
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                     preexec_fn=limits, start_new_session=True)
            rec["pid"] = child.pid
            try:
                rec["exit_code"] = child.wait(timeout=a.wall)
            except subprocess.TimeoutExpired:
                rec["timeout"] = True
                # Only our unreaped direct child and its freshly created group.
                os.killpg(child.pid, signal.SIGKILL)
                rec["exit_code"] = child.wait()
            rec["parent_launch_through_reap_seconds"] = time.monotonic()-start
        rec["child_reaped"] = True
        rec["logs"] = {k: {"sha256": hashlib.sha256(paths[k].read_bytes()).hexdigest(),
                           "bytes": paths[k].stat().st_size}
                       for k in ("stdout.log", "stderr.log")}
        json.dump(rec, receipt, indent=2)
        receipt.write("\n")
    print(json.dumps(rec))
    return rec["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
