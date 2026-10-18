"""Remote execution of a Codex certificate shard packet on a task-owned pod (Task 34; Claude's provisioning side).

The packet (for example codex/certificate-ladder-task34/calibration-001.json) fixes the attempt, plan, exact backend,
pinned upstream files and assets, the command and the resource envelope. Every step refuses rather than adapts.

prepare    copy the uploaded upstream files and assets to their exact absolute paths under a real data root, then
           verify every byte count and SHA-256 against the packet
preflight  exact backend (full sys.version, NumPy, Torch, python-flint, FLINT), a real data root (no symlink on its
           path), all hashes, the packet and plan hashes, every source pin of the plan against the checkout, Git HEAD
           at the dispatch commit with origin/main containing it, the fresh attempt absent, available RAM and disk
run        the packet's command, unchanged, in the checkout: aggregate RSS of its whole process tree (launch.py starts
           the worker in its own session) polled every 2 s against a limit, a hard deadline, cleanup of exactly that
           tree; records start/end, exit, peak aggregate RSS
pack       tar the whole new attempt directory and the records, with SHA-256

Each step writes a JSON record and exits nonzero on refusal or failure.
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))


def asset_paths(packet, data_root):
    snap = (data_root / "models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
            "7ae557604adf67be50417f59c2c2f167def9a775")
    return {"model.safetensors": snap / "model.safetensors", "config.json": snap / "config.json",
            "vectors.npz": data_root / "explore/llm-phase-map/vectors-l12/vectors.npz"}


def check_hashes(packet, data_root):
    bad = []
    for f in packet["upstream_files"]:
        p = Path(f["path"])
        if not p.is_file():
            bad.append({"path": str(p), "problem": "missing"})
        elif p.stat().st_size != f["bytes"] or sha(p) != f["sha256"]:
            bad.append({"path": str(p), "problem": "bytes or sha256 differ"})
    for name, p in asset_paths(packet, data_root).items():
        want = packet["expected_assets"][name]
        if not p.is_file() or sha(p) != want:
            bad.append({"path": str(p), "problem": "asset missing or sha256 differs"})
    return bad


def prepare(a):
    packet = json.loads(Path(a.packet).read_text())
    data_root, src = Path(a.data_root), Path(a.src_data)
    copied = []
    for f in packet["upstream_files"]:
        dst = Path(f["path"])
        rel = dst.relative_to(data_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src / rel, dst)
            copied.append(str(rel))
    hub = "models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    if not (data_root / hub).exists():
        shutil.copytree(src / hub, data_root / hub, symlinks=True)
    vec = "explore/llm-phase-map/vectors-l12/vectors.npz"
    (data_root / vec).parent.mkdir(parents=True, exist_ok=True)
    if not (data_root / vec).exists():
        shutil.copy2(src / vec, data_root / vec)
    bad = check_hashes(packet, data_root)
    rec = {"step": "prepare", "at": now(), "copied_upstream": len(copied), "hash_problems": bad, "ok": not bad}
    write(a.record, rec)
    print(json.dumps({k: rec[k] for k in ("step", "copied_upstream", "ok")}))
    sys.exit(0 if not bad else 1)


def preflight(a):
    packet = json.loads(Path(a.packet).read_text())
    data_root, repo = Path(a.data_root), Path(a.repo)
    checks = {}
    exp = packet["expected_backend"]
    import numpy
    import torch
    import flint
    checks["sys_version"] = (sys.version == exp["python"], sys.version)
    checks["numpy"] = (numpy.__version__ == exp["numpy"], numpy.__version__)
    checks["torch"] = (torch.__version__ == exp["torch"], torch.__version__)
    checks["python_flint"] = (flint.__version__ == exp["python_flint"], flint.__version__)
    checks["flint"] = (getattr(flint, "__FLINT_VERSION__", None) == exp["flint"], getattr(flint, "__FLINT_VERSION__", None))
    parts = [Path(*data_root.parts[:i + 1]) for i in range(1, len(data_root.parts))]
    links = [str(p) for p in parts if p.is_symlink()]
    checks["data_root_real"] = (not links and data_root.is_dir() and os.path.realpath(data_root) == str(data_root),
                                {"symlinks_on_path": links, "realpath": os.path.realpath(data_root)})
    bad = check_hashes(packet, data_root)
    checks["hashes"] = (not bad, bad)
    checks["packet_sha256"] = (sha(a.packet) == a.packet_sha256, sha(a.packet))
    plan_path = repo / packet["plan"]
    plan = json.loads(plan_path.read_text())
    checks["plan_sha256"] = (sha(plan_path) == packet["plan_sha256"], sha(plan_path))
    pins = plan.get("source_sha256", {})
    pin_bad = [k for k, v in pins.items() if not (repo / k).is_file() or sha(repo / k) != v]
    checks["source_pins"] = (bool(pins) and not pin_bad, {"pins": len(pins), "mismatched": pin_bad})
    git = lambda *x: subprocess.run(["git", *x], cwd=repo, capture_output=True, text=True)  # noqa: E731
    head = git("rev-parse", "HEAD").stdout.strip()
    anc = git("merge-base", "--is-ancestor", head, "origin/main").returncode == 0
    checks["git_head"] = (head == a.commit and anc, {"head": head, "origin_main_contains": anc})
    attempt = data_root / "codex/llm-p8-certificate" / packet["attempt"]
    checks["fresh_attempt_absent"] = (not attempt.exists(), str(attempt))
    mem = {line.split(":")[0]: int(line.split()[1]) for line in open("/proc/meminfo")}
    avail_gib = mem["MemAvailable"] / 1024 ** 2
    checks["ram_available_gib"] = (avail_gib >= a.min_ram_gib, round(avail_gib, 2))
    du = shutil.disk_usage(data_root)
    checks["disk_free_gib"] = (du.free / 1024 ** 3 >= a.min_disk_gib, round(du.free / 1024 ** 3, 2))
    ok = all(v[0] for v in checks.values())
    rec = {"step": "preflight", "at": now(), "ok": ok, "commit": a.commit,
           "checks": {k: {"pass": bool(v[0]), "value": v[1]} for k, v in checks.items()}}
    write(a.record, rec)
    print(json.dumps({"step": "preflight", "ok": ok, "failed": [k for k, v in checks.items() if not v[0]]}))
    sys.exit(0 if ok else 1)


def tree(root):
    """PIDs of root and all its descendants (launch.py starts the worker in its own session, so a process group alone
    would miss it)."""
    kids = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            st = open(f"/proc/{d}/stat").read()
            ppid = int(st[st.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(kids.get(p, []))
    return out


def rss_kib(pids):
    total = 0
    for p in pids:
        try:
            for line in open(f"/proc/{p}/status"):
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
        except OSError:
            continue
    return total


def kill_tree(pids, sig):
    groups = set()
    for p in pids:
        try:
            groups.add(os.getpgid(p))
        except OSError:
            pass
    for g in groups:
        try:
            os.killpg(g, sig)
        except OSError:
            pass


def run(a):
    limit_kib = int(a.rss_limit_gib * 1024 ** 2)
    t0, started = time.time(), now()
    proc = subprocess.Popen(a.cmd, shell=True, cwd=a.repo, start_new_session=True,
                            stdout=open(a.stdout, "w"), stderr=subprocess.STDOUT)
    peak, reason, peak_procs = 0, None, 0
    while proc.poll() is None:
        pids = tree(proc.pid)
        rss = rss_kib(pids)
        if rss > peak:
            peak, peak_procs = rss, len(pids)
        if rss > limit_kib:
            reason = "aggregate RSS of the process tree above the limit"
        elif time.time() - t0 > a.deadline_seconds:
            reason = "hard deadline"
        if reason:
            kill_tree(pids, signal.SIGTERM)
            time.sleep(10)
            kill_tree(tree(proc.pid) + pids, signal.SIGKILL)
            break
        time.sleep(2)
    rc = proc.wait()
    left = [p for p in tree(proc.pid) if p != proc.pid]
    rec = {"step": "run", "command": a.cmd, "cwd": a.repo, "started": started, "ended": now(),
           "elapsed_seconds": round(time.time() - t0, 3), "exit_code": rc, "killed_by_watchdog": reason,
           "peak_aggregate_rss_kib": peak, "processes_at_peak": peak_procs, "rss_limit_kib": limit_kib,
           "deadline_seconds": a.deadline_seconds, "owned_processes_left": left}
    write(a.record, rec)
    print(json.dumps({k: rec[k] for k in ("step", "exit_code", "elapsed_seconds", "peak_aggregate_rss_kib",
                                          "killed_by_watchdog")}))
    sys.exit(0 if rc == 0 and reason is None else 1)


def pack(a):
    packet = json.loads(Path(a.packet).read_text())
    attempt = Path(a.data_root) / "codex/llm-p8-certificate" / packet["attempt"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / f"{packet['attempt']}.tar"
    with tarfile.open(tar_path, "w") as t:
        if attempt.exists():
            t.add(attempt, arcname=str(attempt.relative_to(a.data_root)))
    rec = {"step": "pack", "at": now(), "attempt_present": attempt.exists(), "archive": tar_path.name,
           "archive_sha256": sha(tar_path), "archive_bytes": tar_path.stat().st_size}
    write(out / "pack.json", rec)
    print(json.dumps(rec))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    for name in ("prepare", "preflight", "pack"):
        s = sub.add_parser(name)
        s.add_argument("--packet", required=True)
        s.add_argument("--data-root", default="/data2/neural-shrimp-data")
    sub.choices["prepare"].add_argument("--src-data", required=True)
    sub.choices["prepare"].add_argument("--record", required=True)
    p = sub.choices["preflight"]
    p.add_argument("--repo", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--packet-sha256", required=True)
    p.add_argument("--min-ram-gib", type=float, default=24)
    p.add_argument("--min-disk-gib", type=float, default=16)
    p.add_argument("--record", required=True)
    sub.choices["pack"].add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", required=True)
    r.add_argument("--cmd", required=True)
    r.add_argument("--deadline-seconds", type=float, default=3900)
    r.add_argument("--rss-limit-gib", type=float, default=18)
    r.add_argument("--stdout", required=True)
    r.add_argument("--record", required=True)
    a = ap.parse_args()
    {"prepare": prepare, "preflight": preflight, "run": run, "pack": pack}[a.step](a)


if __name__ == "__main__":
    main()
