"""Remote execution of a Codex certificate shard packet on a task-owned pod (Task 34; Claude's provisioning side).

The packet (for example codex/certificate-ladder-task34/calibration-001.json) fixes the attempt, plan, exact backend,
pinned upstream files and assets, the command and the resource envelope. Every step refuses rather than adapts.

prepare    copy the uploaded upstream files and assets to their exact absolute paths under a real data root, then
           verify every byte count and SHA-256 against the packet
preflight  exact backend (full sys.version, NumPy, Torch, python-flint, FLINT), a real data root (no symlink on its
           path), all hashes, the packet and plan hashes, every source pin of the plan against the checkout, Git HEAD
           at the dispatch commit with origin/main containing it, the fresh attempt absent, available RAM and disk
run        the packet's command, unchanged, in the checkout. Owned processes are identified by (pid, start time, uid)
           when observed: descendants by parentage and, after reparenting, members of a recorded session or group
           (launch.py starts the worker in its own session). Aggregate RSS of the live owned processes is polled
           every 2 s against a limit, with a hard deadline; signals go through pidfds re-verified against the recorded
           identity, never to a whole group. After exit, a closure audit; any owned survivor makes the run fail.
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


def describe_file(p):
    p = Path(p)
    d = {"path": str(p), "exists": p.exists(), "is_symlink": p.is_symlink()}
    if p.is_symlink():
        d["link_target"] = os.readlink(p)
    if p.is_file():
        d.update(bytes=p.stat().st_size, sha256=sha(p))
    return d


def check_hashes(packet, data_root):
    bad = []
    for f in packet["upstream_files"]:
        p = Path(f["path"])
        if not p.is_file():
            bad.append({"problem": "missing", **describe_file(p)})
        elif p.stat().st_size != f["bytes"] or sha(p) != f["sha256"]:
            bad.append({"problem": "bytes or sha256 differ", "want_bytes": f["bytes"], "want_sha256": f["sha256"],
                        **describe_file(p)})
    for name, p in asset_paths(packet, data_root).items():
        want = packet["expected_assets"][name]
        if not p.is_file():
            bad.append({"problem": "asset missing", **describe_file(p)})
        elif sha(p) != want:
            bad.append({"problem": "asset sha256 differs", "want_sha256": want, **describe_file(p)})
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
    snap = ("models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
            "7ae557604adf67be50417f59c2c2f167def9a775")
    source_files = {}
    (data_root / snap).mkdir(parents=True, exist_ok=True)
    if (src / snap).is_dir():
        for f in sorted((src / snap).iterdir()):           # real files, symlinks into the blob store dereferenced
            source_files[f.name] = describe_file(f)
            if f.is_file() and not (data_root / snap / f.name).exists():
                shutil.copy2(f, data_root / snap / f.name, follow_symlinks=True)
    vec = "explore/llm-phase-map/vectors-l12/vectors.npz"
    (data_root / vec).parent.mkdir(parents=True, exist_ok=True)
    if not (data_root / vec).exists():
        shutil.copy2(src / vec, data_root / vec)
    bad = check_hashes(packet, data_root)
    rec = {"step": "prepare", "at": now(), "copied_upstream": len(copied), "model_source_snapshot": source_files,
           "hash_problems": bad, "ok": not bad}
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
    # every third-party module the certifier imports anywhere (also inside functions), at the local pinned versions
    import importlib
    import importlib.metadata as md
    want = {"flint": ("python-flint", "0.9.0"), "numpy": ("numpy", "2.5.3"), "safetensors": ("safetensors", "0.8.0"),
            "threadpoolctl": ("threadpoolctl", "3.7.0"), "tokenizers": ("tokenizers", "0.23.3"),
            "torch": ("torch", "2.14.1")}           # build string (+cu130) checked above
    imp = {}
    for mod, (dist, ver) in want.items():
        try:
            importlib.import_module(mod)
            got = md.version(dist)
            # pip records torch's local label (2.14.1+cu130) in its metadata, uv did not; the exact build string is
            # checked separately through torch.__version__, so here only the public version must match
            imp[mod] = (got.split("+")[0] == ver, got)
        except Exception as e:                       # noqa: BLE001 - any import failure is a refusal
            imp[mod] = (False, f"import failed: {e}")
    checks["third_party_imports"] = (all(v[0] for v in imp.values()), {k: v[1] for k, v in imp.items()})
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


def snapshot():
    """pid -> identity fields (ppid, pgid, sid, start in clock ticks since boot, real uid, command) of every readable
    process."""
    procs = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            st = open(f"/proc/{d}/stat").read()
            rest = st[st.rindex(")") + 2:].split()
            ppid, pgid, sid, start = int(rest[1]), int(rest[2]), int(rest[3]), int(rest[19])
            uid = next(int(line.split()[1]) for line in open(f"/proc/{d}/status") if line.startswith("Uid:"))
            cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()[:300]
        except (OSError, ValueError, IndexError, StopIteration):
            continue
        procs[int(d)] = {"ppid": ppid, "pgid": pgid, "sid": sid, "start": start, "uid": uid, "cmd": cmd}
    return procs


def adopt(procs, reg, uid):
    """Add to the registry every descendant (by parentage) of a live registered process, and every process of the same
    uid in a recorded session or process group (members reparented after their parent exited)."""
    kids = {}
    for pid, q in procs.items():
        kids.setdefault(q["ppid"], []).append(pid)
    todo = [k[0] for k in reg if k[0] in procs and procs[k[0]]["start"] == k[1]]
    seen = set()
    while todo:
        pid = todo.pop()
        if pid in seen or pid not in procs:
            continue
        seen.add(pid)
        q = procs[pid]
        if q["uid"] == uid and (pid, q["start"]) not in reg:
            reg[(pid, q["start"])] = dict(pid=pid, first_seen=now(), via="parentage", **q)
        todo.extend(kids.get(pid, []))
    sids = {v["sid"] for v in reg.values()}
    pgids = {v["pgid"] for v in reg.values()}
    for pid, q in procs.items():
        if q["uid"] == uid and (pid, q["start"]) not in reg and (q["sid"] in sids or q["pgid"] in pgids):
            reg[(pid, q["start"])] = dict(pid=pid, first_seen=now(), via="recorded session or group", **q)


def alive(reg, procs):
    return [k for k, v in reg.items() if k[0] in procs and procs[k[0]]["start"] == k[1]
            and procs[k[0]]["uid"] == v["uid"]]


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


def signal_verified(reg, sig):
    """Signal each registered process through a pidfd opened first and then re-verified (start time and uid), so a
    reused PID is never signalled; process groups are never signalled as a whole."""
    sent = []
    for (pid, start), ident in list(reg.items()):
        try:
            fd = os.pidfd_open(pid)
        except OSError:
            continue
        try:
            q = snapshot_one(pid)
            if q and q["start"] == start and q["uid"] == ident["uid"]:
                signal.pidfd_send_signal(fd, sig)
                sent.append(pid)
        except OSError:
            pass
        finally:
            os.close(fd)
    return sent


def snapshot_one(pid):
    try:
        st = open(f"/proc/{pid}/stat").read()
        rest = st[st.rindex(")") + 2:].split()
        uid = next(int(line.split()[1]) for line in open(f"/proc/{pid}/status") if line.startswith("Uid:"))
        return {"start": int(rest[19]), "uid": uid}
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def stop(reg, uid):
    """TERM, a 10 s grace, re-adoption of any new members, KILL; all through verified pidfds."""
    term = signal_verified(reg, signal.SIGTERM)
    time.sleep(10)
    adopt(snapshot(), reg, uid)
    kill = signal_verified(reg, signal.SIGKILL)
    time.sleep(1)
    return {"term_sent": term, "kill_sent": kill}


def run(a):
    limit_kib = int(a.rss_limit_gib * 1024 ** 2)
    uid = os.getuid()
    t0, started = time.time(), now()
    proc = subprocess.Popen(a.cmd, shell=True, cwd=a.repo, start_new_session=True,
                            stdout=open(a.stdout, "w"), stderr=subprocess.STDOUT)
    reg = {}
    q = snapshot_one(proc.pid)
    if q:
        procs = snapshot()
        if proc.pid in procs:
            reg[(proc.pid, procs[proc.pid]["start"])] = dict(pid=proc.pid, first_seen=now(), via="root",
                                                            **procs[proc.pid])
    peak, peak_procs, reason, stopped = 0, 0, None, None
    while proc.poll() is None:
        procs = snapshot()
        adopt(procs, reg, uid)
        live = alive(reg, procs)
        rss = rss_kib([k[0] for k in live])
        if rss > peak:
            peak, peak_procs = rss, len(live)
        if rss > limit_kib:
            reason = "aggregate RSS of the owned processes above the limit"
        elif time.time() - t0 > a.deadline_seconds:
            reason = "hard deadline"
        if reason:
            stopped = stop(reg, uid)
            break
        time.sleep(2)
    rc = proc.wait()
    time.sleep(5)                                   # let exiting members finish before the closure audit
    procs = snapshot()
    adopt(procs, reg, uid)
    left = [reg[k] for k in alive(reg, procs)]
    cleanup = None
    if left:
        cleanup = stop(reg, uid)
        procs = snapshot()
        adopt(procs, reg, uid)
    left_after = [reg[k] for k in alive(reg, procs)]
    closed = not left_after
    ok = rc == 0 and reason is None and not left
    rec = {"step": "run", "command": a.cmd, "cwd": a.repo, "uid": uid, "started": started, "ended": now(),
           "elapsed_seconds": round(time.time() - t0, 3), "exit_code": rc, "killed_by_watchdog": reason,
           "watchdog_stop": stopped, "peak_aggregate_rss_kib": peak, "processes_at_peak": peak_procs,
           "rss_limit_kib": limit_kib, "deadline_seconds": a.deadline_seconds,
           "owned_identities": [dict(v, start_ticks=k[1]) for k, v in reg.items()],
           "alive_after_exit": left, "cleanup_after_exit": cleanup, "alive_after_cleanup": left_after,
           "closure_verified": closed, "success": ok,
           "closure_note": ("Identities are (pid, start ticks, uid) recorded while observed; members of recorded sessions "
                            "and groups are adopted after reparenting. Success requires exit 0, no watchdog stop and no "
                            "owned process alive after exit. The pod's verified termination is the final closure.")}
    write(a.record, rec)
    print(json.dumps({k: rec[k] for k in ("step", "exit_code", "elapsed_seconds", "peak_aggregate_rss_kib",
                                          "killed_by_watchdog", "closure_verified", "success")}))
    sys.exit(0 if ok else 1)


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
