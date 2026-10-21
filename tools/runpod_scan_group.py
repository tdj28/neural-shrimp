"""Launch a cost-bounded scan group with an independent ownership watchdog.

The JSON configuration is private (rates and ceilings must never enter Git).
Only exact names, approval and commit matching new ledger creation records are
owned. Existing IDs are excluded. Each normal worker retrieves and terminates
its pod; the detached watchdog salvages and terminates stalled/orphaned work.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import runpod_pod as rp
from runpod_phase0 import KEY, ssh_base

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def validate(c):
    names = c["names"]
    if not names or len(set(names)) != len(names) or len(names) != len(c["commands"]):
        raise ValueError("unique name per command required")
    if not all(n.startswith("neural-shrimp-") for n in names):
        raise ValueError("project names required")
    for k in ("max_rate", "disk_hourly_rate", "ceiling", "hard_minutes", "retrieval_minutes", "no_progress_minutes"):
        if not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError(f"invalid {k}")
    if c["hard_minutes"] <= c["retrieval_minutes"] + 30:
        raise ValueError("insufficient retrieval reserve")
    worst = len(names) * (c["max_rate"] + c["disk_hourly_rate"]) * c["hard_minutes"] / 60
    if worst > c["ceiling"]:
        raise ValueError("worst-case lifetime exceeds the approved ceiling")
    if len(c["source_commit"]) != 40 or not c["approval"] or c["approval"] == "PENDING":
        raise ValueError("full source commit and approval required")
    for name, cmd in zip(names, c["commands"]):
        if len(cmd) < 2 or cmd[1] != "tools/runpod_soft_float64.py":
            raise ValueError("only the existing scan orchestrator is allowed")
        for key, expected in (("--name", name), ("--commit", c["source_commit"]), ("--approval", c["approval"])):
            if key not in cmd or cmd[cmd.index(key) + 1] != expected:
                raise ValueError(f"command mismatch: {key}")
        jobset = cmd[cmd.index("--jobset") + 1]
        if jobset == "toolseq":
            validate_orbit_job(cmd[cmd.index("--jobset-args") + 1], source_commit=c["source_commit"])
        elif jobset != "scanband":
            raise ValueError("only scanband or the bounded orbit probe is allowed")
        if float(cmd[cmd.index("--max-rate") + 1]) > c["max_rate"]:
            raise ValueError("command rate exceeds group rate")
        if "--gpu-count" in cmd and cmd[cmd.index("--gpu-count") + 1] != "1":
            raise ValueError("one GPU per pod required")
        if float(cmd[cmd.index("--max-cost") + 1]) > c["ceiling"] / len(names):
            raise ValueError("worker ceiling exceeds its share")
    return worst


def validate_orbit_job(arguments, source_commit=None):
    job = shlex.split(arguments)
    allowed = {"baby_shrimp_orbits.py": 90, "baby_shrimp_geometry.py": 195, "baby_shrimp_crossing.py": 75,
               "baby_shrimp_transect.py": 45, "baby_shrimp_saddle_memory.py": 120}
    if len(job) < 7 or job[2] not in allowed or "::" in job:
        raise ValueError("one bounded baby-shrimp orbit worker required")
    minutes = float(job[1])
    if not math.isfinite(minutes) or not 0 < minutes <= allowed[job[2]]:
        raise ValueError("invalid orbit worker lifetime")
    # BSC3's immutable anchor is bound by its plan, rather than an --input
    # argument. Its archive has no .git, so the explicit source must match
    # the controller's uploaded commit. Other workers retain their old gates.
    saddle = job[2] == "baby_shrimp_saddle_memory.py"
    if saddle:
        for key in ("--plan", "--source-commit", "--mode"):
            if job.count(key) != 1 or job.index(key) + 1 >= len(job):
                raise ValueError(f"saddle worker requires one {key}")
        plan = job[job.index("--plan") + 1]
        if not plan.startswith("docs/baby-shrimp-zoom/") or ".." in Path(plan).parts:
            raise ValueError("saddle plan must remain in the declared repository plan directory")
        commit = job[job.index("--source-commit") + 1]
        if not re.fullmatch(r"[0-9a-f]{40}", commit) or source_commit is not None and commit != source_commit:
            raise ValueError("saddle source commit differs from the uploaded source")
        if job[job.index("--mode") + 1] not in ("pilot", "continue"):
            raise ValueError("unknown saddle worker mode")
    for key in (("--out",) if saddle else ("--input", "--out")):
        if key not in job:
            raise ValueError(f"orbit worker missing {key}")
        if saddle and (job.count(key) != 1 or job.index(key) + 1 >= len(job)):
            raise ValueError(f"saddle worker requires one {key}")
        value = job[job.index(key)+1]
        prefix = "llm-soft-loop/pod64/" if key == "--out" else "llm-soft-loop/"
        if not value.startswith(prefix) or ".." in Path(value).parts:
            raise ValueError("orbit input must remain in the data tree and output in the retrieved tree")


def owned(c, entries):
    found = {}
    for e in entries:
        pid = e.get("pod_id")
        if e.get("event") == "created" and e.get("name") in c["names"]:
            if (e.get("commit") != c["source_commit"] or e.get("approval") != c["approval"]
                    or pid in c["preexisting_ids"]):
                continue
            created = datetime.fromisoformat(e["time"]).timestamp()
            if created < c["started_epoch"]:
                continue
            found[pid] = {**e, "created_epoch": created, "terminated": False}
        elif e.get("event") == "terminated_verified" and pid in found:
            found[pid]["terminated"] = True
    return found


def ledger():
    rows = []
    for line in rp.LEDGER.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a concurrent append is re-read next poll
    return rows


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else int(fields[19])
    except FileNotFoundError:
        return None


def event(out, state, **kw):
    with (out / "watchdog-events.jsonl").open("a") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "state": state, **kw}) + "\n")


def cleanup(out, pod, why):
    pid = pod["pod_id"]
    event(out, "salvage_started", pod_id=pid, reason=why)
    try:
        status = rp.pod_status(pid)
        ports = ((status or {}).get("runtime") or {}).get("ports") or []
        port = next((p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")), None)
        if port:
            base = ssh_base(port["ip"], port["publicPort"])
            packed = subprocess.run(base + ["tar czf /root/scan-salvage.tgz -C /root/neural-shrimp-data "
                                            "explore/llm-soft-loop/pod64 && sha256sum /root/scan-salvage.tgz"],
                                    capture_output=True, text=True, timeout=120)
            dest = out / f"salvage-{pid}.tgz"
            subprocess.run(["scp", "-i", str(KEY), "-P", str(port["publicPort"]), "-o", "BatchMode=yes",
                            "-o", f"UserKnownHostsFile={rp.DATA / 'runpod/known_hosts'}",
                            f"root@{port['ip']}:/root/scan-salvage.tgz", str(dest)],
                           capture_output=True, timeout=120, check=True)
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            event(out, "salvaged", pod_id=pid, sha256=sha,
                  hash_verified=bool(packed.stdout.split()) and packed.stdout.split()[0] == sha)
    except BaseException as error:
        event(out, "salvage_failed", pod_id=pid, error=str(error))
    finally:
        try:
            rp.cmd_terminate(argparse.Namespace(pod=pid))
            event(out, "terminated", pod_id=pid)
        except BaseException as error:
            event(out, "termination_unconfirmed", pod_id=pid, error=str(error))


def progress(out, name):
    logs = sorted((out / "workers" / name).glob("*/runner-stdout.txt"))
    if not logs:
        return None, False
    lines = logs[-1].read_text(errors="replace").splitlines()
    finished = any(re.fullmatch(r"[0-9a-f]{64}\s+/root/results\.tgz", line.strip()) for line in lines)
    values = []
    for line in lines:
        for marker in ('{"chunk_start"', '{"cell_index"'):
            if marker in line:
                values.append(line[line.index(marker):])
                break
    return values[-1] if values else None, finished


def watch(out):
    c = json.loads((out / "config.json").read_text())
    validate(c)
    save(out / "watchdog-ready.json", {"pid": os.getpid(), "started": time.time()})
    changes, cleaning = {}, {}
    with ThreadPoolExecutor(max_workers=len(c["names"])) as pool:
        while True:
            state = json.loads((out / "state.json").read_text())
            pods = owned(c, ledger())
            now = time.time()
            alive = identity(state["controller_pid"]) == state["controller_ticks"]
            save(out / "watchdog-heartbeat.json", {"time": now, "pid": os.getpid(),
                                                   "owned_ids": list(pods), "controller_alive": alive})
            for pid, pod in pods.items():
                if pod["terminated"]:
                    continue
                if pid in cleaning:
                    if not cleaning[pid].done():
                        continue
                    cleaning.pop(pid)
                val, finished = progress(out, pod["name"])
                old, changed = changes.get(pid, (None, pod["created_epoch"]))
                if val != old:
                    changed = now
                changes[pid] = val, changed
                age = now - pod["created_epoch"]
                reason = None
                rate = pod.get("cost_per_hr")
                if rate is None or rate > c["max_rate"]:
                    reason = "missing or excessive provider rate"
                elif age >= 60 * (c["hard_minutes"] - c["retrieval_minutes"]):
                    reason = "lifetime retrieval reserve reached"
                elif not alive and now - state["updated_epoch"] > 120:
                    reason = "controller orphaned"
                elif age > 1800 and val is None and not finished:
                    reason = "no numerical progress after setup allowance"
                elif val is not None and not finished and now - changed > 60 * c["no_progress_minutes"]:
                    reason = "numerical progress stalled"
                elif state.get("finished") and not pod["terminated"]:
                    reason = "worker exited without verified termination"
                if reason:
                    cleaning[pid] = pool.submit(cleanup, out, pod, reason)
            if (state.get("finished") or not alive) and all(p["terminated"] for p in pods.values()):
                event(out, "watchdog_complete", owned_ids=list(pods))
                return
            time.sleep(15)


def launch(config, out):
    c = json.loads(config.read_text())
    validate(c)
    out.mkdir(parents=True, exist_ok=False)
    c.update(started_epoch=time.time(), preexisting_ids=[p["id"] for p in rp.inventory()["pods"]])
    save(out / "config.json", c)
    state = {"controller_pid": os.getpid(), "controller_ticks": identity(os.getpid()),
             "updated_epoch": time.time(), "finished": False, "workers": []}
    save(out / "state.json", state)
    with (out / "watchdog.log").open("x") as log:
        watchdog = subprocess.Popen([sys.executable, __file__, "--watch", "--out", str(out)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 30
    while not (out / "watchdog-ready.json").exists():
        if watchdog.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError("watchdog failed before launch; no pods created")
        time.sleep(0.2)
    children = []
    try:
        for name, command in zip(c["names"], c["commands"]):
            if watchdog.poll() is not None:
                raise RuntimeError("watchdog stopped; refusing another pod")
            cmd = list(command)
            cmd[0] = sys.executable
            cmd[cmd.index("--out") + 1] = str(out / "workers" / name)
            with (out / f"{name}.log").open("x") as log:
                child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            state["workers"].append({"name": name, "pid": child.pid, "ticks": identity(child.pid)})
            state["updated_epoch"] = time.time()
            save(out / "state.json", state)
            time.sleep(3)
        while any(p.poll() is None for p in children):
            heartbeat = json.loads((out / "watchdog-heartbeat.json").read_text())
            if watchdog.poll() is not None or time.time() - heartbeat["time"] > 120:
                raise RuntimeError("watchdog lost; controller exits to trigger independent cleanup")
            state["updated_epoch"] = time.time()
            save(out / "state.json", state)
            time.sleep(15)
        state["returncodes"] = [p.returncode for p in children]
    except BaseException:
        if watchdog.poll() is not None:
            # If the independent guard itself dies, the live controller owns
            # this emergency path. Only this task's new provider IDs qualify.
            with ThreadPoolExecutor(max_workers=len(c["names"])) as pool:
                jobs = [pool.submit(cleanup, out, p, "watchdog lost")
                        for p in owned(c, ledger()).values() if not p["terminated"]]
                for job in jobs:
                    job.result()
        raise
    finally:
        state.update(finished=True, updated_epoch=time.time())
        save(out / "state.json", state)
    watchdog.wait(timeout=1800)
    pods = owned(c, ledger())
    present = {p["id"] for p in rp.inventory()["pods"]}
    if len(pods) != len(c["names"]) or any(p in present or not x["terminated"] for p, x in pods.items()):
        raise RuntimeError("incomplete launch or unconfirmed termination; inspect preserved records")
    if any(state["returncodes"]):
        raise RuntimeError("one or more scan workers failed; no automatic retries")
    save(out / "complete.json", {"time": time.time(), "owned_ids": list(pods), "terminated": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    watch(args.out) if args.watch else launch(args.config, args.out)
