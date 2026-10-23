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
TERMINATION_RESERVE_SECONDS = 60


def huginn_command(command):
    """Identify the separately protected worker without substring matching."""
    if "--jobset" not in command or command[command.index("--jobset") + 1] != "toolseq":
        return False
    if "--jobset-args" not in command:
        return False
    job = shlex.split(command[command.index("--jobset-args") + 1])
    return len(job) >= 3 and job[2] == "huginn_chaos_map.py"


def protected_huginn(c, name):
    return huginn_command(c["commands"][c["names"].index(name)])


def worker_command(c, name, out):
    """Bind the protected worker to this controller before it can create a pod."""
    cmd = list(c["commands"][c["names"].index(name)])
    cmd[0] = sys.executable
    cmd[cmd.index("--out") + 1] = str(out / "workers" / name)
    if huginn_command(cmd):
        if "--controller-root" in cmd:
            if (cmd.count("--controller-root") != 1 or
                    cmd.index("--controller-root") + 1 >= len(cmd) or
                    Path(cmd[cmd.index("--controller-root") + 1]).resolve() != out.resolve()):
                raise ValueError("Huginn controller root differs from this scan group")
        else:
            cmd += ["--controller-root", str(out.resolve())]
    return cmd


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
               "baby_shrimp_transect.py": 45, "baby_shrimp_saddle_memory.py": 120,
               "baby_shrimp_memory_scan.py": 75, "huginn_chaos_map.py": 720}
    if len(job) < 7 or job[2] not in allowed or "::" in job:
        raise ValueError("one bounded baby-shrimp orbit worker required")
    minutes = float(job[1])
    if not math.isfinite(minutes) or not 0 < minutes <= allowed[job[2]]:
        raise ValueError("invalid orbit worker lifetime")
    # BSC3's immutable anchor is bound by its plan, rather than an --input
    # argument. Its archive has no .git, so the explicit source must match
    # the controller's uploaded commit. Other workers retain their old gates.
    huginn = job[2] == "huginn_chaos_map.py"
    memory_scan = job[2] == "baby_shrimp_memory_scan.py" or huginn
    saddle = job[2] == "baby_shrimp_saddle_memory.py" or memory_scan
    if saddle:
        for key in ("--plan", "--source-commit", "--mode"):
            if job.count(key) != 1 or job.index(key) + 1 >= len(job):
                raise ValueError(f"saddle worker requires one {key}")
        plan = job[job.index("--plan") + 1]
        if (plan not in ("docs/huginn-chaos-map-plan.json", "docs/huginn-chaos-zoom-plan.json", "docs/huginn-chaos-tight-zoom-plan.json", "docs/huginn-chaos-continuation-plan.json") if huginn else not plan.startswith("docs/baby-shrimp-zoom/")) or ".." in Path(plan).parts:
            raise ValueError("saddle plan must remain in the declared repository plan directory")
        commit = job[job.index("--source-commit") + 1]
        if not re.fullmatch(r"[0-9a-f]{40}", commit) or source_commit is not None and commit != source_commit:
            raise ValueError("saddle source commit differs from the uploaded source")
        modes = ("scan",) if memory_scan else ("pilot", "continue")
        if job[job.index("--mode") + 1] not in modes:
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


def terminate_owned(out, pod):
    """Avoid a second delete failing after the normal worker already finished."""
    pid = pod["pod_id"]
    current = rp.pod_status(pid)
    if current is None:
        if pid in {p["id"] for p in rp.inventory()["pods"]}:
            raise RuntimeError("direct lookup and inventory disagree about owned pod")
        rp.record({"event": "terminated_verified", "pod_id": pid,
                   "direct_lookup": None, "in_inventory": False})
    else:
        if current.get("id") != pid or current.get("name") != pod["name"]:
            raise ValueError("live pod identity differs from the exact creation record")
        rp.cmd_terminate(argparse.Namespace(pod=pid))
    event(out, "terminated", pod_id=pid)


def cleanup(out, pod, why, c):
    pid = pod["pod_id"]
    # Revalidate the immutable creation binding in the cleanup thread, rather
    # than relying only on a prior poll. Never adopt a similarly named resource.
    current = owned(c, ledger()).get(pid)
    if current is None or current["created_epoch"] != pod["created_epoch"]:
        event(out, "cleanup_refused", pod_id=pid, reason="ownership changed")
        return
    if current["terminated"]:
        return
    event(out, "salvage_started", pod_id=pid, reason=why)
    try:
        if protected_huginn(c, pod["name"]):
            # The mandatory collector has already been copying immutable files
            # since the run started. Repacking large, compressed tails here was
            # the cause of HCM2's evidence loss. Preserve its files and failures;
            # never compete with it by starting another archive or collector.
            snapshot = retrieval_snapshot(out, pod, c)
            event(out, "incremental_evidence_preserved", pod_id=pid,
                  attempt=snapshot.get("attempt"),
                  retrieval_complete=snapshot.get("complete", False),
                  progress_bytes=snapshot.get("progress_bytes", 0),
                  verified_bytes=snapshot.get("verified_bytes", 0),
                  error=snapshot.get("error"))
            return
        status = rp.pod_status(pid)
        if status is not None and (status.get("id"), status.get("name")) != (pid, pod["name"]):
            raise ValueError("live pod identity mismatch before salvage")
        ports = ((status or {}).get("runtime") or {}).get("ports") or []
        port = next((p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")), None)
        if port:
            def allowance():
                remaining = (pod["created_epoch"] + 60 * c["hard_minutes"] -
                             TERMINATION_RESERVE_SECONDS - time.time())
                if remaining <= 0:
                    raise TimeoutError("original lifetime leaves no salvage time")
                return min(120, remaining)

            base = ssh_base(port["ip"], port["publicPort"])
            packed = subprocess.run(base + ["tar czf /root/scan-salvage.tgz -C /root/neural-shrimp-data "
                                            "explore/llm-soft-loop/pod64 && sha256sum /root/scan-salvage.tgz"],
                                    capture_output=True, text=True, timeout=allowance(), check=True)
            dest = out / f"salvage-{pid}.tgz"
            subprocess.run(["scp", "-i", str(KEY), "-P", str(port["publicPort"]), "-o", "BatchMode=yes",
                            "-o", f"UserKnownHostsFile={rp.DATA / 'runpod/known_hosts'}",
                            f"root@{port['ip']}:/root/scan-salvage.tgz", str(dest)],
                           capture_output=True, timeout=allowance(), check=True)
            digest = hashlib.sha256()
            with dest.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    allowance()
                    digest.update(block)
            sha = digest.hexdigest()
            event(out, "salvaged", pod_id=pid, sha256=sha,
                  hash_verified=bool(packed.stdout.split()) and packed.stdout.split()[0] == sha)
    except BaseException as error:
        event(out, "salvage_failed", pod_id=pid, error=str(error))
    finally:
        try:
            terminate_owned(out, pod)
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


def numerical_token(value):
    """Elapsed-time chatter must not reset the numerical inactivity timer."""
    if value is None:
        return None
    try:
        row = json.loads(value)
    except json.JSONDecodeError:
        return value
    return json.dumps({k: row[k] for k in ("chunk_start", "cell_index", "stage", "phase", "offset",
                                         "step", "completed", "total", "cells") if k in row}, sort_keys=True)


def retrieval_snapshot(out, pod, c):
    """Read only the protected attempt bound to this exact creation.

    A numerical `phase=complete` or an archive checksum is deliberately not a
    retrieval-complete signal. These records are authored by the local worker
    and collector, whose process identities and fixed deadlines are checked.
    """
    result = {"bound": False, "complete": False, "compute_stopped": False,
              "active": False, "progress_bytes": 0, "verified_bytes": 0,
              "verification_bytes": 0, "verified_files": 0}
    try:
        candidates = []
        for path in sorted((out / "workers" / pod["name"]).glob("*/phase.json")):
            phase = json.loads(path.read_text())
            if phase.get("pod_id") == pod["pod_id"]:
                candidates.append((path.parent, phase))
        if not candidates:
            return result
        if len(candidates) != 1:
            raise ValueError("multiple protected attempts claim this pod")
        attempt, phase = candidates[0]
        result["attempt"] = str(attempt)
        expected = {"pod_id": pod["pod_id"], "name": pod["name"],
                    "source": c["source_commit"], "approval": c["approval"],
                    "hard_deadline_epoch": pod["created_epoch"] + 60 * c["hard_minutes"],
                    "compute_deadline_epoch": pod["created_epoch"] +
                    60 * (c["hard_minutes"] - c["retrieval_minutes"])}
        if any(phase.get(k) != v for k, v in expected.items()):
            raise ValueError("protected worker identity or fixed deadline mismatch")
        result.update(bound=True, phase=phase.get("phase"),
                      compute_stopped=phase.get("compute_stopped") is True)
        folder = attempt / "incremental"
        binding = json.loads((folder / "identity.json").read_text())
        if (any(binding.get(k) != v for k, v in expected.items() if k != "compute_deadline_epoch") or
                binding.get("created_epoch") != pod["created_epoch"] or
                not isinstance(binding.get("deadline_epoch"), (int, float)) or
                not pod["created_epoch"] < binding["deadline_epoch"] <= expected["hard_deadline_epoch"]):
            raise ValueError("collector identity or original lifetime mismatch")
        status = json.loads((folder / "status.json").read_text())
        if status.get("pod_id") != pod["pod_id"] or status.get("pid") != phase.get("collector_pid"):
            raise ValueError("collector status belongs to a different process or resource")
        ticks = phase.get("collector_ticks")
        result["active"] = (type(ticks) is int and ticks > 0 and
                            identity(phase["collector_pid"]) == ticks)
        # Never use status timestamps as progress. The collector's high-water
        # counters are based on actual streamed bytes and completed local hashes.
        for key in ("progress_bytes", "verification_bytes"):
            number = status.get(key, 0)
            if type(number) is not int or number < 0:
                raise ValueError("invalid collector progress counter")
            result[key] = number
        files, final = {}, None
        for line in (folder / "receipts.jsonl").read_text().splitlines(keepends=True):
            if not line.endswith("\n"):
                continue  # a concurrent append is completed on the next poll
            record = json.loads(line)
            if record.get("event") in ("verified", "skip_verified"):
                name, sha, size = record.get("file"), record.get("sha256"), record.get("bytes")
                if not isinstance(name, str) or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                    raise ValueError("invalid verified-file receipt")
                if type(size) is not int or size < 0:
                    raise ValueError("invalid verified-file size")
                if name in files and files[name] != (sha, size):
                    raise ValueError("conflicting immutable file receipts")
                files[name] = sha, size
            elif record.get("event") == "all_manifest_files_verified":
                final = record
        result.update(verified_files=len(files), verified_bytes=sum(size for _, size in files.values()))
        result["complete"] = (status.get("state") == "complete" and status.get("complete") is True and
                              isinstance(status.get("manifest_sha256"), str) and
                              bool(re.fullmatch(r"[0-9a-f]{64}", status["manifest_sha256"])) and
                              final is not None and final.get("manifest_sha256") == status["manifest_sha256"] and
                              type(final.get("files")) is int and final["files"] > 0)
    except FileNotFoundError:
        # The worker emits phase.json before the collector's first atomic
        # checkpoint; startup remains subject to the ordinary bounded allowance.
        pass
    except (ValueError, TypeError, KeyError) as error:
        result.update(bound=False, active=False, complete=False, error=str(error))
    return result


class ProgressWatch:
    """Monotonic observations; chatter or a retry cannot buy more idle time."""
    def __init__(self, created_epoch):
        self.numeric = None
        self.numerical_changed = created_epoch
        self.retrieval_changed = None
        self.compute_stopped = False
        self.highwater = {k: 0 for k in ("progress_bytes", "verification_bytes", "verified_bytes", "verified_files")}

    def observe(self, value, snapshot, now):
        numeric = numerical_token(value)
        if numeric != self.numeric:
            self.numeric, self.numerical_changed = numeric, now
        if not snapshot.get("bound"):
            return
        if snapshot.get("compute_stopped") and not self.compute_stopped:
            self.compute_stopped = True
            self.retrieval_changed = now
        for key, previous in self.highwater.items():
            number = snapshot.get(key, 0)
            if number > previous:
                self.highwater[key] = number
                self.retrieval_changed = now


def watch_reason(c, pod, state, alive, now, tracker, snapshot, *, protected, archive_ready):
    """Pure lifecycle decision, exercised with simulated failures and clocks."""
    age = now - pod["created_epoch"]
    rate = pod.get("cost_per_hr")
    if rate is None or rate > c["max_rate"]:
        return "missing or excessive provider rate"
    if age >= 60 * c["hard_minutes"] - TERMINATION_RESERVE_SECONDS:
        return "original hard lifetime termination reserve reached"
    if protected:
        if snapshot.get("error"):
            return "protected retrieval binding invalid"
        if snapshot.get("bound") and snapshot.get("compute_stopped"):
            if snapshot.get("complete"):
                if state.get("finished") or not alive:
                    return "verified incremental retrieval ready for termination"
                # Give the normal worker a short bounded final receipt/teardown
                # interval, never an entire numerical no-progress window.
                if now - tracker.retrieval_changed > 120:
                    return "verified retrieval awaiting worker termination"
                return None
            if not snapshot.get("active"):
                return "protected collector stopped before verified retrieval"
            if now - tracker.retrieval_changed > 60 * c["no_progress_minutes"]:
                return "artifact retrieval stalled"
            return None  # retrieval can use its original reserve, not more time
        if age >= 60 * (c["hard_minutes"] - c["retrieval_minutes"]):
            # The worker's remote KILL is scheduled no later than this cutoff.
            # Allow one poll to publish its local stopped-compute proof.
            if (snapshot.get("bound") and snapshot.get("active") and
                    age < 60 * (c["hard_minutes"] - c["retrieval_minutes"]) + 15):
                return None
            return "compute cutoff reached without stopped-compute proof"
    elif age >= 60 * (c["hard_minutes"] - c["retrieval_minutes"]):
        return "lifetime retrieval reserve reached"
    if not alive and now - state["updated_epoch"] > 120:
        return "controller orphaned"
    # Protected jobs never interpret an archive hash or numerical completion as
    # successful retrieval; only the bound collector manifest can establish it.
    finished = archive_ready and not protected
    if age > 1800 and tracker.numeric is None and not finished:
        return "no numerical progress after setup allowance"
    if tracker.numeric is not None and not finished and now - tracker.numerical_changed > 60 * c["no_progress_minutes"]:
        return "numerical progress stalled"
    if state.get("finished"):
        return "worker exited without verified termination"
    return None


def watch(out):
    c = json.loads((out / "config.json").read_text())
    validate(c)
    initial = json.loads((out / "state.json").read_text())
    save(out / "watchdog-heartbeat.json", {"time": time.time(), "pid": os.getpid(), "owned_ids": [],
                                           "controller_alive": identity(initial["controller_pid"]) ==
                                           initial["controller_ticks"]})
    # Worker preflight reads both files; readiness must not race its first
    # heartbeat. Bind the process start ticks as well as the recyclable PID.
    save(out / "watchdog-ready.json", {"pid": os.getpid(), "ticks": identity(os.getpid()), "started": time.time()})
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
                protected = protected_huginn(c, pod["name"])
                snapshot = retrieval_snapshot(out, pod, c) if protected else {}
                tracker = changes.setdefault(pid, ProgressWatch(pod["created_epoch"]))
                tracker.observe(val, snapshot, now)
                reason = watch_reason(c, pod, state, alive, now, tracker, snapshot,
                                      protected=protected, archive_ready=finished)
                if reason:
                    cleaning[pid] = pool.submit(cleanup, out, pod, reason, c)
            if (state.get("finished") or not alive) and all(p["terminated"] for p in pods.values()):
                event(out, "watchdog_complete", owned_ids=list(pods))
                return
            time.sleep(15)


def launch(config, out):
    c = json.loads(config.read_text())
    validate(c)
    # Resolve every worker command before creating directories or querying the
    # provider. A mismatched protected controller must fail before any rental.
    commands = [worker_command(c, name, out) for name in c["names"]]
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
        for name, cmd in zip(c["names"], commands):
            if watchdog.poll() is not None:
                raise RuntimeError("watchdog stopped; refusing another pod")
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
                jobs = [pool.submit(cleanup, out, p, "watchdog lost", c)
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
