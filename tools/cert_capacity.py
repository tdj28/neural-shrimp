"""Effective container capacity for packing certificate workers on one pod (Task 34; Codex review, 18 Oct 2026).

Inside a container, /proc/meminfo, nproc and lscpu describe the host (calibration-002 saw 96 CPUs and 503 GiB). This
module reads the limits that actually apply to this process:
- its cgroup membership (/proc/self/cgroup) and the cgroup mounts (/proc/self/mountinfo), which locate the process's own
  cgroup directory in the unified (v2) hierarchy or in the v1 memory, cpu and cpuset hierarchies;
- at that directory and at every visible ancestor up to the mount root: the memory limit and current usage (v2
  memory.max and memory.current; v1 memory.limit_in_bytes and memory.usage_in_bytes, where values of 2^62 or more mean
  unlimited) and the CPU bandwidth quota and period (v2 cpu.max; v1 cpu.cfs_quota_us and cpu.cfs_period_us, -1
  unlimited);
- the effective cpuset (nearest level that has one) and the process's CPU affinity.

Memory: every finite limit contributes its headroom max(0, limit - current), and the effective memory is the minimum
of these headrooms and the host's MemAvailable. CPU: the effective CPU count is the minimum of the affinity count, the
cpuset size and every finite quota/period. Levels above the mount root lie outside the cgroup namespace and cannot be
read; the scope is recorded and nothing is assumed about them. A missing or unparsable value at a visible level that
could carry a limit is listed under "unresolved", and packing admission refuses while anything is unresolved. Only the
topmost visible level may lack limit files without that counting as unresolved, and only when it is not the process's
own cgroup: the root of a cgroup hierarchy carries no limit files.

packing_admission(cap, workers, per_worker_gib=18, shared_gib=4, threads_per_worker=1) admits only if nothing is
unresolved, the effective memory covers workers * per_worker_gib + shared_gib, and the effective CPUs cover
workers * threads_per_worker. Every path is read relative to `root` so that tests can supply a fake tree.

Usage: python tools/cert_capacity.py [--workers N] [--per-worker-gib 18] [--shared-gib 4] [--threads 1]
"""
import argparse
import json
import os
from pathlib import Path

GIB = 1024 ** 3
V1_UNLIMITED = 2 ** 62


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def count_cpus(spec):
    """Number of CPUs in a list such as '0-3,8,10-11'; None if unparsable."""
    if spec is None or spec == "":
        return None
    n = 0
    try:
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-")
                n += int(b) - int(a) + 1
            else:
                int(part)
                n += 1
    except ValueError:
        return None
    return n


def _mounts(root):
    """cgroup mounts from mountinfo: list of dicts with fstype, mount root, mount point and v1 controllers."""
    out = []
    text = _read(Path(root) / "proc/self/mountinfo") or ""
    for line in text.splitlines():
        if " - " not in line:
            continue
        pre, post = line.split(" - ", 1)
        f, g = pre.split(), post.split()
        if len(f) < 5 or len(g) < 3 or g[0] not in ("cgroup", "cgroup2"):
            continue
        ctrl = [o for o in g[2].split(",") if o in ("memory", "cpu", "cpuacct", "cpuset")] if g[0] == "cgroup" else []
        out.append({"fstype": g[0], "mount_root": f[3], "mount_point": f[4], "controllers": ctrl})
    return out


def _membership(root):
    """{'v2': path} and/or {'memory': path, 'cpu': path, 'cpuset': path} from /proc/self/cgroup."""
    out = {}
    text = _read(Path(root) / "proc/self/cgroup") or ""
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, ctrls, path = parts
        if ctrls == "":
            out["v2"] = path
        else:
            for c in ctrls.split(","):
                if c in ("memory", "cpu", "cpuset"):
                    out[c] = path
    return out


def _levels(root, mount, path):
    """Directories from the process's cgroup up to the mount root (inclusive), as host paths under `root`; None if the
    process's path is not inside the mount."""
    mroot = mount["mount_root"].rstrip("/") or "/"
    if mroot != "/" and not (path == mroot or path.startswith(mroot + "/")):
        return None
    rel = path[len(mroot):] if mroot != "/" else path
    rel = rel.strip("/")
    base = Path(root) / mount["mount_point"].lstrip("/")
    parts = [p for p in rel.split("/") if p]
    dirs = [base.joinpath(*parts[:k]) for k in range(len(parts), -1, -1)]
    return dirs


def capacity(root="/", affinity=None):
    root = Path(root)
    meminfo = _read(root / "proc/meminfo") or ""
    host_avail = None
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            host_avail = int(line.split()[1]) * 1024
    if affinity is None:
        affinity = len(os.sched_getaffinity(0))
    member, mounts = _membership(root), _mounts(root)
    unresolved, levels = [], []
    if host_avail is None:
        unresolved.append("host MemAvailable not readable")
    mem_dirs = cpu_dirs = set_dirs = None
    version = None
    if "v2" in member and any(m["fstype"] == "cgroup2" for m in mounts):
        version = "v2"
        m2 = next(m for m in mounts if m["fstype"] == "cgroup2")
        mem_dirs = cpu_dirs = set_dirs = _levels(root, m2, member["v2"])
        if mem_dirs is None:
            unresolved.append(f"process cgroup {member['v2']} is outside the cgroup2 mount root {m2['mount_root']}")
    elif member:
        version = "v1"

        def v1(ctrl):
            m = next((m for m in mounts if m["fstype"] == "cgroup" and ctrl in m["controllers"]), None)
            if m is None or ctrl not in member:
                unresolved.append(f"no v1 {ctrl} hierarchy or membership visible")
                return None
            d = _levels(root, m, member[ctrl])
            if d is None:
                unresolved.append(f"process {ctrl} cgroup {member[ctrl]} is outside its mount root")
            return d

        mem_dirs, cpu_dirs, set_dirs = v1("memory"), v1("cpu"), v1("cpuset")
    else:
        unresolved.append("no cgroup membership readable")
    headrooms, quotas = [], []
    if mem_dirs:
        for k, d in enumerate(mem_dirs):
            top, leaf = k == len(mem_dirs) - 1, k == 0
            if version == "v2":
                lim, cur = _read(d / "memory.max"), _read(d / "memory.current")
                finite = None if lim is None else (None if lim == "max" else lim)
            else:
                lim, cur = _read(d / "memory.limit_in_bytes"), _read(d / "memory.usage_in_bytes")
                finite = None if lim is None else (None if lim.isdigit() and int(lim) >= V1_UNLIMITED else lim)
            rec = {"level": str(d.relative_to(root)), "memory_limit": lim, "memory_current": cur}
            if lim is None:
                if top and not leaf:
                    rec["note"] = "hierarchy root: no limit file"
                else:
                    unresolved.append(f"memory limit unreadable at {rec['level']}")
            elif finite is not None:
                try:
                    h = max(0, int(finite) - int(cur))
                    rec["headroom_bytes"] = h
                    headrooms.append(h)
                except (TypeError, ValueError):
                    unresolved.append(f"finite memory limit without readable usage at {rec['level']}")
            levels.append(rec)
    if cpu_dirs:
        for k, d in enumerate(cpu_dirs):
            top, leaf = k == len(cpu_dirs) - 1, k == 0
            if version == "v2":
                raw = _read(d / "cpu.max")
                q = p = None
                if raw is not None:
                    try:
                        qs, ps = raw.split()
                        q, p = (None if qs == "max" else int(qs)), int(ps)
                    except ValueError:
                        unresolved.append(f"cpu.max unparsable at {d.relative_to(root)}: {raw}")
            else:
                qs, ps = _read(d / "cpu.cfs_quota_us"), _read(d / "cpu.cfs_period_us")
                raw = None if qs is None else f"{qs} {ps}"
                q = p = None
                if qs is not None:
                    try:
                        q, p = (None if int(qs) < 0 else int(qs)), int(ps)
                    except (TypeError, ValueError):
                        unresolved.append(f"cfs quota/period unparsable at {d.relative_to(root)}")
            rec = next((r for r in levels if r["level"] == str(d.relative_to(root))), None)
            if rec is None:
                rec = {"level": str(d.relative_to(root))}
                levels.append(rec)
            rec["cpu_quota_period"] = raw
            if raw is None and not (top and not leaf):
                unresolved.append(f"CPU quota unreadable at {rec['level']}")
            if q is not None and p:
                quotas.append(q / p)
    cpuset, set_level = None, None
    for d in (set_dirs or []):
        s = _read(d / "cpuset.cpus.effective") if version == "v2" else (_read(d / "cpuset.effective_cpus")
                                                                       or _read(d / "cpuset.cpus"))
        if s:
            cpuset, set_level = s, str(d.relative_to(root))
            break
    if set_dirs is not None and cpuset is None:
        unresolved.append("no effective cpuset readable at any visible level")
    ncpuset = count_cpus(cpuset)
    eff_mem = None if host_avail is None else min([host_avail] + headrooms)
    cpu_terms = [affinity] + ([ncpuset] if ncpuset else []) + [int(x) for x in quotas]
    return {"cgroup_version": version, "membership": member, "mounts": mounts, "levels": levels,
            "scope": "the process's cgroup and its ancestors up to the mount root; levels above the mount root lie "
                     "outside the cgroup namespace and were not read",
            "host_mem_available_bytes": host_avail, "finite_memory_headrooms_bytes": headrooms,
            "effective_memory_bytes": eff_mem, "cpu_quotas_cpus": quotas, "cpuset": cpuset, "cpuset_level": set_level,
            "cpuset_count": ncpuset, "affinity_count": affinity, "effective_cpus": min(cpu_terms),
            "unresolved": unresolved}


def packing_admission(cap, workers, per_worker_gib=18.0, shared_gib=4.0, threads_per_worker=1):
    need_mem = int((workers * per_worker_gib + shared_gib) * GIB)
    need_cpu = workers * threads_per_worker
    reasons = []
    if cap["unresolved"]:
        reasons.append("unresolved capacity data: " + "; ".join(cap["unresolved"]))
    if cap["effective_memory_bytes"] is None or cap["effective_memory_bytes"] < need_mem:
        reasons.append(f"effective memory {cap['effective_memory_bytes']} < required {need_mem}")
    if cap["effective_cpus"] < need_cpu:
        reasons.append(f"effective CPUs {cap['effective_cpus']} < required {need_cpu}")
    return {"admit": not reasons, "workers": workers, "per_worker_gib": per_worker_gib, "shared_gib": shared_gib,
            "threads_per_worker": threads_per_worker, "required_memory_bytes": need_mem, "required_cpus": need_cpu,
            "reasons": reasons}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--per-worker-gib", type=float, default=18.0)
    ap.add_argument("--shared-gib", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--root", default="/")
    a = ap.parse_args()
    cap = capacity(a.root)
    print(json.dumps({"capacity": cap, "admission": packing_admission(cap, a.workers, a.per_worker_gib, a.shared_gib,
                                                                       a.threads)}, indent=1))


if __name__ == "__main__":
    main()
