"""Committed manifest of every RunPod pod this project created: GPU type and cloud, pod name, commit, job set and its
arguments, wall time, exit status and the hashes binding uploads and results (from runpod/soft64/*/result.json),
plus pods that ended without a result (refused, failed, or terminated in setup) from runpod/pods.jsonl. Dollar
amounts are left out on purpose (they live in the private ledger only).

Usage: python tools/runpod_manifest.py OUT_JSON
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def outputs(tgz, uploaded=()):
    """Top-level output folders in a run's results archive: each result file's folder truncated to five path
    components (explore/<area>/<sub>/<run>/<item>), logs and uploaded inputs left out."""
    import tarfile

    if not tgz.exists():
        return []
    found = set()
    with tarfile.open(tgz) as tar:
        for m in tar.getmembers():
            parts = m.name.split("/")
            if not m.isfile() or "logs" in parts or any(m.name.startswith(u.rstrip("/") + "/") for u in uploaded):
                continue
            found.add("/".join(parts[:min(len(parts) - 1, 5)]))
    return sorted(found)


def main():
    import explore_llm_attractors as ela

    runpod = ela.DATA / "runpod"
    pods = {}
    for line in (runpod / "pods.jsonl").read_text().splitlines():
        d = json.loads(line)
        pid = d.get("pod_id")
        if not pid:
            continue
        p = pods.setdefault(pid, {"events": []})
        p["events"].append(d.get("event"))
        if d.get("event") == "created":
            p.update(created=d["time"], name=d.get("name"), gpu_type=d.get("gpu_type"), cloud=d.get("cloud"),
                     image=d.get("image"))
        if str(d.get("event", "")).startswith("terminated"):
            p["terminated"] = d["time"]
    ours = {pid: p for pid, p in pods.items() if "created" in p}
    runs = []
    for res in sorted((runpod / "soft64").glob("*/result.json")):
        r = json.loads(res.read_text())
        up = r.get("upload") or {}
        pid = r.get("pod_id")
        meta = ours.get(pid, {})
        runs.append({
            "run": res.parent.name, "pod_id": pid, "name": meta.get("name"),
            "gpu_type": r.get("gpu_type"), "commit": r.get("commit"), "jobset": r.get("jobset"),
            "jobset_args": up.get("jobset_args"), "extra_data": up.get("extra_data"),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["started"])) if r.get("started") else None,
            "wall_seconds": r.get("wall_seconds"), "runner_returncode": r.get("runner_returncode"),
            "hash_verified": r.get("hash_verified"), "results_sha256": r.get("local_sha256"),
            "code_sha256": up.get("code_sha256"), "data_sha256": up.get("data_sha256"),
            "extract_clash": bool(r.get("extract_clash")), "outputs": outputs(res.parent / "results.tgz", up.get("extra_data") or ())})
    with_result = {r["pod_id"] for r in runs}
    no_result = [{"pod_id": pid, "name": p.get("name"), "gpu_type": p.get("gpu_type"), "cloud": p.get("cloud"),
                  "created": p.get("created"), "terminated": p.get("terminated"), "events": p["events"]}
                 for pid, p in sorted(ours.items(), key=lambda kv: kv[1]["created"]) if pid not in with_result]
    gpu = Counter((r["gpu_type"] or "?").split(":")[-1] for r in runs)
    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "GPU type, cloud and provenance of every pod; costs are kept out of this file by policy",
           "runs_with_results": runs, "pods_without_results": no_result,
           "summary": {"pods_created": len(ours), "runs_with_results": len(runs),
                       "runs_by_gpu": dict(gpu.most_common()),
                       "runs_by_jobset": dict(Counter(r["jobset"] for r in runs).most_common()),
                       "pods_without_results": len(no_result)}}
    text = json.dumps(out, indent=1)
    for bad in ("rate", "cost", "price", "$"):
        if f'"{bad}"' in text or (bad == "$" and "$" in text):
            raise SystemExit(f"manifest would contain a cost field ({bad})")
    Path(sys.argv[1]).write_text(text + "\n")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
