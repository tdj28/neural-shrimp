"""Continue an interrupted EXP-028 attempt from its saved evidence.

Adaptive-protocol continuation (see docs/adaptive-research-protocol.md). The
interrupted attempt is never modified. Its receipt inventory is verified; its
assembled coarse maps, zoom specifications and every completed zoom chunk
whose hash matches the receipt are copied byte-for-byte into a new attempt
directory; missing zoom chunks are computed with the unchanged frozen
`run_exp028.chunk_worker`; zoom maps are assembled; then stage 3
(certification, ring confirmation, random chaos reruns) runs exactly as in the
frozen runner. The new receipt records the parent receipt hash and which
files were reused or recomputed.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_exp028 as frozen  # noqa: E402


def verified_parent_files(parent):
    receipt_path = parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    good = {}
    for item in receipt["files"]:
        path = parent / item["path"]
        if path.exists() and frozen.sha256_file(path) == item["sha256"]:
            good[item["path"]] = item["sha256"]
    return receipt, frozen.sha256_file(receipt_path), good


def copy_reused(parent, out_dir, relative, reused):
    target = out_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent / relative, target)
    reused.append(relative)


def continue_attempt(plan, plan_sha, parent, out_dir, guard_record=None, preflight_record=None,
                     workers=None):
    parent, out_dir = Path(parent), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    parent_receipt, parent_receipt_sha, good = verified_parent_files(parent)
    if parent_receipt["plan_sha256"] != plan_sha:
        raise frozen.RunFailure("parent attempt used a different plan")

    def log(event):
        with open(out_dir / "events.jsonl", "a") as handle:
            handle.write(json.dumps(event, default=frozen._json_default) + "\n")

    workers = workers or plan["resources"]["workers"]
    frozen.write_json(out_dir / "runtime-start.json", {
        "experiment": plan["experiment"], "plan_sha256": plan_sha, "git": guard_record,
        "preflight": preflight_record, "environment": frozen.environment_record(),
        "workers": workers, "start_utc": frozen.utc_now(),
        "continuation_of": {"path": str(parent), "receipt_sha256": parent_receipt_sha,
                            "parent_status": parent_receipt["status"]}})
    previous = signal.signal(signal.SIGTERM, frozen._raise_interrupt)
    executor = frozen.Executor(workers, plan["resources"]["wall_ceiling_seconds"], started)
    status, error, interrupted, counts = "failed", None, None, {}
    reused, recomputed = [], []
    try:
        maps = []
        for spec in plan["maps"]:
            relative = f"stage1/{spec['map_id']}.npz"
            if relative not in good:
                raise frozen.RunFailure(f"coarse map missing or altered in parent: {relative}")
            copy_reused(parent, out_dir, relative, reused)
            shaped = dict(np.load(out_dir / relative))
            maps.append((spec, spec["map_index"], spec["map_id"], shaped))
        zoom_jobs = []
        for spec in plan["maps"]:
            if spec["pipeline"] != "full":
                continue
            relative = f"stage1/{spec['map_id']}-zoom.json"
            if relative not in good:
                raise frozen.RunFailure(f"zoom specification missing or altered: {relative}")
            copy_reused(parent, out_dir, relative, reused)
            zoom_spec = json.loads((out_dir / relative).read_text())
            if zoom_spec["window"] is not None:
                zoom_jobs.append((spec, f"{spec['map_id']}-zoom",
                                  np.asarray(zoom_spec["log10_hidden"]),
                                  np.asarray(zoom_spec["log10_output"])))
        all_tasks, missing = [], []
        for spec, map_id, zh, zo in zoom_jobs:
            tasks = frozen.chunk_tasks(spec, map_id, zh, zo, out_dir / "stage2", plan)
            for task in tasks:
                relative = f"stage2/{map_id}/chunk-{task['chunk']:04d}.npz"
                if relative in good:
                    copy_reused(parent, out_dir, relative, reused)
                else:
                    missing.append(task)
                    recomputed.append(relative)
            all_tasks += tasks
        log({"event": "continuation_plan", "reused": len(reused), "recompute": len(missing),
             "time": frozen.utc_now()})
        executor.map(frozen.chunk_worker, missing)
        records = [{"map_id": t["map_id"], "chunk": t["chunk"], "path": str(t["path"])}
                   for t in all_tasks]
        for spec, map_id, zh, zo in zoom_jobs:
            mine = [r for r in records if r["map_id"] == map_id]
            shaped = frozen.assemble(map_id, mine, zh, zo, out_dir / "stage2" / f"{map_id}.npz")
            maps.append((spec, spec["zoom_map_index"], map_id, shaped))
            log({"event": "map_complete", "map_id": map_id, "time": frozen.utc_now()})
        (out_dir / "stage3").mkdir()
        cert_tasks = []
        for spec, _, map_id, shaped in maps:
            if spec["pipeline"] == "full":
                cert_tasks += frozen.certification_tasks(spec, map_id, shaped, plan)
        certificates = executor.map(frozen.certify_worker, cert_tasks)
        frozen.write_json(out_dir / "stage3" / "certificates.json", certificates)
        ring = plan["ring_confirmation"]
        by_id = {map_id: (spec, shaped) for spec, _, map_id, shaped in maps}
        ring_tasks = []
        for record in certificates:
            cert = record["certificate"] or {}
            if record["kind"] != "embedded" or not (cert.get("verified") and cert.get("attracting")):
                continue
            spec, shaped = by_id[record["map_id"]]
            for cell in sorted(record["ring_chaotic_cells"])[:ring["max_cells_per_component"]]:
                ring_tasks.append(frozen.rerun_task(spec, record["map_id"], shaped, cell, ring, plan,
                                                    "ring_confirmation", record["component"]))
        ring_results = executor.map(frozen.chaos_worker, ring_tasks)
        frozen.write_json(out_dir / "stage3" / "ring-confirmation.json", ring_results)
        d = plan["chaos_diagnostics"]
        diag_tasks = []
        for spec, map_index, map_id, shaped in maps:
            for cell in frozen.chaos_sample(shaped["status"], d["count"], d["seed_base"] + map_index):
                diag_tasks.append(frozen.rerun_task(spec, map_id, shaped, cell, d, plan, "random_chaos"))
        diagnostics = executor.map(frozen.chaos_worker, diag_tasks)
        frozen.write_json(out_dir / "stage3" / "chaos-diagnostics.json", diagnostics)
        counts = {"maps": len(maps), "reused_files": len(reused), "recomputed_chunks": len(recomputed),
                  "certificates": len(certificates), "ring_reruns": len(ring_results),
                  "random_chaos_reruns": len(diagnostics)}
        status = "complete"
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        log({"event": "failure", **error, "time": frozen.utc_now()})
        interrupted = exc if isinstance(exc, (KeyboardInterrupt, frozen.RunInterrupted)) else None
    finally:
        cleanup = executor.close(terminate=status != "complete")
        signal.signal(signal.SIGTERM, previous)
        frozen.write_json(out_dir / "runtime-end.json", {
            "status": status, "error": error, "end_utc": frozen.utc_now(), "counts": counts,
            "wall_seconds": time.monotonic() - started, "cleanup": cleanup,
            "reused": reused, "recomputed": recomputed})
        frozen.write_json(out_dir / "receipt.json", {
            "experiment": plan["experiment"], "plan_sha256": plan_sha, "status": status,
            "git": guard_record, "continuation_of": {"path": str(parent),
                                                     "receipt_sha256": parent_receipt_sha},
            "files": frozen.inventory(out_dir)})
    if interrupted is not None:
        raise interrupted
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, help="interrupted attempt directory")
    parser.add_argument("--attempt", required=True, help="new attempt directory name, e.g. attempt-002")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    guard = frozen.git_guard(str(frozen.DEFAULT_PLAN))
    plan_bytes = frozen.DEFAULT_PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    plan_sha = frozen.sha256_bytes(plan_bytes)
    frozen.verify_sources(plan)
    snapshot = frozen.preflight(plan)
    out_dir = (ROOT / plan["output"]["raw_root"] / args.attempt).resolve()
    print(json.dumps({"plan_sha256": plan_sha, "git": guard, "preflight": snapshot,
                      "parent": args.parent, "output": str(out_dir)}, indent=2))
    if args.preflight_only:
        return
    if out_dir.exists():
        raise frozen.RunFailure(f"attempt directory already exists: {out_dir}")
    status = continue_attempt(plan, plan_sha, args.parent, out_dir, guard, snapshot)
    print(json.dumps({"status": status, "output": str(out_dir)}))
    raise SystemExit(0 if status == "complete" else 2)


if __name__ == "__main__":
    main()
