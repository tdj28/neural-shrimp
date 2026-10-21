"""Replay endpoint precision at every six-step-spaced cyclic phase locally.

Preserve the failed A100 phase-zero finite-difference check. Reuse its saved
matrices and exact endpoint; change neither the numerical gates nor the orbit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "8"
from baby_shrimp_orbits import ROOT, save

SOURCES = ("tools/baby_shrimp_crossing_precision.py", "tools/baby_shrimp_orbits.py",
           "tools/bifurcation_curves.py", "tools/batch_fate_finish.py", "tools/batch_fate_summary.py",
           "tools/explore_llm_soft_loop.py", "tools/explore_llm_attractors.py", "tools/explore_llm_phase_map.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_phase(rows):
    if [r["phase"] for r in rows] != [0, 6, 12]:
        raise ValueError("all three declared cyclic precision checks required")
    passed = [r["phase"] for r in rows if r["primitive"]["closed"] and r["primitive"]["no_lower_period"]
              and r["precision"]["passed"]]
    if not passed:
        raise ValueError("no cyclic phase passes the unchanged endpoint precision gates")
    return passed[0]


def worker(args):
    import numpy as np
    import torch
    import explore_llm_soft_loop as soft
    import baby_shrimp_orbits as checks
    original = json.loads((args.run / "result.json").read_text())
    if not original["complete"] or original["endpoint_precision"]["passed"]:
        raise ValueError("requires a completed saddle search with a failed endpoint precision gate")
    row = original["endpoint"]
    source = args.run / f"point-{row['index']:03d}"
    raw = np.load(source / "points.npy", allow_pickle=False)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.85)
    loop = soft.SoftLoop("qwen0.5b", 12, 8, "open", "Hello!", 1, "float64", "cuda", "native")
    if loop.sm.engine.revision != "7ae557604adf67be50417f59c2c2f167def9a775":
        raise ValueError("model revision differs")
    norm = loop.model.model.norm
    test = torch.linspace(-1, 1, norm.weight.numel(), dtype=torch.float64, device=loop.device)[None, None, :]
    sensitivity = float((norm(test * (1 + 1e-12)) - norm(test)).abs().max())
    if not (0 < sensitivity < 1e-9 and hasattr(loop.model.model.rotary_emb, "_inv_freq64")):
        raise ValueError("effective local precision gate failed")
    vector = soft.elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"]
    orb = soft.SoftOrbit(loop, row["alpha"] * vector, 10.0 ** row["log10_temperature"], chunk=1, jacobian_method="ad")
    rows = []
    for phase in (0, 6, 12):
        dest = args.out / f"phase-{phase}"
        dest.mkdir()
        X = torch.as_tensor(raw[phase].reshape((1,) + orb.shape), dtype=torch.float64, device=loop.device)
        with np.load(source / f"phase-{phase}.npz", allow_pickle=False) as z:
            M = z["matrix"]
        points = orb.points(X, 18)
        P = np.stack([p.reshape(-1).double().cpu().numpy() for p in points])
        np.save(dest / "points.npy", P)
        drift = float(np.max(np.linalg.norm(P[:-1] - np.roll(raw, -phase, axis=0), axis=1)
                             / np.maximum(np.linalg.norm(P[:-1], axis=1), 1)))
        if drift >= 1e-11:
            raise ValueError("local cyclic orbit disagrees with saved states")
        primitive = checks.primitive_gate(orb.return_distances(X, 18), 18)
        def progress(stage, **kw):
            record = {"phase": phase, "stage": stage, **kw}
            with (args.out / "progress.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        precision = checks.precision_checks(orb, X, M, points[:-1], dest, progress,
            relative_steps=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8])
        local = {"phase": phase, "primitive": primitive, "precision": precision,
                 "maximum_relative_orbit_replay_difference": drift,
                 "source_matrix_archive_sha256": sha(source / f"phase-{phase}.npz")}
        rows.append(local)
        save(dest / "result.json", local)
        save(args.out / "extension.json", {"complete": False, "rows": rows})
    extension = {"complete": True, "source_commit": args.source_commit,
                 "original_results_sha256": sha(args.run / "result.json"),
                 "source_points_sha256": sha(source / "points.npy"), "rows": rows,
                 "norm_sensitivity": sensitivity, "rotary_float64": True,
                 "device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
                 "selected_phase": None,
                 "scope": "All three cyclic phases, unchanged precision gates and offsets; original failed check retained."}
    save(args.out / "extension.json", extension)
    extension["selected_phase"] = select_phase(rows)
    save(args.out / "extension.json", extension)
    print(json.dumps({"selected_phase": extension["selected_phase"],
                      "passed_phases": [r["phase"] for r in rows if r["precision"]["passed"]]}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--worker", action="store_true")
    args = ap.parse_args()
    if not 0 < args.timeout <= 900:
        ap.error("local replay is bounded to900seconds")
    if args.worker:
        worker(args)
        return
    import batch_fate_finish as guard
    args.out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", args.source_commit], cwd=ROOT, text=True).strip()
    hashes = {}
    for source in SOURCES:
        blob = subprocess.check_output(["git", "show", commit + ":" + source], cwd=ROOT)
        hashes[source] = hashlib.sha256(blob).hexdigest()
        if hashes[source] != sha(ROOT / source):
            raise ValueError("uncommitted source: " + source)
    os.environ["HF_HUB_OFFLINE"] = "1"
    status = {"source_commit": commit, "source_sha256": hashes,
              "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    def event(state, **kw):
        with (args.out / "events.jsonl").open("a") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "state": state, **kw}) + "\n")
    def interrupted(signum, frame):
        raise InterruptedError(f"crossing precision parent received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with guard.gpu_lock(guard.summary.DATA / "render-tmp/gpu.lock", timeout=1):
            status["resources"] = guard.resource_snapshot()
            save(args.out / "status.json", status)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--run", str(args.run.resolve()),
                       "--out", str(args.out.resolve()), "--source-commit", commit, "--timeout", str(args.timeout)]
            guard.run_bounded(command, args.out / "worker.log", args.timeout, event, monitor_memory=True)
        status["status"] = "complete"
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.out / "status.json", status)


if __name__ == "__main__":
    main()
