"""Bounded local acquisition. Never launches remote compute or overwrites a run."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_record():
    paths = list(HERE.glob("*.py")) + [HERE / "plan.json", HERE / "README.md"]
    paths += [ROOT / "tools" / f for f in ["explore_llm_soft_loop.py", "explore_llm_attractors.py", "explore_llm_phase_map.py"]]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["benchmark", "collect"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError("Raw outputs must be outside the repository")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    plan = json.loads((HERE / "plan.json").read_text())
    source = source_record()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    meta = {"status": "running", "mode": args.mode, "source_commit": head, "source_sha256": source,
            "plan": plan, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pid": os.getpid(), "platform": platform.platform(), "python": sys.version,
            "numerical_threads": plan["threads"], "device": "cpu", "batch_rows": 3 * len(plan["rows"])}
    write(out / "run.json", meta)

    def progress(phase, step, **extra):
        item = {"phase": phase, "step": step, "seconds": round(time.monotonic() - start, 3), **extra}
        with (out / "progress.jsonl").open("a") as f:
            f.write(json.dumps(item) + "\n")
        print(json.dumps(item), flush=True)
        if time.monotonic() - start > plan["wall_seconds"]:
            raise TimeoutError("Run reached its local wall-time allowance; partial arrays retained")

    try:
        import explore_llm_soft_loop as esl
        import explore_llm_attractors as ela
        import explore_llm_phase_map as elpm
        loop = esl.SoftLoop(plan["model"], plan["layer"], plan["threads"], "open", ela.OPENER, 1, "float64", "cpu")
        torch = loop.torch
        meta["torch"] = torch.__version__
        meta["numpy"] = np.__version__
        meta["model_revision"] = loop.sm.engine.revision
        meta["model_name"] = loop.sm.engine.name
        meta["model_dtype"] = str(loop.model.dtype)
        meta["norms"] = [{"type": type(m).__name__, "forward": m.forward.__name__}
                         for m in loop.model.modules() if type(m).__name__.endswith("RMSNorm")]
        meta["rotary_float64"] = [str(m._inv_freq64.dtype) for m in loop.model.modules() if hasattr(m, "_inv_freq64")]
        if not meta["rotary_float64"] or any(x != "torch.float64" for x in meta["rotary_float64"]):
            raise RuntimeError("Float64 rotary patch missing")
        if loop.model.dtype != torch.float64 or loop.state_dtype != torch.float64:
            raise RuntimeError("Float64 model and states required")
        vector = np.asarray(elpm.load_vectors(plan["vectors"])[plan["concept"]], dtype=np.float64)
        vector_path = ela.DATA / "explore" / plan["vectors"] / "vectors.npz"
        meta["vector_source"] = {"path": str(vector_path), "sha256": sha(vector_path)}
        n = len(plan["rows"])
        steering = np.array([r["alpha"] * vector for r in plan["rows"]])
        steer3 = np.tile(steering, (3, 1))
        temps = torch.tensor([10 ** r["log10_T"] for r in plan["rows"]] * 3, dtype=torch.float64)
        x = loop.prepare(steer3)
        rng = np.random.default_rng(20261021)
        direction = torch.tensor(rng.normal(size=x.shape), dtype=torch.float64)
        direction /= direction.flatten(1).norm(dim=1)[:, None, None]
        y, _, jv = loop.step_jvp(x, temps, direction)
        primal, _ = loop.step(x, temps)
        primal_gap = float((y - primal).norm() / primal.norm())
        errors = []
        for factor in [1e-3, 5e-4, 2.5e-4]:
            h = x.flatten(1).norm(dim=1)[:, None, None] * factor
            plus, _ = loop.step(x + h * direction, temps)
            minus, _ = loop.step(x - h * direction, temps)
            fd = (plus - minus) / (2 * h)
            error = ((fd - jv).flatten(1).norm(dim=1) / jv.flatten(1).norm(dim=1)).numpy()
            errors.append(error)
        errors = np.array(errors)
        ratios = errors[:-1] / np.maximum(errors[1:], 1e-300)
        qualified = (errors[-1] < 1e-3) & np.all((ratios > 2) | (errors[1:] < 1e-8), axis=0)
        precision = {"primal_relative_gap": primal_gap, "relative_central_difference_errors": errors.tolist(),
                     "halving_ratios": ratios.tolist(), "qualified_rows": qualified.tolist(),
                     "passed": bool(qualified.all() and primal_gap < 1e-10)}
        write(out / "precision.json", precision)
        if not precision["passed"]:
            raise RuntimeError("Effective precision qualification failed; see precision.json")
        x = loop.prepare(steer3)
        t0 = time.monotonic()
        for k in range(12):
            x, _ = loop.step(x, temps)
        throughput = (time.monotonic() - t0) / 12
        meta["benchmark_seconds_per_step"] = throughput
        meta["estimated_acquisition_seconds"] = throughput * (plan["transient"] + plan["feature_steps"] + plan["reference_steps"])
        write(out / "run.json", meta)
        progress("benchmark", 12, seconds_per_step=throughput, estimated_acquisition_seconds=meta["estimated_acquisition_seconds"])
        if args.mode == "collect":
            x = loop.prepare(steer3)
            for k in range(plan["transient"]):
                x, _ = loop.step(x, temps)
                if (k + 1) % 32 == 0 or k + 1 == plan["transient"]:
                    progress("transient", k + 1)
            base = x[:n].clone()
            delta = torch.tensor(rng.normal(size=base.shape), dtype=torch.float64)
            delta /= delta.flatten(1).norm(dim=1)[:, None, None]
            eps = plan["offsets"]
            x = torch.cat([base] + [base + e * delta for e in eps])
            nf, nr = plan["feature_steps"], plan["reference_steps"]
            total = nf + nr
            shape = (n, nf, 24, 4864)
            acts = np.lib.format.open_memmap(out / "activations.npy", mode="w+", dtype=np.float32, shape=shape)
            states = np.lib.format.open_memmap(out / "states.npy", mode="w+", dtype=np.float64, shape=(total + 1, n, 896))
            growth = np.lib.format.open_memmap(out / "growth.npy", mode="w+", dtype=np.float64, shape=(total, n, 2))
            growth[:] = np.nan
            recording = [-1]
            hooks = []
            def hook(layer):
                def save(module, inp):
                    t = recording[0]
                    if 0 <= t < nf:
                        acts[:, t, layer, :] = inp[0][:n, -1].detach().to(torch.float32).cpu().numpy()
                return save
            for li, layer in enumerate(loop.model.model.layers):
                hooks.append(layer.mlp.down_proj.register_forward_pre_hook(hook(li)))
            collapses = []
            try:
                for k in range(total):
                    recording[0] = k
                    states[k] = x[:n, -1].numpy()
                    y, _ = loop.step(x, temps)
                    base = y[:n]
                    twins = []
                    for j, e in enumerate(eps):
                        diff = y[n * (j + 1):n * (j + 2)] - base
                        dist = diff.flatten(1).norm(dim=1)
                        valid = torch.isfinite(dist) & (dist > e * 1e-9)
                        lg = torch.full((n,), float("nan"), dtype=torch.float64)
                        lg[valid] = torch.log(dist[valid] / e)
                        growth[k, :, j] = lg.numpy()
                        direction_next = diff / torch.where(valid, dist, torch.ones_like(dist))[:, None, None]
                        if not bool(valid.all()):
                            bad = (~valid).nonzero().flatten().tolist()
                            collapses.append({"step": k, "offset": e, "rows": bad})
                            direction_next[~valid] = delta[~valid]
                        twins.append(base + e * direction_next)
                    x = torch.cat([base] + twins)
                    if (k + 1) % 32 == 0:
                        acts.flush(); states.flush(); growth.flush()
                        progress("record", k + 1, feature_steps=nf, reference_steps=nr)
                states[total] = x[:n, -1].numpy()
            finally:
                for h in hooks:
                    h.remove()
                acts.flush(); states.flush(); growth.flush()
            meta["offset_collapses"] = collapses
            meta["activation_storage_dtype"] = "float32"
            meta["reference_dtype"] = "float64"
            meta["completed_steps"] = total
        current = source_record()
        changed = [p for p in source if source[p] != current.get(p)]
        meta["source_changed_during_run"] = changed
        if changed:
            raise RuntimeError("A source file changed during execution; retain run and audit source binding")
        meta["status"] = "complete"
        meta["seconds"] = time.monotonic() - start
        write(out / "run.json", meta)
        files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
        write(out / "manifest.json", {"files": files, "source_commit": head, "source_sha256": source})
        progress("complete", meta.get("completed_steps", 0))
        # The final journal entry must be included in the manifest's journal hash.
        files["progress.jsonl"] = {"sha256": sha(out / "progress.jsonl"), "bytes": (out / "progress.jsonl").stat().st_size}
        write(out / "manifest.json", {"files": files, "source_commit": head, "source_sha256": source})
    except BaseException as exc:
        meta["status"] = "failed"
        meta["error"] = str(exc)
        meta["seconds"] = time.monotonic() - start
        write(out / "run.json", meta)
        (out / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
