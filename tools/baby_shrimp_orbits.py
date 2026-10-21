"""Float64 Newton, primitive-period and monodromy gates for selected BSP1 seeds.

This worker runs under a bounded orchestrator. It never assumes that a scan's
period is primitive or that a Newton solve succeeded. Every attempted cell has
its own saved initial/refined state, monodromy, eigenvalues and numerical gates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[thread_variable] = "8"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def refine(orb, X, period, tol, max_newton, progress):
    residuals = []
    n = orb.n
    for iteration in range(max_newton+1):
        points = orb.points(X, period)
        r = (points[-1]-X).reshape(-1).double().cpu().numpy()
        residuals.append(float(np.linalg.norm(r) / max(float(X.norm()), 1.0)))
        M = np.eye(n)
        for phase, x in enumerate(points[:-1]):
            J = orb.jacobian(x)
            if not np.isfinite(J).all():
                raise ValueError("nonfinite Jacobian")
            M = J @ M
            progress("jacobian", iteration=iteration, phase=phase+1, period=period)
        if residuals[-1] < tol or iteration == max_newton:
            return X, residuals, M, points[:-1]
        correction = np.linalg.solve(M - np.eye(n), -r)
        X = X + orb.torch.as_tensor(correction.reshape((1,)+orb.shape), dtype=X.dtype, device=X.device)
    raise RuntimeError("unreachable Newton state")


def primitive_gate(returns, period, tol=1e-8):
    if len(returns) != period or not np.isfinite(returns).all():
        raise ValueError("a finite return at every lag through the proposed period is required")
    return {"closed": bool(returns[-1] < 1e-11),
            "no_lower_period": bool(all(d > tol for d in returns[:-1])),
            "minimum_lower_return": min(returns[:-1]) if period > 1 else None,
            "all_returns": returns, "primitive_separation_tolerance": tol}


def precision_checks(orb, X, M, points, dest, progress, relative_steps=None):
    import bifurcation_curves as bc

    torch = orb.torch
    rng = np.random.default_rng(4421)
    v = rng.normal(size=orb.n)
    v /= np.linalg.norm(v)

    def tangent(vector):
        w = torch.as_tensor(vector.reshape((1,)+orb.shape), dtype=X.dtype, device=X.device)
        for point in points:
            w = orb.jvp_rows(point, w)
        return w.reshape(-1).double().cpu().numpy()

    exact = tangent(v)
    dense = M @ v
    composition_error = float(np.linalg.norm(exact-dense) / max(np.linalg.norm(exact), np.linalg.norm(dense), 1e-300))
    # Check central differences on the composed map, not merely a single step.
    steps = relative_steps if relative_steps is not None else [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    if len(steps) < 3 or not all(0 < s <= 1e-2 for s in steps) or not all(a > b for a,b in zip(steps, steps[1:])):
        raise ValueError("finite-difference steps must be positive, bounded and strictly decreasing")
    scale = max(float(X.norm()), 1.0)
    direction = torch.as_tensor(v.reshape((1,)+orb.shape), dtype=X.dtype, device=X.device)
    differences, errors = [], []
    for relative in steps:
        h = relative*scale
        fd = ((orb.iterate(X+h*direction, len(points))-orb.iterate(X-h*direction, len(points)))/(2*h))
        fd = fd.reshape(-1).double().cpu().numpy()
        differences.append(fd)
        errors.append(float(np.linalg.norm(fd-exact) / max(np.linalg.norm(exact), 1e-300)))
        progress("central_difference", relative_step=relative, relative_error=errors[-1])
    slopes = [float(np.log(errors[i]/errors[i+1])/np.log(steps[i]/steps[i+1]))
              if errors[i] > 0 and errors[i+1] > 0 else None for i in range(len(steps)-1)]
    quadratic = any(slopes[i] is not None and slopes[i+1] is not None and
                    1.5 <= slopes[i] <= 2.5 and 1.5 <= slopes[i+1] <= 2.5 for i in range(len(slopes)-1))
    eigenvalues = np.linalg.eigvals(M)
    rho = float(np.abs(eigenvalues).max())
    refined = bc.refine_multipliers(tangent, M, m=8, count=2)
    leading = refined[0][0]
    radius_error = abs(abs(leading)-rho) / max(rho, 1e-300)
    residual = refined[0][1] / max(abs(leading), 1e-300)
    np.savez_compressed(dest / "derivative-check.npz", direction=v, direct_jvp=exact, dense_jvp=dense,
                        relative_steps=steps, finite_differences=np.array(differences), relative_errors=errors)
    return {"dense_vs_composed_jvp_relative_error": composition_error,
            "central_difference_relative_steps": steps, "central_difference_relative_errors": errors,
            "observed_orders": slopes, "quadratic_regime_found": bool(quadratic),
            "dense_spectral_radius": rho, "refined_leading_multiplier": [leading.real, leading.imag],
            "refined_leading_relative_residual": float(residual),
            "spectral_radius_relative_difference": float(radius_error),
            "passed": bool(composition_error < 1e-7 and quadratic and min(errors) < 1e-5 and
                           radius_error < 1e-5 and residual < 1e-6)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="explore-relative seed directory")
    parser.add_argument("--out", required=True, help="new explore-relative output directory")
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--max-newton", type=int, default=3)
    parser.add_argument("--select", nargs="*", default=None)
    args = parser.parse_args()
    if not (1 <= args.chunk <= 512 and 0 <= args.max_newton <= 6):
        parser.error("invalid numerical resource bounds")
    import torch
    import explore_llm_soft_loop as soft
    import explore_llm_attractors as ela

    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.85)
    src, out = DATA / "explore" / args.input, DATA / "explore" / args.out
    # The shell's minute reporter may create this directory before Python starts.
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("orbits.json", "protocol.json")):
        raise ValueError("output attempt already exists")
    info = json.loads((src / "seeds.json").read_text())
    states = np.load(src / "seeds.npz", allow_pickle=False)["states"]
    if hashlib.sha256((src / "seeds.npz").read_bytes()).hexdigest() != info["states_sha256"]:
        raise ValueError("seed hash differs")
    if args.select and set(args.select) - {r["name"] for r in info["cells"]}:
        raise ValueError("requested cell is absent")
    loop = soft.SoftLoop("qwen0.5b", 12, 8, "open", ela.OPENER, 1, "float64", "cuda", "native")
    if loop.sm.engine.revision != "7ae557604adf67be50417f59c2c2f167def9a775":
        raise ValueError("unexpected model revision")
    norm = loop.model.model.norm
    test = torch.linspace(-1, 1, norm.weight.numel(), dtype=torch.float64, device=loop.device)[None,None,:]
    sensitivity = float((norm(test*(1+1e-12))-norm(test)).abs().max())
    if not (0 < sensitivity < 1e-9 and hasattr(loop.model.model.rotary_emb, "_inv_freq64")):
        raise ValueError("normalization/rotary effective-precision check failed")
    vector = soft.elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"]
    protocol = {"args": vars(args), "seed_sha256": info["states_sha256"], "seed_provenance": info["provenance"],
                "model_revision": loop.sm.engine.revision, "dtype": str(loop.state_dtype),
                "newton_tolerance": 1e-11, "native_float64_jacobians": True,
                "norm_sensitivity_to_1e_12": sensitivity, "rotary_float64": True,
                "batch_layout_change": "trajectory batch to singleton orbit; retain every mismatch"}
    save(out / "protocol.json", protocol)
    results = []
    for index, cell in enumerate(info["cells"]):
        if args.select and cell["name"] not in args.select:
            continue
        period = cell["period_candidate"]
        if not period:
            results.append({**cell, "status": "no recurrence candidate; no Newton orbit attempted"})
            save(out / "orbits.json", {"protocol": protocol, "cells": results})
            continue
        if not 1 <= period <= 192:
            raise ValueError("candidate period outside the declared search")
        dest = out / cell["name"]
        dest.mkdir(exist_ok=False)
        started = time.monotonic()
        row = {**cell, "status": "running", "verified_primitive_orbit": False}

        def progress(stage, **kw):
            r = {"cell_index": index, "cell": cell["name"], "stage": stage,
                 "seconds": round(time.monotonic()-started, 3), **kw}
            with (out / "progress.jsonl").open("a") as f:
                f.write(json.dumps(r)+"\n")

        try:
            orb = soft.SoftOrbit(loop, cell["alpha"]*vector, 10.0**cell["log10_temperature"],
                                 chunk=args.chunk, jacobian_method="ad")
            X = torch.as_tensor(states[index].reshape((1,)+orb.shape), dtype=loop.state_dtype, device=loop.device)
            np.save(dest / "initial-state.npy", states[index])
            X, residuals, M, points = refine(orb, X, period, 1e-11, args.max_newton, progress)
            row["newton_residuals"] = residuals
            np.save(dest / "orbit-state.npy", X.double().cpu().numpy())
            np.save(dest / "orbit-points.npy", np.stack([p.double().cpu().numpy().reshape(-1) for p in points]))
            np.save(dest / "monodromy.npy", M)
            vals = np.linalg.eigvals(M)
            vals = vals[np.argsort(-np.abs(vals))]
            np.save(dest / "multipliers.npy", vals)
            row["multipliers"] = [[float(x.real),float(x.imag),float(abs(x))] for x in vals[:8]]
            row["spectral_radius"] = float(abs(vals[0]))
            row["asymptotic_log_contraction_per_step"] = float(np.log(max(abs(vals[0]),1e-300))/period)
            row["primitive_gate"] = primitive_gate(orb.return_distances(X, period), period)
            if residuals[-1] >= 1e-11 or not row["primitive_gate"]["closed"] or not row["primitive_gate"]["no_lower_period"]:
                row["status"] = "Newton or primitive-period gate failed"
            else:
                row["derivative_precision"] = precision_checks(orb, X, M, points, dest, progress)
                row["verified_primitive_orbit"] = row["derivative_precision"]["passed"]
                row["attracting"] = bool(abs(vals[0]) < 1-1e-8)
                row["status"] = "verified numerical primitive orbit" if row["verified_primitive_orbit"] else "derivative qualification failed"
            row["tokens"] = []
            for point in points:
                p = orb.probabilities(point).double().cpu().numpy()
                token = int(p.argmax())
                row["tokens"].append({"id": token, "text": loop.sm.tok.decode([token]), "probability": float(p[token])})
        except Exception as error:
            row.update(status="failed; retained", error=f"{type(error).__name__}: {error}")
        row["seconds"] = time.monotonic()-started
        save(dest / "orbit.json", row)
        results.append(row)
        save(out / "orbits.json", {"protocol": protocol, "cells": results})
        progress("cell_complete", status=row["status"])
        print(json.dumps({k:row.get(k) for k in ("name","period_candidate","status","spectral_radius","seconds")}), flush=True)
    save(out / "orbits.json", {"protocol": protocol, "cells": results, "complete": True})


if __name__ == "__main__":
    main()
