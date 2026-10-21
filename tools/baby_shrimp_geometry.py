"""Continue the selected period-18 orbit and retain its full float64 spectrum.

Exploratory BSG1. Full-state continuation and all-lag primitive checks precede
every spectral label. A sign change of the largest-modulus multiplier is not
called a zero: real pairs can exchange dominance or become complex. The saved
leading eigenspaces support a separate trace/avoidance analysis.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "8"
import numpy as np

from baby_shrimp_orbits import DATA, ROOT, precision_checks, primitive_gate, refine, save
from doubly_superstable import Frontier
import llm_feigenbaum as lfb


def spectrum(matrix):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("finite square monodromy required")
    values, vectors = np.linalg.eig(matrix)
    order = np.argsort(-np.abs(values))
    values, vectors = values[order], vectors[:, order]
    count = min(6, len(values))
    vectors = vectors[:, :count]
    residual = np.linalg.norm(matrix @ vectors - vectors * values[:count], axis=0)
    relative = residual / np.maximum(np.abs(values[:count]) * np.linalg.norm(vectors, axis=0), 1e-300)
    return values, vectors, relative


def validate_plan(plan, band):
    if plan["period"] != 18 or plan["resolution"] != [21, 31]:
        raise ValueError("only the declared period-18 geometry grid is supported")
    bounds = plan["bands"][band]["rows"]
    if not (0 <= bounds[0] < bounds[1] <= plan["resolution"][0]):
        raise ValueError("invalid row range")
    for key in ("alpha", "log10_temperature"):
        lo, hi = plan[key]
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError("finite increasing axes required")
    if plan["attempts"] != 2 or plan["max_newton"] != 10 or plan["newton_tolerance"] != 1e-11:
        raise ValueError("continuation gates or bounds changed")
    if plan.get("propagation_gate", "full") not in ("full", "primitive_orbit"):
        raise ValueError("unknown continuation propagation gate")
    return bounds


def propagation_passed(record, plan):
    """An orbit can supply a Newton seed without qualifying its spectrum.

    BSG2 discloses this separation. Spectrally failed rows remain failed and
    cannot be used in geometry claims merely because they seed a neighbor.
    """
    if plan.get("propagation_gate", "full") == "primitive_orbit":
        return bool(record.get("orbit_passed") and record.get("same_cyclic_text"))
    return bool(record["passed"])


def completion_status(rows, queue_remaining, planned_cells):
    orbital = {(r["row"], r["column"]) for r in rows if r.get("orbit_passed")}
    qualified = {(r["row"], r["column"]) for r in rows if r["passed"]}
    complete = not queue_remaining and bool(qualified)
    reason = ("no qualified spectral cell" if not qualified else
              "frontier exhausted" if not queue_remaining else "bounded numerical time expired")
    return {"complete": complete, "reason": reason, "attempts": len(rows),
            "orbit_solved": len(orbital), "solved": len(qualified),
            "planned_cells": planned_cells, "remaining_frontier": queue_remaining}


def composed_check(orb, points, matrix, vectors):
    """Qualify the composed operator in random and leading spectral directions."""
    rng = np.random.default_rng(4421)
    directions = [rng.normal(size=orb.n)]
    for v in vectors.T:
        directions.append(v.real)
        if np.linalg.norm(v.imag) > 1e-8:
            directions.append(v.imag)
    directions = [d / np.linalg.norm(d) for d in directions if np.linalg.norm(d) > 1e-12]
    D = np.stack(directions, axis=1)
    W = lfb._rows(D, points[0])
    for point in points:
        W = orb.jvp_rows(point, W)
    direct = lfb._cols(W)
    dense = matrix @ D
    denominator = np.maximum(np.maximum(np.linalg.norm(direct, axis=0), np.linalg.norm(dense, axis=0)), 1e-300)
    errors = np.linalg.norm(direct - dense, axis=0) / denominator
    return D, direct, dense, errors


def qualified_spectral_count(values):
    """Watch both leading directions and any comparable/conjugate companions.

    The seed's third multiplier is about 1e-12. Demanding relative accuracy
    below its numerical floor would reject an otherwise resolved leading pair.
    Such tiny stored directions are not eligible for the tracked block.
    """
    count = max(2, int(np.sum(np.abs(values[:6]) >= .01 * abs(values[0]))))
    while count < min(6, len(values)):
        included = values[:count]
        if all(abs(v.imag) <= 1e-14 or any(abs(w - v.conjugate()) < 1e-10 * max(abs(v), 1e-20)
                                         for w in included) for v in included):
            break
        count += 1
    return min(count, len(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="explore-relative BSP1 seed directory")
    ap.add_argument("--out", required=True, help="fresh explore-relative output")
    ap.add_argument("--band", type=int, choices=(0, 1), required=True)
    ap.add_argument("--plan", default="docs/baby-shrimp-zoom/geometry-plan.json")
    args = ap.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    lo, hi = validate_plan(plan, args.band)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists() or (out / "cells").exists():
        raise ValueError("preserve existing attempts; choose a fresh output")
    (out / "cells").mkdir()
    start = time.monotonic()
    deadline = start + plan["numerical_seconds"]

    def emit(stage, **fields):
        record = {"cell_index": fields.pop("cell_index", -1), "stage": stage,
                  "seconds": round(time.monotonic() - start, 3), **fields}
        with (out / "progress.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    import torch
    import explore_llm_soft_loop as soft
    import explore_llm_attractors as ela

    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.85)
    src = DATA / "explore" / args.input
    metadata = json.loads((src / "seeds.json").read_text())
    seed_hash = hashlib.sha256((src / "seeds.npz").read_bytes()).hexdigest()
    if seed_hash != metadata["states_sha256"] or seed_hash != plan["seed_sha256"]:
        raise ValueError("seed hash mismatch")
    index = next(i for i, c in enumerate(metadata["cells"]) if c["name"] == "baby-core")
    seed = metadata["cells"][index]
    if seed["period_candidate"] != 18 or [seed["alpha"], seed["log10_temperature"]] != plan["seed"]:
        raise ValueError("seed coordinates or period changed")
    loop = soft.SoftLoop("qwen0.5b", 12, 8, "open", ela.OPENER, 1, "float64", "cuda", "native")
    if loop.sm.engine.revision != plan["model_revision"]:
        raise ValueError("unexpected model revision")
    norm = loop.model.model.norm
    test = torch.linspace(-1, 1, norm.weight.numel(), dtype=torch.float64, device=loop.device)[None, None, :]
    sensitivity = float((norm(test * (1 + 1e-12)) - norm(test)).abs().max())
    if not (0 < sensitivity < 1e-9 and hasattr(loop.model.model.rotary_emb, "_inv_freq64")):
        raise ValueError("effective float64 normalization/rotary check failed")
    vector = soft.elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"]

    def make_cell(a, t):
        return soft.SoftOrbit(loop, a * vector, 10.0 ** t, chunk=512, jacobian_method="ad")

    p = plan["period"]
    cell = make_cell(*plan["seed"])
    states = np.load(src / "seeds.npz", allow_pickle=False)["states"]
    X = torch.as_tensor(states[index].reshape((1,) + cell.shape), dtype=loop.state_dtype, device=loop.device)
    seed_dir = out / "seed"
    seed_dir.mkdir()
    X, hist, M, points = refine(cell, X, p, 1e-11, 3, emit)
    primitive = primitive_gate(cell.return_distances(X, p), p)
    precision = precision_checks(cell, X, M, points, seed_dir, emit)
    if hist[-1] >= 1e-11 or not all(primitive[k] for k in ("closed", "no_lower_period")) or not precision["passed"]:
        save(seed_dir / "failed.json", {"newton": hist, "primitive": primitive, "precision": precision})
        raise ValueError("seed numerical qualification failed")
    np.save(seed_dir / "monodromy.npy", M)
    np.save(seed_dir / "state.npy", lfb._flat(X))
    save(out / "protocol.json", {"plan": plan, "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                                 "args": vars(args), "seed_sha256": seed_hash, "seed_provenance": metadata["provenance"],
                                 "seed_newton": hist, "seed_primitive": primitive, "seed_precision": precision,
                                 "norm_sensitivity": sensitivity, "rotary_float64": True,
                                 "started_utc": datetime.now(timezone.utc).isoformat()})
    # Low-rank Newton is only a solver; every accepted point gets a dense
    # float64 monodromy and direct composed-JVP checks afterwards.
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(cell.n, 8)))[0]
    alpha = np.linspace(*plan["alpha"], plan["resolution"][1])
    temperature = np.linspace(*plan["log10_temperature"], plan["resolution"][0])[lo:hi]
    first = (int(np.argmin(abs(temperature - seed["log10_temperature"]))),
             int(np.argmin(abs(alpha - seed["alpha"]))))
    # Bridge to the band's nearest cell in parameter steps no larger than one
    # grid interval, retaining every intermediate Newton history.
    target = (float(alpha[first[1]]), float(temperature[first[0]]))
    count = max(1, int(np.ceil(max(abs(target[0] - seed["alpha"]) / (alpha[1] - alpha[0]),
                                     abs(target[1] - seed["log10_temperature"]) / (temperature[1] - temperature[0])))))
    for k in range(1, count + 1):
        u = k / count
        a = seed["alpha"] + u * (target[0] - seed["alpha"])
        t = seed["log10_temperature"] + u * (target[1] - seed["log10_temperature"])
        cell = make_cell(a, t)
        X, Q, hist = lfb.newton(cell, X, p, Q, 1e-11, 10)
        gate = primitive_gate(cell.return_distances(X, p), p)
        emit("bridge", step=k, alpha=a, log10_temperature=t, residuals=hist, primitive=gate)
        if hist[-1] >= 1e-11 or not gate["closed"] or not gate["no_lower_period"]:
            raise ValueError("bridge lost the primitive period-18 orbit")

    frontier = Frontier((len(temperature), len(alpha)), first, plan["attempts"])
    warm = {None: (X, Q)}
    rows = []
    while frontier.queue and time.monotonic() < deadline:
        (i, j), parent = frontier.pop()
        number = len(rows)
        dest = out / "cells" / f"r{i+lo:02d}-c{j:02d}-a{len(frontier.tries[(i,j)]):02d}"
        dest.mkdir()
        record = {"cell_index": number, "row": i + lo, "column": j, "alpha": float(alpha[j]),
                  "log10_temperature": float(temperature[i]), "parent": None if parent is None else [parent[0]+lo, parent[1]],
                  "attempt": len(frontier.tries[(i, j)]), "passed": False}
        tick = time.monotonic()
        try:
            Xs, Qs = warm[parent]
            cell = make_cell(alpha[j], temperature[i])
            Xn, Qn, hist = lfb.newton(cell, Xs.clone(), p, Qs.copy(), 1e-11, 10)
            record["newton_residuals"] = [float(v) if np.isfinite(v) else str(v) for v in hist]
            np.save(dest / "state.npy", lfb._flat(Xn))
            if not np.isfinite(hist).all() or hist[-1] >= 1e-11:
                raise ValueError("Newton did not converge")
            points_all = cell.points(Xn, p)
            state_scale = max(float(Xn.norm()), 1.0)
            returns = [float((point - Xn).norm()) / state_scale for point in points_all[1:]]
            record["primitive"] = primitive_gate(returns, p)
            if not record["primitive"]["closed"] or not record["primitive"]["no_lower_period"]:
                raise ValueError("full-state primitive-period check failed")
            M = np.eye(cell.n)
            for phase, point in enumerate(points_all[:-1]):
                M = cell.jacobian(point) @ M
                emit("jacobian", cell_index=number, row=i+lo, column=j, phase=phase+1)
            values, vectors, eigen_residual = spectrum(M)
            watched = qualified_spectral_count(values)
            D, direct, dense, errors = composed_check(cell, points_all[:-1], M, vectors[:, :watched])
            orbit = np.stack([lfb._flat(point) for point in points_all[:-1]])
            tokens = [int(cell.probabilities(point).argmax()) for point in points_all[:-1]]
            parent_state = lfb._flat(Xs)
            distances = np.linalg.norm(orbit - parent_state, axis=1) / max(np.linalg.norm(parent_state), 1.0)
            np.savez_compressed(dest / "spectral.npz", points=orbit, multipliers=values, vectors=vectors,
                                eigen_relative_residual=eigen_residual, directions=D, direct_jvp=direct,
                                dense_jvp=dense, jvp_relative_error=errors, tokens=tokens)
            record.update(spectral_radius=float(abs(values[0])),
                          qualified_spectral_count=watched,
                          multipliers=[[float(v.real), float(v.imag)] for v in values[:6]],
                          eigen_relative_residual=eigen_residual.tolist(), jvp_relative_error=errors.tolist(),
                          parent_orbit_distance=float(distances.min()), closest_parent_phase=int(distances.argmin()),
                          same_cyclic_text=canonical(tokens) == canonical([271, 785, 3405, 271, 785, 13885] * 3))
            if (not np.isfinite(errors).all() or float(errors.max()) >= 1e-7
                    or not np.isfinite(eigen_residual[:watched]).all() or float(eigen_residual[:watched].max()) >= 1e-6):
                raise ValueError("dense/composed spectral qualification failed")
            if not record["same_cyclic_text"]:
                raise ValueError("continued text loop changed")
            record["passed"] = True
            record["attracting"] = bool(abs(values[0]) < 1 - 1e-8)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        record["orbit_passed"] = bool(record.get("primitive", {}).get("closed")
                                       and record.get("primitive", {}).get("no_lower_period"))
        record["propagation_passed"] = propagation_passed(record, plan)
        if record["propagation_passed"]:
            warm[i, j] = (Xn, Qn)
        record["seconds"] = time.monotonic() - tick
        save(dest / "cell.json", record)
        rows.append(record)
        frontier.done((i, j), record["propagation_passed"])
        live = frontier.parents_still_useful()
        for key in list(warm):
            if key is not None and key not in live:
                del warm[key]
        result = completion_status(rows, len(frontier.queue), len(temperature)*len(alpha))
        save(out / "grid.json", {**result, "rows": rows})
        emit("cell_complete", **record)
    result = completion_status(rows, len(frontier.queue), len(temperature)*len(alpha))
    save(out / "status.json", {**result, "elapsed_seconds": time.monotonic() - start})
    if not result["solved"]:
        raise ValueError("no qualified spectral cell; retained failed setup")
    if not result["complete"]:
        raise TimeoutError("partial continuation retained at its numerical time limit")


def canonical(tokens):
    return min(tuple(tokens[i:] + tokens[:i]) for i in range(len(tokens)))


if __name__ == "__main__":
    main()
