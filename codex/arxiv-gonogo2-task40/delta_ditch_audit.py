"""Independent Task40 saved-grid analysis; no producer imports or model calls.

Tracking: SVD bases for real invariant planes, projection Frobenius overlap,
maximum-overlap breadth-first propagation. Producer uses QR/principal angles.
Quadratic fits: explicit six-column least squares in cell coordinates.
Retrospective numerical audit, not a new proof or an original-run replay.
"""
import argparse
from collections import deque
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path("/data2/neural-shrimp")
BASE = Path("/data2/neural-shrimp-data/explore/llm-soft-loop")
OUT = ROOT / "codex/arxiv-gonogo2-task40"
PINS = {}


def read(path):
    path = Path(path)
    a = path.stat()
    data = path.read_bytes()
    b = path.stat()
    if (a.st_size, a.st_mtime_ns, a.st_ino) != (b.st_size, b.st_mtime_ns, b.st_ino):
        raise ValueError(f"Concurrent mutation: {path}")
    pin = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if str(path) in PINS and PINS[str(path)] != pin:
        raise ValueError(f"Changed input: {path}")
    PINS[str(path)] = pin
    return data


def grid(folder):
    with np.load(io.BytesIO(read(folder / "grid.npz")), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def combine(folders, ritz=False):
    blocks = []
    for folder in folders:
        z = grid(folder)
        if ritz:
            with np.load(io.BytesIO(read(folder / "ritz.npz")), allow_pickle=False) as v:
                for k in ("alpha", "log10_temperature", "multipliers"):
                    if not np.array_equal(z[k], v[k], equal_nan=True):
                        raise ValueError(f"Ritz/grid axis or spectrum mismatch: {folder}/{k}")
                z["vectors"] = v["vectors"].copy()
        blocks.append(z)
    blocks.sort(key=lambda z: z["log10_temperature"][0])
    a = blocks[0]["alpha"]
    if any(not np.array_equal(b["alpha"], a) for b in blocks):
        raise ValueError("Different alpha axes")
    t = np.concatenate([b["log10_temperature"] for b in blocks])
    if not np.all(np.diff(t) > 0):
        raise ValueError("Duplicate/unordered temperature axis")
    names = ["ok", "mu", "mu_imag", "next_modulus", "newton", "same_tokens"]
    if ritz:
        names += ["multipliers", "vectors"]
    result = {k: np.concatenate([b[k] for b in blocks], axis=0) for k in names}
    return dict(result, alpha=a, log10_temperature=t)


def plane(values, vectors, pair):
    x, y = pair
    if max(abs(values[x].imag), abs(values[y].imag)) > 1e-14:
        if abs(values[x] - values[y].conjugate()) > 1e-9 * max(1, abs(values[x])):
            return None
        real = np.column_stack((vectors[:, x].real, vectors[:, x].imag))
    else:
        columns = []
        for k in pair:
            v = vectors[:, k]
            v = v * np.exp(-1j * np.angle(v[np.argmax(abs(v))]))
            columns.append(v.real)
        real = np.column_stack(columns)
    u, s, _ = np.linalg.svd(real, full_matrices=False)
    if not np.isfinite(s).all() or s[-1] <= 1e-14 * s[0]:
        return None
    return u[:, :2]


def stationary(field, valid, a, t):
    di, dj = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing="ij")
    x, y = di.ravel(), dj.ravel()
    design = np.column_stack((np.ones(25), x, y, x*x, x*y, y*y))
    candidates = []
    for i in range(2, len(t)-2):
        for j in range(2, len(a)-2):
            if not valid[i-2:i+3, j-2:j+3].all():
                continue
            obs = field[i-2:i+3, j-2:j+3].ravel()
            coeff, _, rank, _ = np.linalg.lstsq(design, obs, rcond=None)
            h = np.array([[2*coeff[3], coeff[4]], [coeff[4], 2*coeff[5]]])
            if rank != 6 or abs(np.linalg.det(h)) < 1e-20:
                continue
            offset = np.linalg.solve(h, -coeff[1:3])
            if np.max(abs(offset)) > .75:
                continue
            ev = np.linalg.eigvalsh(h)
            candidates.append({
                "cell": [i, j], "alpha": float(a[j] + offset[1]*(a[1]-a[0])),
                "log10T": float(t[i] + offset[0]*(t[1]-t[0])),
                "value": float(coeff[0] + coeff[1:3]@offset + .5*offset@h@offset),
                "H_cell_eigenvalues": ev.tolist(), "saddle": bool(ev[0] < 0 < ev[1]),
                "rms": float(np.sqrt(np.mean((design@coeff - obs)**2))),
                "offset_linf": float(np.max(abs(offset))),
            })
    selected = []
    for c in sorted(candidates, key=lambda c: c["offset_linf"]):
        if all(abs(c["alpha"]-p["alpha"]) > abs(a[1]-a[0]) or
               abs(c["log10T"]-p["log10T"]) > abs(t[1]-t[0]) for p in selected):
            selected.append(c)
    return {"candidates": len(candidates), "deduplicated": selected}


def edges(field, valid):
    out = []
    for i, j in np.argwhere(valid):
        for di, dj in ((1, 0), (0, 1)):
            ni, nj = i+di, j+dj
            if ni < valid.shape[0] and nj < valid.shape[1] and valid[ni, nj]:
                if np.sign(field[i, j]) != np.sign(field[ni, nj]):
                    out.append((i+di/2, j+dj/2))
    return np.array(out, dtype=float).reshape(-1, 2)


def matches(first, second):
    if len(first) == 0 or len(second) == 0:
        return 0
    return int(np.sum(np.min(np.max(abs(first[:, None] - second[None]), axis=2), axis=1) <= 1))


def delta():
    z = combine([BASE / f"pod64/dss/delta2d-d{k}" for k in range(4)], True)
    values = z["multipliers"][..., 0] + 1j*z["multipliers"][..., 1]
    vectors = z["vectors"]
    good = z["ok"] & np.isfinite(values).all(axis=-1)
    ratio = abs(values[..., 1]) / np.maximum(abs(values[..., 2]), 1e-300)
    seed = tuple(np.unravel_index(np.argmax(np.where(good, ratio, -np.inf)), good.shape))
    pairs = ((0, 1), (0, 2), (1, 2))
    bases = {}
    for i, j in np.argwhere(good):
        bases[i, j] = [(p, plane(values[i, j], vectors[i, j], p)) for p in pairs]
        bases[i, j] = [(p, q) for p, q in bases[i, j] if q is not None]
    # The declared producer traversal selects the same mathematical objective;
    # implementation and orthogonalization are independent.
    picked = {seed: (0, 1)}
    chosen = {seed: dict(bases[seed])[(0, 1)]}
    overlaps = np.full(good.shape, np.nan)
    overlaps[seed] = 1
    queue = deque([seed])
    while queue:
        i, j = queue.popleft()
        for ni, nj in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
            key = ni, nj
            if key in chosen or key not in bases or not bases[key]:
                continue
            options = [(float(np.sum((chosen[i, j].T @ q)**2)/2), p, q) for p, q in bases[key]]
            overlap, pair, basis = max(options, key=lambda v: v[0])
            overlaps[key] = overlap
            chosen[key], picked[key] = basis, pair
            queue.append(key)
    tau = np.full(good.shape, np.nan)
    det = tau.copy()
    for key, pair in picked.items():
        vals = values[key][list(pair)]
        tau[key], det[key] = np.sum(vals).real, np.prod(vals).real
    tracked = np.isfinite(tau)
    real = good & (z["mu_imag"] == 0)
    real_edges = edges(z["mu"], real)
    trace_edges = edges(tau, tracked)
    delta_edges = edges(det, tracked)
    complex_cells = np.argwhere(good & (abs(z["mu_imag"]) > 0))
    measured = np.concatenate((real_edges, complex_cells))
    agreement = np.sign(tau) == np.sign(z["mu"])
    return {
        "shape": list(good.shape), "solved": int(good.sum()), "tracked": int(tracked.sum()),
        "seed": list(map(int, seed)), "overlap_below_09": int(np.sum(overlaps < .9)),
        "overlap_min": float(np.nanmin(overlaps)), "sign_agreement": int(np.sum(agreement & tracked)),
        "disagreement_cells": np.argwhere(~agreement & tracked).tolist(),
        "selected_pairs": {str(p): sum(v == p for v in picked.values()) for p in pairs},
        "real_zero_edges": len(real_edges), "trace_zero_edges": len(trace_edges),
        "delta_zero_edges": len(delta_edges), "complex_leading_cells": len(complex_cells),
        "real_near_trace": matches(real_edges, trace_edges),
        "trace_near_measured": matches(trace_edges, measured),
        "real_near_delta": matches(real_edges, delta_edges),
        "delta_near_real": matches(delta_edges, real_edges),
        "trace_range": [float(np.nanmin(tau)), float(np.nanmax(tau))],
        "delta_range": [float(np.nanmin(det)), float(np.nanmax(det))],
        "trace_stationary": stationary(tau, tracked & (overlaps >= .9), z["alpha"], z["log10_temperature"]),
        "delta_stationary": stationary(det, tracked & (overlaps >= .9), z["alpha"], z["log10_temperature"]),
        "leading_real_stationary": stationary(z["mu"], real, z["alpha"], z["log10_temperature"]),
        "newton_max": float(np.max(z["newton"][good])),
        "same_tokens_false_solved": int(np.sum(good & ~z["same_tokens"])),
        "limits": [
            "Finite mesh, selected rank-two Ritz planes, local quadratic least squares, not certified critical points.",
            "No model, original Newton, or composed derivative replay.",
            "Original grid stores no Ritz residuals; saved Ritz values/vectors are audited, not independently generated.",
            "Absence of a fitted stationary point is only for disclosed fit/filter/mesh.",
        ],
    }


def stats(z):
    ok = z["ok"]
    real = ok & (z["mu_imag"] == 0)
    idx = tuple(np.unravel_index(np.argmin(np.where(ok, abs(z["mu"] + 1j*z["mu_imag"]), np.inf)), ok.shape))
    return {"cells": int(ok.size), "solved": int(ok.sum()), "complex": int(np.sum(ok & ~real)),
            "leading_real_range": [float(z["mu"][ok].min()), float(z["mu"][ok].max())],
            "leading_modulus_min": float(abs(z["mu"][idx] + 1j*z["mu_imag"][idx])),
            "minimum_index": list(map(int, idx)), "minimum_alpha": float(z["alpha"][idx[1]]),
            "minimum_log10T": float(z["log10_temperature"][idx[0]]),
            "next_max": float(z["next_modulus"][ok].max()),
            "newton_max": float(z["newton"][ok].max()),
            "same_tokens_false_solved": int(np.sum(ok & ~z["same_tokens"])),
            "real_sign_change_edges": len(edges(z["mu"], real))}


def ditches():
    out = {}
    for name in ("pod64/dss/p6-body", "p6s/ditch-top-a", "p6s/ditch-main-b"):
        out[name] = stats(grid(BASE / name))
    patch = combine([BASE / f"pod64/dss/ditch2d-b{k}" for k in range(4)])
    out["ditch2d"] = stats(patch)
    gpu = combine([BASE / f"pod64/dss/ditchfine-g{k}" for k in range(2)])
    out["fine_gpu"] = stats(gpu)
    col = int(np.argmin(abs(gpu["alpha"] - .07318)))
    out["fine_gpu"]["alpha_07318_column"] = col
    out["fine_gpu"]["column_min"] = float(gpu["mu"][:, col].min())
    out["fine_gpu"]["row_minimum_columns"] = np.argmin(gpu["mu"], axis=1).tolist()
    folders = [BASE / f"p6s/ditch-fine-b{k}" for k in range(3)]
    complete = all((f / "grid.json").is_file() for f in folders)
    out["fine_cpu_complete"] = complete
    if complete:
        metas = [json.loads(read(f / "grid.json")) for f in folders]
        cpu = combine(folders)
        out["fine_cpu"] = stats(cpu)
        axes = all(np.array_equal(cpu[k], gpu[k]) for k in ("alpha", "log10_temperature"))
        out["fine_comparison"] = {"identical_axes": axes, "cpu_cells_from_receipts": [r.get("cells") for r in metas]}
        if not axes:
            raise ValueError("CPU/GPU fine axes differ")
        common = cpu["ok"] & gpu["ok"]
        out["fine_comparison"].update(
            common_solved=int(common.sum()), mismatched_ok=int(np.sum(cpu["ok"] != gpu["ok"])),
            max_leading_complex_abs_diff=float(np.max(abs((cpu["mu"]+1j*cpu["mu_imag"] -
                                                         gpu["mu"]-1j*gpu["mu_imag"])[common]))),
            max_next_abs_diff=float(np.max(abs((cpu["next_modulus"]-gpu["next_modulus"])[common]))),
            same_tokens_mismatches=int(np.sum(cpu["same_tokens"][common] != gpu["same_tokens"][common])))
    return out


def main():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert os.environ.get(key) == "1", key
    assert os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=["delta", "ditches"], required=True)
    args = ap.parse_args()
    start = time.monotonic()
    target = OUT / (args.section + "-evidence.json")
    with target.open("x") as f:
        result = globals()[args.section]()
        json.dump({"section": args.section, "result": result, "input_manifest": PINS,
                   "numpy": np.__version__, "elapsed_seconds": time.monotonic()-start},
                  f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(target), "seconds": time.monotonic()-start}))


if __name__ == "__main__":
    main()
