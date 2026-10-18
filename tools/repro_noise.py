"""Do the incidental numerical differences of ordinary inference change what the loop does? (run REPRO)

Production inference is not bitwise reproducible: the batch a request shares, the kernel chosen for a matrix shape,
and the precision all change the rounding. This tool runs the same cells twice from the same start, in lockstep,
differing only in one such detail, and measures how far the two trajectories drift apart:
    batch     the cells alone in a batch of n rows, against the same cells followed by n other rows (batch 2n): the
              same arithmetic on different matrix shapes;
    precision float32 against float64 (model and state);
    bf16      the model cast to bfloat16 (state kept in float32), the cells in batch n against batch 2n.
Fixed-shape controls (run FS, external review 18 Oct 2026; float64, the matrix shapes identical in both runs):
    repeat    the same n rows in the same order twice: the computation's own run-to-run determinism;
    fixedperm the same n cells in a batch of n, the second run with its rows permuted (seeded), so each cell has other
              neighbors and another row index but every kernel sees the same shapes;
    dose      n x L rows in both runs (each cell once per level); in the second run the start of the block for level
              d is displaced by d |x| along a seeded random unit direction, for d in --doses. Per cell and level, the
              first step at which the distance exceeds --eps, against ln(eps / d) / lambda with lambda the cell's map
              exponent.
Per comparison, cell and step: the relative distance |x_A - x_B| / |x_A| of the soft tokens and whether the two most
probable next tokens agree. Per cell: the first step at which the distance exceeds 1e-2, the first step after which
the most probable tokens never agree again within the run, and the fraction of disagreeing steps over the last
`tail` steps. Cells come from a labelled file (docs/h2-horizon-cells.json: chaotic, periodic and other cells of two
float64 maps), so divergence can be read against the dynamical regime.

Usage: python tools/repro_noise.py --cells-file docs/h2-horizon-cells.json --steps 1500 --device cuda \
           --compare batch64 batch32 precision --out llm-soft-loop/pod64/repro/x
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def make_loop(esl, ela, args, dtype, bf16=False):
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, dtype, args.device,
                        args.weights if dtype == "float64" else "native")
    if bf16:
        torch = loop.torch
        loop.model = loop.model.to(torch.bfloat16)
        loop.dtype = torch.bfloat16
    return loop


def twin_runs(loopA, loopB, steer, padB, temps, steps, log=print, label=""):
    """Run loopA on the n steered rows and loopB on the same rows followed by padB extra rows, in lockstep; returns
    per-step relative distances (steps, n) and token agreement (steps, n) of the first n rows."""
    torch = loopA.torch
    n = len(steer)
    XA = loopA.prepare(steer)
    rowsB = np.concatenate([steer, padB]) if padB is not None else steer
    XB = loopB.prepare(rowsB)
    tA = torch.as_tensor(temps, dtype=loopA.state_dtype, device=loopA.device)
    tB = torch.as_tensor(np.concatenate([temps, temps[:len(rowsB) - n]]) if padB is not None else temps,
                         dtype=loopB.state_dtype, device=loopB.device)
    dist = np.zeros((steps, n), dtype=np.float32)
    agree = np.zeros((steps, n), dtype=bool)
    tic = time.perf_counter()
    for i in range(steps):
        XA, pA = loopA.step(XA, tA)
        XB, pB = loopB.step(XB, tB)
        a = XA[:, -1].double()
        b = XB[:n, -1].double().to(a.device)
        dist[i] = ((a - b).norm(dim=1) / a.norm(dim=1)).cpu().numpy()
        agree[i] = (pA.argmax(-1) == pB[:n].argmax(-1).to(pA.device)).cpu().numpy()
        if (i + 1) % 250 == 0:
            log(json.dumps({"compare": label, "step": i + 1, "of": steps, "median_dist": float(np.median(dist[i])),
                            "seconds": round(time.perf_counter() - tic, 1)}))
    return dist, agree


def fixed_runs(loopA, loopB, steerA, steerB, tempsA, tempsB, mapB, steps, perturb=None, log=print, label=""):
    """Lockstep runs with the same number of rows; row i of A is compared with row mapB[i] of B. perturb: (rows,
    rel) displaces the starts of those rows of B by rel |x| along seeded random unit directions. Returns the
    per-step relative distances and token agreement (steps, rows of A)."""
    torch = loopA.torch
    XA = loopA.prepare(steerA)
    XB = loopB.prepare(steerB)
    if perturb is not None:
        rows, rel = perturb
        g = torch.Generator(device=XB.device).manual_seed(20261018)
        U = torch.randn(XB[:, -1].shape, generator=g, dtype=XB.dtype, device=XB.device)
        U = U / U.norm(dim=1, keepdim=True)
        r = torch.as_tensor(rows, device=XB.device)
        scale = torch.as_tensor(rel, dtype=XB.dtype, device=XB.device)
        XB = XB.clone()
        XB[r, -1] = XB[r, -1] + (scale[:, None] * XB[r, -1].norm(dim=1, keepdim=True)) * U[r]
    tA = torch.as_tensor(tempsA, dtype=loopA.state_dtype, device=loopA.device)
    tB = torch.as_tensor(tempsB, dtype=loopB.state_dtype, device=loopB.device)
    m = torch.as_tensor(mapB, device=loopA.device)
    n = len(steerA)
    dist = np.zeros((steps, n), dtype=np.float64)
    agree = np.zeros((steps, n), dtype=bool)
    tic = time.perf_counter()
    for i in range(steps):
        XA, pA = loopA.step(XA, tA)
        XB, pB = loopB.step(XB, tB)
        a = XA[:, -1].double()
        b = XB[m, -1].double().to(a.device)
        dist[i] = ((a - b).norm(dim=1) / a.norm(dim=1)).cpu().numpy()
        agree[i] = (pA.argmax(-1) == pB[m].argmax(-1).to(pA.device)).cpu().numpy()
        if (i + 1) % 250 == 0:
            log(json.dumps({"compare": label, "step": i + 1, "of": steps, "median_dist": float(np.median(dist[i])),
                            "max_dist": float(dist[i].max()), "seconds": round(time.perf_counter() - tic, 1)}))
    return dist, agree


def summarize(dist, agree, tail):
    steps, n = dist.shape
    out = []
    for c in range(n):
        over = np.nonzero(dist[:, c] > 1e-2)[0]
        dis = np.nonzero(~agree[:, c])[0]
        never_again = None
        if len(dis) and agree[-1, c] == False:                                  # noqa: E712
            ok = np.nonzero(agree[:, c])[0]
            never_again = int(ok[-1] + 1) if len(ok) else 0
        out.append({"first_step_dist_above_1e-2": int(over[0]) + 1 if len(over) else None,
                    "first_disagreement": int(dis[0]) + 1 if len(dis) else None,
                    "disagreeing_for_good_from": never_again,
                    "tail_disagreement_fraction": float(np.mean(~agree[-tail:, c])),
                    "final_dist": float(dist[-1, c]), "max_dist": float(dist[:, c].max()),
                    "min_nonzero_dist": float(dist[:, c][dist[:, c] > 0].min()) if np.any(dist[:, c] > 0) else 0.0})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12", help="'none': unsteered, alpha 0 everywhere")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--cells-file", required=True, help="JSON with 'cells' ([log10T, alpha] pairs) and 'label'")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--tail", type=int, default=500)
    ap.add_argument("--compare", nargs="+", default=["batch64", "batch32", "precision", "bf16"],
                    choices=("batch64", "batch32", "precision", "bf16", "repeat", "fixedperm", "dose"))
    ap.add_argument("--doses", type=float, nargs="+", default=[1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3])
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--weights", default="native", choices=("native", "bf16"),
                    help="float64 runs of large models: bf16 storage of the decoder weights (exact), with --device auto")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    spec = json.loads(Path(args.cells_file).read_text())
    cells = spec["cells"]
    labels = spec.get("label", [None] * len(cells))
    temps = 10.0 ** np.array([t for t, _ in cells])
    steer = pad = None
    if args.vectors not in (None, "none"):
        v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
        steer = np.array([a * v for _, a in cells])
        pad = steer[::-1].copy()                 # the other rows of the larger batch: the same cells, reversed
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"

    def log(line):
        print(line, flush=True)
        with open(progress, "a") as f:
            f.write(line + "\n")

    results = {"args": vars(args), "labels": labels, "cells": cells, "comparisons": {}}
    arrays = {}
    loops = {}
    def rows_for(loop):
        nonlocal steer, pad
        if steer is None:                        # unsteered (--vectors none): zero rows of the model's width
            if any(a != 0 for _, a in cells):
                raise SystemExit("--vectors none needs alpha = 0 in every cell")
            steer = np.zeros((len(cells), loop.E.shape[1]))
            pad = steer.copy()
        return steer, pad

    for comp in args.compare:
        if comp in ("batch64", "batch32"):
            dt = "float64" if comp == "batch64" else "float32"
            if dt not in loops:
                loops[dt] = make_loop(esl, ela, args, dt)
            A = loops[dt]
            B = copy.copy(A)                     # shares the model; prepare gives B its own cache and rows
            st, pd = rows_for(A)
            dist, agree = twin_runs(A, B, st, pd, temps, args.steps, log, comp)
        elif comp == "precision":
            for dt in ("float64", "float32"):
                if dt not in loops:
                    loops[dt] = make_loop(esl, ela, args, dt)
            st, _ = rows_for(loops["float64"])
            dist, agree = twin_runs(loops["float64"], loops["float32"], st, None, temps, args.steps, log, comp)
        elif comp in ("repeat", "fixedperm", "dose"):
            if "float64" not in loops:
                loops["float64"] = make_loop(esl, ela, args, "float64")
            A = loops["float64"]
            B = copy.copy(A)
            st, _ = rows_for(A)
            nc = len(st)
            if comp == "repeat":
                dist, agree = fixed_runs(A, B, st, st, temps, temps, np.arange(nc), args.steps, None, log, comp)
            elif comp == "fixedperm":
                perm = np.random.default_rng(20261018).permutation(nc)          # row j of B holds cell perm[j]
                inv = np.argsort(perm)                                         # cell i sits in row inv[i] of B
                dist, agree = fixed_runs(A, B, st, st[perm], temps, temps[perm], inv, args.steps, None, log, comp)
                results["fixedperm_permutation"] = perm.tolist()
            else:
                L = len(args.doses)
                stL = np.concatenate([st] * L)
                tL = np.concatenate([temps] * L)
                rel = np.repeat(np.asarray(args.doses, dtype=np.float64), nc)
                dist, agree = fixed_runs(A, B, stL, stL, tL, tL, np.arange(nc * L), args.steps,
                                         (np.arange(nc * L), rel), log, comp)
                lam = np.asarray(spec.get("map_exponent", [np.nan] * nc), dtype=np.float64)
                rows = []
                for k, d0 in enumerate(args.doses):
                    for c in range(nc):
                        col = dist[:, k * nc + c]
                        over = np.nonzero(col > args.eps)[0]
                        t = int(over[0]) + 1 if len(over) else None
                        pred = float(np.log(args.eps / d0) / lam[c]) if lam[c] > 0 else None
                        rows.append({"dose": d0, "cell": c, "label": labels[c], "lambda": float(lam[c]),
                                     "first_step_above_eps": t, "predicted": pred,
                                     "max_dist": float(col.max()), "final_dist": float(col[-1])})
                results["dose_passages"] = rows
        else:                                    # bf16: batch n against batch 2n with a bfloat16 model
            A = make_loop(esl, ela, args, "float32", bf16=True)
            B = copy.copy(A)
            st, pd = rows_for(A)
            dist, agree = twin_runs(A, B, st, pd, temps, args.steps, log, comp)
            del A, B
        arrays[f"{comp}_dist"] = dist
        arrays[f"{comp}_agree"] = agree
        results["comparisons"][comp] = summarize(dist, agree, args.tail)
        log(json.dumps({"compare": comp, "done": True,
                        "cells_ever_above_1e-2": int(sum(r["first_step_dist_above_1e-2"] is not None
                                                         for r in results["comparisons"][comp]))}))
    np.savez_compressed(out / "repro.npz", **arrays)
    (out / "repro.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
