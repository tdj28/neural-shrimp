"""Do the incidental numerical differences of ordinary inference change what the loop does? (run REPRO)

Production inference is not bitwise reproducible: the batch a request shares, the kernel chosen for a matrix shape,
and the precision all change the rounding. This tool runs the same cells twice from the same start, in lockstep,
differing only in one such detail, and measures how far the two trajectories drift apart:
    batch     the cells alone in a batch of n rows, against the same cells followed by n other rows (batch 2n): the
              same arithmetic on different matrix shapes;
    precision float32 against float64 (model and state);
    bf16      the model cast to bfloat16 (state kept in float32), the cells in batch n against batch 2n.
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
                    choices=("batch64", "batch32", "precision", "bf16"))
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
