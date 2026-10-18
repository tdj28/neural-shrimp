"""Run LC (exploratory; human, 18 Oct 2026: follow the shrimp's fourth leg more finely): the period-5 attractor along
the centre line of a leg too thin for a grid, with controls just outside it.

The fourth (lower-right) leg of the period-5 window in the gratitude@4 / hostility@16 plane is the strip between a
continued fold curve and a continued flip curve; at fixed alpha2 it is 3e-5 wide near the body and 1.4e-6 near its tip,
so the follow run (tools/follow_attractor.py, grid 1.25e-4) never entered it, and no grid fine enough to follow it cell by
cell is affordable. Instead, at every point of the fold curve inside the alpha2 range:
- the centre cell lies midway between the fold and the flip curve at that alpha2 (the flip interpolated linearly);
- two control cells lie one leg width outside, beyond the fold and beyond the flip;
- each cell starts from the fold point's continued period-5 state (the curve's states.npy row), runs --transient steps,
  then --record steps with one twin for the exponent, exactly as tools/follow_attractor.py does;
- membership is judged by follow_attractor's rule against the body's reference word: label 1 (period a multiple of 5),
  3 (no period found, the text repeats exactly, exponent below --max-slow-exponent: slow convergence next to a fold or
  flip), 2 (chaos behind the repeating text) or 0 (not a member).
Expected: centre cells are members; beyond the fold the orbit is gone; beyond the flip the period doubles.

Output: DATA/explore/OUT/leg.json (settings, per cell: alphas, role, width, period, exponent, word, agreement, label)
and progress.jsonl.

Usage: python tools/leg_centerline.py --fold llm-soft-loop/pod64/fm5x/curve-a1-up-fold-plus \
         --flip llm-soft-loop/pod64/fm5x/curve-a2-down-flip-minus --alpha2 0.0890 0.0976 --log-temperature -0.1 \
         --reference-word 56568,13,358,13,13 --device cuda --out llm-soft-loop/leg/lower-right-t-0.100
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from follow_attractor import best_match, dominant_word  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fold", required=True, help="explore-relative folder with curve.json and states.npy")
    ap.add_argument("--flip", required=True, help="explore-relative folder with curve.json")
    ap.add_argument("--alpha2", type=float, nargs=2, required=True)
    ap.add_argument("--scale", type=float, default=10.0, help="curve theta = scale * alpha")
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--block1", type=int, default=4)
    ap.add_argument("--vectors1", default="llm-phase-map/vectors-l4")
    ap.add_argument("--concept1", default="gratitude")
    ap.add_argument("--block2", type=int, default=16)
    ap.add_argument("--vectors2", default="llm-phase-map/vectors-l16")
    ap.add_argument("--concept2", default="hostility")
    ap.add_argument("--log-temperature", type=float, required=True)
    ap.add_argument("--reference-word", required=True, help="comma-separated token ids of the body's reference word")
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--transient", type=int, default=250)
    ap.add_argument("--record", type=int, default=300)
    ap.add_argument("--token-steps", type=int, default=200)
    ap.add_argument("--max-slow-exponent", type=float, default=0.005)
    ap.add_argument("--max-period", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--min-match", type=int, default=4)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ring = 2 * args.max_period + 1
    if args.record < max(ring, args.token_steps):
        raise SystemExit("--record must cover 2 * max_period + 1 steps and --token-steps")

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "progress.jsonl", "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    fold = json.loads((esl.DATA / "explore" / args.fold / "curve.json").read_text())
    states = np.load(esl.DATA / "explore" / args.fold / "states.npy")
    flip = json.loads((esl.DATA / "explore" / args.flip / "curve.json").read_text())
    if len(states) != len(fold["points"]):
        raise SystemExit(f"{len(states)} states for {len(fold['points'])} fold points")
    fl = np.array([[q["theta"][0] / args.scale, q["theta"][1] / args.scale] for q in flip["points"]])
    fl = fl[(fl[:, 1] >= args.alpha2[0] - 1e-3) & (fl[:, 1] <= args.alpha2[1] + 1e-3)]
    order = np.argsort(fl[:, 1])
    if np.any(np.diff(fl[order, 1]) <= 0):
        raise SystemExit("the flip curve is not single-valued in alpha2 over this range")
    cells = []                                           # (alpha1, alpha2, role, width, state row)
    for k, q in enumerate(fold["points"]):
        x, y = q["theta"][0] / args.scale, q["theta"][1] / args.scale
        if not (args.alpha2[0] <= y <= args.alpha2[1]):
            continue
        xf = float(np.interp(y, fl[order, 1], fl[order, 0]))
        width = x - xf
        for role, a1c in (("centre", (x + xf) / 2), ("beyond fold", x + abs(width)), ("beyond flip", xf - abs(width))):
            cells.append((a1c, y, role, width, k))
    if not cells:
        raise SystemExit("no fold points in the alpha2 range")

    tic = time.perf_counter()
    loop = esl.SoftLoop(args.model, args.block1, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    loop.sm.add_steer_layer(args.block2)
    torch, dev = loop.torch, loop.device
    u = np.asarray(elpm.load_vectors(args.vectors1)[args.concept1], dtype=np.float64)
    w = np.asarray(elpm.load_vectors(args.vectors2)[args.concept2], dtype=np.float64)
    T = 10.0 ** args.log_temperature
    ref = np.array([int(t) for t in args.reference_word.split(",")], dtype=np.int64)
    rng = np.random.default_rng(0)
    emit({"cells": len(cells), "anchors": len(cells) // 3, "alpha2": args.alpha2, "reference_word": ref.tolist()})

    def run(batch):
        """As tools/follow_attractor.py run(), from the fold states of the batch's anchors."""
        n = len(batch)
        steer = np.array([c[0] * u for c in batch])
        loop.extra = (args.block2, np.array([c[1] * w for c in batch]))
        X0 = loop.prepare(steer).clone()                 # one row per cell, as follow_attractor's seed
        X0[:, -1, :] = torch.as_tensor(states[[c[4] for c in batch]], dtype=X0.dtype, device=X0.device)
        loop.prepare(np.concatenate([steer, steer]))     # steering of the base and twin rows, as follow_attractor's run
        temps = np.full(2 * n, T)
        with torch.no_grad():
            X = torch.cat([X0, X0])
            for _ in range(args.transient):
                X, _ = loop.step(X, temps)
            base = X[:n]
            d = torch.as_tensor(rng.normal(size=tuple(base.shape)), dtype=loop.state_dtype, device=dev)
            X = torch.cat([base, base + d / d.flatten(1).norm(dim=1)[:, None, None] * args.eps])
            growth = np.zeros(n)
            buf = torch.zeros((ring, n, X.shape[2]), dtype=loop.state_dtype, device=dev)
            toks = np.zeros((args.record, n), dtype=np.int64)
            for t in range(args.record):
                X, p = loop.step(X, temps)
                base, tw = X[:n], X[n:]
                diff = tw - base
                dist = np.maximum(diff.flatten(1).norm(dim=1).double().cpu().numpy(), args.eps * 1e-30)
                growth += np.log(dist / args.eps)
                tw = base + diff * torch.as_tensor(args.eps / dist, dtype=loop.state_dtype, device=dev)[:, None, None]
                X = torch.cat([base, tw])
                buf[t % ring] = base[:, -1, :]
                toks[t] = p[:n].argmax(-1).cpu().numpy()
            t = args.record - 1
            cur = buf[t % ring]
            scale = torch.clamp(cur.norm(dim=1), min=1.0) * args.tol
            found = torch.zeros(n, dtype=torch.long, device=dev)
            for q in range(args.max_period, 0, -1):          # descending, so the smallest period wins
                ok = ((buf[(t - q) % ring] - cur).norm(dim=1) < scale) & ((buf[(t - 2 * q) % ring] - cur).norm(dim=1)
                                                                           < scale)
                found = torch.where(ok, torch.full_like(found, q), found)
        loop.extra = None
        return growth / args.record, found.cpu().numpy(), toks

    rows = []
    for s in range(0, len(cells), args.max_batch):
        batch = cells[s:s + args.max_batch]
        lam, per, toks = run(batch)
        wd, ag = dominant_word(toks[-args.token_steps:], args.period)
        for k, c in enumerate(batch):
            exact = bool(np.all(ag[k] >= 1.0))
            if best_match(wd[k], ref) < args.min_match:
                label = 0
            elif per[k] > 0 and per[k] % args.period == 0:
                label = 1
            elif per[k] == 0 and exact and lam[k] < args.max_slow_exponent:
                label = 3
            elif per[k] == 0 and exact:
                label = 2
            else:
                label = 0
            rows.append({"alpha1": c[0], "alpha2": c[1], "role": c[2], "width": c[3], "fold_point": c[4],
                         "period": int(per[k]), "exponent": float(lam[k]), "word": wd[k].tolist(),
                         "agreement": ag[k].tolist(), "match": best_match(wd[k], ref), "label": label})
        emit({"done": len(rows), "of": len(cells), "seconds": round(time.perf_counter() - tic, 1)})
    summary = {}
    for role in ("centre", "beyond fold", "beyond flip"):
        rr = [r for r in rows if r["role"] == role]
        summary[role] = {"cells": len(rr), "members": sum(r["label"] in (1, 2, 3) for r in rr),
                         "labels": {str(v): sum(r["label"] == v for r in rr) for v in (0, 1, 2, 3)},
                         "periods": {str(p): sum(r["period"] == p for r in rr) for p in sorted({r["period"] for r in rr})}}
    rec = {"args": vars(args), "summary": summary, "cells": rows, "seconds": round(time.perf_counter() - tic, 1)}
    (out / "leg.json").write_text(json.dumps(rec, indent=1))
    emit({"summary": summary})


if __name__ == "__main__":
    main()
