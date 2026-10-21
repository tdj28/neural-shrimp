"""Batch layout and the attractor a start reaches (run BF, exploratory, 20 Oct 2026).

At one cell (alpha, log10 T) of a steered plane, the same starts are iterated in several batch layouts: each start
alone (batch 1), in batches of a middle size, and all together. The starts follow the recipe of attractor_portrait.py,
the fixed start displaced by --spread |x| along random directions, but the directions are drawn once with numpy
(seed --seed) and reused in every layout, so the layouts differ only in how the rows are grouped. Grouping changes
the matrix shapes the kernels see and so the float64 rounding, never the map.

Every step, each trajectory's state is compared with its last --max-period states (relative distance below --tol).
The first step at which a period q is found is the start's capture step; the q states and the most probable tokens
of that cycle are kept. A start alone stops when it is captured; a batch runs until every row is captured or the
horizon ends, so its layout never changes during the run. Every --record-every steps all states are recorded, for
comparing the layouts' trajectories.

Output: DATA/explore/OUT/fate.npz (directions, start states, and per layout the capture steps, periods, cycle
tokens, cycle states, final states, recorded states) and fate.json (cell, settings, per-layout census and timings,
SHA-256 of the start states). Chunks are saved as they finish and reused on restart.

Usage: python tools/batch_fate.py --alpha 0.07248 --log-t -0.148 --starts 64 --steps 5000 --layouts 1,32,64 \
           --device cuda --out llm-soft-loop/batch-fate/bf1
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402
import explore_llm_soft_loop as esl  # noqa: E402


def start_states(loop, steer, n, spread, seed):
    """The fixed start displaced by spread |x| along n seeded random unit directions (numpy, float64)."""
    X = loop.prepare(steer[None])
    x = X[0, -1].double().cpu().numpy()
    U = np.random.default_rng(seed).standard_normal((n, x.size))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    return U, x[None] + spread * np.linalg.norm(x) * U


def run_chunk(loop, steer, T, x0, steps, max_period, tol, record_every, stop_alone=True):
    """Iterate the rows of x0 as one batch; returns per-row capture step (-1: none), period, cycle tokens and states,
    final states, the smallest relative recurrence distance at the end, and the recorded states."""
    torch = loop.torch
    B, d = x0.shape
    X = loop.prepare(np.repeat(steer[None], B, axis=0))
    X = X.clone()
    X[:, -1] = torch.as_tensor(x0, dtype=X.dtype, device=X.device)
    temps = torch.full((B,), T, dtype=loop.state_dtype, device=loop.device)
    Q = max_period
    R = torch.empty((Q, B, d), dtype=torch.float64, device=loop.device)     # ring of the last Q states
    Rt = torch.empty((Q, B), dtype=torch.long, device=loop.device)          # their most probable tokens
    cap = np.full(B, -1, dtype=np.int64)
    per = np.zeros(B, dtype=np.int64)
    cyc_tok, cyc_state = {}, {}
    rec = np.full((steps // record_every + 1, B, d), np.nan)
    rec[0] = x0
    mind = np.full(B, np.nan)
    t = 0
    with torch.no_grad():
        for t in range(1, steps + 1):
            X, p = loop.step(X, temps)
            x = X[:, -1].double()
            tok = p.argmax(-1)
            k = min(Q, t - 1)
            if k > 0:
                idx = torch.as_tensor([(t - q) % Q for q in range(1, k + 1)], device=loop.device)
                dist = (R[idx] - x[None]).norm(dim=2) / x.norm(dim=1)[None]    # (k, B), row q-1: x_t vs x_{t-q}
                hit = (dist < tol).cpu().numpy()
                new = np.nonzero(hit.any(0) & (cap < 0))[0]
                for r in new:
                    q = int(np.argmax(hit[:, r])) + 1
                    cap[r], per[r] = t, q
                    slots = [(t - j) % Q for j in range(q - 1, 0, -1)]       # x_{t-q+1} .. x_{t-1}, then x_t
                    prev = R[torch.as_tensor(slots, dtype=torch.long, device=loop.device), r] if slots else x[r][:0]
                    prevt = Rt[torch.as_tensor(slots, dtype=torch.long, device=loop.device), r] if slots else tok[:0]
                    cyc_state[int(r)] = np.concatenate([prev.reshape(-1, d).cpu().numpy(), x[r][None].cpu().numpy()])
                    cyc_tok[int(r)] = [int(v) for v in prevt.cpu().tolist()] + [int(tok[r])]
                if t == steps:
                    mind = dist.min(0).values.cpu().numpy()
            R[t % Q] = x
            Rt[t % Q] = tok
            if t % record_every == 0:
                rec[t // record_every] = x.cpu().numpy()
            if (B == 1 and stop_alone and cap[0] >= 0) or (cap >= 0).all():
                break
    final = X[:, -1].double().cpu().numpy()
    return cap, per, cyc_tok, cyc_state, final, mind, rec, t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--log-t", type=float, required=True)
    ap.add_argument("--starts", type=int, default=64)
    ap.add_argument("--spread", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--layouts", default="1,32,64", help="batch sizes, comma-separated; a repeated size is run again "
                    "as a determinism check (labelled with a suffix)")
    ap.add_argument("--max-period", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--record-every", type=int, default=100)
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = esl.DATA / "explore" / args.out
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, args.dtype, args.device)
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    steer = args.alpha * v
    T = 10.0 ** args.log_t
    U, x0 = start_states(loop, steer, args.starts, args.spread, args.seed)
    x0_sha = hashlib.sha256(np.ascontiguousarray(x0).tobytes()).hexdigest()
    sizes = [int(s) for s in args.layouts.split(",")]
    labels, seen = [], {}
    for s in sizes:
        seen[s] = seen.get(s, 0) + 1
        labels.append(f"b{s}" if seen[s] == 1 else f"b{s}-run{seen[s]}")
    results, summary = {}, {}
    for s, lab in zip(sizes, labels):
        if args.starts % s:
            raise SystemExit(f"--starts {args.starts} is not a multiple of batch size {s}")
        t0 = time.perf_counter()
        parts = []
        for c0 in range(0, args.starts, s):
            path = out / "chunks" / f"{lab}-{c0:05d}.npz"
            if path.exists():
                z = np.load(path, allow_pickle=True)
                parts.append({k: z[k] for k in z.files})
                continue
            cap, per, ctok, cst, final, mind, rec, t_end = run_chunk(loop, steer, T, x0[c0:c0 + s], args.steps,
                                                                    args.max_period, args.tol, args.record_every)
            part = {"cap": cap, "per": per, "final": final, "mind": mind, "rec": rec, "t_end": np.int64(t_end),
                    "cyc_tok": np.array([json.dumps({str(k + c0): v for k, v in ctok.items()})]),
                    "cyc_keys": np.array(sorted(k + c0 for k in cst), dtype=np.int64)}
            for k, a in cst.items():
                part[f"cyc_state_{k + c0}"] = a
            np.savez_compressed(path, **part)
            parts.append(part)
            print(json.dumps({"layout": lab, "done": c0 + s, "of": args.starts, "captured": int((cap >= 0).sum()),
                              "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
        cap = np.concatenate([p["cap"] for p in parts])
        per = np.concatenate([p["per"] for p in parts])
        cyc_tok = {}
        for p in parts:
            cyc_tok.update(json.loads(str(p["cyc_tok"][0])))
        cyc_state = {}
        for p in parts:
            for k in p["cyc_keys"]:
                cyc_state[int(k)] = p[f"cyc_state_{int(k)}"]
        results[lab] = {"cap": cap, "per": per, "cyc_tok": cyc_tok, "cyc_state": cyc_state,
                        "final": np.concatenate([p["final"] for p in parts]),
                        "mind": np.concatenate([p["mind"] for p in parts]),
                        "rec": np.concatenate([p["rec"] for p in parts], axis=1)}
        caught = cap >= 0
        summary[lab] = {"batch": s, "captured": int(caught.sum()), "not_captured": int((~caught).sum()),
                        "periods": {str(q): int((per == q).sum()) for q in sorted(set(per[caught].tolist()))},
                        "capture_step_quartiles": (np.percentile(cap[caught], [0, 25, 50, 75, 100]).tolist()
                                                   if caught.any() else None),
                        "seconds": round(time.perf_counter() - t0, 1)}
    save = {"directions": U, "starts": x0}
    for lab, r in results.items():
        save[f"{lab}_cap"] = r["cap"]
        save[f"{lab}_per"] = r["per"]
        save[f"{lab}_final"] = r["final"]
        save[f"{lab}_mind"] = r["mind"]
        save[f"{lab}_rec"] = r["rec"]
        save[f"{lab}_cyc_tok"] = np.array([json.dumps(r["cyc_tok"])])
        for k, a in r["cyc_state"].items():
            save[f"{lab}_cyc_state_{k}"] = a
    np.savez_compressed(out / "fate.npz", **save)
    rec = {"args": vars(args), "labels": labels, "starts_sha256": x0_sha, "summary": summary,
           "seconds": round(time.perf_counter() - tic, 1)}
    (out / "fate.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
