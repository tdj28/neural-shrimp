"""Basins of coexisting attractors of the soft-token map on a plane of starting states (human, 18 Oct 2026: "image
them precisely").

At one cell (alpha, log10 T), the start's soft-token state x0 (the last slot of the fixed start) is displaced to
x0 + |x0| (s u + t v) for (s, t) on an N x N grid of [-R, R]^2. The plane is spanned by two orthonormal directions:
by default u points from x0 towards a point of the first periodic attractor that the census finds, and v is the
component towards the second; otherwise two seeded random directions. Every grid start is iterated --transient steps
(float64 by default), and its final state is classified by full-state recurrence (period up to --max-period, tolerance --tol)
and then by orbit: the attractor whose orbit passes closest to it (same rule as tools/attractor_portrait.py). Starts
that settle on no periodic orbit within the transient are marked 0.

Output: DATA/explore/OUT/basin.npz (grid, labels, periods, steps-to-settle proxy) and basin.json (cell, plane,
attractor list with token loops, label counts).

Usage: python tools/basin_slice.py --alpha 0.06468 --log-t -0.148 --grid 128 --radius 0.08 --device cuda \
         --out llm-soft-loop/pod64/portraits/pair-basin
"""
import argparse
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


def settle(loop, X, temps, transient, max_period, tol, torch):
    """Iterate, then return the period of each final state (0: none up to max_period) and the final orbit states."""
    with torch.no_grad():
        for _ in range(transient):
            X, _ = loop.step(X, temps)
        Y = [X[:, -1].double().clone()]
        toks = []
        Z = X
        for _ in range(max_period):
            Z, p = loop.step(Z, temps)
            Y.append(Z[:, -1].double().clone())
            toks.append(p.argmax(-1).cpu().numpy())
        Y = torch.stack(Y)
        per = np.zeros(X.shape[0], dtype=int)
        for q in range(1, max_period + 1):
            dq = ((Y[q] - Y[0]).norm(dim=1) / Y[0].norm(dim=1)).cpu().numpy()
            per[(per == 0) & (dq < tol)] = q
    return per, Y.cpu().numpy(), np.stack(toks, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--log-t", type=float, required=True)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--radius", type=float, default=0.08, help="half-width of the plane, relative to |x0|")
    ap.add_argument("--plane", default="attractors", choices=("attractors", "random"))
    ap.add_argument("--census-starts", type=int, default=64)
    ap.add_argument("--transient", type=int, default=400)
    ap.add_argument("--max-period", type=int, default=12)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"),
                    help="float32 for imaging only: keep --tol above its error (1e-4)")
    ap.add_argument("--match-tol", type=float, default=None,
                    help="relative distance for 'same orbit' (default 1e-6 in float64, 1e-3 in float32)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    match = args.match_tol if args.match_tol is not None else (1e-6 if args.dtype == "float64" else 1e-3)
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, args.dtype, args.device)
    torch = loop.torch
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    steer = args.alpha * v
    T = 10.0 ** args.log_t

    def starts(n):
        return loop.prepare(np.repeat(steer[None], n, axis=0)).clone()

    X0 = starts(1)
    x0 = X0[0, -1].double().cpu().numpy()
    r0 = float(np.linalg.norm(x0))
    # census: which periodic attractors do random nearby starts reach?
    rng = np.random.default_rng(0)
    Xc = starts(args.census_starts)
    U = rng.normal(size=(args.census_starts, x0.size))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    Xc[:, -1] = Xc[:, -1] + torch.as_tensor(args.radius * r0 * U, dtype=Xc.dtype, device=Xc.device)
    temps = torch.full((args.census_starts,), T, dtype=loop.state_dtype, device=loop.device)
    per, Y, toks = settle(loop, Xc, temps, args.transient, args.max_period, args.tol, torch)
    attractors = []                                     # (period, orbit states (q, d), token loop)
    for i in np.nonzero(per > 0)[0]:
        q = int(per[i])
        orb = Y[:q, i]
        if not any(a[0] == q and np.min(np.linalg.norm(a[1][:, None] - orb[None], axis=-1)) < match * r0 for a in attractors):
            attractors.append((q, orb, toks[i, :q].tolist()))
    attractors.sort(key=lambda a: a[0])
    if args.plane == "attractors":
        if len(attractors) < 2:
            raise SystemExit(f"the census found {len(attractors)} periodic attractor(s); --plane attractors needs two")
        a = attractors[0][1][np.argmin(np.linalg.norm(attractors[0][1] - x0, axis=1))] - x0
        b = attractors[1][1][np.argmin(np.linalg.norm(attractors[1][1] - x0, axis=1))] - x0
        u = a / np.linalg.norm(a)
        w = b - (b @ u) * u
        if np.linalg.norm(w) < 1e-12 * np.linalg.norm(b):
            raise SystemExit("the two attractor directions are parallel")
        w /= np.linalg.norm(w)
    else:
        u, w = np.linalg.qr(rng.normal(size=(x0.size, 2)))[0].T
    s = np.linspace(-args.radius, args.radius, args.grid)
    S, Tt = np.meshgrid(s, s)                           # rows: t (second direction), columns: s (first)
    D = (S.reshape(-1, 1) * u[None] + Tt.reshape(-1, 1) * w[None]) * r0
    labels = np.zeros(D.shape[0], dtype=int)
    periods = np.zeros(D.shape[0], dtype=int)
    for b0 in range(0, len(D), args.batch):
        n = min(args.batch, len(D) - b0)
        X = starts(n)
        X[:, -1] = X[:, -1] + torch.as_tensor(D[b0:b0 + n], dtype=X.dtype, device=X.device)
        temps = torch.full((n,), T, dtype=loop.state_dtype, device=loop.device)
        per, Y, _ = settle(loop, X, temps, args.transient, args.max_period, args.tol, torch)
        periods[b0:b0 + n] = per
        for k in range(n):
            if per[k] == 0:
                continue
            q = per[k]
            dist = [np.min(np.linalg.norm(a[1] - Y[0, k], axis=1)) if a[0] == q else np.inf for a in attractors]
            j = int(np.argmin(dist))
            labels[b0 + k] = j + 1 if dist[j] < match * r0 else -1        # -1: periodic, but not a census attractor
        print(json.dumps({"done": b0 + n, "of": len(D), "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
    counts = {str(k): int((labels == k).sum()) for k in np.unique(labels)}
    np.savez_compressed(out / "basin.npz", s=s, labels=labels.reshape(S.shape), periods=periods.reshape(S.shape),
                        u=u, w=w, x0=x0)
    rec = {"args": vars(args), "x0_norm": r0,
           "attractors": [{"label": k + 1, "period": a[0], "token_ids": a[2]} for k, a in enumerate(attractors)],
           "label_counts": counts, "plane_angle_deg": None, "seconds": round(time.perf_counter() - tic, 1)}
    if args.plane == "attractors":
        rec["plane_angle_deg"] = float(np.degrees(np.arccos(np.clip((a @ b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
    (out / "basin.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("attractors", "label_counts", "seconds")}))


if __name__ == "__main__":
    main()
