"""Unstable manifold of a saddle periodic orbit of the soft-token map: the skeleton of a strange attractor (human, 18 Oct
2026: "image them precisely").

1. The period-p orbit is found where it attracts (the fixed start at --from-alpha reaches it), then continued by Newton
   to the target cell in --march-steps steps of alpha (Newton follows the orbit through its flip, where it becomes a
   saddle). Each step is gated: relative residual below --tol, or the run stops.
2. At the target, the leading multipliers and Ritz vectors (llm_feigenbaum.ritz) give the unstable multiplier mu
   (real, |mu| > 1, the largest) and its direction v.
3. A fundamental domain of the local unstable manifold, x_k = x* +- s_k v with s_k log-spaced in [--delta, delta mu^2],
   is iterated --iterates times under G = F^(2p) (even, so that a negative mu keeps its orientation). The images cover
   the unstable manifold out to the size of the attractor. Its closure contains the attractor when the saddle lies on
   it.

The points are projected on the principal axes of a portrait run of the same cell (tools/attractor_portrait.py) for
overlays. Output: DATA/explore/OUT/manifold.npz (projections, branch, iterate, s) and manifold.json (orbit, multipliers,
gates).

Usage: python tools/unstable_manifold.py --alpha 0.06626 --log-t -0.148 --period 3 --from-alpha 0.0650 \
         --portrait llm-soft-loop/portraits/cascade --device cuda --out llm-soft-loop/portraits/cascade-wu
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
import llm_feigenbaum as lfb  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--log-t", type=float, required=True)
    ap.add_argument("--period", type=int, required=True)
    ap.add_argument("--from-alpha", type=float, required=True, help="where the fixed start reaches the attracting orbit")
    ap.add_argument("--march-steps", type=int, default=60)
    ap.add_argument("--transient", type=int, default=900)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-11)
    ap.add_argument("--delta", type=float, default=1e-7, help="inner edge of the fundamental domain (absolute)")
    ap.add_argument("--points", type=int, default=2048, help="points per side of the fundamental domain")
    ap.add_argument("--iterates", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--portrait", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    torch = loop.torch
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    T = 10.0 ** args.log_t
    P = args.period
    tol = lfb.tol_for(P, args.tol)
    cell = esl.SoftOrbit(loop, args.from_alpha * v, T, chunk=512)
    X = cell.iterate(cell.start, args.transient)
    per = cell.detect_period(X, 64, 1e-6)
    if per != P:
        raise SystemExit(f"the fixed start at --from-alpha reaches period {per}, not {P}")
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), args.k)))[0]
    march = []
    for k, a in enumerate(np.linspace(args.from_alpha, args.alpha, args.march_steps + 1)):
        cell = esl.SoftOrbit(loop, a * v, T, chunk=512)
        X, Q, hist = lfb.newton(cell, X, P, Q, tol, 12)
        march.append({"alpha": float(a), "residual": float(hist[-1])})
        if not hist[-1] < tol:
            raise SystemExit(f"Newton lost the period-{P} orbit at alpha {a}: residual {hist[-1]:.3g}")
    vals, vecs, res, _, Q = lfb.ritz(cell, X, P, Q)
    real = [(i, z) for i, z in enumerate(vals) if abs(z.imag) < 1e-9 and abs(z) > 1.0]
    if not real:
        raise SystemExit(f"no real multiplier outside the unit circle at the target: {vals[:4]}")
    i_u, mu = max(real, key=lambda t: abs(t[1]))
    mu = float(mu.real)
    vu = np.real(vecs[:, i_u])
    vu /= np.linalg.norm(vu)
    x0 = X.reshape(-1).double().cpu().numpy()                   # one point of the orbit (phase 0)
    s = args.delta * np.logspace(0, np.log10(mu * mu), args.points, endpoint=False)
    seeds = np.concatenate([x0[None] + s[:, None] * vu[None], x0[None] - s[:, None] * vu[None]])
    branch = np.concatenate([np.ones(args.points), -np.ones(args.points)])
    if args.portrait:
        z = np.load(esl.DATA / "explore" / args.portrait / "portrait.npz")
        mean, basis = z["mean"], z["basis"]
    else:
        mean, basis = x0, np.linalg.qr(np.stack([vu, np.random.default_rng(1).normal(size=x0.size)], 1))[0]
    proj, it_idx, s_idx = [], [], []
    for b0 in range(0, len(seeds), args.batch):
        Y = torch.as_tensor(seeds[b0:b0 + args.batch].reshape((-1,) + tuple(X.shape[1:])), dtype=X.dtype,
                            device=X.device)
        n = Y.shape[0]
        cellb = esl.SoftOrbit(loop, args.alpha * v, T, chunk=512)
        with torch.no_grad():
            for it in range(args.iterates + 1):
                y = Y.reshape(n, -1).double().cpu().numpy()
                proj.append((y - mean) @ basis)
                it_idx.append(np.full(n, it))
                s_idx.append(np.arange(b0, b0 + n))
                if it < args.iterates:
                    for _ in range(2 * P):
                        Y = cellb.f(Y)
        print(json.dumps({"done": b0 + n, "of": len(seeds), "seconds": round(time.perf_counter() - tic, 1)}),
              flush=True)
    np.savez_compressed(out / "manifold.npz", proj=np.concatenate(proj), iterate=np.concatenate(it_idx),
                        seed=np.concatenate(s_idx), branch=branch, s=np.concatenate([s, s]), orbit_point=x0,
                        direction=vu)
    rec = {"args": vars(args), "march": march[-3:], "march_steps": len(march), "orbit_residual": march[-1]["residual"],
           "multipliers": [[float(z.real), float(z.imag)] for z in vals[:args.k]], "ritz_residuals": [float(r) for r in res[:args.k]],
           "unstable_multiplier": mu, "seconds": round(time.perf_counter() - tic, 1)}
    (out / "manifold.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("orbit_residual", "unstable_multiplier", "seconds")}))


if __name__ == "__main__":
    main()
