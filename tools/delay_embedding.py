"""Takens delay reconstruction of the soft-token loop's attractor from one observable of its output (human, 18 Oct 2026:
"should we try Takens' theorem too?").

At one cell, --trajectories trajectories from the displaced fixed start run --transient steps and are then recorded for
--steps steps. Each step records scalars an outside observer could compute from the output distribution p alone: the
largest probability p_max, the entropy H(p), the logit gap between the two leading tokens, and the leading token, plus
the state's first principal coordinates (axes of a portrait run of the same cell, for comparison only).
Then, for the chosen observable s (default log p_max):
- the delay vectors (s_n, s_{n+1}, ..., s_{n+m-1}); by Takens' theorem (and Sauer, Yorke and Casdagli for fractal
  sets) m > 2 D suffices for a generic observable, D the attractor's box-counting dimension;
- the false-nearest-neighbour fraction against m (Kennel, Brown and Abarbanel 1992): a nearest neighbour in m delays is
  false when adding the (m+1)-th delay increases its distance by more than --rtol times;
- a continuity check: how far apart in the full state are points that are nearest neighbours in the m = 3
  reconstruction, against random pairs.

Output: DATA/explore/OUT/delay.npz (series, tokens, principal coordinates) and delay.json (settings, false-neighbour
fractions, continuity statistics).

Usage: python tools/delay_embedding.py --alpha 0.07248 --log-t -0.148 --portrait llm-soft-loop/pod64/portraits/f64-strong \
         --device cuda --dtype float32 --out llm-soft-loop/portraits/takens-strong
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


def fnn(X, m_max, rtol, rng, n=6000):
    from scipy.spatial import cKDTree
    out = []
    N = len(X) - m_max
    idx = rng.choice(N, size=min(n, N), replace=False)
    for m in range(1, m_max):
        E = np.stack([X[k:k + N] for k in range(m)], axis=1)
        tree = cKDTree(E)
        d, j = tree.query(E[idx], k=2)
        d1, j1 = d[:, 1], j[:, 1]
        extra = np.abs(X[idx + m] - X[j1 + m])
        ok = d1 > 0
        out.append({"m": m, "false_fraction": float(np.mean(extra[ok] / d1[ok] > rtol))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--log-t", type=float, required=True)
    ap.add_argument("--portrait", default=None, help="explore-relative portrait run whose axes are used for comparison")
    ap.add_argument("--trajectories", type=int, default=16)
    ap.add_argument("--transient", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--spread", type=float, default=0.05)
    ap.add_argument("--observable", default="log_pmax", choices=("log_pmax", "entropy", "gap"))
    ap.add_argument("--m-max", type=int, default=7)
    ap.add_argument("--rtol", type=float, default=10.0)
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, args.dtype, args.device)
    torch = loop.torch
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    n = args.trajectories
    X = loop.prepare(np.repeat((args.alpha * v)[None], n, axis=0)).clone()
    g = torch.Generator(device=X.device).manual_seed(7)
    U = torch.randn(X[:, -1].shape, generator=g, dtype=X.dtype, device=X.device)
    U = U / U.norm(dim=1, keepdim=True)
    X[:, -1] = X[:, -1] + args.spread * X[:, -1].norm(dim=1, keepdim=True) * U
    temps = torch.full((n,), 10.0 ** args.log_t, dtype=loop.state_dtype, device=loop.device)
    proj = None
    if args.portrait:
        zp = np.load(esl.DATA / "explore" / args.portrait / "portrait.npz", allow_pickle=True)
        proj = (torch.as_tensor(zp["mean"], dtype=torch.float64, device=X.device),
                torch.as_tensor(zp["basis"][:, :12].copy(), dtype=torch.float64, device=X.device))
    series = {k: np.zeros((n, args.steps)) for k in ("log_pmax", "entropy", "gap")}
    toks = np.zeros((n, args.steps), dtype=np.int32)
    pcs = np.zeros((n, args.steps, 12), dtype=np.float32) if proj else None
    full = []
    with torch.no_grad():
        for _ in range(args.transient):
            X, _ = loop.step(X, temps)
        for t in range(args.steps):
            X, p = loop.step(X, temps)
            pd = p.double()
            top2 = torch.topk(pd, 2, dim=-1)
            series["log_pmax"][:, t] = torch.log(top2.values[:, 0]).cpu().numpy()
            series["entropy"][:, t] = (-(pd * torch.log(pd.clamp_min(1e-300))).sum(-1)).cpu().numpy()
            series["gap"][:, t] = (torch.log(top2.values[:, 0]) - torch.log(top2.values[:, 1])).cpu().numpy()
            toks[:, t] = top2.indices[:, 0].cpu().numpy()
            x = X[:, -1].double()
            if proj:
                pcs[:, t] = ((x - proj[0]) @ proj[1]).float().cpu().numpy()
            if t < 4000:
                full.append(x[0].cpu().numpy().astype(np.float32))
    s = series[args.observable]
    rng = np.random.default_rng(0)
    fn = fnn(s[0], args.m_max, args.rtol, rng)
    # continuity: nearest neighbours in the m = 3 reconstruction of trajectory 0, distance in the full state
    from scipy.spatial import cKDTree
    F = np.stack(full)
    N = len(F) - 2
    E = np.stack([s[0][k:k + N] for k in range(3)], axis=1)
    d, j = cKDTree(E).query(E, k=2)
    nn = np.linalg.norm(F[:N] - F[j[:, 1]], axis=1)
    rnd = np.linalg.norm(F[:N] - F[rng.permutation(N)], axis=1)
    np.savez_compressed(out / "delay.npz", tokens=toks, pcs=pcs if proj else np.zeros(0), **series)
    rec = {"args": vars(args), "false_nearest_neighbours": fn,
           "continuity_m3": {"median_state_distance_of_delay_neighbours": float(np.median(nn)),
                             "median_state_distance_of_random_pairs": float(np.median(rnd)), "pairs": int(N)},
           "series_stats": {k: [float(series[k].min()), float(series[k].max())] for k in series},
           "seconds": round(time.perf_counter() - tic, 1)}
    (out / "delay.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("false_nearest_neighbours", "continuity_m3", "seconds")}))


if __name__ == "__main__":
    main()
