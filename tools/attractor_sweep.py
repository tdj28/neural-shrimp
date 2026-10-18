"""Attractors along a line of the steering-temperature plane, for a movie of the attractor changing shape (human, 18 Oct
2026: "a movie that scans from periodic points to chaotic points back to periodic points, showing the attractor shape
evolving").

Frames are --frames values of alpha from --alpha-from to --alpha-to at fixed log10 T. At each frame, --starts
trajectories start from the fixed start displaced by --spread |x| along seeded random directions (seed 1000 + frame), run
--transient steps, and are recorded for --record steps in float64. Frames are independent, so several share one batch
(one steering row per trajectory).
- Pass 1 pools the recorded states of every --basis-stride-th frame (--basis-starts trajectories each) and fixes one
  set of principal axes for the whole sweep, so that the shape changes are seen in fixed coordinates.
- Pass 2 records every frame's states on the leading --pcs axes (float32), the most probable next token, and the period
  of each final state by full-state recurrence (up to --max-period, tolerance --tol; 0: none).

Output: DATA/explore/OUT/sweep.npz (alpha, coords, tokens, periods, basis, mean, eigenvalues) and sweep.json.

Usage: python tools/attractor_sweep.py --alpha-from 0.0625 --alpha-to 0.0775 --log-t -0.148 --frames 480 \
         --device cuda --out llm-soft-loop/pod64/portraits/sweep
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


def batch_run(loop, v, alphas, frame_ids, T, spread, transient, record, torch, project=None, max_period=0, tol=0.0):
    """T: one temperature per row."""
    """Trajectories for the given (alpha, frame) rows. Returns the recorded states (record, n, d) when project is None,
    else coordinates (record, n, pcs), tokens (record, n) and periods (n,)."""
    n = len(alphas)
    X = loop.prepare(np.asarray(alphas)[:, None] * v[None])
    X = X.clone()
    U = np.empty((n, X.shape[-1]))
    for f in np.unique(frame_ids):               # the same displacements for a frame whichever batch it lands in
        rows = np.nonzero(frame_ids == f)[0]
        U[rows] = np.random.default_rng(1000 + int(f)).normal(size=(len(rows), X.shape[-1]))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    Ut = torch.as_tensor(U, dtype=X.dtype, device=X.device)
    X[:, -1] = X[:, -1] + spread * X[:, -1].norm(dim=1, keepdim=True) * Ut
    temps = torch.as_tensor(np.asarray(T, dtype=np.float64), dtype=loop.state_dtype, device=loop.device)
    out, toks = [], []
    with torch.no_grad():
        for _ in range(transient):
            X, _ = loop.step(X, temps)
        for _ in range(record):
            X, p = loop.step(X, temps)
            x = X[:, -1].double()
            if project is None:
                out.append(x)
            else:
                mean, basis = project
                out.append(((x - mean) @ basis).float().cpu().numpy())
                toks.append(p.argmax(-1).cpu().numpy().astype(np.int32))
        if project is None:
            return torch.stack(out)
        Y0 = X[:, -1].double().clone()
        per = np.zeros(n, dtype=int)
        Z = X
        for q in range(1, max_period + 1):
            Z, _ = loop.step(Z, temps)
            dq = ((Z[:, -1].double() - Y0).norm(dim=1) / Y0.norm(dim=1)).cpu().numpy()
            per[(per == 0) & (dq < tol)] = q
    return np.stack(out), np.stack(toks), per


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--alpha-from", type=float, default=None)
    ap.add_argument("--alpha-to", type=float, default=None)
    ap.add_argument("--log-t", type=float, default=None)
    ap.add_argument("--path", default=None, help="'a1,t1/a2,t2/...' (or ';'): a polyline in (alpha, log10 T); '/' is safe "
                    "in the remote shell of the pod launcher, ';' is not. Frames are spread "
                    "along it in proportion to length, alpha in units of 0.015 and log10 T in units of 0.045 (the cross-512 "
                    "map's spans); replaces --alpha-from/--alpha-to/--log-t")
    ap.add_argument("--basis-from", default=None, help="explore-relative .npz with mean and basis: use these axes (no pass 1)")
    ap.add_argument("--frames", type=int, default=480)
    ap.add_argument("--starts", type=int, default=32)
    ap.add_argument("--spread", type=float, default=0.05)
    ap.add_argument("--transient", type=int, default=300)
    ap.add_argument("--record", type=int, default=200)
    ap.add_argument("--pcs", type=int, default=6)
    ap.add_argument("--basis-stride", type=int, default=8)
    ap.add_argument("--basis-starts", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-period", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
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
    if args.path:
        pts = np.array([[float(x) for x in q.split(",")] for q in args.path.replace(";", "/").split("/")])
        if len(pts) < 2:
            raise SystemExit(f"--path needs at least two points, got {args.path!r}")
        seg = np.hypot(np.diff(pts[:, 0]) / 0.015, np.diff(pts[:, 1]) / 0.045)
        u = np.concatenate([[0.0], np.cumsum(seg)]) / seg.sum()
        f = np.linspace(0.0, 1.0, args.frames)
        alphas, logts = np.interp(f, u, pts[:, 0]), np.interp(f, u, pts[:, 1])
    else:
        if None in (args.alpha_from, args.alpha_to, args.log_t):
            raise SystemExit("give --path, or --alpha-from, --alpha-to and --log-t")
        alphas = np.linspace(args.alpha_from, args.alpha_to, args.frames)
        logts = np.full(args.frames, args.log_t)
    temps_all = 10.0 ** logts
    # pass 1: one basis for the whole sweep
    bf = np.arange(0, args.frames, args.basis_stride)
    rows_a = np.repeat(alphas[bf], args.basis_starts)
    rows_f = np.repeat(bf, args.basis_starts)
    d = None
    for b0 in range(0, 0 if args.basis_from else len(rows_a), args.batch):
        S = batch_run(loop, v, rows_a[b0:b0 + args.batch], rows_f[b0:b0 + args.batch], temps_all[rows_f[b0:b0 + args.batch]],
                      args.spread, args.transient, args.record, torch)
        S = S.reshape(-1, S.shape[-1])
        if d is None:
            d = S.shape[1]
            s1 = torch.zeros(d, dtype=torch.float64, device=S.device)
            s2 = torch.zeros((d, d), dtype=torch.float64, device=S.device)
            cnt = 0
        s1 += S.sum(0)
        s2 += S.T @ S
        cnt += S.shape[0]
    if args.basis_from:
        zb = np.load(esl.DATA / "explore" / args.basis_from)
        mean = torch.as_tensor(zb["mean"], dtype=torch.float64, device=loop.device)
        V = np.asarray(zb["basis"])
        w = np.asarray(zb["eigenvalues"]) if "eigenvalues" in zb.files else np.ones(V.shape[1])
        cnt = 0
        if V.shape[1] < args.pcs:
            raise SystemExit(f"--basis-from has {V.shape[1]} axes, fewer than --pcs {args.pcs}")
    else:
        mean = s1 / cnt
        C = (s2 / cnt - torch.outer(mean, mean)).cpu().numpy()
        w, V = np.linalg.eigh(C)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
    meant = mean
    basist = torch.as_tensor(V[:, :args.pcs].copy(), dtype=torch.float64, device=mean.device)
    print(json.dumps({"basis": "done", "states": int(cnt), "variance_top": (w[:6] / w.sum()).tolist(),
                      "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
    # pass 2: every frame, several frames per batch
    coords = np.zeros((args.frames, args.starts, args.record, args.pcs), dtype=np.float32)
    tokens = np.zeros((args.frames, args.starts, args.record), dtype=np.int32)
    periods = np.zeros((args.frames, args.starts), dtype=np.int32)
    fpb = max(1, args.batch // args.starts)
    for f0 in range(0, args.frames, fpb):
        fr = np.arange(f0, min(f0 + fpb, args.frames))
        ra, rf = np.repeat(alphas[fr], args.starts), np.repeat(fr, args.starts)
        c, t, per = batch_run(loop, v, ra, rf, temps_all[rf], args.spread, args.transient, args.record, torch,
                              (meant, basist), args.max_period, args.tol)
        for k, f in enumerate(fr):
            sl = slice(k * args.starts, (k + 1) * args.starts)
            coords[f] = np.transpose(c[:, sl], (1, 0, 2))
            tokens[f] = t[:, sl].T
            periods[f] = per[sl]
        print(json.dumps({"frames_done": int(fr[-1] + 1), "of": args.frames,
                          "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
    np.savez_compressed(out / "sweep.npz", alpha=alphas, log_t=logts, coords=coords, tokens=tokens,
                        periods=periods, basis=V[:, :args.pcs], mean=mean.cpu().numpy(), eigenvalues=w[:64])
    rec = {"args": vars(args), "basis_states": int(cnt), "variance_explained_top": (w[:10] / w.sum()).tolist(),
           "path_alpha_range": [float(alphas.min()), float(alphas.max())], "path_log_t_range": [float(logts.min()), float(logts.max())],
           "aperiodic_share_by_frame": (periods == 0).mean(1).tolist(), "seconds": round(time.perf_counter() - tic, 1)}
    (out / "sweep.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("basis_states", "seconds")}))


if __name__ == "__main__":
    main()
