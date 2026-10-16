"""Run E2E, Q3a: how close to rank one are the readout covariance Sigma_p and the step derivative DF along orbits.

spectra  at each listed cell, iterate to the attractor (from the fixed start, a token embedding or a saved state),
         then at n consecutive states: the leading eigenvalues of Sigma_p = Cov_{t~p}(e_t) (with its trace and the
         token pair whose embedding difference is best aligned with the leading eigenvector), the leading singular
         values of the dense DF (forward-mode JVPs, float64), and for the composed map over the n steps (a period,
         or a stretch of a chaotic orbit) its singular values and eigenvalues. DF = (1/T) Sigma_p J_h is checked at
         the first state against the dense AD Jacobian (relative error of the product on random directions).

Usage: python tools/e2e_rank.py spectra --cell NAME ALPHA LOG10T N [--seed-token is | --seed-state PATH] ... --out rank
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e2e_model as em  # noqa: E402
import e2e_numerics as en  # noqa: E402


def dense_DF(cell, x, chunk=128):
    """Dense one-step Jacobian at x by forward-mode JVPs (columns e_j)."""
    d = cell.d
    cols = []
    for s in range(0, d, chunk):
        V = np.zeros((d, min(chunk, d - s)))
        V[np.arange(s, s + V.shape[1]), np.arange(V.shape[1])] = 1.0
        cols.append(cell.apply_M_ad([x], V))
    return np.hstack(cols)


def best_pair(E, u, p, top=8):
    """Among the `top` most probable tokens, the pair (a, b) whose e_a - e_b is best aligned with u."""
    idx = np.argsort(-p)[:top]
    best = (0.0, None, None)
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            c = en.pair_alignment(u, E[idx[i]], E[idx[j]])
            if c > best[0]:
                best = (c, int(idx[i]), int(idx[j]))
    return best


def spectra(args):
    model = em.Model(args.threads)
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    E = model.E
    results = []
    seeds = dict(s.split("=", 1) for s in args.seed)
    for name, alpha, log_t, n in args.cell:
        alpha, log_t, n = float(alpha), float(log_t), int(n)
        tic = time.perf_counter()
        cell = model.cell(alpha, log_t)
        seed = seeds.get(name)
        if seed is None:
            x = cell.start.copy()
        elif seed in em.TOKENS:
            x = E[em.TOKENS[seed]].copy()
        else:
            x = np.load(seed).reshape(-1)
        x = cell.Fp(x, args.transient)
        pts = cell.orbit_points(x, n + 1)
        closure = float(np.linalg.norm(pts[-1] - pts[0]) / np.linalg.norm(pts[0]))
        P = cell.probs(np.array(pts[:n]))
        steps, Js = [], []
        for i in range(n):
            p = P[i]
            w, U, tr, ns = en.covariance_spectrum(E, p, tol=1e-12, k=6)
            c, a, b = best_pair(E, U[:, 0], p)
            J = dense_DF(cell, pts[i])
            Js.append(J)
            s = np.linalg.svd(J, compute_uv=False)
            top = np.argsort(-p)[:4]
            steps.append({"top": [[model.decode(t), float(p[t])] for t in top],
                          "entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum()),
                          "sigma_eigs": [float(v) for v in w], "sigma_trace": tr, "sigma_support": ns,
                          "sigma_ratio_2_1": float(w[1] / w[0]), "sigma_lead_share_of_trace": float(w[0] / tr),
                          "sigma_lead_pair": [model.decode(a), model.decode(b)], "sigma_lead_pair_alignment": c,
                          "DF_singular": [float(v) for v in s[:6]], "DF_ratio_2_1": float(s[1] / s[0])})
            print(json.dumps({"cell": name, "step": i} | {k: steps[-1][k] for k in (
                "sigma_ratio_2_1", "sigma_lead_pair", "sigma_lead_pair_alignment", "DF_ratio_2_1")} |
                {"top": steps[-1]["top"][:2]}, ensure_ascii=False), flush=True)
        # identity check at the first state: DF u = (1/T) Sigma_p J_h u is not recomputed here (paper: 1.35e-13);
        # instead the dense AD Jacobian's product is compared with a fresh JVP along random directions
        rng = np.random.default_rng(0)
        R = rng.normal(size=(model.d, 4))
        jv = cell.apply_M_ad([pts[0]], R)
        consistency = float(np.abs(Js[0] @ R - jv).max() / np.abs(jv).max())
        M = np.eye(model.d)
        for J in Js:
            M = J @ M
        sv = np.linalg.svd(M, compute_uv=False)
        ev = np.linalg.eigvals(M)
        ev = ev[np.argsort(-np.abs(ev))]
        rec = {"cell": name, "alpha": alpha, "log10_temperature": log_t, "states": n, "closure": closure,
               "dense_vs_jvp": consistency, "steps": steps,
               "composed_singular": [float(v) for v in sv[:6]], "composed_ratio_2_1": float(sv[1] / sv[0]),
               "composed_eigs": [[float(z.real), float(z.imag)] for z in ev[:6]],
               "seconds": round(time.perf_counter() - tic, 1)}
        results.append(rec)
        np.savez_compressed(out / f"{name}.npz", states=np.array(pts), jacobians=np.array(Js))
        print(json.dumps({k: rec[k] for k in ("cell", "closure", "dense_vs_jvp", "composed_ratio_2_1", "seconds")}),
              flush=True)
        (out / "spectra.json").write_text(json.dumps({"args": vars(args), "cells": results}, indent=1,
                                                     ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spectra")
    s.add_argument("--cell", nargs=4, action="append", required=True, metavar=("NAME", "ALPHA", "LOG10T", "N"))
    s.add_argument("--seed", nargs="*", default=[], help="NAME=token or NAME=path.npy (default: the fixed start)")
    s.add_argument("--transient", type=int, default=1000)
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"spectra": spectra}[args.cmd](args)


if __name__ == "__main__":
    main()
