"""Analysis of run DELTA2D: the smoothly tracked leading two-dimensional block of the period-6 orbit's monodromy over
the top of the period-6 body (run card in docs/research-log.md, 20 Oct 2026).

Input: band directories of tools/doubly_superstable.py grid --store-ritz (grid.npz and ritz.npz each), stacked by
log10 T. At every solved cell the three leading Ritz pairs (lambda_k, v_k) are known. The block is tracked breadth first
from the cell whose two leading multipliers are best separated from the third: at each cell the candidate
two-dimensional invariant subspaces are the spans of two of the three Ritz vectors (a complex-conjugate pair counts as
one span, of its real and imaginary parts), and the one with the largest overlap (mean squared cosine of the principal
angles) with the parent's tracked span is taken. Overlaps below 0.9 mark cells where the pair mixes with the third
direction.

Reported: tau and delta of the tracked block on the grid; the sign changes of delta between neighbouring cells, and of
the real leading multiplier (the measured ditches); and every critical point of delta (tracked cells with overlap at
least 0.9) and of the real leading multiplier mu_1: cells whose least-squares quadratic fit over the 5 x 5 window
centered on them has its stationary point within 0.75 cells of the center, with the value there, the signs of the
Hessian's eigenvalues (a saddle has one of each) and the fit residual (DELTA2D amendment 1).

Usage: python tools/delta2d_analysis.py --bands llm-soft-loop/pod64/dss/delta2d-d0 ... --json docs/delta2d/analysis.json
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_llm_soft_loop import DATA  # noqa: E402


def load(bands):
    grids, ritz = [], []
    for b in bands:
        grids.append(np.load(DATA / "explore" / b / "grid.npz", allow_pickle=True))
        ritz.append(np.load(DATA / "explore" / b / "ritz.npz", allow_pickle=True))
    order = np.argsort([float(g["log10_temperature"][0]) for g in grids])
    grids, ritz = [grids[k] for k in order], [ritz[k] for k in order]
    al = grids[0]["alpha"]
    lt = np.concatenate([g["log10_temperature"] for g in grids])
    ok = np.concatenate([g["ok"] for g in grids])
    mu = np.concatenate([g["mu"] for g in grids])
    mi = np.concatenate([g["mu_imag"] for g in grids])
    m2 = np.concatenate([g["multipliers"] for g in grids])
    vec = np.concatenate([r["vectors"] for r in ritz])
    lam = m2[..., 0] + 1j * m2[..., 1]
    return al, lt, ok, mu, mi, lam, vec


def spans(lam, vec):
    """Candidate two-dimensional real spans at one cell: (eigenvalue indices, orthonormal basis n x 2)."""
    out = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        la, lb = lam[a], lam[b]
        if not (np.isfinite(la) and np.isfinite(lb)):
            continue
        cpx_a, cpx_b = abs(la.imag) > 1e-14, abs(lb.imag) > 1e-14
        if cpx_a or cpx_b:
            if not (cpx_a and cpx_b and abs(la - np.conj(lb)) < 1e-9 * max(1.0, abs(la))):
                continue                                  # a complex multiplier must stay with its conjugate
            v = vec[:, a]
            M = np.stack([v.real, v.imag], axis=1)
        else:
            va, vb = vec[:, a], vec[:, b]
            va = np.real(va * np.exp(-1j * np.angle(va[np.argmax(np.abs(va))])))
            vb = np.real(vb * np.exp(-1j * np.angle(vb[np.argmax(np.abs(vb))])))
            M = np.stack([va, vb], axis=1)
        Q = np.linalg.qr(M)[0]
        out.append(((a, b), Q))
    return out


def overlap(Q1, Q2):
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.mean(s ** 2))


def critical_points(field, valid, al, lt, reach=0.75):
    """Cells whose 5 x 5 quadratic fit (cell units) has its stationary point within `reach` cells of the center."""
    nt, na = field.shape
    ii, jj = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing="ij")
    A = np.stack([np.ones(25), ii.ravel(), jj.ravel(), ii.ravel() ** 2, ii.ravel() * jj.ravel(), jj.ravel() ** 2], axis=1)
    out = []
    for i in range(2, nt - 2):
        for j in range(2, na - 2):
            blk, ok = field[i - 2:i + 3, j - 2:j + 3], valid[i - 2:i + 3, j - 2:j + 3]
            if not (np.all(ok) and np.all(np.isfinite(blk))):
                continue
            coef = np.linalg.lstsq(A, blk.ravel(), rcond=None)[0]
            H = np.array([[2 * coef[3], coef[4]], [coef[4], 2 * coef[5]]])
            if abs(np.linalg.det(H)) == 0:
                continue
            x0 = np.linalg.solve(H, -coef[1:3])
            if np.max(np.abs(x0)) > reach:
                continue
            ev = np.linalg.eigvalsh(H)
            out.append({"cell": [int(i), int(j)], "alpha": float(al[j] + x0[1] * (al[1] - al[0])),
                        "log10T": float(lt[i] + x0[0] * (lt[1] - lt[0])),
                        "value": float(coef[0] + coef[1:3] @ x0 + 0.5 * x0 @ H @ x0),
                        "hessian_eigenvalues_per_cell2": ev.tolist(),
                        "type": "saddle" if ev[0] < 0 < ev[1] else ("minimum" if ev[0] > 0 else "maximum"),
                        "residual_rms": float(np.sqrt(np.mean((A @ coef - blk.ravel()) ** 2))),
                        "field_scale": float(np.ptp(blk)), "offset": float(np.max(np.abs(x0)))})
    kept = []                                    # one entry per critical point: the window centered closest to it
    for c in sorted(out, key=lambda c: c["offset"]):
        if all(abs(c["alpha"] - k["alpha"]) > abs(al[1] - al[0]) or abs(c["log10T"] - k["log10T"]) > abs(lt[1] - lt[0])
               for k in kept):
            kept.append(c)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bands", nargs="+", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    al, lt, ok, mu, mi, lam, vec = load(args.bands)
    nt, na = ok.shape
    good = ok & np.all(np.isfinite(lam), axis=-1)
    sep = np.where(good, np.abs(lam[..., 1]) / np.maximum(np.abs(lam[..., 2]), 1e-300), -np.inf)
    start = np.unravel_index(int(np.argmax(sep)), sep.shape)
    tau = np.full((nt, na), np.nan); delta = np.full((nt, na), np.nan); ov = np.full((nt, na), np.nan)
    basis, pair = {}, {}
    cand = spans(lam[start], vec[start])
    (a0, b0), Q0 = next(c for c in cand if c[0] == (0, 1))
    basis[start], pair[start] = Q0, (a0, b0)
    ov[start] = 1.0
    q = deque([start])
    while q:
        c = q.popleft()
        i, j = c
        la, lb = lam[c][pair[c][0]], lam[c][pair[c][1]]
        tau[c], delta[c] = float(np.real(la + lb)), float(np.real(la * lb))
        for d in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= d[0] < nt and 0 <= d[1] < na and good[d] and d not in basis:
                cs = spans(lam[d], vec[d])
                if not cs:
                    continue
                scores = [overlap(basis[c], Q) for _, Q in cs]
                k = int(np.argmax(scores))
                basis[d], pair[d], ov[d] = cs[k][1], cs[k][0], scores[k]
                q.append(d)
    # sign changes of delta and of the real leading multiplier between neighbours
    real_lead = good & (np.abs(np.nan_to_num(mi)) == 0)
    dz, mz = [], []
    for i in range(nt):
        for j in range(na):
            for k, l in ((i, j + 1), (i + 1, j)):
                if k < nt and l < na:
                    if np.isfinite(delta[i, j]) and np.isfinite(delta[k, l]) and np.sign(delta[i, j]) != np.sign(delta[k, l]):
                        dz.append(((i + k) / 2, (j + l) / 2))
                    if real_lead[i, j] and real_lead[k, l] and np.sign(mu[i, j]) != np.sign(mu[k, l]):
                        mz.append(((i + k) / 2, (j + l) / 2))
    near = lambda p, S: any(abs(p[0] - s[0]) <= 1.0 and abs(p[1] - s[1]) <= 1.0 for s in S)
    match = {"delta_zero_pairs": len(dz), "leading_sign_change_pairs": len(mz),
             "delta_pairs_within_one_cell_of_a_leading_change": int(sum(near(p, mz) for p in dz)),
             "leading_changes_within_one_cell_of_a_delta_pair": int(sum(near(p, dz) for p in mz))}
    real_mu = np.where(real_lead, mu, np.nan)
    crit = {"delta": critical_points(delta, ov >= 0.9, al, lt), "mu1": critical_points(real_mu, real_lead, al, lt)}
    # coupling in a basis transported from the start along the tracking tree
    rec = {"grid": [int(nt), int(na)], "solved": int(good.sum()), "tracked": int(np.isfinite(delta).sum()),
           "start_cell": [int(start[0]), int(start[1])], "start_alpha": float(al[start[1]]), "start_log10T": float(lt[start[0]]),
           "mixing_cells_overlap_below_0.9": int(np.nansum(ov < 0.9)), "min_overlap": float(np.nanmin(ov)),
           "tau_range": [float(np.nanmin(tau)), float(np.nanmax(tau))], "delta_range": [float(np.nanmin(delta)), float(np.nanmax(delta))],
           "zero_sets": match, "critical_points": crit,
           "complex_cells": int(np.sum(np.isfinite(delta) & (np.nan_to_num(tau) ** 2 < 4 * np.nan_to_num(delta))))}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(rec, indent=1) + "\n")
    np.savez_compressed(Path(args.json).with_suffix(".npz"), alpha=al, log10_temperature=lt, tau=tau, delta=delta, overlap=ov,
                        mu=mu, mu_imag=np.nan_to_num(mi), ok=ok)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
