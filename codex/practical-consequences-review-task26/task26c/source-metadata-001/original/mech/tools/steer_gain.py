"""Why does steering make the soft-token loop chaotic? The gain split along steering, steering block and a readout-only
intervention (run MECH).

tools/gain_split.py (runs G1, G2) splits the growth of a twin on unsteered temperature lines exactly into a network
and a readout factor. This tool does the same for any grid of cells (alpha, beta, log10 T):
    alpha  steering alpha * v added after block L (v a concept vector at that block, scaled to the residual norm);
    beta   a logit bias beta * b added to the next-token logits, b = W v (the direct logit effect of v) centred and
           scaled to unit standard deviation over the vocabulary: a readout-only intervention that leaves the network
           untouched (alpha = 0 for a pure bias).
Each cell runs the loop (window 1, float64) for a transient, then records steps with twins at relative offsets
eps (renormalised every step). With h the final normalised hidden state at the soft token's position, F the next soft
token and x the current one, each step's growth of the first twin splits exactly as in gain_split.py:
    g_net  = (|dh| / |h|) / (|dx| / |x|),   g_read = (|dF| / |F|) / (|dh| / |h|),
and the log-averages of g_net and g_read add up to the twin exponent up to the boundary term (1/N) log(|x_0|/|x_N|).
Per step, from the base trajectory, also tangent-free readout statistics:
    entropy of p; gap = (z_1 - z_2) / T between the two largest logits; probability of the most probable token;
    for the readout Jacobian dF/dh = A = (1/T) E^T (diag p - p p^T) E over the whole vocabulary (tied embeddings):
    its trace (exact), its largest eigenvalue r1 (power iteration, `iters` products per step, warm-started from the
    previous step's vector; the last relative change of the Rayleigh quotient is reported), made relative like
    g_read (r1_rel = r1 |h| / |F|, so g_read <= r1_rel), and its stable rank trace / r1 (the number of directions in
    which the readout amplifies comparably: the number of competing tokens, roughly, minus one).
A cell is chaotic if both twin exponents exceed 1e-3 and no period up to 64 is found (tolerance 1e-8 relative), as
in the paper's maps.

Usage: python tools/steer_gain.py --layer 12 --vectors llm-phase-map/vectors-l12 --concept philosophy \
           --alpha -0.12 0.12 241 --log-temperature -0.6 -0.3 0.0 0.15 0.25 --device cuda --out llm-soft-loop/pod64/mech/x
       (--beta LO HI N for the logit bias; --tgrid LO HI N instead of a list of temperatures)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

KEEP_STEPS = 1000            # per-step arrays are saved for the first KEEP_STEPS recorded steps of every cell


def readout_stats(p, z, F, E, e2, temps, u, iters):
    """Trace and largest eigenvalue of A = (1/T) (E^T diag(p) E - F F^T) for every row (F = E^T p), by `iters` steps
    of power iteration from u (n, d); the gap between the two largest logits divided by T. e2 (V,) = |e_i|^2.
    Returns (r1, trace, gap, u, last relative change of the Rayleigh quotient)."""
    import torch
    trace = ((p @ e2) - (F * F).sum(-1)) / temps
    q_old = None
    for _ in range(iters):
        Au = ((p * (u @ E.T)) @ E - F * (F * u).sum(-1, keepdim=True)) / temps[:, None]
        q = (u * Au).sum(-1)
        u = Au / torch.clamp(Au.norm(dim=-1, keepdim=True), min=1e-300)
        change = None if q_old is None else torch.abs(q - q_old) / torch.clamp(torch.abs(q), min=1e-300)
        q_old = q
    top2 = torch.topk(z, 2, dim=-1).values
    return q, trace, (top2[:, 0] - top2[:, 1]) / temps, u, change


def h_F(loop, X, temps, beta=None, bvec=None):
    """Final normalised hidden state, logits (with the optional bias), next-token distribution and next soft token."""
    torch = loop.torch
    loop.sm.steer = loop.steer_rows
    try:
        with torch.no_grad():
            out = loop.model.model(inputs_embeds=X.to(loop.dtype), past_key_values=loop.cache, use_cache=True)
            h = out.last_hidden_state[:, -1]
            z = loop.model.lm_head(h).to(loop.device, loop.state_dtype)
    finally:
        loop.sm.steer = None
        loop.reset_cache()
    if beta is not None:
        z = z + beta[:, None] * bvec[None, :]
    p = torch.softmax(z / temps[:, None], dim=-1)
    return h.to(loop.device, loop.state_dtype), z, p, p @ loop.E


def run_cells(loop, v, cells, transient, record, eps_list, iters, bvec=None, seed=0, log=print):
    """cells: list of (alpha, beta, log10 T). Returns per-cell summaries and per-step arrays."""
    torch = loop.torch
    n, m = len(cells), len(eps_list)
    alphas = np.array([c[0] for c in cells])
    betas = np.array([c[1] for c in cells])
    log_ts = np.array([c[2] for c in cells])
    steer = np.concatenate([alphas[:, None] * v[None, :]] * (m + 1))
    temps = torch.as_tensor(np.tile(10.0 ** log_ts, m + 1), dtype=loop.state_dtype, device=loop.device)
    beta = None
    if bvec is not None and np.any(betas != 0):
        beta = torch.as_tensor(np.tile(betas, m + 1), dtype=loop.state_dtype, device=loop.device)
    X = loop.prepare(steer)
    tic = time.perf_counter()
    for i in range(transient):
        _, _, _, F = h_F(loop, X, temps, beta, bvec)
        X = F[:, None, :]
    base = X[:n, -1]
    rng = np.random.default_rng(seed)
    twins = []
    for e in eps_list:
        g = torch.as_tensor(rng.normal(size=tuple(base.shape)), dtype=loop.state_dtype, device=loop.device)
        twins.append(base + g / g.norm(dim=1, keepdim=True) * e * base.norm(dim=1, keepdim=True))
    X = torch.cat([base] + twins)[:, None, :]
    x0 = base.norm(dim=1).double().cpu().numpy()
    nx = lambda a: a.norm(dim=1).double().cpu().numpy()                        # noqa: E731
    growth = np.zeros((m, n))
    keep = min(record, KEEP_STEPS)
    names = ("log_g_net", "log_g_read", "entropy", "gap", "p_top", "log_r1_rel", "stable_rank", "rayleigh_change")
    arr = {k: np.zeros((keep, n), dtype=np.float32) for k in names}
    sums = {k: np.zeros(n) for k in names}
    frac_gap1, frac_gap2 = np.zeros(n), np.zeros(n)
    tail = []
    T0 = temps[:n]
    e2 = (loop.E * loop.E).sum(-1)
    u = torch.as_tensor(rng.normal(size=(n, loop.E.shape[1])), dtype=loop.state_dtype, device=loop.device)
    u = u / u.norm(dim=-1, keepdim=True)
    for i in range(record):
        h, z, p, F = h_F(loop, X, temps, beta, bvec)
        xb = X[:n, -1]
        hb, Fb = h[:n], F[:n]
        new = [Fb]
        for k, e in enumerate(eps_list):
            sl = slice(n * (k + 1), n * (k + 2))
            dx, dh, dF = nx(X[sl, -1] - xb), nx(h[sl] - hb), nx(F[sl] - Fb)
            growth[k] += np.log(dF / dx)
            if k == 0:
                rx, rh, rF = dx / nx(xb), dh / nx(hb), dF / nx(Fb)
                step = {"log_g_net": np.log(rh / rx), "log_g_read": np.log(rF / rh)}
            scale = torch.as_tensor(e * nx(Fb) / dF, dtype=loop.state_dtype, device=loop.device)[:, None]
            new.append(Fb + (F[sl] - Fb) * scale)
        pb, zb = p[:n], z[:n]
        r1, tr, gap, u, change = readout_stats(pb, zb, Fb, loop.E, e2, T0, u, iters)
        r1 = r1.double().cpu().numpy()
        step.update({
            "entropy": (-(pb * torch.log(torch.clamp(pb, min=1e-300))).sum(-1)).double().cpu().numpy(),
            "gap": gap.double().cpu().numpy(),
            "p_top": pb.max(dim=-1).values.double().cpu().numpy(),
            "log_r1_rel": np.log(np.maximum(r1 * nx(hb) / nx(Fb), 1e-300)),
            "stable_rank": tr.double().cpu().numpy() / np.maximum(r1, 1e-300),
            "rayleigh_change": change.double().cpu().numpy() if change is not None else np.zeros(n)})
        for k in names:
            sums[k] += step[k]
            if i < keep:
                arr[k][i] = step[k]
        frac_gap1 += step["gap"] < 1.0
        frac_gap2 += step["gap"] < 2.0
        X = torch.cat(new)[:, None, :]
        tail.append(Fb.double().cpu().numpy())
        if len(tail) > 66:
            tail.pop(0)
        if (i + 1) % 500 == 0:
            log(json.dumps({"recorded": i + 1, "of": record, "seconds": round(time.perf_counter() - tic, 1)}))
    S = np.array(tail)
    out = []
    for c in range(n):
        s = S[:, c]
        period = next((q for q in range(1, min(65, len(s))) if np.linalg.norm(s[-1 - q] - s[-1])
                       < 1e-8 * max(1.0, np.linalg.norm(s[-1]))), 0)
        lyap = [float(g[c] / record) for g in growth]
        chaotic = period == 0 and all(x > 1e-3 for x in lyap)
        rec = {"alpha": float(alphas[c]), "beta": float(betas[c]), "log10_temperature": float(log_ts[c]),
               "lyapunov": lyap, "eps": list(eps_list), "period": int(period), "chaotic": bool(chaotic),
               "boundary_term": float(np.log(np.linalg.norm(s[-1]) / x0[c]) / record),
               "frac_gap_below_1": float(frac_gap1[c] / record), "frac_gap_below_2": float(frac_gap2[c] / record)}
        rec.update({f"mean_{k}": float(sums[k][c] / record) for k in names})
        rec["max_rayleigh_change"] = float(arr["rayleigh_change"][:, c].max())
        out.append(rec)
    return out, arr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--alpha", type=float, nargs=3, default=[0.0, 0.0, 1], metavar=("LO", "HI", "N"))
    ap.add_argument("--beta", type=float, nargs=3, default=[0.0, 0.0, 1], metavar=("LO", "HI", "N"))
    ap.add_argument("--log-temperature", type=float, nargs="+")
    ap.add_argument("--tgrid", type=float, nargs=3, metavar=("LO", "HI", "N"))
    ap.add_argument("--cells-file", help="JSON file with a list 'cells' of [log10T, alpha] pairs (instead of a grid)")
    ap.add_argument("--transient", type=int, default=1000)
    ap.add_argument("--record", type=int, default=2000)
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-6, 1e-9])
    ap.add_argument("--iters", type=int, default=6, help="power-iteration products per step for the readout's r1")
    ap.add_argument("--chunk", type=int, default=1300, help="cells per batch")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    betas = np.linspace(args.beta[0], args.beta[1], int(args.beta[2]))
    if args.cells_file:
        cells = [(float(a), 0.0, float(t)) for t, a in json.loads(Path(args.cells_file).read_text())["cells"]]
    else:
        if (args.log_temperature is None) == (args.tgrid is None):
            raise SystemExit("give either --log-temperature or --tgrid (or --cells-file)")
        log_ts = args.log_temperature if args.log_temperature is not None else \
            list(np.linspace(args.tgrid[0], args.tgrid[1], int(args.tgrid[2])))
        alphas = np.linspace(args.alpha[0], args.alpha[1], int(args.alpha[2]))
        cells = [(float(a), float(b), float(t)) for t in log_ts for b in betas for a in alphas]
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    torch = loop.torch
    if loop.model.lm_head.weight.data_ptr() != loop.model.get_input_embeddings().weight.data_ptr():
        raise SystemExit("steer_gain: the readout spectrum assumes tied input and output embeddings")
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    bvec, bnote = None, None
    if np.any(betas != 0):
        W = loop.model.lm_head.weight.detach().to(loop.state_dtype)
        b = W @ torch.as_tensor(v, dtype=loop.state_dtype, device=W.device)
        b = (b - b.mean()) / b.std()
        bvec = b.to(loop.device)
        bnote = {"definition": "W v centred and scaled to unit standard deviation over the vocabulary",
                 "top_tokens": [loop.sm.tok.decode([int(i)]) for i in torch.topk(b, 10).indices.tolist()],
                 "bottom_tokens": [loop.sm.tok.decode([int(i)]) for i in torch.topk(-b, 10).indices.tolist()]}
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    progress = out / "progress.jsonl"

    def log(line):
        print(line, flush=True)
        with open(progress, "a") as f:
            f.write(line + "\n")

    results, arrays = [], {}
    for s0 in range(0, len(cells), args.chunk):
        part = cells[s0:s0 + args.chunk]
        res, arr = run_cells(loop, v, part, args.transient, args.record, args.eps, args.iters, bvec, log=log)
        results += res
        for k, a in arr.items():
            arrays.setdefault(k, []).append(a)
        log(json.dumps({"done": s0 + len(part), "of": len(cells), "chaotic": int(sum(r["chaotic"] for r in res)),
                        "seconds": round(time.perf_counter() - tic, 1)}))
    np.savez_compressed(out / "steps.npz", **{k: np.concatenate(v_, axis=1) for k, v_ in arrays.items()},
                        cells=np.array(cells))
    (out / "gain.json").write_text(json.dumps({"args": vars(args), "bias": bnote, "cells": results,
                                               "seconds": round(time.perf_counter() - tic, 1)}, indent=1))


if __name__ == "__main__":
    main()
