"""Exploratory (run SYNC): synchronization of two diffusively coupled copies of the chaotic soft-token loop.

The single map is the soft-token loop F of tools/explore_llm_soft_loop.py (window K = 1): x -> E^T softmax(z(x)/T),
Qwen2.5-0.5B-Instruct, philosophy direction on layer 12, the "open" prompt with opener "Hello!", float64 on the CPU.
Two copies are coupled symmetrically in their outputs,

    x_A' = (1 - c) F(x_A) + c F(x_B),      x_B' = (1 - c) F(x_B) + c F(x_A),

with both rows of a pair in one batch (pairs interleaved: rows 2i and 2i+1). A transverse perturbation u = x_A - x_B
obeys u' = (1 - 2c) (F(x_A) - F(x_B)) = (1 - 2c) DF u + O(|u|^2), so along a typical synchronized orbit the transverse
exponent is lambda_1 + ln|1 - 2c| and synchronization is linearly stable for c > c* = (1 - exp(-lambda_1)) / 2.

Subcommands
  check       batch-position invariance of F in float64, invariance of the synchronized manifold (identical rows
              give identical outputs), and central differences against the exact JVP (error ~ h^2)
  sweep       for each cell and coupling c: start A on the attractor and B at a relative offset (near) or at another
              point of the attractor (far), iterate the coupled map, and store the relative distance
              |x_A - x_B| / |x_A| and the top-1 tokens of both rows at every step
  transverse  transverse exponent measured on the coupled map (a twin pair renormalized to a relative distance delta
              after every step), and lambda_1 of the single loop from exact JVPs on the same kind of orbit
  analyze     summary of the sweep and transverse outputs (docs/sync/sync-summary.json)
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

VECTORS, CONCEPT, LAYER, MODEL = "llm-phase-map/vectors-l12", "philosophy", 12, "qwen0.5b"
CELLS_FILE = ROOT / "docs" / "e2e" / "lyap-cells.json"
SYNC_TOL = 1e-8                          # relative distance below which a pair counts as synchronized


# ----------------------------------------------------------------------------------------------- pure functions

def c_star(lam):
    """Coupling at which lambda_1 + ln|1 - 2c| = 0 (c < 1/2); refuses a non-positive exponent."""
    if not lam > 0:
        raise ValueError(f"c* needs a positive leading exponent, got {lam}")
    return 0.5 * (1.0 - np.exp(-lam))


def predicted_transverse(lam, c):
    c = np.asarray(c, dtype=float)
    return lam + np.log(np.abs(1.0 - 2.0 * c))


def couple(F, c):
    """Coupled outputs for pair-interleaved rows F (2n, d) and couplings c (n,)."""
    torch = __import__("torch")
    c = torch.as_tensor(np.asarray(c, dtype=float), dtype=F.dtype, device=F.device)[:, None]
    Fa, Fb = F[0::2], F[1::2]
    out = torch.empty_like(F)
    out[0::2] = (1 - c) * Fa + c * Fb
    out[1::2] = (1 - c) * Fb + c * Fa
    return out


def coupled_step(loop, X, temps, c):
    """One step of the coupled map for pair-interleaved windows X (2n, K, d): the loop's step, then the new
    embeddings (last window slot) are mixed within each pair."""
    Y, p = loop.step(X, temps)
    Y = Y.clone()
    Y[:, -1, :] = couple(Y[:, -1, :], c)
    return Y, p


def rel_distance(X):
    """|x_A - x_B| / |x_A| per pair (full window state)."""
    A, B = X[0::2].reshape(X.shape[0] // 2, -1), X[1::2].reshape(X.shape[0] // 2, -1)
    return ((A - B).norm(dim=1) / A.norm(dim=1)).double().cpu().numpy()


def run_lengths(mask):
    """Lengths of maximal runs of True in a boolean sequence, and whether the last run reaches the end (censored)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(0, dtype=int), False
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return ends - starts, bool(mask[-1])


def powerlaw_slope(lengths, min_len=2, min_count=100):
    """Slope of the log-binned density of run lengths (least squares in log-log over bins with counts); None when
    fewer than min_count lengths >= min_len."""
    x = np.asarray(lengths, dtype=float)
    x = x[x >= min_len]
    if len(x) < min_count:
        return None
    edges = np.unique(np.floor(np.logspace(np.log10(min_len), np.log10(x.max() + 1), 15)).astype(int))
    if len(edges) < 4:
        return None
    h, _ = np.histogram(x, bins=edges)
    dens = h / np.diff(edges) / len(x)
    mid = np.sqrt(edges[:-1] * edges[1:])
    ok = h >= 3
    if ok.sum() < 3:
        return None
    slope, _ = np.polyfit(np.log(mid[ok]), np.log(dens[ok]), 1)
    return {"slope": float(slope), "bins_used": int(ok.sum()), "n": int(len(x)), "min_len": int(min_len)}


def threshold_interval(cs, synced):
    """Observed threshold from a c grid and per-c synchronization flags: the bracket (largest unsynchronized c below
    the smallest c from which every larger grid value is synchronized). None entries when no such c exists."""
    order = np.argsort(cs)
    cs, synced = np.asarray(cs, dtype=float)[order], np.asarray(synced, dtype=bool)[order]
    if not synced.any():
        return {"c_lo": float(cs[-1]), "c_hi": None, "status": "no synchronized c on the grid"}
    first = len(cs)
    for i in range(len(cs) - 1, -1, -1):
        if synced[i]:
            first = i
        else:
            break
    if first == len(cs):
        return {"c_lo": float(cs[-1]), "c_hi": None, "status": "largest c not synchronized"}
    lo = float(cs[first - 1]) if first > 0 else None
    return {"c_lo": lo, "c_hi": float(cs[first]), "status": "bracketed" if lo is not None else "all synchronized",
            "non_monotone": bool(synced[:first].any())}


def twin_logs(loop, X, temps, cs, delta, record):
    """Per-step log growth of the transverse distance of pair-interleaved twins X (row 2i+1 near row 2i) under the
    coupled map, with row 2i+1 reset to relative distance delta from row 2i after every step. Fails loudly when a
    twin collapses onto its partner."""
    n = len(cs)
    logs = np.zeros((n, record))
    for t in range(record):
        X, _ = coupled_step(loop, X, temps, cs)
        A = X[0::2]
        U = X[1::2] - A
        un = U.reshape(n, -1).norm(dim=1)
        an = A.reshape(n, -1).norm(dim=1)
        if not bool((un > 0).all()):
            raise RuntimeError(f"step {t + 1}: a twin collapsed onto its partner (zero transverse distance)")
        logs[:, t] = np.log((un / an).double().cpu().numpy() / delta)
        X = X.clone()
        X[1::2] = A + (delta * an / un)[:, None, None] * U
    return logs


def jvp_logs(loop, X, temps, record, rng):
    """Per-step log growth of one renormalized exact tangent per row (single loop, no coupling)."""
    torch = loop.torch
    n = X.shape[0]
    V = torch.as_tensor(rng.normal(size=tuple(X.shape)), dtype=X.dtype)
    V = V / V.reshape(n, -1).norm(dim=1)[:, None, None]
    out = np.zeros((n, record))
    for t in range(record):
        X, _, JV = loop.step_jvp(X, temps, V)
        nv = JV.reshape(n, -1).norm(dim=1)
        out[:, t] = np.log(nv.double().cpu().numpy())
        V = JV / nv[:, None, None]
    return out


def iterate_pairs(loop, X, temps, cs, record, on_save=None, save_every=250):
    """Iterate the coupled map; relative pair distances (n, record + 1) and top-1 tokens (2n, record)."""
    n = len(cs)
    dist = np.zeros((n, record + 1))
    tok = np.zeros((2 * n, record), dtype=np.int32)
    dist[:, 0] = rel_distance(X)
    for t in range(record):
        X, p = coupled_step(loop, X, temps, cs)
        dist[:, t + 1] = rel_distance(X)
        tok[:, t] = p.argmax(dim=-1).cpu().numpy()
        if on_save is not None and ((t + 1) % save_every == 0 or t + 1 == record):
            on_save(dist[:, :t + 2], tok[:, :t + 1], X, t + 1)
    return dist, tok, X


# ----------------------------------------------------------------------------------------------- model helpers

def load_cells():
    pairs = json.loads(CELLS_FILE.read_text())["cells"]
    return [{"index": i, "log10_temperature": float(lt), "alpha": float(a)} for i, (lt, a) in enumerate(pairs)]


def make_loop(threads):
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    loop = esl.SoftLoop(MODEL, LAYER, threads, "open", ela.OPENER, 1, "float64", "cpu")
    v = elpm.load_vectors(VECTORS)[CONCEPT]
    norms = sorted({type(m).forward.__qualname__ for m in loop.model.modules() if type(m).__name__.endswith("RMSNorm")})
    rot = [type(m).__name__ for m in loop.model.modules() if hasattr(m, "_inv_freq64")]
    if loop.state_dtype != loop.torch.float64 or loop.dtype != loop.torch.float64:
        raise SystemExit("the loop is not float64")
    if norms != ["keep_float64_norms.<locals>.forward"] or not rot:
        raise SystemExit(f"float64 patches missing: norms {norms}, float64 rotary modules {rot}")
    return loop, v


def attractor_states(loop, cells, v, transient, extra=0):
    """Single-loop states of every cell after `transient` steps (and after `transient + extra` steps), one batch row
    per cell; returns tensors (n_cells, K, d)."""
    X = loop.prepare(np.stack([c["alpha"] * v for c in cells]))
    temps = np.array([10.0 ** c["log10_temperature"] for c in cells])
    for _ in range(transient):
        X, _ = loop.step(X, temps)
    X0 = X.clone()
    for _ in range(extra):
        X, _ = loop.step(X, temps)
    return X0, X.clone()


def git_commit():
    try:
        sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "tools"],
                                             text=True).strip())
        return {"commit": sha, "tools_dirty": dirty}
    except Exception as e:          # noqa: BLE001
        return {"commit": None, "error": str(e)}


def out_dir(name):
    import explore_llm_attractors as ela

    d = ela.DATA / "explore" / "sync" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------------- subcommands

def check(args):
    """Batch-position invariance, synchronized-manifold invariance and h^2 convergence of central differences."""
    torch = __import__("torch")
    loop, v = make_loop(args.threads)
    cells = load_cells()
    res = {"commit": git_commit(), "threads": args.threads, "cells": []}
    for cell in cells:
        steer, T = cell["alpha"] * v, 10.0 ** cell["log10_temperature"]
        X0, X1 = attractor_states(loop, [cell], v, args.transient, extra=37)
        a, b = X0[0], X1[0]

        def F(rows):
            loop.prepare(np.tile(steer, (len(rows), 1)))
            Y, _ = loop.step(torch.stack(rows), np.full(len(rows), T))
            return Y[:, -1, :]

        single_a, single_b = F([a])[0], F([b])[0]
        ab, ba = F([a, b]), F([b, a])
        aa = F([a, a])
        big = F([b, a, b, a, a, b, b, a])
        scale = float(single_a.norm())
        diffs = {
            "pair_vs_single_a": float((ab[0] - single_a).abs().max()) / scale,
            "pair_vs_single_b": float((ab[1] - single_b).abs().max()) / scale,
            "swapped_layout_a": float((ab[0] - ba[1]).abs().max()) / scale,
            "swapped_layout_b": float((ab[1] - ba[0]).abs().max()) / scale,
            "batch8_vs_pair_a": float((big[1] - ab[0]).abs().max()) / scale,
            "identical_rows_differ": float((aa[0] - aa[1]).abs().max()) / scale,
        }
        bitwise = {"pair_vs_single": bool(torch.equal(ab[0], single_a) and torch.equal(ab[1], single_b)),
                   "swapped_layout": bool(torch.equal(ab[0], ba[1]) and torch.equal(ab[1], ba[0])),
                   "batch8_vs_pair": bool(torch.equal(big[1], ab[0]) and torch.equal(big[0], ab[1])),
                   "identical_rows_equal": bool(torch.equal(aa[0], aa[1]))}
        # central differences against the exact JVP along a random direction of norm |a|
        g = torch.as_tensor(np.random.default_rng(1).normal(size=a.shape), dtype=a.dtype)
        g = g / g.norm() * a.norm()
        loop.prepare(steer[None])
        _, _, J = loop.step_jvp(a[None], np.array([T]), g[None])
        J = J[0, -1]
        hs = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 1e-7]
        errs = []
        for h in hs:
            Fp, Fm = F([a + h * g, a - h * g])
            errs.append(float(((Fp - Fm) / (2 * h) - J).norm() / J.norm()))
        ratios = [errs[i] / errs[i + 2] for i in range(0, 6, 2)]    # factor-10 steps in h: ~100 for h^2
        cell_res = {**cell, "max_abs_diff_rel": diffs, "bitwise": bitwise, "fd_h": hs, "fd_rel_err": errs,
                    "fd_ratio_per_decade": ratios}
        print(json.dumps(cell_res), flush=True)
        res["cells"].append(cell_res)
    o = out_dir(args.out)
    (o / "check.json").write_text(json.dumps(res, indent=1))


def parse_cs(args):
    if args.c:
        return [float(c) for c in args.c]
    cs = np.round(np.arange(args.c_min, args.c_max + 1e-12, args.c_step), 6)
    return [float(c) for c in cs]


def sweep(args):
    torch = __import__("torch")
    loop, v = make_loop(args.threads)
    cells = [c for c in load_cells() if c["index"] in args.cells]
    if len(cells) != len(args.cells):
        raise SystemExit(f"unknown cell index in {args.cells}")
    cs = parse_cs(args)
    per_cell = {int(k): [float(c) for c in v_] for k, v_ in json.loads(args.c_per_cell).items()} \
        if args.c_per_cell else None
    if per_cell and set(per_cell) != {c["index"] for c in cells}:
        raise SystemExit("--c-per-cell must list exactly the cells of --cells")
    X0, X1 = attractor_states(loop, cells, v, args.transient, extra=args.far_extra if args.start == "far" else 0)
    rows, temps, steer, pair_c, meta = [], [], [], [], []
    rng = np.random.default_rng(args.seed)
    for ci, cell in enumerate(cells):
        a = X0[ci]
        if args.start == "near":
            g = torch.as_tensor(rng.normal(size=a.shape), dtype=a.dtype)
            b = a + args.offset * a.norm() * g / g.norm()
        else:
            b = X1[ci]
        for c in (per_cell[cell["index"]] if per_cell else cs):
            rows += [a, b]
            temps += [10.0 ** cell["log10_temperature"]] * 2
            steer += [cell["alpha"] * v] * 2
            pair_c.append(c)
            meta.append({"cell": cell["index"], "c": c})
    X = torch.stack(rows).clone()
    temps = np.array(temps)
    loop.prepare(np.stack(steer))
    tic = time.perf_counter()

    def on_save(dist, tok, Xt, steps):
        save_sweep(args, meta, dist, tok, Xt, steps, time.perf_counter() - tic)
        print(json.dumps({"steps": steps, "elapsed_s": round(time.perf_counter() - tic, 1)}), flush=True)

    iterate_pairs(loop, X, temps, pair_c, args.record, on_save, args.save_every)


def save_sweep(args, meta, dist, tok, X, steps, elapsed):
    o = out_dir(args.out)
    np.savez_compressed(o / "sweep.npz", dist=dist, tok=tok, final=X.cpu().numpy())
    h = hashlib.sha256((o / "sweep.npz").read_bytes()).hexdigest()
    (o / "sweep.json").write_text(json.dumps({"args": vars(args), "commit": git_commit(), "pairs": meta,
                                              "steps": steps, "complete": steps == args.record,
                                              "elapsed_s": elapsed, "sweep_npz_sha256": h}, indent=1))


def transverse(args):
    """Transverse exponent on the coupled map by a renormalized twin pair: B = A + delta |A| u after every step;
    and lambda_1 of the single loop by exact JVPs (k = 1, renormalized) on its own orbit from the same start."""
    torch = __import__("torch")
    loop, v = make_loop(args.threads)
    cells = [c for c in load_cells() if c["index"] in args.cells]
    cs = parse_cs(args)
    X0, _ = attractor_states(loop, cells, v, args.transient)
    rng = np.random.default_rng(args.seed)
    rows, temps, steer, pair_c, meta = [], [], [], [], []
    for ci, cell in enumerate(cells):
        a = X0[ci]
        g = torch.as_tensor(rng.normal(size=a.shape), dtype=a.dtype)
        for c in cs:
            rows += [a, a + args.delta * a.norm() * g / g.norm()]
            temps += [10.0 ** cell["log10_temperature"]] * 2
            steer += [cell["alpha"] * v] * 2
            pair_c.append(c)
            meta.append({"cell": cell["index"], "c": c})
    X, temps = torch.stack(rows).clone(), np.array(temps)
    loop.prepare(np.stack(steer))
    tic = time.perf_counter()
    logs = twin_logs(loop, X, temps, pair_c, args.delta, args.record)
    el = time.perf_counter() - tic
    # single-loop lambda_1 from exact JVPs along the orbit of every cell from the same transient start
    Xs = X0.clone()
    loop.prepare(np.stack([c["alpha"] * v for c in cells]))
    ts = np.array([10.0 ** c["log10_temperature"] for c in cells])
    jl = jvp_logs(loop, Xs, ts, args.record_jvp, rng)
    o = out_dir(args.out)
    np.savez_compressed(o / "transverse.npz", logs=logs, jvp_logs=jl)
    h = hashlib.sha256((o / "transverse.npz").read_bytes()).hexdigest()
    (o / "transverse.json").write_text(json.dumps({"args": vars(args), "commit": git_commit(), "pairs": meta,
                                                   "cells": cells, "elapsed_twin_s": el,
                                                   "elapsed_total_s": time.perf_counter() - tic,
                                                   "transverse_npz_sha256": h}, indent=1))
    for i, m in enumerate(meta):
        print(json.dumps({**m, "lambda_perp": round(float(logs[i].mean()), 4)}), flush=True)
    for i, c in enumerate(cells):
        print(json.dumps({"cell": c["index"], "lambda1_jvp": round(float(jl[i].mean()), 4)}), flush=True)


def excursions(d, low=SYNC_TOL, high=1e-3):
    """Number of bursts: passages from below `low` to above `high`."""
    n, state = 0, None
    for x in d:
        if x < low:
            state = "low"
        elif x > high and state == "low":
            n, state = n + 1, "high"
    return n


def token_hist(tok, V):
    h = np.bincount(tok.ravel(), minlength=V).astype(float)
    return h / h.sum()


def analyze(args):
    import explore_llm_attractors as ela
    from transformers import AutoTokenizer

    tok_dec = AutoTokenizer.from_pretrained(ela.MODELS[MODEL], cache_dir=ela.CACHE)
    base = ela.DATA / "explore" / "sync"
    cells = {c["index"]: c for c in load_cells()}
    pub = json.loads((ela.DATA / "explore" / "e2e" / "lyap-spectrum" / "lyapunov-spectrum.json").read_text())
    pub = {c["index"]: c["exponents"][0] for c in pub["cells"]}
    summary = {"commit": git_commit(), "sync_tol": SYNC_TOL, "sync_rule": f"max distance over the last "
               f"{args.tail} steps below {SYNC_TOL}", "cells": {}}
    ck = json.loads((base / "check" / "check.json").read_text())
    summary["check"] = [{"index": c["index"], "bitwise": c["bitwise"], "max_abs_diff_rel": c["max_abs_diff_rel"],
                         "fd_rel_err": dict(zip(map(str, c["fd_h"]), c["fd_rel_err"]))} for c in ck["cells"]]
    tj = json.loads((base / "transverse" / "transverse.json").read_text())
    tz = np.load(base / "transverse" / "transverse.npz")
    if tj.get("commit", {}).get("tools_dirty"):
        summary["warnings"] = ["transverse run had uncommitted tool changes"]
    for i, c in enumerate(tj["cells"]):
        jl = tz["jvp_logs"][i]
        lam, se = float(jl.mean()), block_se(jl)
        ci = c["index"]
        cs_ = c_star(lam)
        rows = []
        for k, m in enumerate(tj["pairs"]):
            if m["cell"] != ci:
                continue
            lg = tz["logs"][k]
            meas, mse = float(lg.mean()), block_se(lg)
            pred = float(predicted_transverse(lam, m["c"]))
            rows.append({"c": m["c"], "measured": meas, "measured_se": mse, "predicted": pred,
                         "measured_minus_predicted": meas - pred,
                         "measured_minus_ln_factor": meas - float(np.log(abs(1 - 2 * m["c"])))})
        summary["cells"][str(ci)] = {**cells[ci], "lambda1_published": pub.get(ci),
                                     "lambda1_jvp": lam, "lambda1_jvp_se": se, "lambda1_jvp_steps": int(len(jl)),
                                     "c_star_published": c_star(pub[ci]), "c_star_jvp": cs_,
                                     "c_star_jvp_2se": [c_star(lam - 2 * se), c_star(lam + 2 * se)],
                                     "transverse": rows, "sweeps": {}}
    for name in args.sweeps:
        sj = json.loads((base / name / "sweep.json").read_text())
        sz = np.load(base / name / "sweep.npz")
        dist, tok = sz["dist"], sz["tok"]
        steps = sj["steps"]
        V = int(tok.max()) + 1
        for ci in sorted({m["cell"] for m in sj["pairs"]}):
            idx = [k for k, m in enumerate(sj["pairs"]) if m["cell"] == ci]
            c0 = [k for k in idx if sj["pairs"][k]["c"] == 0.0]
            ref = None
            if c0:
                h0a = token_hist(tok[2 * c0[0], steps // 2:], V)
                h0b = token_hist(tok[2 * c0[0] + 1, steps // 2:], V)
                ref = (h0a + h0b) / 2
            rows = []
            for k in idx:
                d = dist[k, 1:]
                tail = d[-args.tail:]
                synced = bool(tail.max() < SYNC_TOL)
                row = {"c": sj["pairs"][k]["c"], "mean_dist": float(d.mean()),
                       "mean_dist_second_half": float(d[len(d) // 2:].mean()),
                       "frac_below_tol": float((d < SYNC_TOL).mean()), "synced": synced,
                       "tail_max": float(tail.max()), "tail_median": float(np.median(tail)),
                       "first_below_tol": int(np.argmax(d < SYNC_TOL)) + 1 if (d < SYNC_TOL).any() else None,
                       "bursts_1e-8_to_1e-3": excursions(d)}
                for th in (1e-4, 1e-6):
                    lens, cens = run_lengths(d < th)
                    row[f"laminar_{th:g}"] = {"n": int(len(lens)), "censored_last": cens,
                                              "max": int(lens.max()) if len(lens) else 0,
                                              "fit": powerlaw_slope(lens[:-1] if cens else lens)}
                ta, tb = tok[2 * k, -args.text_len:], tok[2 * k + 1, -args.text_len:]
                row["text_A_last"] = tok_dec.decode(ta.tolist())
                row["text_B_last"] = tok_dec.decode(tb.tolist())
                row["top1_equal_frac_second_half"] = float((tok[2 * k, steps // 2:] == tok[2 * k + 1, steps // 2:]).mean())
                if ref is not None:
                    hA = token_hist(tok[2 * k, steps // 2:], V)
                    row["tv_A_vs_c0_tokens"] = float(0.5 * np.abs(hA - ref).sum())
                rows.append(row)
            if c0:
                h0a = token_hist(tok[2 * c0[0], steps // 2:], V)
                h0b = token_hist(tok[2 * c0[0] + 1, steps // 2:], V)
                tv0 = float(0.5 * np.abs(h0a - h0b).sum())
            else:
                tv0 = None
            th = threshold_interval([r["c"] for r in rows], [r["synced"] for r in rows])
            summary["cells"][str(ci)]["sweeps"][name] = {"start": sj["args"]["start"], "steps": steps,
                                                         "complete": sj["complete"], "commit": sj["commit"],
                                                         "sweep_npz_sha256": sj["sweep_npz_sha256"],
                                                         "threshold": th, "tv_c0_A_vs_B_tokens": tv0, "pairs": rows}
    out = ROOT / "docs" / "sync" / "sync-summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    for ci, c in summary["cells"].items():
        print(ci, "lambda1", round(c["lambda1_jvp"], 4), "+-", round(c["lambda1_jvp_se"], 4), "c*",
              round(c["c_star_jvp"], 4), {n: s["threshold"] for n, s in c["sweeps"].items()})


def block_se(x, nblocks=10):
    """Standard error of the mean from nblocks contiguous blocks."""
    x = np.asarray(x, dtype=float)
    m = len(x) // nblocks
    if m < 1:
        return None
    b = x[:m * nblocks].reshape(nblocks, m).mean(1)
    return float(b.std(ddof=1) / np.sqrt(nblocks))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "sweep", "transverse"):
        s = sub.add_parser(name)
        s.add_argument("--threads", type=int, default=4)
        s.add_argument("--transient", type=int, default=1000)
        s.add_argument("--out", required=True)
        if name != "check":
            s.add_argument("--cells", type=int, nargs="+", default=[0, 1, 2])
            s.add_argument("--c", type=float, nargs="+", default=None)
            s.add_argument("--c-min", type=float, default=0.0)
            s.add_argument("--c-max", type=float, default=0.3)
            s.add_argument("--c-step", type=float, default=0.01)
            s.add_argument("--record", type=int, default=2000)
            s.add_argument("--seed", type=int, default=0)
    sw = sub.choices["sweep"]
    sw.add_argument("--start", choices=("near", "far"), default="near")
    sw.add_argument("--offset", type=float, default=1e-3)
    sw.add_argument("--far-extra", type=int, default=500)
    sw.add_argument("--save-every", type=int, default=250)
    sw.add_argument("--c-per-cell", default=None, help='JSON {"cell": [c, ...]}: per-cell grids (overrides --c)')
    tr = sub.choices["transverse"]
    tr.add_argument("--delta", type=float, default=1e-7)
    tr.add_argument("--record-jvp", type=int, default=2000)
    an = sub.add_parser("analyze")
    an.add_argument("--sweeps", nargs="+", default=["sweep-near"])
    an.add_argument("--tail", type=int, default=500)
    an.add_argument("--text-len", type=int, default=12)
    args = ap.parse_args()
    {"check": check, "sweep": sweep, "transverse": transverse, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
