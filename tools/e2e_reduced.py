"""Run E2E, Q3b: scalar maps built from the model, compared with the full soft-token map.

Where two tokens A and B compete at one phase of a loop (the section), the loop is followed through the log-odds
s = (z_A - z_B) / T of that pair, with q = sigma(s) = p_A / (p_A + p_B). Two reductions:

R1 (invariant curve)  the full map's own one-dimensional curve: a segment of section states is pushed through the
                      return map G = F^p a few times, which contracts it onto the attracting curve (transverse
                      multipliers are small), and the map s -> s' is read off on that curve. No model of the
                      dynamics is assumed; what it tests is whether the dynamics is one-dimensional.
R2 (two-token model)  only the share q of the pair varies: the soft token after the section is
                      x(q) = w (q e_A + (1 - q) e_B) + (1 - w) m_rest with w = p_A + p_B and the mean embedding of
                      the other tokens m_rest frozen at a reference state; then h(q) = z_A - z_B after the p - 1
                      remaining steps of the network, and q' = sigma(h(q) / T). A sigmoid with slope 1/T composed with
                      a smooth network response: its multiplier at a fixed point is q (1 - q) h'(q) / T.

section  iterate a cell to its attractor and record the section states (top token A or B), their log-odds and
         the return pairs (s_n, s_{n+1}); the full map's twin exponent is measured along the same record
r1       R1 maps at given cells from a segment between two saved section states (or around a periodic state)
r2       R2 maps at given cells from a reference section state
flips    flip parameters of a reduced family along a cut (secant on the multiplier of the period-2^k orbit)

Usage: python tools/e2e_reduced.py section --alpha 0.0652 --log-t -0.156 --pair existence question --period 3 --out red-sec
       python tools/e2e_reduced.py r2 --ref red-sec --pair existence question --period 3 --cells 0.0645 -0.156 ... --out red-r2
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


def logodds(cell, X, a, b):
    Z = cell.logits(np.atleast_2d(X))
    return (Z[:, a] - Z[:, b]) / cell.T, Z


def section(args):
    model = em.Model(args.threads)
    a, b = em.TOKENS[args.pair[0]], em.TOKENS[args.pair[1]]
    cell = model.cell(args.alpha, args.log_t)
    x = np.load(args.seed_state).reshape(-1) if args.seed_state else cell.start.copy()
    tic = time.perf_counter()
    x = cell.Fp(x, args.transient)
    rng = np.random.default_rng(0)
    y = x + rng.normal(size=x.shape) * 1e-7 * np.linalg.norm(x) / np.sqrt(len(x))
    X, logs = [], 0.0
    for _ in range(args.steps):
        both = cell.F(np.array([x, y]))
        x, y = both[0], both[1]
        dist = np.linalg.norm(y - x)
        logs += np.log(dist / (1e-7 * np.linalg.norm(x)))
        y = x + (y - x) * (1e-7 * np.linalg.norm(x) / dist)
        X.append(x.copy())
    X = np.array(X)
    s, Z = logodds(cell, X, a, b)
    top = Z.argmax(axis=1)
    sec = np.where((top == a) | (top == b))[0]
    gaps = np.diff(sec)
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "section.npz", states=X[sec], s=s[sec], index=sec)
    pairs = [(float(s[i]), float(s[j])) for i, j in zip(sec[:-1], sec[1:]) if j - i == args.period]
    rec = {"args": vars(args), "lyapunov_twin": float(logs / args.steps), "section_points": int(len(sec)),
           "return_times": {int(k): int(v) for k, v in zip(*np.unique(gaps, return_counts=True))},
           "s_range": [float(s[sec].min()), float(s[sec].max())], "pairs": pairs,
           "seconds": round(time.perf_counter() - tic, 1)}
    (out / "section.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("lyapunov_twin", "section_points", "return_times", "s_range")}))


class R2:
    """The two-token model at one cell, frozen at a reference section state x_ref (computed at the same cell or
    another one; w and m_rest come from p(x_ref) at the reference cell)."""

    def __init__(self, model, a, b, period, ref_p):
        self.m, self.a, self.b, self.period = model, a, b, period
        E = model.E
        w = ref_p[a] + ref_p[b]
        rest = ref_p.copy()
        rest[[a, b]] = 0.0
        self.w = float(w)
        self.m_rest = rest @ E / rest.sum()
        self.ea, self.eb = E[a], E[b]

    def soft_token(self, q):
        q = np.atleast_1d(q)
        return self.w * (q[:, None] * self.ea + (1 - q[:, None]) * self.eb) + (1 - self.w) * self.m_rest

    def h(self, cell, q):
        X = self.soft_token(q)
        if self.period > 1:
            X = cell.Fp(X, self.period - 1)
        Z = cell.logits(X)
        return Z[:, self.a] - Z[:, self.b]

    def gridmap(self, cell, n=401, lo=1e-6, hi=1 - 1e-6):
        """q -> sigma(h(q)/T) on a grid uniform in the log-odds of q (resolves both saturated ends)."""
        s = np.linspace(np.log(lo / (1 - lo)), np.log(hi / (1 - hi)), n)
        q = 1 / (1 + np.exp(-s))
        h = self.h(cell, q)
        s2 = h / cell.T
        return s, s2, h


def r2(args):
    model = em.Model(args.threads)
    a, b = em.TOKENS[args.pair[0]], em.TOKENS[args.pair[1]]
    ref = json.loads((em.OUT / args.ref / "section.json").read_text())["args"]
    refcell = model.cell(ref["alpha"], ref["log_t"])
    S = np.load(em.OUT / args.ref / "section.npz")
    i = int(np.argmin(np.abs(S["s"] - np.median(S["s"])))) if args.ref_index < 0 else args.ref_index
    p_ref = refcell.probs(S["states"][i][None])[0]
    model_r2 = R2(model, a, b, args.period, p_ref)
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = [(args.cells[j], args.cells[j + 1]) for j in range(0, len(args.cells), 2)]
    for alpha, log_t in cells:
        cell = model.cell(alpha, log_t)
        s, s2, h = model_r2.gridmap(cell, n=args.n)
        g = en.GridMap(s, s2)
        rec = {"alpha": alpha, "log10_temperature": log_t, "w": model_r2.w, "extrema": g.extrema()}
        for per in args.periods:
            try:
                rec[f"orbits_{per}"] = g.periodic_orbits(per, grid=args.grid)
            except ValueError as err:
                rec[f"orbits_{per}"] = str(err)
        try:
            rec["lyapunov_per_return"] = g.lyapunov(float(np.median(s)), transient=2000, steps=20000)
        except ValueError as err:
            rec["lyapunov_per_return"] = str(err)
        rows.append(rec)
        np.savez_compressed(out / f"r2-a{alpha:.7f}-t{log_t:.5f}.npz", s=s, s2=s2, h=h)
        print(json.dumps({k: rec[k] for k in ("alpha", "log10_temperature", "extrema", "lyapunov_per_return")}),
              flush=True)
    (out / "r2.json").write_text(json.dumps({"args": vars(args), "ref_index": i, "ref_state_s": float(S["s"][i]),
                                             "rows": rows}, indent=1))


def r2_fixed_points(model_r2, cell, n, grid):
    s, s2, h = model_r2.gridmap(cell, n=n)
    g = en.GridMap(s, s2)
    return g, g.periodic_orbits(1, grid=grid)


def r2scan(args):
    """R2 fixed points along a cut (free alpha at fixed log10 T, or free log10 T at fixed alpha) and the events
    where their number changes (a fold of the reduced map: two fixed points meet, multiplier +1) or a multiplier
    crosses -1 (a flip), each refined by bisection on the cut. The reduced map's events are then compared with the
    full map's (continued fold curves, located flips)."""
    model = em.Model(args.threads)
    a, b = em.TOKENS[args.pair[0]], em.TOKENS[args.pair[1]]
    ref = json.loads((em.OUT / args.ref / "section.json").read_text())["args"]
    refcell = model.cell(ref["alpha"], ref["log_t"])
    S = np.load(em.OUT / args.ref / "section.npz")
    i = int(np.argmin(np.abs(S["s"] - np.median(S["s"])))) if args.ref_index < 0 else args.ref_index
    model_r2 = R2(model, a, b, args.period, refcell.probs(S["states"][i][None])[0])

    def at(lam):
        cell = model.cell(lam, args.value) if args.fixed == "log_t" else model.cell(args.value, lam)
        g, fps = r2_fixed_points(model_r2, cell, args.n, args.grid)
        return {"lambda": float(lam), "fixed_points": [[f["points"][0], f["multiplier"]] for f in fps]}

    lams = np.linspace(args.range[0], args.range[1], args.steps)
    rows = []
    for lam in lams:
        rows.append(at(lam))
        print(json.dumps(rows[-1]), flush=True)

    def stable_count(r):
        return sum(1 for _, m in r["fixed_points"] if abs(m) < 1)

    def signature(r):
        return (len(r["fixed_points"]), stable_count(r))

    events = []
    for r0, r1 in zip(rows, rows[1:]):
        if signature(r0) == signature(r1):
            continue
        lo, hi = r0, r1
        for _ in range(args.bisect):
            mid = at(0.5 * (lo["lambda"] + hi["lambda"]))
            if signature(mid) == signature(lo):
                lo = mid
            else:
                hi = mid
        kind = ("fold (fixed points appear or vanish in a pair)" if len(lo["fixed_points"]) != len(hi["fixed_points"])
                else "stability change at fixed count (flip if the multiplier passes -1)")
        mults = [m for _, m in lo["fixed_points"] + hi["fixed_points"]]
        events.append({"between": [lo["lambda"], hi["lambda"]], "kind": kind, "lo": lo, "hi": hi,
                       "near_minus_one": bool(any(abs(m + 1) < 0.2 for m in mults)),
                       "near_plus_one": bool(any(abs(m - 1) < 0.2 for m in mults))})
        print(json.dumps({"event": events[-1]["between"], "kind": kind}), flush=True)
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "r2scan.json").write_text(json.dumps({"args": vars(args), "ref_index": i, "w": model_r2.w,
                                                 "rows": rows, "events": events}, indent=1))


def arclength(X, k=8, start=None):
    """Geodesic (arclength) coordinate along a sampled curve: shortest-path distances in the k-nearest-neighbour
    graph of the points X (n, d) from the point `start` (index), normalized to [0, 1]. Raises when the graph is
    disconnected (the points do not sample one arc)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    tree = cKDTree(X)
    dist, idx = tree.query(X, k=k + 1)
    n = len(X)
    rows = np.repeat(np.arange(n), k)
    G = csr_matrix((dist[:, 1:].ravel(), (rows, idx[:, 1:].ravel())), shape=(n, n))
    G = G.maximum(G.T)
    d = dijkstra(G, indices=start)
    if not np.all(np.isfinite(d)):
        raise ValueError("the neighbour graph is disconnected")
    return d / d.max()


def project_to_chain(C, Y):
    """For each row of Y, the arclength position on the polyline C (m, d) (cumulative length normalized to [0, 1])
    of the nearest point, and the distance to it."""
    seg = np.diff(C, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    out_t, out_d = [], []
    for y in Y:
        w = y[None] - C[:-1]
        u = np.clip(np.einsum("ij,ij->i", w, seg) / np.maximum(L ** 2, 1e-300), 0, 1)
        P = C[:-1] + u[:, None] * seg
        dd = np.linalg.norm(P - y[None], axis=1)
        j = int(np.argmin(dd))
        out_t.append((cum[j] + u[j] * L[j]) / cum[-1])
        out_d.append(dd[j])
    return np.array(out_t), np.array(out_d), cum[-1]


def curve1d(args):
    """R1 on the attractor's own curve: arclength coordinate tau along the section states of a chaotic record,
    the data return map (tau_n, tau_{n+1}), and the model's map on a dense tau grid: states interpolated along the
    ordered curve, pushed once through G = F^period, projected back onto the curve (with the transverse distance)."""
    model = em.Model(args.threads)
    a, b = em.TOKENS[args.pair[0]], em.TOKENS[args.pair[1]]
    S = np.load(em.OUT / args.ref / "section.npz")
    sec = json.loads((em.OUT / args.ref / "section.json").read_text())
    X, s = S["states"], S["s"]
    tau = arclength(X, k=args.k, start=int(np.argmax(s)))
    order = np.argsort(tau)
    C = X[order]
    # thin the chain to points at least `spacing` apart in arclength (noise in ordering at tiny spacing)
    keep = [0]
    for i in range(1, len(C)):
        if np.linalg.norm(C[i] - C[keep[-1]]) > args.spacing * np.linalg.norm(C[-1] - C[0]) / len(C):
            keep.append(i)
    C = C[keep]
    t_data, d_data, length = project_to_chain(C, X)
    pairs = np.column_stack([t_data[:-1], t_data[1:]])
    cell = model.cell(sec["args"]["alpha"], sec["args"]["log_t"])
    tg = np.linspace(0, 1, args.n)
    seg_len = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    seg_len /= seg_len[-1]
    Xg = np.array([np.array([np.interp(t, seg_len, C[:, j]) for j in range(C.shape[1])]) for t in tg]) \
        if C.shape[1] < 0 else np.stack([np.interp(tg, seg_len, C[:, j]) for j in range(C.shape[1])], axis=1)
    Yg = cell.Fp(Xg, args.period)
    t_img, d_img, _ = project_to_chain(C, Yg)
    sg, _ = logodds(cell, Xg, a, b)
    g = en.GridMap(tg, t_img)
    rec = {"args": vars(args), "chain_points": int(len(C)), "curve_length": float(length),
           "curve_length_relative": float(length / np.linalg.norm(X.mean(0))),
           "data_transverse_max_relative": float(d_data.max() / length),
           "model_transverse_max_relative": float(d_img.max() / length),
           "model_transverse_median_relative": float(np.median(d_img) / length),
           "lyapunov_twin_per_step": sec["lyapunov_twin"], "lyapunov_twin_per_return": 3 * sec["lyapunov_twin"]
           if args.period == 3 else args.period * sec["lyapunov_twin"],
           "extrema": g.extrema()}
    try:
        rec["lyapunov_R1_per_return"] = g.lyapunov(0.37, transient=2000, steps=200000)
    except ValueError as err:
        rec["lyapunov_R1_per_return"] = str(err)
    for per in (1, 2, 3):
        rec[f"orbits_{per}"] = g.periodic_orbits(per, grid=4000)
    # data map residual against the model map
    rec["data_vs_model_max_abs"] = float(np.abs(g(pairs[:, 0]) - pairs[:, 1]).max())
    rec["data_vs_model_median_abs"] = float(np.median(np.abs(g(pairs[:, 0]) - pairs[:, 1])))
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "curve1d.npz", tau_grid=tg, tau_image=t_img, transverse=d_img, s_grid=sg,
                        pairs=pairs, chain=C)
    (out / "curve1d.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in rec if k not in ("args", "orbits_3")}))


def smooth_curve(X, tau, m=10, n=2001):
    """A smooth version of a sampled curve: PCA to m components, smoothing splines (GCV) of each component against
    the geodesic coordinate tau, re-parametrized by the arclength of the smooth curve. Returns (mean, V (m, d),
    dense smooth points in PC coordinates (n, m), their arclength parameter sigma in [0, 1])."""
    from scipy.interpolate import make_smoothing_spline

    mean = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    V = Vt[:m]
    P = (X - mean) @ V.T
    o = np.argsort(tau)
    t = tau[o]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, Po = t[keep], P[o][keep]
    tg = np.linspace(0, 1, n)
    Pg = np.stack([make_smoothing_spline(t, Po[:, j])(tg) for j in range(m)], axis=1)
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Pg, axis=0), axis=1))])
    return mean, V, Pg, L / L[-1]


def curve1d_smooth(args):
    """R1 on a smoothed attractor curve (see smooth_curve): the model's return map sigma -> sigma' on a dense
    grid, its fixed point and period-2 orbit with multipliers, and the full map's period-3 and period-6 orbits
    solved from the curve points at those positions (Newton-Picard, AD multipliers) for comparison; the R1
    Lyapunov exponent against the twin exponent of the section record."""
    model = em.Model(args.threads)
    S = np.load(em.OUT / args.ref / "section.npz")
    sec = json.loads((em.OUT / args.ref / "section.json").read_text())
    X, s = S["states"], S["s"]
    tau = arclength(X, k=args.k, start=int(np.argmax(s)))
    mean, V, Pg, sig = smooth_curve(X, tau, m=args.m)
    sg = np.linspace(0, 1, args.n)
    Pn = np.stack([np.interp(sg, sig, Pg[:, j]) for j in range(Pg.shape[1])], axis=1)
    Xn = mean + Pn @ V
    cell = model.cell(sec["args"]["alpha"], sec["args"]["log_t"])
    Yn = cell.Fp(Xn, args.period)

    def chart(Y):
        Q = (Y - mean) @ V.T
        off = np.linalg.norm((Y - mean) - Q @ V, axis=1)          # outside the m-dimensional subspace
        t, d, _ = project_to_chain(Pg, Q)
        return t, d, off

    t_img, d_img, off_img = chart(Yn)
    t_dat, d_dat, off_dat = chart(X)
    Lc = float(np.sum(np.linalg.norm(np.diff(Pg, axis=0), axis=1)))
    g = en.GridMap(sg, t_img)
    rec = {"args": vars(args), "curve_length": Lc,
           "model_transverse_median_relative": float(np.median(np.hypot(d_img, off_img)) / Lc),
           "model_transverse_max_relative": float(np.max(np.hypot(d_img, off_img)) / Lc),
           "data_offcurve_median_relative": float(np.median(np.hypot(d_dat, off_dat)) / Lc),
           "data_vs_model_median_abs": float(np.median(np.abs(g(t_dat[:-1]) - t_dat[1:]))),
           "data_vs_model_max_abs": float(np.max(np.abs(g(t_dat[:-1]) - t_dat[1:]))),
           "extrema": g.extrema(), "lyapunov_twin_per_return": args.period * sec["lyapunov_twin"]}
    xs, x = [], 0.3
    for _ in range(200000):
        x = float(np.clip(g(x), 0, 1))
        xs.append(x)
    xs = np.array(xs[1000:])
    rec["lyapunov_R1_per_return"] = float(np.mean(np.log(np.abs(g.deriv(xs[:-1])))))
    rec["orbits_1"] = g.periodic_orbits(1)
    rec["orbits_2"] = g.periodic_orbits(2)
    full = []
    for per, key in ((1, "orbits_1"), (2, "orbits_2")):
        for o in rec[key][:1]:
            t0 = o["points"][0]
            x0 = mean + np.array([np.interp(t0, sg, Pn[:, j]) for j in range(Pn.shape[1])]) @ V
            xo, info, _ = cell.solve(x0, per * args.period, k=6, ad=True)
            full.append({"reduced_period": per, "full_period": per * args.period, "R1_multiplier": o["multiplier"],
                         "full_converged": info["converged"], "full_residual": info["residuals"][-1],
                         "full_multipliers": info.get("multipliers", [])[:4],
                         "full_classified": info.get("crossing_side")})
            print(json.dumps(full[-1]), flush=True)
    rec["full_orbits"] = full
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "curve1d-smooth.npz", sigma=sg, sigma_image=t_img, data_sigma=t_dat, Yn=Yn, Xn=Xn,
                        transverse=np.hypot(d_img, off_img))
    (out / "curve1d-smooth.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in rec if k not in ("args", "orbits_2", "extrema")}))


def along(args):
    """What varies along the attractor curve: at every section state (ordered by the geodesic coordinate tau),
    the pair's log-odds s, the pair's total probability w, the leading tokens, and the same at the next two
    phases; binned summaries in tau."""
    model = em.Model(args.threads)
    a, b = em.TOKENS[args.pair[0]], em.TOKENS[args.pair[1]]
    S = np.load(em.OUT / args.ref / "section.npz")
    sec = json.loads((em.OUT / args.ref / "section.json").read_text())
    X, s = S["states"], S["s"]
    tau = arclength(X, k=8, start=int(np.argmax(s)))
    cell = model.cell(sec["args"]["alpha"], sec["args"]["log_t"])
    phases = [X]
    for _ in range(args.period - 1):
        phases.append(cell.F(phases[-1]))
    bins = np.linspace(0, 1, args.bins + 1)
    rows = []
    for k, Xk in enumerate(phases):
        P = np.concatenate([cell.probs(Xk[i:i + 100]) for i in range(0, len(Xk), 100)])
        for j in range(args.bins):
            m = (tau >= bins[j]) & (tau <= bins[j + 1])
            if not m.any():
                continue
            pm = P[m].mean(0)
            top = np.argsort(-pm)[:5]
            rows.append({"phase": k, "tau_bin": [float(bins[j]), float(bins[j + 1])], "n": int(m.sum()),
                         "s_mean": float(s[m].mean()), "w_mean": float((P[m][:, a] + P[m][:, b]).mean()),
                         "top": [[model.decode(t), float(pm[t])] for t in top]})
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "along.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1, ensure_ascii=False))
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


def r1_map(cell, a, b, period, ends, n, project):
    """R1: n states on the segment between the two section states `ends`, pushed `project` times through
    G = F^period, then the log-odds before and after one more G. Returns (s, s', spread) with s sorted; spread is
    the largest relative state distance between neighbours after projection (how far the curve was resolved)."""
    t = np.linspace(0.0, 1.0, n)
    X = ends[0][None] * (1 - t[:, None]) + ends[1][None] * t[:, None]
    for _ in range(project):
        X = cell.Fp(X, period)
    s, _ = logodds(cell, X, a, b)
    Y = cell.Fp(X, period)
    s2, _ = logodds(cell, Y, a, b)
    return s, s2, X, Y


def r1(args):
    model = em.Model(args.threads)
    a, b = em.TOKENS[args.pair[0]], em.TOKENS[args.pair[1]]
    S = np.load(em.OUT / args.ref / "section.npz")
    lo, hi = int(np.argmin(S["s"])), int(np.argmax(S["s"]))
    ends = (S["states"][lo], S["states"][hi])
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = [(args.cells[j], args.cells[j + 1]) for j in range(0, len(args.cells), 2)]
    for alpha, log_t in cells:
        cell = model.cell(alpha, log_t)
        s, s2, X, Y = r1_map(cell, a, b, args.period, ends, args.n, args.project)
        order = np.argsort(s)
        s, s2 = s[order], s2[order]
        # single-valuedness: neighbours in s whose images differ (the curve folded back on itself)
        keep = np.concatenate([[True], np.diff(s) > 1e-9 * (s.max() - s.min())])
        s, s2 = s[keep], s2[keep]
        smooth = np.polyfit((s - s.mean()) / s.std(), s2, 12)
        resid = float(np.abs(np.polyval(smooth, (s - s.mean()) / s.std()) - s2).max() / (s2.max() - s2.min()))
        rec = {"alpha": alpha, "log10_temperature": log_t, "s_range": [float(s.min()), float(s.max())],
               "points": int(len(s)), "poly12_residual_of_range": resid}
        try:
            g = en.GridMap(s, s2)
            rec["extrema"] = g.extrema()
            for per in args.periods:
                rec[f"orbits_{per}"] = g.periodic_orbits(per, grid=args.grid)
            rec["lyapunov_per_return"] = g.lyapunov(float(np.median(s)), transient=2000, steps=20000)
        except ValueError as err:
            rec["error"] = str(err)
        rows.append(rec)
        np.savez_compressed(out / f"r1-a{alpha:.7f}-t{log_t:.5f}.npz", s=s, s2=s2)
        print(json.dumps({k: rec.get(k) for k in ("alpha", "log10_temperature", "s_range", "poly12_residual_of_range",
                                                     "extrema", "lyapunov_per_return", "error")}), flush=True)
    (out / "r1.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("section")
    s.add_argument("--alpha", type=float, required=True)
    s.add_argument("--log-t", type=float, required=True)
    s.add_argument("--pair", nargs=2, required=True, choices=sorted(em.TOKENS))
    s.add_argument("--period", type=int, required=True, help="return time of the section (loop length)")
    s.add_argument("--seed-state", default=None)
    s.add_argument("--transient", type=int, default=2000)
    s.add_argument("--steps", type=int, default=3000)
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--out", required=True)
    for name in ("r1", "r2"):
        r = sub.add_parser(name)
        r.add_argument("--ref", required=True, help="a section run (explore/e2e-relative)")
        r.add_argument("--ref-index", type=int, default=-1)
        r.add_argument("--pair", nargs=2, required=True, choices=sorted(em.TOKENS))
        r.add_argument("--period", type=int, required=True)
        r.add_argument("--cells", type=float, nargs="+", required=True)
        r.add_argument("--n", type=int, default=401)
        r.add_argument("--project", type=int, default=3)
        r.add_argument("--periods", type=int, nargs="*", default=[1, 2, 4])
        r.add_argument("--grid", type=int, default=4000)
        r.add_argument("--threads", type=int, default=8)
        r.add_argument("--out", required=True)
    c = sub.add_parser("curve1d")
    c.add_argument("--ref", required=True)
    c.add_argument("--pair", nargs=2, required=True, choices=sorted(em.TOKENS))
    c.add_argument("--period", type=int, required=True)
    c.add_argument("--k", type=int, default=8)
    c.add_argument("--spacing", type=float, default=0.3)
    c.add_argument("--n", type=int, default=801)
    c.add_argument("--threads", type=int, default=8)
    c.add_argument("--out", required=True)
    c = sub.add_parser("curve1d-smooth")
    c.add_argument("--ref", required=True)
    c.add_argument("--period", type=int, required=True)
    c.add_argument("--k", type=int, default=8)
    c.add_argument("--m", type=int, default=10)
    c.add_argument("--n", type=int, default=801)
    c.add_argument("--threads", type=int, default=8)
    c.add_argument("--out", required=True)
    c = sub.add_parser("along")
    c.add_argument("--ref", required=True)
    c.add_argument("--pair", nargs=2, required=True, choices=sorted(em.TOKENS))
    c.add_argument("--period", type=int, required=True)
    c.add_argument("--bins", type=int, default=10)
    c.add_argument("--threads", type=int, default=8)
    c.add_argument("--out", required=True)
    r = sub.add_parser("r2scan")
    r.add_argument("--ref", required=True)
    r.add_argument("--ref-index", type=int, default=-1)
    r.add_argument("--pair", nargs=2, required=True, choices=sorted(em.TOKENS))
    r.add_argument("--period", type=int, required=True)
    r.add_argument("--fixed", choices=("alpha", "log_t"), required=True)
    r.add_argument("--value", type=float, required=True)
    r.add_argument("--range", type=float, nargs=2, required=True)
    r.add_argument("--steps", type=int, default=25)
    r.add_argument("--bisect", type=int, default=12)
    r.add_argument("--n", type=int, default=401)
    r.add_argument("--grid", type=int, default=4000)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    {"section": section, "r1": r1, "r2": r2, "r2scan": r2scan, "curve1d": curve1d,
     "curve1d-smooth": curve1d_smooth, "along": along}[args.cmd](args)


if __name__ == "__main__":
    main()
