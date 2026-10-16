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
    {"section": section, "r1": r1, "r2": r2, "r2scan": r2scan}[args.cmd](args)


if __name__ == "__main__":
    main()
