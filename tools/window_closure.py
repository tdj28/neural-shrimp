"""Is the return dynamics near the period-5 window operationally two-dimensional? (run CL2; consult 10, direction 2)

At each frozen location (docs/cl/cl-locations.json: development and evaluation locations), the attracting period-5
orbit x* of the two-block soft-token map (float64; tools/window_census.setup) is solved by the low-rank Newton from a
seed state, and the leading Ritz pairs of its return map P = F^5 give a real orthonormal basis B of the leading
invariant subspace (a complex pair contributes its real and imaginary parts). Finite perturbations
x_0 = x* + A |x*| d (d random unit vectors; A in AMPLITUDES) are followed for R returns, SEEDS seeds per amplitude, half
to fit and half to test. Coordinates y = B_k^T (x - x*) for k = 1, 2, 4. Models, fitted on the training seeds of a
location (both amplitudes pooled):
- the k-coordinate return model y_{r+1} = f(y_r), a quadratic polynomial (least squares, in units of the training
  coordinates' RMS);
- a lifting map from y back to the state deviation, linear plus quadratic in y (least squares);
- baselines: the linearized return map on the Ritz subspace (delta_{r+1} = M delta_r with M = V diag(mu) V^+), and the
  fixed-orbit predictor (delta = 0, error 1 by definition).
Forecasts on test seeds start from the first return's coordinates y_1 and run autonomously to return R. Error per seed:
E = sum_{r=2..R} |dhat_r - d_r|^2 / sum_{r=2..R} |d_r|^2. Operational gates (consult 10): for k = 2, the median E below
0.05, at least a twofold improvement over k = 1, and less than 20 % further improvement from k = 4. The same pipeline
runs on the synthetic controls of fit_and_score (tests/test_window_closure.py): a Henon map embedded in R^10 with
strongly contracting extra coordinates (two coordinates must pass) and a map whose slow third coordinate drives the
other two (two coordinates must not).

Usage:
  python tools/window_closure.py run --locations docs/cl/cl-locations.json --device cuda --out llm-soft-loop/cl/run
  python tools/window_closure.py summary --run llm-soft-loop/cl/run [--json OUT]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

AMPLITUDES = (1e-3, 1e-2)
SEEDS = 64
RETURNS = 8
KS = (1, 2, 4)


def quad_features(Y):
    """[1, y_i, y_i y_j (i <= j)] for rows of Y (n, k)."""
    n, k = Y.shape
    cols = [np.ones((n, 1)), Y]
    for i in range(k):
        for j in range(i, k):
            cols.append((Y[:, i] * Y[:, j])[:, None])
    return np.hstack(cols)


def modal_coordinates(vals, V, k):
    """Coordinates on the leading k-dimensional invariant subspace, as a (k, n) real matrix P with y = P delta: the
    modal amplitudes a = V^+ delta of the leading eigenvectors (sorted by modulus), i.e. the spectral projection along
    the other invariant directions, not an orthogonal projection (the return map is non-normal, and an orthogonal
    projection would mix faster modes into the leading coordinate). A complex pair contributes the real and imaginary
    parts of its amplitude; the rows are trimmed to k."""
    Vp = np.linalg.pinv(V)
    rows, i = [], 0
    while len(rows) < k and i < len(vals):
        if abs(vals[i].imag) > 1e-12 * max(1.0, abs(vals[i])):
            rows += [Vp[i].real, Vp[i].imag]
            i += 2
        else:
            rows.append(Vp[i].real)
            i += 1
    return np.array(rows[:k])


def fit_and_score(D_train, D_test, P, M=None, ks=KS):
    """D: deviation trajectories (seeds, R+1, n) from the orbit, return 0 to R. P: (kmax, n) coordinate map (leading
    first, from modal_coordinates). M: optional (n, n) or callable linear return map for the linear baseline. Returns
    per-k test errors per seed (forecast from return 1), and the linear baseline's."""
    out = {}
    for k in ks:
        Pk = P[:k]
        Ytr = np.einsum("srn,kn->srk", D_train, Pk)
        Yte = np.einsum("srn,kn->srk", D_test, Pk)
        scale = np.sqrt(np.mean(Ytr[:, 1:] ** 2)) or 1.0
        Ytr, Yte = Ytr / scale, Yte / scale
        Xin = Ytr[:, 1:-1].reshape(-1, k)
        Xout = Ytr[:, 2:].reshape(-1, k)
        F = quad_features(Xin)
        coef, *_ = np.linalg.lstsq(F, Xout, rcond=None)
        G = quad_features(Ytr[:, 1:].reshape(-1, k))[:, 1:]          # lifting: no constant term
        lift, *_ = np.linalg.lstsq(G, D_train[:, 1:].reshape(-1, D_train.shape[2]), rcond=None)
        errs = []
        for s in range(D_test.shape[0]):
            y = Yte[s, 1]
            num = den = 0.0
            for r in range(2, D_test.shape[1]):
                y = quad_features(y[None])[0] @ coef
                dhat = quad_features(y[None])[0, 1:] @ lift
                num += float(np.sum((dhat - D_test[s, r]) ** 2))
                den += float(np.sum(D_test[s, r] ** 2))
            errs.append(num / den if den > 0 else np.nan)
        out[k] = errs
    if M is not None:
        errs = []
        for s in range(D_test.shape[0]):
            d = D_test[s, 1]
            num = den = 0.0
            for r in range(2, D_test.shape[1]):
                d = M(d) if callable(M) else M @ d
                num += float(np.sum((d - D_test[s, r]) ** 2))
                den += float(np.sum(D_test[s, r] ** 2))
            errs.append(num / den if den > 0 else np.nan)
        out["linear"] = errs
    return out


def gates(errs):
    """Operational two-dimensional closure (consult 10): median E(k=2) < 0.05, E1/E2 >= 2, and E4 not below 0.8 E2.
    A median error above 1 (worse than forecasting zero) or not finite marks a degenerate fit, not a measurement: the
    iterated quadratic model diverged (added after run CL2, where E4 reached 1e122 and so passed "E4 >= 0.8 E2"). Then
    no closure flag is set."""
    m = {k: float(np.nanmedian(v)) for k, v in errs.items()}
    degenerate = sorted(k for k in m if not np.isfinite(m[k]) or m[k] > 1.0)
    ok = not degenerate
    dim = lambda eps: next((k for k in sorted(m) if m[k] < eps), None)          # noqa: E731
    return {"median": m, "degenerate_k": degenerate, "e2_below_0.05": m[2] < 0.05, "two_better_than_one_2x": m[1] >= 2 * m[2],
            "four_adds_less_than_20pct": m[4] >= 0.8 * m[2],
            "closure_2d": bool(ok and m[2] < 0.05 and m[1] >= 2 * m[2] and m[4] >= 0.8 * m[2]),
            "closure_1d": bool(ok and m[1] < 0.05 and m[2] >= 0.8 * m[1]),
            # secondary (added after the CPU smoke test at the body location; disclosed): the smallest number of
            # coordinates whose median error is below an absolute level
            "dimension_at_0.05": dim(0.05) if ok else None, "dimension_at_0.005": dim(0.005) if ok else None}


def run(args):
    import explore_llm_attractors as ela
    import llm_feigenbaum as lfb
    import window_census as wc
    spec = json.loads(Path(args.locations).read_text())
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "closure.jsonl"
    if path.exists():
        raise SystemExit(f"{path} exists; outputs are never overwritten")
    loop, make_cell = wc.setup(args.device, args.threads)
    torch = loop.torch
    rng = np.random.default_rng(args.seed)
    tol = lfb.tol_for(wc.PERIOD, 1e-12)
    tic = time.perf_counter()
    locations = []
    for loc in spec["locations"]:                        # census-seeded entries: one location per attracting orbit
        if "census_point" not in loc:
            locations.append(loc)
            continue
        src = ela.DATA / "explore" / loc["seed_run"]
        rec = next(json.loads(line) for line in open(src / "census.jsonl")
                   if json.loads(line)["name"] == loc["census_point"])
        for k, o in enumerate(rec["orbits"]):
            if o["attracting"]:
                locations.append({**loc, "name": f"{loc['name']}-orbit{k}", "alpha": rec["alpha"],
                                  "alpha2": rec["alpha2"], "seed_file": f"orbits-{loc['census_point']}.npy",
                                  "seed_index": k})
    for loc in locations:
        cell = make_cell(loc["alpha"], loc["alpha2"])
        src = ela.DATA / "explore" / loc["seed_run"]
        x0 = np.load(src / loc.get("seed_file", "states.npy"))[loc["seed_index"]].ravel()
        n = x0.size
        Q0 = np.linalg.qr(rng.normal(size=(n, args.k)))[0]
        X, Q, hist = lfb.newton(cell, wc.to_state(loop, x0), wc.PERIOD, Q0, tol=tol, max_iter=args.max_newton)
        rec = {**loc, "newton_residual": hist[-1], "newton_iterations": len(hist) - 1}
        if not hist[-1] < tol or lfb.return_distance(cell, X, 1) < 1e-6:
            rec["status"] = "no primitive period-5 orbit from the seed"
            with open(path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps({"name": loc["name"], "status": rec["status"]}), flush=True)
            continue
        vals, V, res, hh, Q = lfb.ritz(cell, X, wc.PERIOD, Q, passes=6)
        xs = lfb._flat(X)
        B = modal_coordinates(vals, V, max(KS))
        Vp = np.linalg.pinv(V)
        M = lambda d, V=V, vals=vals, Vp=Vp: np.real(V @ (vals * (Vp @ d)))          # noqa: E731
        rec.update({"tokens": wc.token_loop(cell, X), "multipliers": [[float(z.real), float(z.imag)] for z in vals],
                    "ritz_residuals": [float(r) for r in res], "attracting": bool(abs(vals[0]) < 1)})
        trajs = {}
        for A in AMPLITUDES:
            dirs = rng.normal(size=(SEEDS, n))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            Xb = torch.as_tensor((xs + A * np.linalg.norm(xs) * dirs).reshape(SEEDS, 1, n), dtype=loop.state_dtype,
                                 device=loop.device)
            D = np.zeros((SEEDS, RETURNS + 1, n))
            D[:, 0] = Xb.reshape(SEEDS, n).double().cpu().numpy() - xs
            for r in range(1, RETURNS + 1):
                for _ in range(wc.PERIOD):
                    Xb = cell.f(Xb)
                D[:, r] = Xb.reshape(SEEDS, n).double().cpu().numpy() - xs
            trajs[A] = D
            rec[f"relative_deviation_by_return_A{A:g}"] = (np.median(np.linalg.norm(D, axis=2), axis=0) /
                                                           np.linalg.norm(xs)).tolist()
        half = SEEDS // 2
        Dtr = np.concatenate([trajs[A][:half] for A in AMPLITUDES])
        scores = {}
        for A in AMPLITUDES:
            e = fit_and_score(Dtr, trajs[A][half:], B, M)
            scores[f"A{A:g}"] = {"errors": {str(k): v for k, v in e.items()},
                                 "gates": gates({k: v for k, v in e.items() if k != "linear"}),
                                 "linear_median": float(np.nanmedian(e["linear"]))}
        rec["scores"] = scores
        rec["seconds"] = round(time.perf_counter() - tic, 1)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({"name": loc["name"], "role": loc["role"], "lead": [round(abs(z), 4) for z in vals[:3]],
                          **{a: s["gates"]["median"] for a, s in scores.items()}, "seconds": rec["seconds"]}),
              flush=True)


def summary(args):
    import explore_llm_attractors as ela
    recs = [json.loads(line) for line in open(ela.DATA / "explore" / args.run / "closure.jsonl")]
    rows = []
    for r in recs:
        if "scores" not in r:
            rows.append({"name": r["name"], "role": r["role"], "status": r.get("status")})
            continue
        cur = {a: gates({int(k): v for k, v in s["errors"].items() if k != "linear"}) for a, s in r["scores"].items()}
        rows.append({"name": r["name"], "role": r["role"], "region": r.get("region"), "predicted": r.get("predicted"),
                     "leading_moduli": [round(float(np.hypot(*z)), 5) for z in r["multipliers"][:3]],
                     **{a: {"median": cur[a]["median"], "linear_median": s["linear_median"],
                            "degenerate_k": cur[a]["degenerate_k"],
                            "closure_2d": cur[a]["closure_2d"], "closure_1d": cur[a]["closure_1d"],
                            "dimension_at_0.05": cur[a]["dimension_at_0.05"],
                            "dimension_at_0.005": cur[a]["dimension_at_0.005"],
                            # flags as computed on the pod, before the degenerate-fit guard
                            "as_run": {"closure_2d": s["gates"]["closure_2d"],
                                       "closure_1d": s["gates"].get("closure_1d")}}
                        for a, s in r["scores"].items()}})
    text = json.dumps({"locations": rows}, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--locations", required=True)
    r.add_argument("--k", type=int, default=8, help="Ritz subspace dimension")
    r.add_argument("--max-newton", type=int, default=30)
    r.add_argument("--seed", type=int, default=2015)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--device", default="cpu")
    r.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--run", required=True)
    s.add_argument("--json")
    args = ap.parse_args()
    {"run": run, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
