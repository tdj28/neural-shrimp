"""Run E2E, Q1: periodic orbits of the soft-token map continued in one parameter by natural continuation with a
secant predictor and Newton-Picard correction (float64; tools/e2e_model.py), with AD multipliers at every point.

tcont   at fixed alpha, in log10 T: downward toward T = 0 (multipliers, distance of each orbit state from the nearest
        token embedding, logit margins) and upward to the fold, located from (1 - mu)^2 on the stable branch
acont   at fixed log10 T, in alpha, to the fold on either side (the boundaries of the window at that temperature)

The orbit is seeded by iterating the map from a token embedding (--seed-token), from the opener (--seed fixed) or
from a saved state (--seed-state, a .npy of one state), and its period is verified by return distances before the
continuation starts (no default period is substituted).

Usage: python tools/e2e_branches.py tcont --alpha 0.0628 --log-t -0.13 --period 2 --seed-token is --to -2.0 --out p2-tcont-a0.0628-down
       python tools/e2e_branches.py acont --log-t -0.5 --alpha 0.05 --period 2 --seed-token is --to 0.09 --out p2-acont-t-0.5-right
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


def describe(cell, x, p, k_sigma=3):
    """Per phase: leading tokens, distance of the state from the nearest embedding among the previous phase's top
    tokens, logit margin (top-1 minus top-2) in logit units and in units of T, and the leading eigenvalues of
    Sigma_p with the alignment of the first eigenvector with e_top1 - e_top2."""
    pts = cell.orbit_points(x, p)
    Z = cell.logits(np.array(pts))
    E = cell.m.E
    phases = []
    for i, (xi, z) in enumerate(zip(pts, Z)):
        zz = z / cell.T
        zz = zz - zz.max()
        pr = np.exp(zz)
        pr /= pr.sum()
        top = np.argsort(-z)[:3]
        prev = Z[i - 1]
        cand = np.argsort(-prev)[:5]
        dist = [float(np.linalg.norm(xi - E[t]) / np.linalg.norm(E[t])) for t in cand]
        j = int(np.argmin(dist))
        w, U, tr, ns = en.covariance_spectrum(E, pr, tol=1e-12, k=k_sigma)
        phases.append({"top": [[cell.m.decode(t), float(pr[t])] for t in top],
                       "nearest_embedding": cell.m.decode(cand[j]), "relative_distance": dist[j],
                       "margin": float(z[top[0]] - z[top[1]]), "margin_over_T": float((z[top[0]] - z[top[1]]) / cell.T),
                       "sigma_eigs": [float(a) for a in w], "sigma_trace": tr, "support": ns,
                       "sigma_lead_alignment_top12": en.pair_alignment(U[:, 0], E[top[0]], E[top[1]])})
    return phases


def seed_state(cell, args):
    if args.seed_state:
        return np.load(args.seed_state).reshape(-1)
    if args.seed_token:
        return cell.m.E[em.TOKENS[args.seed_token]].copy()
    return cell.start.copy()


def verify_period(cell, x, p, transient, tol=1e-7, max_period=48):
    X = cell.Fp(x, transient) if transient else x
    hist = [X]
    for _ in range(max_period):
        hist.append(cell.F(hist[-1])[0])
    d = [float(np.linalg.norm(hist[q] - X) / np.linalg.norm(X)) for q in range(1, max_period + 1)]
    found = next((q for q, dq in enumerate(d, 1) if dq < tol), 0)
    return X, found, d


def continue_branch(model, fixed, free, lam0, x0, p, to, step, step_min, step_max, out, k=6, mu_stop=0.995,
                    log=print, max_points=400, ad=True, name="branch.json", through=False,
                    merge_tol=0.02):
    """Natural continuation of a period-p orbit in `free` ('log_t' or 'alpha') from lam0 toward `to`."""
    def make(lam):
        return model.cell(lam, fixed) if free == "alpha" else model.cell(fixed, lam)

    sign = 1.0 if to > lam0 else -1.0
    records, states = [], []
    lam, x, Q = lam0, x0, None
    prev = None
    h = step
    stop = "reached the end of the range"
    while len(records) < max_points:
        cell = make(lam)
        if prev is not None and len(states) >= 2:
            l1, l2 = records[-1]["lambda"], records[-2]["lambda"]
            guess = states[-1] + (states[-1] - states[-2]) * (lam - l1) / (l1 - l2)
        else:
            guess = x
        tic = time.perf_counter()
        xs, info, Qn = cell.solve(guess, p, k=k, Q0=Q, ad_iters=6, ad=ad)
        if not info["converged"]:
            if h / 2 < step_min:
                stop = f"Newton-Picard failed at {lam:.10g} with the smallest step"
                break
            h /= 2
            lam = records[-1]["lambda"] + sign * h if records else lam0
            log(json.dumps({"retry": lam, "step": h, "residuals": info["residuals"][-3:]}))
            continue
        vals = np.array([complex(a, b) for a, b in info["multipliers"]])
        lead = vals[np.argsort(-np.abs(vals))][0]
        rec = {"index": len(records), "lambda": float(lam), "free": free, "fixed": float(fixed), "step": float(h),
               "residual": info["residuals"][-1], "newton_picard_iterations": len(info["residuals"]) - 1,
               "multipliers": info["multipliers"], "multipliers_converged": info["multipliers_converged"],
               "leading": [float(lead.real), float(lead.imag)], "crossing_side": info["crossing_side"],
               "phases": describe(cell, xs, p), "seconds": round(time.perf_counter() - tic, 1)}
        if records:                                    # the state must not jump to another orbit
            rec["state_change"] = float(np.linalg.norm(xs - states[-1]) / np.linalg.norm(xs))
        if p % 2 == 0:                                 # distance from the half-period orbit (a flip of it ahead)
            rec["half_period_distance"] = float(np.linalg.norm(cell.Fp(xs, p // 2) - xs) / np.linalg.norm(xs))
        records.append(rec)
        states.append(xs)
        Q = Qn
        prev = lam
        log(json.dumps({k2: rec[k2] for k2 in ("index", "lambda", "residual", "leading", "seconds")} |
                       {"tokens": [ph["top"][0][0] for ph in rec["phases"]],
                        "dist": [round(ph["relative_distance"], 6) for ph in rec["phases"]]}, ensure_ascii=False))
        np.save(out / name.replace(".json", "-states.npy"), np.array(states))
        (out / name).write_text(json.dumps({"records": records, "stop": "running"}, indent=1, ensure_ascii=False))
        mod = abs(lead)
        if mod >= 1.0 and not through:
            stop = f"left the stable side (|mu| = {mod:.6f})"
            break
        if rec.get("half_period_distance", 1.0) < merge_tol:
            stop = f"approaching the half-period orbit (distance {rec['half_period_distance']:.3g})"
            break
        if abs(lead.imag) < 1e-9 and lead.real > mu_stop:
            stop = f"leading multiplier {lead.real:.6f} above {mu_stop}"
            break
        if sign * (to - lam) <= 1e-15:
            break
        # step control: near a fold (1 - mu)^2 is linear in the parameter, so the remaining distance is estimated
        # from the last two points and the step kept to half of it; elsewhere the step grows
        h = min(step_max, h * 1.5)
        if through and len(records) >= 2:      # through instabilities: keep the step where |mu| changes fast
            prev_mod = abs(complex(*records[-2]["leading"]))
            dm = abs(np.log(max(mod, 1e-300)) - np.log(max(prev_mod, 1e-300)))      # relative change of |mu|
            if dm > 0.3:
                h = max(step_min, h / 2)
        elif len(records) >= 2 and mod > 0.6:
            m1, m2 = complex(*records[-2]["leading"]), lead
            if abs(m1.imag) < 1e-9 and abs(m2.imag) < 1e-9 and 0 < m1.real < 1 and 0 < m2.real < 1:
                y1, y2 = (1 - m1.real) ** 2, (1 - m2.real) ** 2
                dl = records[-1]["lambda"] - records[-2]["lambda"]
                if y2 < y1 and dl != 0:
                    remaining = y2 * abs(dl) / (y1 - y2)
                    h = max(step_min, min(h, remaining / 2))
            elif mod > 0.9:
                h = max(step_min, min(h, (1 - mod) * step_max))
        lam = lam + sign * h
        if sign * (lam - to) > 0:
            lam = to
    return records, states, stop


def fold_estimate(records):
    """(1 - mu)^2 extrapolation from the last points of a branch that approaches +1 from below; None (with the
    reason) when the last points do not qualify."""
    pts = [(r["lambda"], complex(*r["leading"])) for r in records]
    good = [(l, m) for l, m in pts if abs(m.imag) < 1e-9 and 0.5 < m.real < 1]
    if len(good) < 3:
        return {"fold": None, "reason": f"only {len(good)} final points with real multiplier in (0.5, 1)"}
    use = good[-4:]
    lam_c, slope, res = en.fold_extrapolate([u[0] for u in use], [u[1] for u in use], use=len(use))
    last = records[-1]
    return {"fold": lam_c, "slope": slope, "fit_residual": res, "points_used": len(use),
            "last_multiplier": last["leading"], "last_next_modulus": last["crossing_side"]["next_modulus"]}


def run(args, free, fixed, lam0):
    model = em.Model(args.threads)
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cell = model.cell(lam0, fixed) if free == "alpha" else model.cell(fixed, lam0)
    x = seed_state(cell, args)
    X, found, d = verify_period(cell, x, args.period, args.transient)
    if found != args.period:
        raise SystemExit(f"seed reached period {found} (wanted {args.period}); return distances {d[:12]}")
    logf = open(out / "progress.jsonl", "w")

    def log(line):
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    records, states, stop = continue_branch(model, fixed, free, lam0, X, args.period, args.to, args.step,
                                            args.step_min, args.step_max, out, k=args.k, mu_stop=args.mu_stop,
                                            log=log, ad=not args.fd, through=args.through)
    fold = fold_estimate(records) if records else {"fold": None, "reason": "no points"}
    res = {"args": vars(args), "free": free, "fixed": fixed, "period": args.period, "stop": stop, "fold": fold,
           "records": records}
    (out / "branch.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({"stop": stop, "points": len(records), "fold": fold}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tcont", "acont"):
        s = sub.add_parser(name)
        s.add_argument("--alpha", type=float, required=True)
        s.add_argument("--log-t", type=float, required=True)
        s.add_argument("--period", type=int, required=True)
        s.add_argument("--seed-token", default=None, choices=sorted(em.TOKENS))
        s.add_argument("--seed-state", default=None)
        s.add_argument("--transient", type=int, default=300)
        s.add_argument("--to", type=float, required=True)
        s.add_argument("--step", type=float, default=0.02 if name == "tcont" else 1e-3)
        s.add_argument("--step-min", type=float, default=1e-6 if name == "tcont" else 1e-8)
        s.add_argument("--step-max", type=float, default=0.1 if name == "tcont" else 2e-3)
        s.add_argument("--mu-stop", type=float, default=0.995)
        s.add_argument("--k", type=int, default=6)
        s.add_argument("--through", action="store_true", help="continue through losses of stability")
        s.add_argument("--fd", action="store_true", help="multipliers by central differences (faster) instead of AD")
        s.add_argument("--threads", type=int, default=8)
        s.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "tcont":
        run(args, "log_t", args.alpha, args.log_t)
    else:
        run(args, "alpha", args.log_t, args.alpha)


if __name__ == "__main__":
    main()
