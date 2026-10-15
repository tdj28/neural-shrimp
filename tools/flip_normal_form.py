"""Flip normal form of the soft-token map at a located period doubling, tested against the doubled orbit.

At a flip of the period-p orbit (a real multiplier -1 of the p-fold map, no other multiplier on the unit circle) the
centre-manifold reduction is exact whatever the size of the other multipliers: on the critical direction r (left
eigenvector l, l^T r = 1, |r| = 1) the p-fold map is eta -> -(1 + beta) eta + c eta^3 + O(eta^4) with

    c = l^T C(r, r, r) / 6 + l^T B(r, h2) / 2,    (I - M) h2 = B(r, r)

(Kuznetsov, Elements of Applied Bifurcation Theory, 3rd ed., sec. 5.4), M the monodromy and B, C the second and third
derivatives of the p-fold map, here central differences of float64 Jacobian-vector products. c is independent of the
orientation of r. c > 0: supercritical (a stable period-2p orbit where the period-p orbit is unstable); c < 0:
subcritical. Where the period-p orbit has multiplier -(1 + beta), beta > 0 small, the normal form predicts the
doubled orbit's half-amplitude along r, eta = sqrt(beta / c), and its multiplier, 1 - 4 beta (independent of c).
Both predictions are tested against Newton solves of the period-2p orbit at offsets past the flip.

The crossing is classified first (AGENTS.md): the tool refuses unless the multiplier nearest -1 is real and within
unit_tol of -1, and every other multiplier has modulus below `max_next`.

Usage: python tools/flip_normal_form.py --log-temperature 0.0319 --alpha-flip 0.07535341734032334 --period 2 \
    --start-alpha 0.0775 --side -1 --offsets 2.5e-5 5e-5 1e-4 2e-4 4e-4 --device cuda --out llm-soft-loop/pod64/flipnf/hiT-p2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import llm_feigenbaum as lfb  # noqa: E402


def flip_cubic(family, x, theta, steps=(1e-2, 3e-3, 1e-3, 3e-4), unit_tol=1e-6, max_next=0.95):
    """Classify the crossing at (x, theta) and return the flip coefficient c with its step table."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    M = family.monodromy(x, theta)
    vals, R = np.linalg.eig(M)
    i = int(np.argmin(np.abs(vals + 1)))
    mu = vals[i]
    others = np.delete(vals, i)
    nxt = float(np.abs(others).max()) if len(others) else 0.0
    crossing = {"multiplier": [float(mu.real), float(mu.imag)], "modulus": float(abs(mu)),
                "argument_turns": float(np.angle(mu) / (2 * np.pi)), "next_modulus": nxt}
    if abs(mu.imag) > unit_tol or abs(mu.real + 1) > unit_tol or nxt >= max_next:
        raise ValueError(f"not a flip: multiplier nearest -1 is {mu!r}, next modulus {nxt:.3g} (refusing coefficients)")
    valsL, L = np.linalg.eig(M.T)
    j = int(np.argmin(np.abs(valsL - mu)))
    r, l = np.real(R[:, i]), np.real(L[:, j])
    r = r / np.linalg.norm(r)
    l = l / float(l @ r)
    scale = float(np.linalg.norm(x)) or 1.0
    j0 = family.jvp_x(x, theta, r)
    IM = np.eye(n) - M
    table = []
    for h in steps:
        s = h * scale
        jp, jm = family.jvp_x(x + s * r, theta, r), family.jvp_x(x - s * r, theta, r)
        brr = (jp - jm) / (2 * s)
        crrr = (jp - 2 * j0 + jm) / s ** 2
        h2 = np.linalg.solve(IM, brr)
        brh = (family.jvp_x(x + s * r, theta, h2) - family.jvp_x(x - s * r, theta, h2)) / (2 * s)
        direct, centre = float(l @ crrr) / 6, 0.5 * float(l @ brh)
        table.append({"h_relative": h, "c": direct + centre, "c_direct": direct, "c_centre_manifold": centre,
                      "h2_norm": float(np.linalg.norm(h2))})
    cs = np.array([t["c"] for t in table])
    k = int(np.argmin(np.abs(np.diff(cs)))) if len(cs) > 1 else 0
    c = float(0.5 * (cs[k] + cs[k + 1])) if len(cs) > 1 else float(cs[0])
    return {"crossing": crossing, "c": c, "c_steps": table, "c_plateau_pair": [steps[k], steps[k + 1]],
            "c_plateau_relative_spread": float(abs(cs[k] - cs[k + 1]) / max(abs(c), 1e-300)), "r": r, "l": l}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log-temperature", type=float, required=True)
    ap.add_argument("--alpha-flip", type=float, required=True)
    ap.add_argument("--period", type=int, required=True)
    ap.add_argument("--start-alpha", type=float, required=True, help="where the period-p orbit attracts")
    ap.add_argument("--side", type=int, choices=(-1, 1), required=True,
                    help="direction in alpha from the flip into the side where the period-p orbit is unstable")
    ap.add_argument("--offsets", type=float, nargs="+", required=True)
    ap.add_argument("--continuation-steps", type=int, default=6)
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--prompt", default="open")
    ap.add_argument("--opener", default=None)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--transient", type=int, default=900)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    tic = time.perf_counter()
    opener = args.opener if args.opener is not None else ela.OPENER
    loop = esl.SoftLoop(args.model, args.layer, args.threads, args.prompt, opener, args.window, "float64", args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    p, lt = args.period, args.log_temperature
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)

    def make_cell(a):
        return esl.SoftOrbit(loop, a * v, 10.0 ** lt, chunk=args.chunk)

    cell = make_cell(args.start_alpha)
    X = cell.iterate(cell.start, args.transient)
    found = cell.detect_period(X, max(2 * p, 32), 1e-6)
    if found != p:
        raise SystemExit(f"the attractor at alpha {args.start_alpha} has period {found}, not {p}: refusing to start")
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), args.k)))[0]
    # continuation to the flip by Newton alone: the guard of lfb.evaluate against a complex pair near -1 belongs to
    # flip location, and along the way the leading pair may be complex (FNF2, 15 Oct 2026: -0.51 +- 0.26i)
    for a in np.linspace(args.start_alpha, args.alpha_flip, args.continuation_steps + 1)[:-1]:
        X, Q, hist = lfb.newton(make_cell(float(a)), X, p, Q, lfb.tol_for(p, args.tol), 12)
        if not hist[-1] < lfb.tol_for(p, args.tol):
            raise SystemExit(f"continuation Newton did not converge at alpha {a}: {hist}")
    flip = lfb.evaluate(make_cell, args.alpha_flip, X, p, Q, args.tol)
    fam = esl.soft_family(loop, v, p, 10.0, args.chunk)
    theta = np.array([lt, args.alpha_flip * 10.0])
    nf = flip_cubic(fam, lfb._flat(flip.X), theta)
    r, l = nf.pop("r"), nf.pop("l")
    c = nf["c"]
    rec = {"args": vars(args), "flip_lowrank_critical": [float(flip.critical.real), float(flip.critical.imag)],
           "normal_form": nf, "kind": "supercritical" if c > 0 else "subcritical", "offsets": []}
    print(json.dumps({k: v for k, v in rec.items() if k != "args"}), flush=True)

    side = args.side if c > 0 else -args.side          # a subcritical doubled orbit lies where the p-orbit is stable
    rec["offset_side"] = side
    prev = flip
    for s in sorted(args.offsets):
        a = args.alpha_flip + side * s
        row = {"offset": s, "alpha": a}
        try:
            ep = lfb.evaluate(make_cell, a, prev.X, p, prev.Q, args.tol)
            prev = ep
            beta = -float(ep.critical.real) - 1.0
            row.update({"mu_p": [float(ep.critical.real), float(ep.critical.imag)], "beta": beta,
                        "next_modulus_p": ep.next_modulus})
            if beta * c <= 0:
                row["note"] = "beta / c <= 0: no doubled orbit predicted on this side"
            else:
                eta = float(np.sqrt(beta / c))
                xs = lfb._flat(ep.X)
                X0 = ep.X + loop.torch.as_tensor((eta * r).reshape(tuple(ep.X.shape)), dtype=ep.X.dtype,
                                                 device=ep.X.device)
                Q2 = np.linalg.qr(np.random.default_rng(1).normal(size=(X0.numel(), args.k)))[0]
                e2 = lfb.evaluate(make_cell, a, X0, 2 * p, Q2, args.tol, half=p)
                cellA = make_cell(a)
                xa = lfb._flat(e2.X)
                xb = lfb._flat(cellA.iterate(e2.X, p))
                eta_meas = abs(float(l @ (xa - xb))) / 2
                i1 = int(np.argmin(np.abs(e2.vals - 1)))
                row.update({"eta_predicted": eta, "eta_measured": eta_meas, "ratio": eta_meas / eta,
                            "centre_offset": float(l @ ((xa + xb) / 2 - xs)),
                            "mu_2p": [float(e2.vals[i1].real), float(e2.vals[i1].imag)],
                            "mu_2p_predicted": 1.0 - 4.0 * beta, "next_modulus_2p": e2.next_modulus,
                            "newton_2p": e2.newton_history[-1]})
        except lfb.SearchError as err:
            row["error"] = str(err)
        rec["offsets"].append(row)
        print(json.dumps(row), flush=True)
    rec["seconds"] = round(time.perf_counter() - tic, 1)
    (out / "flip-normal-form.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
