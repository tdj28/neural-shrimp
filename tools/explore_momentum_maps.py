"""Exploratory E1: does the edge-of-stability route survive heavy-ball momentum?

Map (PyTorch SGD-momentum convention, per-layer rates P):
    m' = beta * m + grad L(theta),    theta' = theta - P m'.
This equals heavy ball, theta_{t+1} = theta_t - P grad L(theta_t) + beta (theta_t - theta_{t-1}).
At a minimum every Hessian eigenvalue h gives mu^2 - (1 + beta - eta h) mu + beta = 0, so
complex pairs have modulus sqrt(beta) and the minimum first loses stability by a flip at
eta h = 2 (1 + beta). The question here is what the resulting 2-cycle does next: a
Neimark-Sacker bifurcation (quasi-periodic training, locking windows) as in plain gradient
descent, or something else.

Subcommands:
  scan     Along a ray of rates (fixed output/hidden ratio), iterate from the shared initial
           parameters, find the attracting 2-cycle by Newton on F^2 in the 26-dimensional
           state, and report its two-step multipliers; the first unit-circle crossing type is
           the answer for that ray.
  rotation-map  Rotation numbers of two-step samples of theta on a rate rectangle (CPU).
  branch   Basin-independent version of scan: find the minimum (rate independent), locate its
           flip on the ray exactly (largest eigenvalue of P H equal to 2 (1 + beta)), seed the
           2-cycle from the flip eigenvector and continue it along the ray, recording its
           two-step multipliers until (and after) the first unit-circle crossing.
  arc      Same question with pseudo-arclength continuation of the 2-cycle branch (state and
           ray parameter together), started a little past the flip, so the branch cannot slide
           back onto the fixed point; records the first unit-circle crossing.
  onset    Just past that crossing: Newton for the (now unstable) 2-cycle at given distances
           past the crossing, start next to it along the critical eigenvector, iterate, and
           measure the attractor's distance from the cycle and its rotation around it (soft
           onset: amplitude ~ sqrt(distance), rotation near the crossing angle; hard onset: a
           jump far away or divergence).
  criticality  First Lyapunov coefficient d of the 2-cycle's Neimark-Sacker
           crossing on the ray (two-step map in the full (theta, m) state, at the
           bisected crossing): d < 0 soft (supercritical), d > 0 hard
           (subcritical). Separates a genuinely subcritical crossing from
           hysteresis caused by a coexisting attractor.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.trainability import SYSTEMS  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


def make_width_system(seed, width):
    """S2 data and initialization rule with a different number of hidden units."""
    from neural_shrimp.gdmaps import TwoLayerNet
    from neural_shrimp.models import TanhRegressor, make_teacher_regression

    net = TwoLayerNet(d=2, width=int(width), activation="tanh", bias=True, loss_scale=0.5, ridge=1e-3)
    data = make_teacher_regression(seed=int(seed), d=2, teacher_width=3, n_train=24, n_validation=48, n_test=96)
    theta0 = TanhRegressor(2, int(width), ridge=1e-3).initialize(seed=int(seed) + 100000, scale=0.2)
    return net, data.train_x, data.train_y, theta0


class MomentumMap:
    def __init__(self, system, seed, beta, width=None, ridge=None):
        if width is not None and width != 3:
            self.net, self.x, self.y, self.theta0 = make_width_system(seed, width)
        else:
            self.net, self.x, self.y, self.theta0 = SYSTEMS[system](seed)
        if ridge is not None:                          # weight-decay override (the initialization is unchanged)
            self.net = dataclasses.replace(self.net, ridge=float(ridge))
        self.beta = float(beta)
        self.n = self.net.n_parameters
        h = self.net.hidden_parameter_count
        self.mask_h = np.zeros(self.n)
        self.mask_h[:h] = 1.0

    def rates(self, lh, lo):
        lh, lo = np.atleast_1d(lh), np.atleast_1d(lo)
        return 10.0 ** lh[:, None] * self.mask_h + 10.0 ** lo[:, None] * (1.0 - self.mask_h)

    def step(self, theta, m, rates):
        g = self.net.gradient(theta, self.x, self.y)
        m = self.beta * m + g
        return theta - rates * m, m

    def jacobian(self, theta, m, rates):
        """26 x 26 Jacobian of one step for a single state."""
        h = self.net.hessian(theta, self.x, self.y)
        n = self.n
        jac = np.zeros((2 * n, 2 * n))
        r = rates[0]
        jac[:n, :n] = np.eye(n) - r[:, None] * h
        jac[:n, n:] = -self.beta * np.diag(r)
        jac[n:, :n] = h
        jac[n:, n:] = self.beta * np.eye(n)
        return jac


def require_same_ridge(mm, record):
    arc_ridge = record.get("ridge", 1e-3)              # arcs written before the option used the built-in 1e-3
    if arc_ridge != mm.net.ridge:
        raise SystemExit(f"ridge mismatch: arc {arc_ridge}, this run {mm.net.ridge}")


def refine_minimum(mm, theta):
    """Trust-region Newton (exact Hessian) from a descent endpoint; None unless a strict minimum is reached."""
    from scipy.optimize import minimize

    fun = lambda t: float(mm.net.loss(t[None, :], mm.x, mm.y)[0])
    jac = lambda t: mm.net.gradient(t[None, :], mm.x, mm.y)[0]
    hess = lambda t: mm.net.hessian(t, mm.x, mm.y)
    res = minimize(fun, theta, jac=jac, hess=hess, method="trust-exact", options={"gtol": 1e-13, "maxiter": 500})
    t = res.x
    for _ in range(5):
        g = jac(t)
        if np.linalg.norm(g) < 1e-13:
            break
        t = t - np.linalg.solve(hess(t), g)
    if np.linalg.norm(jac(t)) > 1e-10 or np.min(np.linalg.eigvalsh(hess(t))) <= 0:
        return None
    return t


def newton_two_cycle(mm, state, rates, tol=1e-12, max_iter=30):
    n = mm.n
    z = state.copy()
    for _ in range(max_iter):
        t1, m1 = mm.step(z[None, :n], z[None, n:], rates)
        j0 = mm.jacobian(z[:n], z[n:], rates)
        t2, m2 = mm.step(t1, m1, rates)
        j1 = mm.jacobian(t1[0], m1[0], rates)
        f2 = np.concatenate([t2[0], m2[0]])
        res = f2 - z
        if np.linalg.norm(res) < tol * max(1.0, np.linalg.norm(z)):
            return z, j1 @ j0
        z = z + np.linalg.solve(j1 @ j0 - np.eye(2 * n), -res)
    return None, None


def scan(args):
    mm = MomentumMap(args.system, args.seed, args.beta)
    rows = []
    state_prev = None
    for s in np.linspace(args.start, args.stop, args.points):
        lh = s + args.offset_hidden
        lo = s + args.offset_output
        rates = mm.rates(lh, lo)
        theta = mm.theta0[None, :].copy()
        m = np.zeros_like(theta)
        for _ in range(args.transient):
            theta, m = mm.step(theta, m, rates)
        state = np.concatenate([theta[0], m[0]])
        if not np.all(np.isfinite(state)):
            rows.append({"s": float(s), "status": "diverged"})
            continue
        z, mono = newton_two_cycle(mm, state, rates)
        if z is None and state_prev is not None:
            z, mono = newton_two_cycle(mm, state_prev, rates)
        if z is None:
            rows.append({"s": float(s), "status": "no 2-cycle"})
            continue
        t1, m1 = mm.step(z[None, :mm.n], z[None, mm.n:], rates)
        separation = float(np.linalg.norm(t1[0] - z[:mm.n]))
        mult = np.linalg.eigvals(mono)
        order = np.argsort(-np.abs(mult))
        top = mult[order[0]]
        attract = float(np.linalg.norm(state - z))
        rows.append({"s": float(s), "status": "fixed point" if separation < 1e-8 else "2-cycle",
                     "separation": separation, "distance_attractor_to_cycle": attract,
                     "top": [[float(mult[i].real), float(mult[i].imag)] for i in order[:4]],
                     "top_modulus": float(abs(top)),
                     "top_turns": float(abs(np.angle(top)) / (2 * np.pi))})
        state_prev = z
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rows[-1].items()
                          if k != "top"} | {"top2": [[round(a, 4), round(b, 4)] for a, b in rows[-1]["top"][:2]]}))
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "scan.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
    print(json.dumps({"out": str(out / "scan.json")}))


def rotation_block(task):
    system, seed, beta, lh, lo, steps, pca, samples, width = task
    mm = MomentumMap(system, seed, beta, width)
    rates = mm.rates(lh, lo)
    theta = np.repeat(mm.theta0[None, :], len(lh), axis=0)
    m = np.zeros_like(theta)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            theta, m = mm.step(theta, m, rates)
        lobe = np.empty((pca, len(lh), mm.n))
        for k in range(pca):
            lobe[k] = theta
            theta, m = mm.step(*mm.step(theta, m, rates), rates)
        alive = np.all(np.isfinite(theta), axis=1) & np.all(np.isfinite(lobe), axis=(0, 2))
        centroid = np.nan_to_num(lobe.mean(0))
        centered = np.nan_to_num(np.transpose(lobe - centroid[None], (1, 0, 2)))
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        plane = vt[:, :2, :]
        spread = singular[:, 0] / np.sqrt(pca)

        def angle(t):
            planar = np.einsum("cp,cjp->cj", np.nan_to_num(t) - centroid, plane)
            return np.arctan2(planar[:, 1], planar[:, 0])

        previous = angle(theta)
        total = np.zeros(len(lh))
        for _ in range(samples):
            theta, m = mm.step(*mm.step(theta, m, rates), rates)
            current = angle(theta)
            total += (current - previous + np.pi) % (2 * np.pi) - np.pi
            previous = current
    alive &= np.all(np.isfinite(theta), axis=1)
    rho = np.where(alive, np.abs(total) / (2 * np.pi * samples), np.nan)
    return {"rho": rho, "spread": np.where(alive, spread, np.nan)}


def rotation_map(args):
    rows, cols = args.resolution
    log_h = np.linspace(*args.hidden, rows)
    log_o = np.linspace(*args.output, cols)
    r, c = np.divmod(np.arange(rows * cols), cols)
    chunks = np.array_split(np.arange(rows * cols), args.workers * 4)
    tasks = [(args.system, args.seed, args.beta, log_h[r[i]], log_o[c[i]], args.steps, args.pca, args.samples,
              args.width) for i in chunks]
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(rotation_block, tasks, chunksize=1))
    merged = {k: np.concatenate([p[k] for p in parts]).reshape(rows, cols) for k in parts[0]}
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "rotation.npz", log10_hidden=log_h, log10_output=log_o, **merged)
    (out / "metadata.json").write_text(json.dumps({"args": vars(args), "seconds": time.perf_counter() - start},
                                                  indent=1))
    print(json.dumps({"out": str(out), "seconds": round(time.perf_counter() - start, 1)}))


def branch(args):
    mm = MomentumMap(args.system, args.seed, args.beta)
    n = mm.n
    p0 = mm.rates(0.0, args.offset_output)[0]           # rates at s = 0 on the ray
    # the minimum: plain gradient descent at a small rate on the ray, then Newton on the gradient
    small = mm.rates(args.s_low, args.s_low + args.offset_output)
    theta = mm.theta0[None, :].copy()
    for _ in range(args.descent_steps):
        theta = theta - small * mm.net.gradient(theta, mm.x, mm.y)
    theta = theta[0]
    for _ in range(20):
        g = mm.net.gradient(theta[None, :], mm.x, mm.y)[0]
        if np.linalg.norm(g) < 1e-13:
            break
        theta = theta - np.linalg.solve(mm.net.hessian(theta, mm.x, mm.y), g)
    if np.linalg.norm(mm.net.gradient(theta[None, :], mm.x, mm.y)) > 1e-10:
        print(json.dumps({"outcome": "descent and Newton did not reach a critical point; lower --s-low"}))
        return
    hess = mm.net.hessian(theta, mm.x, mm.y)
    if np.min(np.linalg.eigvalsh(hess)) <= 0:
        print(json.dumps({"outcome": "critical point is not a minimum"}))
        return
    lam = np.linalg.eigvals(p0[:, None] * hess)
    lam_max = float(np.max(lam.real))
    s_flip = float(np.log10(2.0 * (1.0 + mm.beta) / lam_max))
    record = {"args": vars(args), "minimum_gradient": float(np.linalg.norm(
        mm.net.gradient(theta[None, :], mm.x, mm.y))), "minimum_loss": float(mm.net.loss(theta[None, :], mm.x, mm.y)[0]),
        "lambda_max_at_s0": lam_max, "s_flip": s_flip, "minimum": theta.tolist()}
    print(json.dumps({k: record[k] for k in ("minimum_gradient", "minimum_loss", "lambda_max_at_s0", "s_flip")}))
    # flip eigenvector of the 2n-dimensional step Jacobian just beyond the flip
    rows = []
    for side in (+1, -1):
        s = s_flip + side * args.seed_offset
        rates = mm.rates(s, s + args.offset_output)
        jac = mm.jacobian(theta, np.zeros(n), rates)
        mult, vecs = np.linalg.eig(jac)
        i = int(np.argmin(np.abs(mult + 1.0)))
        v = np.real(vecs[:, i])
        v /= np.linalg.norm(v)
        found = None
        for amp in args.amplitudes:
            z0 = np.concatenate([theta, np.zeros(n)]) + amp * v
            z, mono = newton_two_cycle(mm, z0, rates)
            if z is not None:
                t1, _ = mm.step(z[None, :n], z[None, n:], rates)
                if np.linalg.norm(t1[0] - z[:n]) > 1e-7:
                    found = (z, mono, amp)
                    break
        if found is not None:
            record["seed_side"] = side
            break
    if found is None:
        record["outcome"] = "no 2-cycle near the flip on either side"
        print(json.dumps({"outcome": record["outcome"]}))
    else:
        z, mono, amp = found
        direction = side
        crossed = None
        for k in range(args.steps):
            rates = mm.rates(s, s + args.offset_output)
            z_new, mono = newton_two_cycle(mm, z, rates)
            if z_new is None:
                rows.append({"s": s, "status": "Newton failed"})
                break
            z = z_new
            t1, _ = mm.step(z[None, :n], z[None, n:], rates)
            sep = float(np.linalg.norm(t1[0] - z[:n]))
            mult = np.linalg.eigvals(mono)
            order = np.argsort(-np.abs(mult))
            top = mult[order[0]]
            row = {"s": float(s), "separation": sep, "top_modulus": float(abs(top)),
                   "top_turns": float(abs(np.angle(top)) / (2 * np.pi)),
                   "top": [[float(mult[i].real), float(mult[i].imag)] for i in order[:4]]}
            rows.append(row)
            if crossed is None and abs(top) > 1.0 + 1e-9 and k > 0:
                kind = "complex (Neimark-Sacker)" if abs(top.imag) > 1e-8 else (
                    "real +1 (fold or branch point)" if top.real > 0 else "real -1 (flip)")
                crossed = {"s": float(s), "type": kind, "turns": row["top_turns"]}
                print(json.dumps({"first_crossing": crossed}))
            if k % args.print_every == 0:
                print(json.dumps({"s": round(s, 5), "sep": round(sep, 5), "mod": round(abs(top), 6),
                                  "turns": round(row["top_turns"], 4)}))
            if crossed is not None and abs(s - crossed["s"]) > args.beyond:
                break
            s += direction * args.ds
        record["first_crossing"] = crossed
    record["rows"] = rows
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "branch.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({"out": str(out / "branch.json"), "first_crossing": record.get("first_crossing")}))


class MomentumTwoCycle:
    """F^2(z) - z on a rate ray, unknowns (26-dimensional state, s), for fastcont.arclength."""

    def __init__(self, mm, offset_output):
        self.mm, self.offset = mm, offset_output

    def rates(self, s):
        return self.mm.rates(s, s + self.offset)

    def one(self, z, rates):
        n = self.mm.n
        t, m = self.mm.step(z[None, :n], z[None, n:], rates)
        jac = self.mm.jacobian(z[:n], z[n:], rates)
        ds = np.concatenate([-np.log(10.0) * rates[0] * m[0], np.zeros(n)])
        return np.concatenate([t[0], m[0]]), jac, ds

    def residual_jacobian(self, w):
        z, s = w[:-1], w[-1]
        rates = self.rates(s)
        z1, j0, d0 = self.one(z, rates)
        z2, j1, d1 = self.one(z1, rates)
        mono = j1 @ j0
        dfds = j1 @ d0 + d1
        jac = np.concatenate([mono - np.eye(len(z)), dfds[:, None]], axis=1)
        return z2 - z, jac, mono, np.array([z, z1, z2])


def arc(args):
    from neural_shrimp.fastcont import arclength, newton, null_tangent

    mm = MomentumMap(args.system, args.seed, args.beta, getattr(args, "width", None), getattr(args, "ridge", None))
    n = mm.n
    p0 = mm.rates(0.0, args.offset_output)[0]
    small = mm.rates(args.s_low, args.s_low + args.offset_output)
    theta = mm.theta0[None, :].copy()
    for _ in range(args.descent_steps):
        theta = theta - small * mm.net.gradient(theta, mm.x, mm.y)
    theta = refine_minimum(mm, theta[0])
    if theta is None:
        print(json.dumps({"outcome": "no converged minimum"}))
        return
    lam_max = float(np.max(np.linalg.eigvals(p0[:, None] * mm.net.hessian(theta, mm.x, mm.y)).real))
    s_flip = float(np.log10(2.0 * (1.0 + mm.beta) / lam_max))
    system = MomentumTwoCycle(mm, args.offset_output)
    record = {"args": vars(args), "s_flip": s_flip, "lambda_max_at_s0": lam_max, "ridge": mm.net.ridge,
              "minimum_norm": float(np.linalg.norm(theta)),
              "minimum_loss": float(mm.net.loss(theta[None, :], mm.x, mm.y)[0])}
    seed = None
    for side in (+1, -1):
        s0 = s_flip + side * args.start_offset
        rates = system.rates(s0)
        jac = mm.jacobian(theta, np.zeros(n), rates)
        mult, vecs = np.linalg.eig(jac)
        v = np.real(vecs[:, int(np.argmin(np.abs(mult + 1.0)))])
        v /= np.linalg.norm(v)
        fixed = np.zeros(2 * n + 1)
        fixed[-1] = 1.0
        for amp in args.amplitudes:
            w0 = np.concatenate([theta, np.zeros(n), [s0]]) + np.concatenate([amp * v, [0.0]])
            w, hist = newton(system, w0, max_iter=40, extra=lambda q, s0=s0: (fixed, q[-1] - s0))
            if w is None:
                continue
            _, _, _, pts = system.residual_jacobian(w)
            if np.linalg.norm(pts[1][:n] - pts[0][:n]) > args.min_separation:
                seed = (w, side)
                break
        if seed is not None:
            break
    if seed is None:
        record["outcome"] = "no 2-cycle found near the flip"
        print(json.dumps({"s_flip": s_flip, "outcome": record["outcome"]}))
    else:
        w, side = seed
        _, jac, _, _ = system.residual_jacobian(w)
        scale = np.concatenate([np.ones(2 * n), [args.s_scale]])
        tangent = null_tangent(jac, scale)
        if tangent[-1] * side < 0:
            tangent = -tangent
        crossing = []

        def watch(node):
            mult = np.linalg.eigvals(node["monodromy"])
            top = mult[np.argmax(np.abs(mult))]
            if abs(top) > 1.0 + 1e-9 and not crossing:
                kind = "complex (Neimark-Sacker)" if abs(top.imag) > 1e-8 else (
                    "real +1 (fold or branch point)" if top.real > 0 else "real -1 (flip)")
                crossing.append({"s": float(node["z"][-1]), "type": kind,
                                 "turns": float(abs(np.angle(top)) / (2 * np.pi)), "modulus": float(abs(top))})
            return bool(crossing) and abs(node["z"][-1] - crossing[0]["s"]) > args.beyond or node["z"][-1] > args.s_max

        nodes, reason = arclength(system, w, tangent, scale, steps=args.steps, h0=args.h, h_min=1e-8,
                                  h_max=args.h_max, stop=watch)
        rows = []
        for node in nodes:
            mult = np.linalg.eigvals(node["monodromy"])
            order = np.argsort(-np.abs(mult))
            pts = node["points"]
            rows.append({"s": float(node["z"][-1]), "separation": float(np.linalg.norm(pts[1][:n] - pts[0][:n])),
                         "norm": float(0.5 * (np.linalg.norm(pts[0][:n]) + np.linalg.norm(pts[1][:n]))),
                         "top": [[float(mult[i].real), float(mult[i].imag)] for i in order[:4]]})
        record.update({"seed_side": side, "reason": reason, "first_crossing": crossing[0] if crossing else None,
                       "rows": rows})
        print(json.dumps({"s_flip": s_flip, "seed_side": side, "nodes": len(rows), "reason": reason,
                          "first_crossing": record["first_crossing"],
                          "separation_range": [min(r["separation"] for r in rows), max(r["separation"] for r in rows)]
                          if rows else None}))
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "arc.json").write_text(json.dumps(record, indent=1))
    if seed is not None and nodes:
        np.savez_compressed(out / "nodes.npz", w=np.array([node["z"] for node in nodes]))


def onset(args):
    from neural_shrimp.fastcont import newton

    mm = MomentumMap(args.system, args.seed, args.beta)
    n = mm.n
    base = DATA / "explore" / args.arc
    record = json.loads((base / "arc.json").read_text())
    nodes = np.load(base / "nodes.npz")["w"]
    crossing = record["first_crossing"]
    system = MomentumTwoCycle(mm, args.offset_output)
    k = int(np.argmin(np.abs(nodes[:, -1] - crossing["s"])))
    fixed_row = np.zeros(2 * n + 1)
    fixed_row[-1] = 1.0

    def cycle_at(s_value, guess):
        w0 = guess.copy()
        w0[-1] = s_value
        w, _ = newton(system, w0, max_iter=40, extra=lambda q: (fixed_row, q[-1] - s_value))
        if w is None:
            return None, None
        _, _, mono, _ = system.residual_jacobian(w)
        return w, float(np.max(np.abs(np.linalg.eigvals(mono))))

    # bisection for the crossing between the last stable and first unstable stored nodes
    lo_s, hi_s = nodes[max(k - 1, 0)][-1], nodes[k][-1]
    guess = nodes[max(k - 1, 0)]
    _, r_lo = cycle_at(lo_s, guess)
    if r_lo is not None and r_lo > 1.0:
        lo_s = nodes[max(k - 2, 0)][-1]
    for _ in range(40):
        mid = 0.5 * (lo_s + hi_s)
        w_mid, r_mid = cycle_at(mid, guess)
        if w_mid is None:
            break
        guess = w_mid
        if r_mid > 1.0:
            hi_s = mid
        else:
            lo_s = mid
    s_cross = 0.5 * (lo_s + hi_s)
    print(json.dumps({"refined_crossing": s_cross, "bracket": [lo_s, hi_s]}))
    rows = []
    for offset in args.offsets:
        s_target = s_cross + offset
        fixed = np.zeros(2 * n + 1)
        fixed[-1] = 1.0
        w0 = nodes[k].copy()
        w0[-1] = s_target
        w, hist = newton(system, w0, max_iter=40, extra=lambda q: (fixed, q[-1] - s_target))
        if w is None:
            rows.append({"offset": offset, "outcome": "cycle not found"})
            continue
        z = w[:-1]
        _, _, mono, pts = system.residual_jacobian(w)
        mult, vecs = np.linalg.eig(mono)
        i = int(np.argmax(np.abs(mult)))
        v = vecs[:, i]
        crit = mult[i]
        rates = system.rates(s_target)
        kick = args.kick if offset > 0 or args.hysteresis_kick is None else args.hysteresis_kick
        state = z + kick * np.real(v) / np.linalg.norm(np.real(v))
        t, m = state[None, :n], state[None, n:]
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(args.transient):
                t, m = mm.step(t, m, rates)
            samples = []
            for _ in range(args.samples):
                samples.append(np.concatenate([t[0], m[0]]))
                t, m = mm.step(*mm.step(t, m, rates), rates)
        samples = np.array(samples)
        if not np.all(np.isfinite(samples)):
            rows.append({"offset": offset, "outcome": "diverged", "critical_modulus": float(abs(crit))})
            print(json.dumps(rows[-1]))
            continue
        d = np.linalg.norm(samples[:, :n] - z[:n], axis=1)
        d_other = np.linalg.norm(samples[:, :n] - pts[1][:n], axis=1)
        dist = np.minimum(d, d_other)
        basis = np.stack([np.real(v)[:n], np.imag(v)[:n]], axis=1)
        q, _ = np.linalg.qr(basis)
        planar = (samples[:, :n] - z[:n]) @ q
        angles = np.unwrap(np.arctan2(planar[:, 1], planar[:, 0]))
        rot = float(abs(angles[-1] - angles[0]) / (2 * np.pi * (len(angles) - 1)))
        rows.append({"offset": offset, "s": s_target, "kick": kick, "critical_modulus": float(abs(crit)),
                     "critical_turns": float(abs(np.angle(crit)) / (2 * np.pi)),
                     "amplitude_mean": float(dist.mean()), "amplitude_max": float(dist.max()),
                     "amplitude_min": float(dist.min()), "rotation_two_step": rot,
                     "cycle_separation": float(np.linalg.norm(pts[1][:n] - pts[0][:n]))})
        print(json.dumps({key: (round(val, 6) if isinstance(val, float) else val) for key, val in rows[-1].items()}))
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "onset.json").write_text(json.dumps({"args": vars(args), "crossing": crossing,
                                                "refined_crossing": s_cross, "rows": rows}, indent=1))


def criticality(args):
    from neural_shrimp.criticality import finite_difference_forms, ns_coefficient
    from neural_shrimp.fastcont import newton

    mm = MomentumMap(args.system, args.seed, args.beta, getattr(args, "width", None), getattr(args, "ridge", None))
    n = mm.n
    base = DATA / "explore" / args.arc
    record = json.loads((base / "arc.json").read_text())
    require_same_ridge(mm, record)
    nodes = np.load(base / "nodes.npz")["w"]
    system = MomentumTwoCycle(mm, args.offset_output)
    fixed_row = np.zeros(2 * n + 1)
    fixed_row[-1] = 1.0

    def cycle_at(s_value, guess):
        w0 = guess.copy()
        w0[-1] = s_value
        w, _ = newton(system, w0, max_iter=40, extra=lambda q: (fixed_row, q[-1] - s_value))
        if w is None:
            return None, None, None
        _, _, mono, _ = system.residual_jacobian(w)
        return w, mono, float(np.max(np.abs(np.linalg.eigvals(mono))))

    radii = []
    for w in nodes:
        _, _, mono, _ = system.residual_jacobian(w)
        radii.append(float(np.max(np.abs(np.linalg.eigvals(mono)))))
    k = next((i for i in range(1, len(radii)) if radii[i] > 1.0 >= radii[i - 1]), None)
    if k is None:
        print(json.dumps({"beta": args.beta, "outcome": "no unit-circle crossing within the continued range",
                          "max_radius": max(radii), "s_range": [float(nodes[0][-1]), float(nodes[-1][-1])]}))
        return
    lo_w, hi_s = nodes[k - 1], nodes[k][-1]
    lo_s = lo_w[-1]
    guess = lo_w
    for _ in range(60):
        mid = 0.5 * (lo_s + hi_s)
        w_mid, _, r_mid = cycle_at(mid, guess)
        if w_mid is None:
            break
        guess = w_mid
        if r_mid > 1.0:
            hi_s = mid
        else:
            lo_s = mid
    w_c, mono_c, r_c = cycle_at(0.5 * (lo_s + hi_s), guess)
    mult_c = np.linalg.eigvals(mono_c)
    lead = mult_c[np.argmax(np.abs(mult_c))]
    if abs(lead.imag) < 1e-6:
        out = {"beta": args.beta, "crossing_s": float(w_c[-1]), "outcome": "first crossing is real, not Neimark-Sacker",
               "leading_multiplier": [float(lead.real), float(lead.imag)]}
        dest = DATA / "explore" / args.out
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "criticality.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out))
        return
    x0 = w_c[:-1]
    rates = system.rates(w_c[-1])

    def two_step_jvp(X, V):
        out = []
        for xk, vk in zip(X, V):
            z1, j0, _ = system.one(xk, rates)
            _, j1, _ = system.one(z1, rates)
            out.append(j1 @ (j0 @ vk))
        return np.array(out)

    B, C = finite_difference_forms(two_step_jvp, x0, step_b=args.step_b, step_c=args.step_c)
    res = ns_coefficient(mono_c, B, C, mu=lead)
    # step-size check
    B2, C2 = finite_difference_forms(two_step_jvp, x0, step_b=args.step_b * 2, step_c=args.step_c * 2)
    res2 = ns_coefficient(mono_c, B2, C2, mu=lead)
    out = {"beta": args.beta, "crossing_s": float(w_c[-1]), "spectral_radius_at_crossing": r_c,
           "first_lyapunov": res, "first_lyapunov_doubled_steps": res2,
           "classification": "supercritical (soft)" if res["d"] < 0 else "subcritical (hard)"}
    dest = DATA / "explore" / args.out
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "criticality.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"beta": args.beta, "crossing_s": out["crossing_s"], "radius": r_c,
                      "d": res["d"], "d_doubled_steps": res2["d"], "theta_over_2pi": res["theta_over_2pi"],
                      "classification": out["classification"]}))


def staircase(args):
    """Rotation number versus ray parameter beyond the 2-cycle's first crossing."""
    from neural_shrimp.regimes import rotation_number

    mm = MomentumMap(args.system, args.seed, args.beta, args.width, args.ridge)
    n = mm.n
    base = DATA / "explore" / args.arc
    record = json.loads((base / "arc.json").read_text())
    require_same_ridge(mm, record)
    nodes = np.load(base / "nodes.npz")["w"]
    s_cross = record["first_crossing"]["s"]
    k = int(np.argmin(np.abs(nodes[:, -1] - s_cross)))
    start = nodes[k][:-1]
    rows = []
    for s_val in np.linspace(s_cross + args.start, s_cross + args.stop, args.points):
        rates = mm.rates(s_val, s_val + args.offset_output)
        t = start[None, :n] + 1e-4
        m = start[None, n:].copy()
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(args.transient):
                t, m = mm.step(t, m, rates)
            samples = []
            for _ in range(args.samples):
                samples.append(t[0].copy())
                t, m = mm.step(*mm.step(t, m, rates), rates)
        samples = np.array(samples)
        if not np.all(np.isfinite(samples)):
            rows.append({"s": float(s_val), "rotation": None, "status": "diverged"})
            continue
        scale = max(1.0, float(np.linalg.norm(samples[0])))
        period = next((q for q in range(1, 65) if np.linalg.norm(samples[q] - samples[0]) < 1e-7 * scale), None)
        rho = rotation_number(samples)[0]
        rows.append({"s": float(s_val), "rotation": rho, "two_step_period": period,
                     "amplitude": float(np.max(np.linalg.norm(samples - samples.mean(0), axis=1)))})
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "staircase.json").write_text(json.dumps({"args": vars(args), "crossing": s_cross, "rows": rows}, indent=1))
    rho = [r["rotation"] for r in rows if r["rotation"] is not None]
    locked = [r for r in rows if r.get("two_step_period")]
    print(json.dumps({"points": len(rows), "rotation_range": [min(rho), max(rho)] if rho else None,
                      "locked_points": len(locked),
                      "locked_periods": sorted({r["two_step_period"] for r in locked})[:12],
                      "diverged": sum(1 for r in rows if r.get("status") == "diverged")}))


def crossing_anatomy(args):
    """Which hidden units carry the first crossing's eigenvector, and is the crossing a turning point in s?"""
    mm = MomentumMap(args.system, args.seed, args.beta, args.width, args.ridge)
    n, net = mm.n, mm.net
    base = DATA / "explore" / args.arc
    record = json.loads((base / "arc.json").read_text())
    require_same_ridge(mm, record)
    fc = record.get("first_crossing")
    if fc is None:
        print(json.dumps({"arc": args.arc, "outcome": "no first crossing recorded"}))
        return
    nodes = np.load(base / "nodes.npz")["w"]
    system = MomentumTwoCycle(mm, args.offset_output)
    small = mm.rates(args.s_low, args.s_low + args.offset_output)
    theta = mm.theta0[None, :].copy()
    for _ in range(args.descent_steps):
        theta = theta - small * net.gradient(theta, mm.x, mm.y)
    theta_min = refine_minimum(mm, theta[0])
    width, d = net.width, net.d

    def unit(j):
        return [i * width + j for i in range(d)] + [d * width + j, d * width + width + j]

    s_vals = nodes[:, -1]
    k = int(np.argmin(np.abs(s_vals - fc["s"])))
    ds = np.diff(s_vals[max(0, k - 4):k + 5])
    _, _, mono, pts = system.residual_jacobian(nodes[k])
    mult, vecs = np.linalg.eig(mono)
    order = np.argsort(-np.abs(mult))
    v = vecs[:n, order[0]]
    total = float(np.sum(np.abs(v) ** 2))
    shares = [float(np.sum(np.abs(v[unit(j)]) ** 2) / total) for j in range(width)]
    centre = 0.5 * (pts[0][:n] + pts[1][:n])
    out = {"arc": args.arc, "ridge": net.ridge, "width": width, "crossing": fc,
           "node_s": float(s_vals[k]), "turning_point_nearby": bool(np.any(ds > 0) and np.any(ds < 0)),
           "leading_multipliers": [[float(mult[i].real), float(mult[i].imag)] for i in order[:4]],
           "unit_norms_at_minimum": None if theta_min is None else [float(np.linalg.norm(theta_min[unit(j)]))
                                                                     for j in range(width)],
           "unit_norms_at_cycle_centre": [float(np.linalg.norm(centre[unit(j)])) for j in range(width)],
           "eigenvector_unit_shares": shares,
           "eigenvector_output_bias_share": float(abs(v[-1]) ** 2 / total)}
    print(json.dumps(out))
    dest = DATA / "explore" / args.out
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "crossing-anatomy.json").write_text(json.dumps(out, indent=1))


def rotation_summary(args):
    """Counts and a figure for saved rotation maps (rotation is meaningful only where the attractor has extent)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = sorted({(p_, q) for q in range(2, args.qmax + 1) for p_ in range(1, q) if np.gcd(p_, q) == 1 and p_ / q <= 0.5})
    fig, axes = plt.subplots(1, len(args.maps), figsize=(5.2 * len(args.maps), 4.6), squeeze=False)
    rows = []
    for ax, name in zip(axes[0], args.maps):
        d = np.load(DATA / "explore" / name / "rotation.npz")
        rho, spread, lh, lo = d["rho"], d["spread"], d["log10_hidden"], d["log10_output"]
        alive = np.isfinite(rho)
        moving = alive & (np.nan_to_num(spread) > args.min_spread)
        rotating = moving & (np.nan_to_num(rho) > args.min_rho) & (np.nan_to_num(rho) < 0.5 - args.min_rho)
        near = np.zeros_like(alive)
        for p_, q in fracs:
            near |= rotating & (np.abs(np.nan_to_num(rho) - p_ / q) < args.tol)
        row = {"map": name, "cells": int(rho.size), "alive": int(alive.sum()), "moving": int(moving.sum()),
               "rotating": int(rotating.sum()), "near_rational": int(near.sum()),
               "rho_quantiles_rotating": np.quantile(rho[rotating], [0.1, 0.5, 0.9]).tolist() if rotating.any() else None}
        rows.append(row)
        print(json.dumps(row))
        # categories: white = diverged, grey = settled (no extent), colour = moving (non-cyclic map, 0 dark)
        ax.pcolormesh(lo, lh, np.where(alive & ~moving, 1.0, np.nan), cmap="Greys", vmin=0, vmax=4, shading="auto")
        im = ax.pcolormesh(lo, lh, np.where(moving, rho, np.nan), cmap="viridis", vmin=0, vmax=0.5, shading="auto")
        ax.set_xlabel("log10 output rate")
        ax.set_ylabel("log10 hidden rate")
        ax.set_title(name.split("/")[-1])
        fig.colorbar(im, ax=ax, label="rotation (turns per two steps)")
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "rotation-summary.png", dpi=110)
    (out / "rotation-summary.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sc = sub.add_parser("scan")
    sc.add_argument("--system", default="S2")
    sc.add_argument("--seed", type=int, default=28011)
    sc.add_argument("--beta", type=float, required=True)
    sc.add_argument("--start", type=float, required=True, help="ray parameter start (log10 units)")
    sc.add_argument("--stop", type=float, required=True)
    sc.add_argument("--points", type=int, default=40)
    sc.add_argument("--offset-hidden", type=float, default=0.0)
    sc.add_argument("--offset-output", type=float, default=0.0)
    sc.add_argument("--transient", type=int, default=20000)
    sc.add_argument("--out", required=True)
    rm = sub.add_parser("rotation-map")
    rm.add_argument("--system", default="S2")
    rm.add_argument("--seed", type=int, default=28011)
    rm.add_argument("--beta", type=float, required=True)
    rm.add_argument("--hidden", type=float, nargs=2, required=True)
    rm.add_argument("--output", type=float, nargs=2, required=True)
    rm.add_argument("--resolution", type=int, nargs=2, default=(128, 128))
    rm.add_argument("--steps", type=int, default=20000)
    rm.add_argument("--pca", type=int, default=256)
    rm.add_argument("--samples", type=int, default=8000)
    rm.add_argument("--workers", type=int, default=8)
    rm.add_argument("--width", type=int, default=None)
    rm.add_argument("--out", required=True)
    br = sub.add_parser("branch")
    br.add_argument("--system", default="S2")
    br.add_argument("--seed", type=int, default=28011)
    br.add_argument("--beta", type=float, required=True)
    br.add_argument("--offset-output", type=float, default=-1.828, help="log10 output - log10 hidden on the ray")
    br.add_argument("--s-low", type=float, default=0.5, help="small-rate point for the descent to the minimum")
    br.add_argument("--descent-steps", type=int, default=200000)
    br.add_argument("--seed-offset", type=float, default=2e-4)
    br.add_argument("--amplitudes", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2, 3e-2, 0.1])
    br.add_argument("--ds", type=float, default=5e-4)
    br.add_argument("--steps", type=int, default=4000)
    br.add_argument("--beyond", type=float, default=0.02)
    br.add_argument("--print-every", type=int, default=20)
    br.add_argument("--out", required=True)
    ac = sub.add_parser("arc")
    ac.add_argument("--system", default="S2")
    ac.add_argument("--seed", type=int, default=28011)
    ac.add_argument("--beta", type=float, required=True)
    ac.add_argument("--offset-output", type=float, default=-1.828)
    ac.add_argument("--s-low", type=float, default=0.5)
    ac.add_argument("--descent-steps", type=int, default=200000)
    ac.add_argument("--start-offset", type=float, default=0.01)
    ac.add_argument("--amplitudes", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.4])
    ac.add_argument("--min-separation", type=float, default=1e-3)
    ac.add_argument("--s-scale", type=float, default=10.0)
    ac.add_argument("--h", type=float, default=0.01)
    ac.add_argument("--h-max", type=float, default=0.05)
    ac.add_argument("--steps", type=int, default=6000)
    ac.add_argument("--beyond", type=float, default=0.02)
    ac.add_argument("--s-max", type=float, default=2.3)
    ac.add_argument("--out", required=True)
    on = sub.add_parser("onset")
    on.add_argument("--system", default="S2")
    on.add_argument("--seed", type=int, default=28011)
    on.add_argument("--beta", type=float, required=True)
    on.add_argument("--offset-output", type=float, default=-1.828)
    on.add_argument("--arc", required=True, help="explore-relative folder of an arc run")
    on.add_argument("--offsets", type=float, nargs="+", default=[-0.002, 0.001, 0.002, 0.004, 0.008])
    on.add_argument("--kick", type=float, default=1e-4)
    on.add_argument("--hysteresis-kick", type=float, default=None,
                    help="kick used at negative offsets (before the crossing) to test bistability")
    on.add_argument("--transient", type=int, default=100000)
    on.add_argument("--samples", type=int, default=20000)
    on.add_argument("--out", required=True)
    cr = sub.add_parser("criticality")
    cr.add_argument("--system", default="S2")
    cr.add_argument("--seed", type=int, default=28011)
    cr.add_argument("--beta", type=float, required=True)
    cr.add_argument("--offset-output", type=float, default=-1.828)
    cr.add_argument("--arc", required=True)
    cr.add_argument("--step-b", type=float, default=1e-4)
    cr.add_argument("--step-c", type=float, default=1e-3)
    cr.add_argument("--out", required=True)
    st = sub.add_parser("staircase")
    st.add_argument("--system", default="S2")
    st.add_argument("--seed", type=int, default=28011)
    st.add_argument("--beta", type=float, default=0.0)
    st.add_argument("--width", type=int, default=None)
    st.add_argument("--offset-output", type=float, default=-1.828)
    st.add_argument("--arc", required=True)
    st.add_argument("--start", type=float, default=0.001)
    st.add_argument("--stop", type=float, default=0.15)
    st.add_argument("--points", type=int, default=150)
    st.add_argument("--transient", type=int, default=20000)
    st.add_argument("--samples", type=int, default=3000)
    st.add_argument("--out", required=True)
    for sp in (ac, cr):
        sp.add_argument("--width", type=int, default=None)
    rs = sub.add_parser("rotation-summary")
    rs.add_argument("--maps", nargs="+", required=True, help="explore-relative folders holding rotation.npz")
    rs.add_argument("--min-spread", type=float, default=1e-6)
    rs.add_argument("--min-rho", type=float, default=1e-3)
    rs.add_argument("--qmax", type=int, default=12)
    rs.add_argument("--tol", type=float, default=2e-4)
    rs.add_argument("--out", required=True)
    ca = sub.add_parser("crossing-anatomy")
    ca.add_argument("--system", default="S2")
    ca.add_argument("--seed", type=int, default=28011)
    ca.add_argument("--beta", type=float, default=0.0)
    ca.add_argument("--width", type=int, default=None)
    ca.add_argument("--offset-output", type=float, default=-1.828)
    ca.add_argument("--s-low", type=float, default=0.5)
    ca.add_argument("--descent-steps", type=int, default=200000)
    ca.add_argument("--arc", required=True)
    ca.add_argument("--out", required=True)
    for sp in (ac, cr, st, ca):
        sp.add_argument("--ridge", type=float, default=None, help="weight decay (default: the system's 1e-3)")
    args = parser.parse_args()
    {"scan": scan, "rotation-map": rotation_map, "branch": branch, "arc": arc, "onset": onset,
     "criticality": criticality, "staircase": staircase, "crossing-anatomy": crossing_anatomy,
     "rotation-summary": rotation_summary}[args.command](args)


if __name__ == "__main__":
    main()
