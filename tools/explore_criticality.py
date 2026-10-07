"""Exploratory: criticality of the period-2 Neimark-Sacker curve (safe vs dangerous edges).

Subcommands (outputs under DATA/explore/criticality/<name>/):
  control   engineered planar gradient map with a prescribed 2-cycle
            (neural_shrimp.minimal.EngineeredTwoCycle): first Lyapunov coefficient d
            at r = 1 versus kappa, and a dynamical check just beyond (r = 1 + delta)
  curve     for each output rate of a traced Neimark-Sacker curve: re-find the
            period-2 orbit near the boundary, continue it to the boundary
            (bisection 1e-10) and evaluate d of the two-step map there
  dynamics  at selected output rates: (a) asymptotic amplitude beyond the curve
            for several offsets (circle with amplitude ~ sqrt(delta) or escape),
            (b) escape threshold of perturbations before the curve, (c) a slow
            up-and-down sweep of the hidden rate carrying the state (hysteresis)

The sign of d decides the edge type: d < 0 supercritical (soft: quasi-periodic
training is born continuously), d > 0 subcritical (hard: the period-2 orbit is
surrounded by a repelling circle that shrinks onto it; trajectories leave).
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.continuation import continue_ray  # noqa: E402
from neural_shrimp.criticality import finite_difference_forms, ns_coefficient  # noqa: E402
from neural_shrimp.gdmaps import GridMap, run_grid  # noqa: E402
from neural_shrimp.minimal import EngineeredTwoCycle  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))
OUT = DATA / "explore" / "criticality"


# ---------------------------------------------------------------- control
def engineered_d(kappa, phi):
    f = EngineeredTwoCycle(1.0, phi, kappa)

    def jvp(X, V):
        return np.stack([f.jacobian(f.step(x)) @ (f.jacobian(x) @ v) for x, v in zip(X, V)])

    A = f.jacobian(f.p1) @ f.jacobian(f.p0)
    B, C = finite_difference_forms(jvp, f.p0)
    return ns_coefficient(A, B, C)


def engineered_dynamics(kappa, phi, delta, steps=200000):
    f = EngineeredTwoCycle(1.0 + delta, phi, kappa)
    x = f.p0 + np.array([1e-3, 0.0])
    amplitudes = []
    for k in range(steps):
        x = f.step(f.step(x))
        if not np.all(np.isfinite(x)) or np.linalg.norm(x - f.p0) > 0.5:
            return {"escaped": True, "at": k}
        if k >= steps - 2000:
            amplitudes.append(np.linalg.norm(x - f.p0))
    return {"escaped": False, "amplitude_mean": float(np.mean(amplitudes)),
            "amplitude_spread": float(np.std(amplitudes))}


def engineered_amplitude(kappa, phi, delta, coef, steps):
    """Normal-form amplitude |w| = |<p, x - p0>| of the circle at r = 1 + delta vs sqrt(delta / -d)."""
    base = EngineeredTwoCycle(1.0, phi, kappa)
    A = base.jacobian(base.p1) @ base.jacobian(base.p0)
    from neural_shrimp.criticality import critical_pair
    mu, q, p, _ = critical_pair(A)
    f = EngineeredTwoCycle(1.0 + delta, phi, kappa)
    predicted = np.sqrt(np.log1p(delta) / -coef["d"]) if coef["d"] < 0 else None
    w0 = predicted if predicted else np.sqrt(delta)
    x = f.p0 + 2 * (w0 * q).real
    ws = []
    for k in range(steps):
        x = f.step(f.step(x))
        if not np.all(np.isfinite(x)) or np.linalg.norm(x - f.p0) > 0.5:
            return {"delta": delta, "escaped": True}
        if k >= steps - 3000:
            ws.append(abs(np.vdot(p, x - f.p0)))
    return {"delta": delta, "escaped": False, "w_mean": float(np.mean(ws)), "w_spread": float(np.std(ws)),
            "predicted_w": None if predicted is None else float(predicted)}


def cmd_control(args):
    rows = []
    phi = 2 * np.pi * 0.13
    for kappa in args.kappas:
        coef = engineered_d(kappa, phi)
        dyn = [{"delta": dlt, **engineered_dynamics(kappa, phi, dlt)} for dlt in (1e-3, 4e-3)]
        amp = [engineered_amplitude(kappa, phi, dlt, coef, args.amplitude_steps) for dlt in args.amplitude_deltas]
        rows.append({"kappa": kappa, **coef, "dynamics": dyn, "normal_form_amplitude": amp})
        print(json.dumps({"kappa": kappa, "d": round(coef["d"], 6), "dynamics": dyn, "amplitude": amp}), flush=True)
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "control.json").write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------- network helpers
def network(system, seed):
    return SYSTEMS[system](seed)


def two_step_jvp(grid):
    def jvp(X, V):
        X1 = grid.step(X)
        return grid.jvp(X1, grid.jvp(X, V))
    return jvp


def dense_two_step(grid, x0):
    _, jacobian = grid.cell_map(0)
    x1 = grid.step(x0[None])[0]
    return jacobian(x1) @ jacobian(x0)


def period2_start(net, x, y, theta0, h, o, steps=40000):
    grid = GridMap(net, x, y, [10.0 ** h], [10.0 ** o])
    res = run_grid(grid, theta0, steps, tail_steps=384, ftle_steps=1000, ftle_align=200, sd_horizons=())
    return int(res["status"][0]), int(res["period"][0]), res["final_state"][0]


def boundary_point(task):
    system, seed, o, h_guess, margins = task
    net, x, y, theta0 = network(system, seed)

    def family(p):
        return GridMap(net, x, y, [10.0 ** p[0]], [10.0 ** p[1]]).cell_map(0)

    # margins below the traced boundary; negative entries are absolute start rates (-1.5 -> 1.5)
    for margin in margins:
        if margin < 0:
            margin = h_guess - (-margin)
        h0 = h_guess - margin
        status, period, state = period2_start(net, x, y, theta0, h0, o)
        if period != 2:
            continue
        ray = continue_ray(family, state, 2, (h0, o), (1.0, 0.0), margin + 0.05, min(0.002, margin / 4),
                           bisection_tolerance=1e-10)
        if ray["type"] != "neimark_sacker":
            return {"log10_output": o, "start": h0, "type": ray["type"],
                    "log10_hidden_boundary": None if ray["s_boundary"] is None else h0 + ray["s_boundary"]}
        h_low = h0 + ray["bracket"][0]
        x0 = np.asarray(ray["last_attracting_state"])
        grid = GridMap(net, x, y, [10.0 ** h_low], [10.0 ** o])
        A = dense_two_step(grid, x0)
        B, C = finite_difference_forms(two_step_jvp(grid), x0)
        coef = ns_coefficient(A, B, C)
        scale = float(np.linalg.norm(x0))
        B2, C2 = finite_difference_forms(two_step_jvp(grid), x0, step_b=2e-4, step_c=2e-3)
        coef2 = ns_coefficient(A, B2, C2)
        return {"log10_output": o, "start": h0, "type": "neimark_sacker", "log10_hidden_boundary": h0 + ray["s_boundary"],
                "log10_hidden_last_attracting": h_low, "orbit_point": x0.tolist(), "state_norm": scale, **coef,
                "d_step_check": coef2["d"]}
    return {"log10_output": o, "type": "no_period_2_start"}


def cmd_curve(args):
    rows = [r for r in json.loads(Path(args.ns).read_text())["rows"] if r.get("type") == "neimark_sacker"]
    rows = sorted(rows, key=lambda r: r["log10_output"])[::args.stride]
    if args.only:
        rows = [min(rows, key=lambda r: abs(r["log10_output"] - t)) for t in args.only]
    tasks = [(args.system, args.seed, r["log10_output"], r["log10_hidden_boundary"], tuple(args.margins))
             for r in rows]
    with ProcessPoolExecutor(args.workers) as pool:
        out_rows = list(pool.map(boundary_point, tasks, chunksize=1))
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "curve.json").write_text(json.dumps({"args": vars(args), "rows": out_rows}, indent=1))
    for r in out_rows:
        if r.get("type") == "neimark_sacker":
            print(f"out {r['log10_output']:+.4f} h_b {r['log10_hidden_boundary']:.6f} theta/2pi {r['theta_over_2pi']:.5f} "
                  f"d {r['d']:+.4e} (step check {r['d_step_check']:+.4e}) gap {r['eigen_gap']:.3f}")
        else:
            print(f"out {r['log10_output']:+.4f} {r['type']}")


# ---------------------------------------------------------------- dynamics
def beyond(net, x, y, x0, h_b, o, delta, steps):
    """Iterate the two-step map at h_b + delta from x0 plus a tiny push; amplitude or escape."""
    grid = GridMap(net, x, y, [10.0 ** (h_b + delta)], [10.0 ** o])
    xs = x0 + 1e-6 * np.random.default_rng(1).normal(size=x0.shape)
    theta = xs[None]
    amps = []
    for k in range(steps):
        theta = grid.step(grid.step(theta))
        if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 1e6:
            return {"delta": delta, "escaped": "diverged", "at": k}
        if k >= steps - 4000:
            amps.append(float(np.linalg.norm(theta[0] - x0)))
    return {"delta": delta, "escaped": False, "amplitude_median": float(np.median(amps)),
            "amplitude_min": float(np.min(amps)), "amplitude_max": float(np.max(amps))}


def threshold(net, x, y, x0, h_b, o, delta, q_dir, steps):
    """Smallest perturbation amplitude (along the critical plane) that does not return to the 2-cycle."""
    grid = GridMap(net, x, y, [10.0 ** (h_b - delta)], [10.0 ** o])

    def returns(amplitude):
        # Near the curve the 2-cycle contracts slowly, so "returns" means: never escaped
        # (no divergence, never farther than 10x the start or 0.5) and ends closer than it began.
        theta = (x0 + amplitude * q_dir)[None]
        for _ in range(steps):
            theta = grid.step(grid.step(theta))
            if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 1e6:
                return False
            dist = float(np.linalg.norm(theta[0] - x0))
            if dist > max(10 * amplitude, 0.5):
                return False
        return dist < amplitude

    lo, hi = 1e-6, 1.0
    if not returns(lo):
        return {"delta": -delta, "threshold": "<1e-6"}
    if returns(hi):
        return {"delta": -delta, "threshold": ">1"}
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        if returns(mid):
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.05:
            break
    return {"delta": -delta, "threshold": float(np.sqrt(lo * hi))}


def dynamics_point(task):
    system, seed, row, deltas, steps = task
    net, x, y, theta0 = network(system, seed)
    o, h_b = row["log10_output"], row["log10_hidden_boundary"]
    x0 = np.asarray(row["orbit_point"])
    grid = GridMap(net, x, y, [10.0 ** row["log10_hidden_last_attracting"]], [10.0 ** o])
    A = dense_two_step(grid, x0)
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(np.where(vals.imag > 1e-12, np.abs(vals), -1)))
    q_dir = vecs[:, k].real / np.linalg.norm(vecs[:, k].real)
    out = {"log10_output": o, "log10_hidden_boundary": h_b, "d": row["d"],
           "beyond": [beyond(net, x, y, x0, h_b, o, d, steps) for d in deltas],
           "before": [threshold(net, x, y, x0, h_b, o, d, q_dir, steps // 4) for d in deltas]}
    # hysteresis: sweep up through the boundary and back, carrying the state
    path = np.r_[np.linspace(-max(deltas), max(deltas), 9), np.linspace(max(deltas), -max(deltas), 9)]
    theta = x0[None].copy()
    sweep = []
    for dlt in path:
        g = GridMap(net, x, y, [10.0 ** (h_b + dlt)], [10.0 ** o])
        ok = True
        for _ in range(steps // 8):
            theta = g.step(g.step(theta))
            if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 1e6:
                ok = False
                break
        sweep.append({"delta": float(dlt), "amplitude": float(np.linalg.norm(theta[0] - x0)) if ok else None})
        if not ok:
            theta = x0[None].copy()
    out["sweep"] = sweep
    return out


def cycle_at(net, x, y, h, o, seed_state):
    """Newton-refined period-2 orbit point and multipliers of the two-step map at (h, o)."""
    from neural_shrimp.orbits import certify_orbit
    grid = GridMap(net, x, y, [10.0 ** h], [10.0 ** o])
    map_fn, jacobian = grid.cell_map(0)
    cert = certify_orbit(map_fn, jacobian, seed_state, 2, candidate_tolerance=0.05)
    if not cert["verified"]:
        return None, None, grid
    xc = np.asarray(cert["orbit"][0])
    return xc, dense_two_step(grid, xc), grid


def dynamics2_point(task):
    """Corrected checks: distances and normal-form amplitudes relative to the 2-cycle at each offset."""
    from neural_shrimp.criticality import critical_pair
    system, seed, row, deltas, steps = task
    net, x, y, theta0 = network(system, seed)
    o, h_b, d = row["log10_output"], row["log10_hidden_boundary"], row["d"]
    x0 = np.asarray(row["orbit_point"])
    out = {"log10_output": o, "log10_hidden_boundary": h_b, "d": d, "beyond": [], "before": []}
    for delta in deltas:
        xc, A, grid = cycle_at(net, x, y, h_b + delta, o, x0)
        if xc is None:
            out["beyond"].append({"delta": delta, "cycle": "not found"})
            continue
        mu, q, p, _ = critical_pair(A)
        growth = float(np.log(abs(mu)))
        predicted = float(np.sqrt(growth / -d)) if d < 0 and growth > 0 else None
        w0 = predicted if predicted else 1e-3
        theta = (xc + 2 * (w0 * q).real)[None]
        ws, far = [], 0.0
        status = "settled"
        for k in range(steps):
            theta = grid.step(grid.step(theta))
            if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 1e6:
                status = "diverged"
                break
            dist = float(np.linalg.norm(theta[0] - xc))
            far = max(far, dist)
            if k >= steps - 4000:
                ws.append(abs(np.vdot(p, theta[0] - xc)))
        out["beyond"].append({"delta": delta, "status": status, "log_modulus": growth,
                              "predicted_w": predicted, "w_median": float(np.median(ws)) if ws else None,
                              "w_min": float(np.min(ws)) if ws else None, "w_max": float(np.max(ws)) if ws else None,
                              "max_distance": far})
    for delta in deltas:
        xc, A, grid = cycle_at(net, x, y, h_b - delta, o, x0)
        if xc is None:
            out["before"].append({"delta": -delta, "cycle": "not found"})
            continue
        vals, vecs = np.linalg.eig(A)
        k = int(np.argmax(np.where(vals.imag > 1e-12, np.abs(vals), -1)))
        direction = vecs[:, k].real / np.linalg.norm(vecs[:, k].real)
        decay = float(-np.log(abs(vals[k])))
        horizon = int(min(steps, max(2000, 6.0 / max(decay, 1e-9))))

        def returns(amplitude):
            th = (xc + amplitude * direction)[None]
            for _ in range(horizon):
                th = grid.step(grid.step(th))
                if not np.all(np.isfinite(th)) or np.max(np.abs(th)) > 1e6:
                    return False
                if float(np.linalg.norm(th[0] - xc)) > max(10 * amplitude, 0.5):
                    return False
            return float(np.linalg.norm(th[0] - xc)) < 0.5 * amplitude

        lo, hi = 1e-6, 0.3
        if not returns(lo):
            thr = "<1e-6"
        elif returns(hi):
            thr = ">0.3"
        else:
            for _ in range(24):
                mid = np.sqrt(lo * hi)
                lo, hi = (mid, hi) if returns(mid) else (lo, mid)
                if hi / lo < 1.1:
                    break
            thr = float(np.sqrt(lo * hi))
        predicted_thr = float(np.sqrt(decay / d)) if d > 0 else None
        out["before"].append({"delta": -delta, "decay": decay, "horizon": horizon, "threshold": thr,
                              "predicted_unstable_circle_w": predicted_thr})
    return out


def cmd_dynamics2(args):
    curve = json.loads(Path(args.curve).read_text())["rows"]
    cands = [r for r in curve if r.get("type") == "neimark_sacker"]
    chosen = [min(cands, key=lambda r: abs(r["log10_output"] - t)) for t in args.outputs]
    tasks = [(args.system, args.seed, r, args.deltas, args.steps) for r in chosen]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(dynamics2_point, tasks, chunksize=1))
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "dynamics2.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
    for r in rows:
        print(json.dumps({"out": round(r["log10_output"], 4), "d": round(r["d"], 4),
                          "beyond": [(b["delta"], b.get("status"), b.get("predicted_w") and round(b["predicted_w"], 5),
                                      b.get("w_median") and round(b["w_median"], 5), b.get("max_distance") and round(b["max_distance"], 4))
                                     for b in r["beyond"]],
                          "before": [(b["delta"], b.get("threshold"), b.get("predicted_unstable_circle_w")) for b in r["before"]]}))


def cmd_dynamics(args):
    curve = json.loads(Path(args.curve).read_text())["rows"]
    chosen = []
    for target in args.outputs:
        cands = [r for r in curve if r.get("type") == "neimark_sacker"]
        chosen.append(min(cands, key=lambda r: abs(r["log10_output"] - target)))
    tasks = [(args.system, args.seed, r, args.deltas, args.steps) for r in chosen]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(dynamics_point, tasks, chunksize=1))
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "dynamics.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
    for r in rows:
        print(json.dumps({"out": r["log10_output"], "d": r["d"],
                          "beyond": [(b["delta"], b.get("escaped"), b.get("amplitude_median")) for b in r["beyond"]],
                          "before": [(b["delta"], b["threshold"]) for b in r["before"]]}))


def main():
    os.nice(10)  # shared machine: run below normal priority (inherited by worker processes)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    control = sub.add_parser("control")
    control.add_argument("--kappas", type=float, nargs="+", default=[-4, -2, -1, 0, 1, 2, 4, 8])
    control.add_argument("--name", default="control")
    control.add_argument("--amplitude-deltas", type=float, nargs="+", default=[1e-5, 1e-6])
    control.add_argument("--amplitude-steps", type=int, default=400000)
    curve = sub.add_parser("curve")
    curve.add_argument("--system", default="S2")
    curve.add_argument("--seed", type=int, default=28011)
    curve.add_argument("--ns", default=str(DATA / "explore/a3-ns-curve/s2-28011-merged/ns-curve.json"))
    curve.add_argument("--stride", type=int, default=1)
    curve.add_argument("--workers", type=int, default=8)
    curve.add_argument("--name", default="s2-28011-curve")
    curve.add_argument("--only", type=float, nargs="*", default=None, help="restrict to these output rates")
    curve.add_argument("--margins", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2],
                       help="start margins below the boundary; negative values are absolute start rates")
    dyn = sub.add_parser("dynamics")
    dyn.add_argument("--system", default="S2")
    dyn.add_argument("--seed", type=int, default=28011)
    dyn.add_argument("--curve", default=str(OUT / "s2-28011-curve/curve.json"))
    dyn.add_argument("--outputs", type=float, nargs="+", default=[-0.42, -0.30, -0.20, -0.08, -0.034, -0.018, 0.013, 0.045])
    dyn.add_argument("--deltas", type=float, nargs="+", default=[1e-4, 4e-4, 1.6e-3])
    dyn.add_argument("--steps", type=int, default=40000)
    dyn.add_argument("--workers", type=int, default=8)
    dyn.add_argument("--name", default="s2-28011-dynamics")
    dyn2 = sub.add_parser("dynamics2")
    dyn2.add_argument("--system", default="S2")
    dyn2.add_argument("--seed", type=int, default=28011)
    dyn2.add_argument("--curve", default=str(OUT / "s2-28011-curve/curve.json"))
    dyn2.add_argument("--outputs", type=float, nargs="+", default=[-0.44, -0.30, -0.20, -0.08, -0.034, -0.018, -0.0026, 0.013])
    dyn2.add_argument("--deltas", type=float, nargs="+", default=[1e-4, 4e-4, 1.6e-3])
    dyn2.add_argument("--steps", type=int, default=120000)
    dyn2.add_argument("--workers", type=int, default=8)
    dyn2.add_argument("--name", default="s2-28011-dynamics")
    args = parser.parse_args()
    {"control": cmd_control, "curve": cmd_curve, "dynamics": cmd_dynamics, "dynamics2": cmd_dynamics2}[args.command](args)


if __name__ == "__main__":
    main()
