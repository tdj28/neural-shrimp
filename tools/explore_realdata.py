"""Exploratory real-data generality check: does the quasi-periodic route appear?

Network: a standard-parameterization tanh MLP (64-h-1, biases, PyTorch-default
initialization) on a fixed subset of the UCI digits with +-1 targets and mean
squared error, trained by full-batch float64 gradient descent with ONE global
learning rate (the equal-rates diagonal of the per-layer rate plane).

Subcommands (all write under --out):
  calibrate  sharpness at initialization and at an L-BFGS minimum (flip
             threshold 2 / lambda_max), and the cost of one batched step.
  scan       fixed-start sweep over log10 learning rate (the network's
             initialization, or a saved state with --init): EXP-028
             screening (status, period, FTLE), then ``keep`` further steps
             recording the loss and a fixed 32-dimensional random projection of
             the state; per cell, the rotation number of the two-step map
             (principal-plane angle about the centroid), the fraction of
             forward angle steps, and the main loss-spectrum peaks.
  continue   certify a period-p attracting orbit at a scanned learning rate and
             continue it in log10 learning rate until it stops attracting;
             type the boundary from the dominant multiplier (fold, flip,
             Neimark-Sacker) and report arg(mu) / 2 pi.
  branch     polish a minimum (L-BFGS, then Newton), place the flip at
             2 / lambda_max, seed the period-2 orbit just beyond it along the
             top Hessian eigenvector, and continue that orbit in log10
             learning rate to its stability boundary (independent of the
             fixed-initialization scan).
  sweep      adiabatic sweep in log10 learning rate from a saved state (or the
             initialization): each rate starts from the previous rate's final
             state, so one attractor branch is followed; per rate, rotation
             number, recurrence period of the projected state, FTLE and loss
             spectrum.

Exploratory under the adaptive research protocol. Seeds 29000-29999 are
refused (reserved for confirmatory tests).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.continuation import continue_ray  # noqa: E402
from neural_shrimp.gdmaps import GridMap, iterate_and_screen  # noqa: E402
from neural_shrimp.realdata import make_digits  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))
PLAN = ROOT / "experiments/trainability/EXP-028-plan.json"
PROJECTION_SEED = 28100


def network(config):
    return make_digits(config["seed"], width=config["width"], n=config["n"], task=config["task"],
                       ridge=config["ridge"])


def screening_options():
    it = json.loads(PLAN.read_text())["iteration"]
    options = {k: it[k] for k in ("tail_steps", "ftle_steps", "ftle_align", "max_period", "min_repeats",
                                  "atol", "rtol", "chaos_threshold", "drift_threshold",
                                  "divergence_threshold", "tangent_seed")}
    options["sd_horizons"] = ()
    return options


def rotation(points):
    """Rotation number (turns per step) of points on a closed curve, plus quality measures."""
    centered = points - points.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    planar = centered @ vt[:2].T
    steps = np.diff(np.arctan2(planar[:, 1], planar[:, 0]))
    steps = (steps + np.pi) % (2 * np.pi) - np.pi
    rho = abs(steps.sum()) / (2 * np.pi * len(steps))
    forward = float(np.mean(np.sign(steps) == np.sign(steps.sum()))) if steps.sum() != 0 else 0.0
    share = float((singular[:2] ** 2).sum() / max((singular ** 2).sum(), 1e-300))
    return float(rho), forward, share


def spectrum_peaks(loss, count=6):
    """Frequencies (cycles per step) and relative powers of the largest loss-spectrum peaks."""
    signal = loss - loss.mean()
    if not np.any(signal):
        return []
    power = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    freqs = np.fft.rfftfreq(len(signal))
    interior = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])) + 1
    if power[-1] > power[-2]:
        interior = np.r_[interior, len(power) - 1]
    top = interior[np.argsort(power[interior])[::-1][:count]]
    total = power[1:].sum()
    return [[float(freqs[i]), float(power[i] / total)] for i in top]


def scan_chunk(task):
    config, log_eta, steps, keep, init = task
    net, x, y, theta0 = network(config)
    if init is not None:
        theta0 = np.load(init)
    eta = 10.0 ** np.asarray(log_eta)
    grid = GridMap(net, x, y, eta, eta)

    def step_subset(states, idx):
        grad, loss = grid.net.gradient(states, grid.x, grid.y, return_loss=True)
        return states - grid.rates[idx] * grad, loss

    def jvp_subset(states, idx, vectors):
        return vectors - grid.rates[idx] * grid.net.hvp(states, grid.x, grid.y, vectors)

    initial = np.repeat(theta0[None], len(eta), axis=0)
    labels = iterate_and_screen(step_subset, jvp_subset, initial, steps, **screening_options())
    projection = np.random.default_rng(PROJECTION_SEED).normal(size=(net.n_parameters, 32))
    theta = labels["final_state"].copy()
    alive = np.all(np.isfinite(theta), axis=1)
    theta[~alive] = 0.0
    losses = np.full((keep, len(eta)), np.nan)
    projected = np.full((keep, len(eta), 32), np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(keep):
            grad, loss = net.gradient(theta, x, y, return_loss=True)
            losses[k] = loss
            projected[k] = theta @ projection
            theta = theta - grid.rates * grad
    rows = []
    for i in range(len(eta)):
        row = {"log10_eta": float(log_eta[i]), "status": int(labels["status"][i]),
               "period": int(labels["period"][i]),
               "ftle": None if not np.isfinite(labels["ftle"][i]) else float(labels["ftle"][i]),
               "tail_mean_loss": None if not np.isfinite(labels["tail_mean_loss"][i])
               else float(labels["tail_mean_loss"][i])}
        if alive[i] and np.all(np.isfinite(losses[:, i])):
            row["loss_min"], row["loss_max"] = float(losses[:, i].min()), float(losses[:, i].max())
            if row["status"] in (3, 4, 5) or (row["status"] == 2 and row["period"] > 2):
                for name, points in (("rho2", projected[0::2, i]), ("rho1", projected[:, i])):
                    rho, forward, share = rotation(points)
                    row[name], row[name + "_forward"], row[name + "_share"] = rho, forward, share
            row["peaks"] = spectrum_peaks(losses[:, i])
        rows.append(row)
    return rows, losses.T.astype(np.float32)


def cmd_calibrate(config, args, out):
    from scipy.optimize import minimize

    net, x, y, theta0 = network(config)
    report = {"config": config, "parameters": net.n_parameters}

    def lam_max(theta):
        return float(np.linalg.eigvalsh(net.hessian(theta, x, y))[-1])

    report["lambda_max_init"] = lam_max(theta0)
    fun = lambda t: float(net.loss(t[None], x, y)[0])  # noqa: E731
    jac = lambda t: net.gradient(t[None], x, y)[0]  # noqa: E731
    result = minimize(fun, theta0, jac=jac, method="L-BFGS-B",
                      options={"maxiter": 20000, "gtol": 1e-12, "ftol": 1e-15})
    report["lbfgs"] = {"loss": float(result.fun), "grad_norm": float(np.linalg.norm(jac(result.x))),
                       "iterations": int(result.nit), "message": str(result.message)}
    eig = np.linalg.eigvalsh(net.hessian(result.x, x, y))
    report["lambda_max_min"], report["lambda_min_min"] = float(eig[-1]), float(eig[0])
    report["eta_flip_at_lbfgs_min"] = 2.0 / float(eig[-1])
    report["accuracy_at_lbfgs_min"] = float(np.mean(np.sign(net.predict(result.x[None], x)[:, 0]) == y))
    cells = 50
    grid = GridMap(net, x, y, np.full(cells, 0.01), np.full(cells, 0.01))
    theta = np.repeat(theta0[None], cells, axis=0)
    start = time.perf_counter()
    for _ in range(20):
        theta = grid.step(theta)
    report["seconds_per_step_50_cells"] = (time.perf_counter() - start) / 20
    (out / "calibrate.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def cmd_scan(config, args, out):
    log_eta = np.linspace(args.range[0], args.range[1], args.cells)
    chunks = np.array_split(log_eta, max(1, int(np.ceil(args.cells / args.chunk))))
    tasks = [(config, c, args.steps, args.keep, args.init) for c in chunks]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(scan_chunk, tasks, chunksize=1))
    rows = [r for chunk_rows, _ in results for r in chunk_rows]
    losses = np.concatenate([l for _, l in results], axis=0)
    np.savez_compressed(out / "scan-losses.npz", log10_eta=log_eta, losses=losses)
    (out / "scan.json").write_text(json.dumps({"config": config, "args": vars(args), "rows": rows}, indent=1))
    counts = {}
    for r in rows:
        key = f"{r['status']}/{r['period']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"out": str(out), "status/period counts": counts}))


def cmd_continue(config, args, out):
    net, x, y, theta0 = network(config)
    start = args.start
    grid = GridMap(net, x, y, [10.0 ** start], [10.0 ** start])

    def step_subset(states, idx):
        grad, loss = grid.net.gradient(states, grid.x, grid.y, return_loss=True)
        return states - grid.rates[idx] * grad, loss

    def jvp_subset(states, idx, vectors):
        return vectors - grid.rates[idx] * grid.net.hvp(states, grid.x, grid.y, vectors)

    labels = iterate_and_screen(step_subset, jvp_subset, theta0[None], args.steps, **screening_options())
    screened = int(labels["period"][0])
    period = args.period if args.period is not None else screened
    report = {"config": config, "start_log10_eta": start, "start_status": int(labels["status"][0]),
              "start_period": screened, "certified_period": period}
    if period < 1:
        report["type"] = "start_not_periodic"
    else:
        def family(p):
            eta = 10.0 ** p[0]
            return GridMap(net, x, y, [eta], [eta]).cell_map(0)

        result = continue_ray(family, labels["final_state"][0], period, (start, 0.0),
                              (float(np.sign(args.direction)), 0.0), args.s_max, args.h0,
                              bisection_tolerance=args.tolerance)
        mu = complex(*result["dominant"]) if result["dominant"] else None
        report.update({
            "type": result["type"],
            "log10_eta_boundary": None if result["s_boundary"] is None
            else start + np.sign(args.direction) * result["s_boundary"],
            "bracket": result["bracket"], "dominant": result["dominant"],
            "dominant_modulus": None if mu is None else abs(mu),
            "theta_over_2pi": None if mu is None else abs(np.angle(mu)) / (2 * np.pi),
            "path": result["path"]})
        if result.get("last_attracting_state") is not None:
            np.save(out / f"continue-{start:+.5f}-state.npy", np.array(result["last_attracting_state"]))
    (out / f"continue-{start:+.5f}.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "path"}, indent=1, default=float))


def polished_minimum(net, x, y, theta0, newton_steps=30):
    """L-BFGS from ``theta0`` followed by Newton polishing with the dense Hessian."""
    from scipy.optimize import minimize

    fun = lambda t: float(net.loss(t[None], x, y)[0])  # noqa: E731
    jac = lambda t: net.gradient(t[None], x, y)[0]  # noqa: E731
    theta = minimize(fun, theta0, jac=jac, method="L-BFGS-B",
                     options={"maxiter": 50000, "maxfun": 100000, "gtol": 1e-12, "ftol": 1e-16}).x
    for _ in range(newton_steps):
        g = jac(theta)
        if np.linalg.norm(g) < 1e-13:
            break
        theta = theta - np.linalg.lstsq(net.hessian(theta, x, y), g, rcond=1e-14)[0]
    return theta, float(np.linalg.norm(jac(theta)))


def cmd_branch(config, args, out):
    """Follow the minimum's own branch: flip at 2 / lambda_max, then the period-2 orbit it spawns."""
    net, x, y, theta0 = network(config)
    theta_star, grad_norm = polished_minimum(net, x, y, theta0)
    eig, vec = np.linalg.eigh(net.hessian(theta_star, x, y))
    lam, v = float(eig[-1]), vec[:, -1]
    eta_c = 2.0 / lam
    report = {"config": config, "grad_norm_at_minimum": grad_norm, "loss_at_minimum":
              float(net.loss(theta_star[None], x, y)[0]), "lambda_max": lam, "lambda_min": float(eig[0]),
              "eta_flip": eta_c, "log10_eta_flip": float(np.log10(eta_c)), "attempts": []}
    np.save(out / "branch-minimum.npy", theta_star)

    def family(p):
        eta = 10.0 ** p[0]
        return GridMap(net, x, y, [eta], [eta]).cell_map(0)

    for factor in args.factors:
        eta = eta_c * factor
        grid = GridMap(net, x, y, [eta], [eta])
        theta = (theta_star + args.kick * v)[None]
        for _ in range(args.steps):
            theta = grid.step(theta)
        orbit_check = [theta[0]]
        for _ in range(4):
            orbit_check.append(grid.step(orbit_check[-1][None])[0])
        two_cycle_gap = float(np.max(np.abs(orbit_check[2] - orbit_check[0])))
        one_cycle_gap = float(np.max(np.abs(orbit_check[1] - orbit_check[0])))
        attempt = {"factor": factor, "log10_eta": float(np.log10(eta)), "two_step_gap": two_cycle_gap,
                   "one_step_gap": one_cycle_gap,
                   "distance_from_minimum": float(np.linalg.norm(theta[0] - theta_star))}
        if two_cycle_gap < args.gap and one_cycle_gap > 100 * two_cycle_gap:
            result = continue_ray(family, theta[0], 2, (float(np.log10(eta)), 0.0), (1.0, 0.0), args.s_max,
                                  args.h0, bisection_tolerance=args.tolerance)
            mu = complex(*result["dominant"]) if result["dominant"] else None
            attempt.update({
                "type": result["type"],
                "log10_eta_boundary": None if result["s_boundary"] is None
                else float(np.log10(eta)) + result["s_boundary"],
                "dominant": result["dominant"], "dominant_modulus": None if mu is None else abs(mu),
                "theta_over_2pi": None if mu is None else abs(np.angle(mu)) / (2 * np.pi),
                "path_points": len(result["path"])})
            if result.get("last_attracting_state") is not None:
                np.save(out / f"branch-{factor:.4f}-last-state.npy", np.array(result["last_attracting_state"]))
        report["attempts"].append(attempt)
        print(json.dumps(attempt, default=float), flush=True)
        if "type" in attempt:
            break
    (out / "branch.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "attempts"}, indent=1, default=float))


def recurrence_period(projected, window=2048, max_lag=2048, rtol=1e-7):
    """Smallest lag at which the final ``window`` projected states repeat, or 0."""
    tail = projected[-(window + max_lag):]
    scale = max(float(np.max(np.abs(tail))), 1e-300)
    for lag in range(1, max_lag + 1):
        if np.max(np.abs(tail[lag:lag + window] - tail[:window])) < rtol * scale:
            return lag
    return 0


def cmd_sweep(config, args, out):
    """Adiabatic sweep: each learning rate starts from the previous rate's final state."""
    net, x, y, theta0 = network(config)
    theta = np.load(args.init) if args.init else theta0
    log_eta = np.linspace(args.range[0], args.range[1], args.cells)
    projection = np.random.default_rng(PROJECTION_SEED).normal(size=(net.n_parameters, 32))
    tangent = np.random.default_rng(0).normal(size=net.n_parameters)
    rows, traces = [], []
    for le in log_eta:
        eta = 10.0 ** le
        grid = GridMap(net, x, y, [eta], [eta])
        state = theta[None].copy()
        ok = True
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(args.steps):
                state = grid.step(state)
            losses, projected = np.empty(args.keep), np.empty((args.keep, 32))
            log_growth, ftle_steps = 0.0, min(args.ftle_steps, args.keep)
            vector = tangent / np.linalg.norm(tangent)
            for k in range(args.keep):
                grad, loss = net.gradient(state, x, y, return_loss=True)
                losses[k], projected[k] = loss[0], state[0] @ projection
                if k >= args.keep - ftle_steps:
                    vector = grid.jvp(state, vector[None])[0]
                    norm = np.linalg.norm(vector)
                    log_growth += np.log(norm)
                    vector /= norm
                state = state - grid.rates * grad
            ok = bool(np.all(np.isfinite(state)) and np.all(np.isfinite(losses)))
        row = {"log10_eta": float(le), "finite": ok}
        if ok:
            theta = state[0]
            row["ftle"] = log_growth / ftle_steps
            row["period"] = recurrence_period(projected)
            row["loss_min"], row["loss_max"] = float(losses.min()), float(losses.max())
            row["rho2"], row["rho2_forward"], row["rho2_share"] = rotation(projected[0::2])
            row["turns"] = row["rho2"] * len(projected[0::2])
            row["peaks"] = spectrum_peaks(losses)
            traces.append(losses.astype(np.float32))
        rows.append(row)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items() if k != "peaks"}),
              flush=True)
        if not ok:
            break
    np.savez_compressed(out / "sweep-losses.npz", log10_eta=log_eta[:len(traces)], losses=np.array(traces))
    (out / "sweep.json").write_text(json.dumps({"config": config, "args": vars(args), "rows": rows}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["calibrate", "scan", "continue", "branch", "sweep"])
    parser.add_argument("--ftle-steps", type=int, default=4000, help="tangent steps at the end of each record (sweep)")
    parser.add_argument("--factors", type=float, nargs="+", default=[1.002, 1.01, 1.03, 1.1],
                        help="eta / eta_flip values to seed the period-2 orbit (branch)")
    parser.add_argument("--kick", type=float, default=1e-3, help="initial offset along the top eigenvector (branch)")
    parser.add_argument("--gap", type=float, default=1e-5,
                        help="largest two-step gap accepted as a period-2 seed for Newton (branch)")
    parser.add_argument("--seed", type=int, default=28101)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--task", default="halves")
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--range", type=float, nargs=2, help="log10 learning-rate range (scan)")
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--keep", type=int, default=4096)
    parser.add_argument("--init", default=None,
                        help="optional .npy state to start every scan cell from instead of the initialization")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start", type=float, help="log10 learning rate inside the orbit's region (continue)")
    parser.add_argument("--period", type=int, default=None,
                        help="period to certify when the screen misses it, e.g. slow drift (continue)")
    parser.add_argument("--direction", type=float, default=1.0)
    parser.add_argument("--s-max", type=float, default=0.3)
    parser.add_argument("--h0", type=float, default=0.002)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    config = {"seed": args.seed, "width": args.width, "n": args.n, "task": args.task, "ridge": args.ridge}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    {"calibrate": cmd_calibrate, "scan": cmd_scan, "continue": cmd_continue,
     "branch": cmd_branch, "sweep": cmd_sweep}[args.command](config, args, out)


if __name__ == "__main__":
    main()
