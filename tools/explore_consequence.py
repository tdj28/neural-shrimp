"""Exploratory A3b: training consequences of regimes beyond the edge of stability.

simulate  For every cell of a one-dimensional learning-rate slice, iterate
          full-batch gradient descent from the system's initial parameters
          together with copies perturbed at initialization (sizes DELTAS, a
          few random directions), then:
            - record parameter and function-space disagreement with the base
              run at several horizons (reproducibility);
            - anneal every copy and the base (step decay ANNEAL_STAGES: rates
              x0.1 for 3,000 steps, then x0.01 for 1,000) and record the final
              function-space disagreement (two-phase schedule);
            - continue the base run for TAIL steps, recording training and
              held-out losses, two tangent vectors (largest FTLE twice), two
              on-attractor copies started 1e-10 away (chaos qualification),
              the averaged iterate, the preconditioned sharpness at 16 states,
              and the last two-step states (rotation number);
            - anneal from ANNEAL_STARTS consecutive tail states (stopping-time
              sensitivity of the final model; distinct outcomes counted in
              function space at DISTINCT_TOLERANCE of the target RMS).
analyze   Regime labels (neural_shrimp.regimes.classify), per-regime summaries
          against ordinary period-2 and qualified chaotic controls, nested
          model comparisons (rotation and locking features added to FTLE and
          sharpness) and figures.

All seeds used here are below 29000 (29000-29999 are reserved for
confirmatory tests). Exploratory under docs/adaptive-research-protocol.md.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.gdmaps import GridMap, screen_tails  # noqa: E402
from neural_shrimp.models import make_teacher_regression  # noqa: E402
from neural_shrimp.regimes import (  # noqa: E402
    cycle_rotation, peak_frequencies, rotation_number, spectrum_features,
)
from neural_shrimp.trainability import SYSTEMS  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))
OUT = DATA / "explore" / "consequence"

DELTAS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
DIRECTIONS = 6
HORIZONS = (2000, 10000, 40000)
TAIL = 8192
LOBE_STEPS = 4096
SCREEN_STEPS = 384
SHARPNESS_STATES = 16
ANNEAL_STARTS = 64
ANNEAL_STAGES = ((0.1, 3000), (0.01, 1000))
DISTINCT_TOLERANCE = 1e-4
DIVERGENCE = 1e8
PERTURB_SEED, TANGENT_SEED, SEPARATION_SEED, PROBE_SEED = 28700, 28710, 28720, 28900


def probe_data(system, seed, x, y):
    """Held-out inputs and targets (S2 teacher test split; S1R random inputs, no targets)."""
    if system == "S2":
        data = make_teacher_regression(seed=int(seed), d=2, teacher_width=3, n_train=24,
                                       n_validation=48, n_test=96)
        return data.test_x, data.test_y
    rng = np.random.default_rng(PROBE_SEED)
    return rng.normal(size=(256, x.shape[1])), None


def unit_vectors(count, dim, seed):
    out = []
    for k in range(count):
        v = np.random.default_rng(seed + k).normal(size=dim)
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def step_masked(grid, theta, alive):
    with np.errstate(over="ignore", invalid="ignore"):
        safe = np.where(alive[:, None], theta, 0.0)
        new, loss = grid.step(safe, return_loss=True)
        ok = np.all(np.isfinite(new), axis=1) & (np.max(np.abs(new), axis=1) < DIVERGENCE) & np.isfinite(loss)
    alive = alive & ok
    new[~alive] = np.nan
    loss = np.where(alive, loss, np.nan)
    return new, loss, alive


def run_steps(grid, theta, steps):
    alive = np.all(np.isfinite(theta), axis=1)
    for _ in range(steps):
        theta, _, alive = step_masked(grid, theta, alive)
    return theta, alive


def anneal(net, x, y, log_hidden, log_output, theta, stages):
    """Step-decay schedule: for each (factor, steps), run GD at factor x the rates."""
    for factor, steps in stages:
        grid = GridMap(net, x, y, factor * 10.0 ** log_hidden, factor * 10.0 ** log_output)
        theta, _ = run_steps(grid, theta, steps)
    return theta


def function_rms(net, a, b, probe_x, scale):
    """RMS over probe inputs of prediction differences, per pair of rows."""
    with np.errstate(over="ignore", invalid="ignore"):
        fa = net.predict(np.nan_to_num(a), probe_x)
        fb = net.predict(np.nan_to_num(b), probe_x)
        out = np.sqrt(np.mean((fa - fb) ** 2, axis=0)) / scale
    bad = ~(np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1))
    out[bad] = np.nan
    return out


def mse(net, theta, x, y):
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.mean((net.predict(np.nan_to_num(theta), x) - y[:, None]) ** 2, axis=0)
    out[~np.all(np.isfinite(theta), axis=1)] = np.nan
    return out


def simulate_chunk(task):
    system, seed, log_hidden, log_output, cfg = task
    net, x, y, theta0 = SYSTEMS[system](seed)
    probe_x, probe_y = probe_data(system, seed, x, y)
    scale = float(np.sqrt(np.mean(y ** 2)))
    n, p = len(log_hidden), net.n_parameters
    deltas, dirs_n = cfg["deltas"], cfg["directions"]
    k_copies = len(deltas) * dirs_n
    group = 1 + k_copies
    dirs = unit_vectors(dirs_n, p, PERTURB_SEED)
    init = np.repeat(theta0[None, None, :], group, axis=1).repeat(n, axis=0)
    for i, d in enumerate(deltas):
        init[:, 1 + i * dirs_n:1 + (i + 1) * dirs_n, :] += d * dirs[None, :, :]
    lh = np.repeat(log_hidden, group)
    lo = np.full(n * group, log_output)
    grid = GridMap(net, x, y, 10.0 ** lh, 10.0 ** lo)
    theta = init.reshape(n * group, p)
    alive = np.ones(len(theta), dtype=bool)
    horizons = cfg["horizons"]
    param_dis = np.full((n, k_copies, len(horizons)), np.nan)
    func_dis = np.full((n, k_copies, len(horizons)), np.nan)
    for t in range(1, horizons[-1] + 1):
        theta, _, alive = step_masked(grid, theta, alive)
        if t in horizons:
            h = horizons.index(t)
            g = theta.reshape(n, group, p)
            base = np.repeat(g[:, :1, :], k_copies, axis=1).reshape(-1, p)
            copies = g[:, 1:, :].reshape(-1, p)
            param_dis[:, :, h] = np.linalg.norm(copies - base, axis=1).reshape(n, k_copies)
            func_dis[:, :, h] = function_rms(net, copies, base, probe_x, scale).reshape(n, k_copies)
    # two-phase schedule: anneal base and copies from the last horizon
    annealed = anneal(net, x, y, lh, lo, theta.copy(), cfg["anneal_stages"])
    g = annealed.reshape(n, group, p)
    base = np.repeat(g[:, :1, :], k_copies, axis=1).reshape(-1, p)
    anneal_func_dis = function_rms(net, g[:, 1:, :].reshape(-1, p), base, probe_x, scale).reshape(n, k_copies)
    # tail of the base run with tangents and on-attractor copies
    base0 = theta.reshape(n, group, p)[:, 0, :]
    sep_dirs = unit_vectors(2, p, SEPARATION_SEED)
    tail_init = np.concatenate([base0, base0 + 1e-10 * sep_dirs[0], base0 + 1e-10 * sep_dirs[1]])
    grid3 = GridMap(net, x, y, np.tile(10.0 ** log_hidden, 3), np.full(3 * n, 10.0 ** log_output))
    grid1 = GridMap(net, x, y, 10.0 ** log_hidden, np.full(n, 10.0 ** log_output))
    tangents = [unit_vectors(1, p, TANGENT_SEED + j)[0][None, :].repeat(n, 0) for j in range(2)]
    log_growth = [np.zeros(n), np.zeros(n)]
    tail_steps = cfg["tail"]
    lobe_steps = cfg["lobe_steps"]
    train_loss = np.full((tail_steps, n), np.nan)
    test_mse = np.full((tail_steps, n), np.nan) if probe_y is not None else None
    lobe = np.full((lobe_steps, n, p), np.nan)
    sharp_at = set(np.linspace(0, tail_steps - 1, cfg["sharpness_states"]).astype(int).tolist())
    sharp_states = []
    running = np.zeros((n, p))
    running_sq = np.zeros(n)
    states = tail_init
    alive3 = np.all(np.isfinite(states), axis=1)
    for t in range(tail_steps):
        current = states[:n]
        ok = alive3[:n]
        with np.errstate(over="ignore", invalid="ignore"):
            for j in range(2):
                v = grid1.jvp(np.where(ok[:, None], current, 0.0), tangents[j])
                norm = np.linalg.norm(v, axis=1)
                log_growth[j] += np.log(np.where(norm > 0, norm, np.nan))
                tangents[j] = v / np.where(norm > 0, norm, 1.0)[:, None]
        if t in sharp_at:
            sharp_states.append(current.copy())
        running += np.where(ok[:, None], current, np.nan)
        running_sq += np.where(ok, np.sum(current * current, axis=1), np.nan)
        if probe_y is not None:
            test_mse[t] = mse(net, current, probe_x, probe_y)
        if t >= tail_steps - lobe_steps:
            lobe[t - (tail_steps - lobe_steps)] = current
        states, loss, alive3 = step_masked(grid3, states, alive3)
        train_loss[t] = loss[:n]
    last = lobe[-1]
    ftle = np.stack([g_ / tail_steps for g_ in log_growth], axis=1)
    separation = np.nanmedian(np.stack([np.linalg.norm(states[n:2 * n] - states[:n], axis=1),
                                        np.linalg.norm(states[2 * n:] - states[:n], axis=1)], axis=1), axis=1)
    averaged = running / tail_steps
    amplitude = np.sqrt(np.maximum(running_sq / tail_steps - np.sum(averaged * averaged, axis=1), 0.0))
    period, residual = screen_tails(np.transpose(lobe[-cfg["screen_steps"]:], (1, 0, 2)), max_period=64,
                                    min_repeats=6, atol=1e-8, rtol=1e-6)
    status = np.where(period == -1, 0, np.where(period == 1, 1, np.where(period >= 2, 2, 3)))
    status[~np.all(np.isfinite(last), axis=1)] = 0
    rho = np.full((n, 3), np.nan)
    share = np.full(n, np.nan)
    cycle = np.zeros((n, 2), dtype=int)
    peaks = np.full((n, 3), np.nan)
    lines = np.full((n, 2), np.nan)
    for i in range(n):
        if status[i] == 0:
            continue
        r, r1, r2, s = rotation_number(lobe[::2, i, :])
        rho[i] = (r, r1, r2)
        share[i] = s
        if status[i] == 2 and period[i] >= 3:
            m, q, _ = cycle_rotation(lobe[-period[i]:, i, :])
            cycle[i] = (m, q)
        series = train_loss[:, i]
        if np.all(np.isfinite(series)):
            lines[i] = spectrum_features(series)
            found = peak_frequencies(series[::2], count=3)
            peaks[i, :len(found)] = found
    # preconditioned sharpness lambda_max(P^1/2 H P^1/2) at saved tail states
    sharp = np.full((n, len(sharp_states)), np.nan)
    root = np.sqrt(grid1.rates)
    for j, saved in enumerate(sharp_states):
        for i in range(n):
            if np.all(np.isfinite(saved[i])):
                hess = net.hessian(saved[i], x, y)
                sharp[i, j] = np.linalg.eigvalsh(root[i][:, None] * hess * root[i][None, :])[-1]
    # stopping-time sensitivity: anneal from consecutive tail states
    starts = lobe[-cfg["anneal_starts"]:]                      # (starts, n, p)
    s_count = starts.shape[0]
    flat = np.transpose(starts, (1, 0, 2)).reshape(-1, p)       # cell-major
    final = anneal(net, x, y, np.repeat(log_hidden, s_count), np.full(n * s_count, log_output), flat,
                   cfg["anneal_stages"])
    grad_norm = np.linalg.norm(net.gradient(np.nan_to_num(final), x, y), axis=1)
    final_train = net.loss(np.nan_to_num(final), x, y)
    final_train[~np.all(np.isfinite(final), axis=1)] = np.nan
    final_test = mse(net, final, probe_x, probe_y) if probe_y is not None else np.full(len(final), np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        outputs = net.predict(np.nan_to_num(final), probe_x).T.reshape(n, s_count, -1)
    finite = np.all(np.isfinite(final), axis=1).reshape(n, s_count)
    centered = outputs - np.nanmean(np.where(finite[:, :, None], outputs, np.nan), axis=1, keepdims=True)
    func_spread = np.sqrt(np.nanmean(np.where(finite[:, :, None], centered ** 2, np.nan), axis=(1, 2))) / scale
    distinct = np.zeros(n, dtype=int)
    for i in range(n):
        reps = []
        for j in range(s_count):
            if not finite[i, j]:
                continue
            row = outputs[i, j]
            if not any(np.sqrt(np.mean((row - r) ** 2)) <= DISTINCT_TOLERANCE * scale for r in reps):
                reps.append(row)
        distinct[i] = len(reps)
    avg_train = net.loss(np.nan_to_num(averaged), x, y)
    avg_test = mse(net, averaged, probe_x, probe_y) if probe_y is not None else np.full(n, np.nan)
    return {
        "log10_hidden": log_hidden, "param_dis": param_dis, "func_dis": func_dis,
        "anneal_func_dis": anneal_func_dis, "status": status, "period": period,
        "screen_residual": residual, "ftle": ftle, "separation": separation, "rho": rho,
        "planar_share": share, "cycle": cycle, "loss_peaks": peaks, "line_fraction": lines[:, 0],
        "spectral_entropy": lines[:, 1], "tail_train_mean": np.nanmean(train_loss, axis=0),
        "tail_train_min": np.nanmin(train_loss, axis=0), "tail_train_max": np.nanmax(train_loss, axis=0),
        "tail_test_mean": (np.nanmean(test_mse, axis=0) if test_mse is not None else np.full(n, np.nan)),
        "avg_iterate_train": avg_train, "avg_iterate_test": avg_test,
        "sharpness_mean": np.nanmean(sharp, axis=1), "sharpness_max": np.nanmax(sharp, axis=1),
        "anneal_train": final_train.reshape(n, s_count), "anneal_test": final_test.reshape(n, s_count),
        "anneal_grad_norm": grad_norm.reshape(n, s_count), "anneal_distinct": distinct,
        "anneal_func_spread": func_spread, "amplitude": amplitude,
    }


def cmd_simulate(args):
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=False)
    cfg = {"deltas": list(DELTAS), "directions": DIRECTIONS, "horizons": list(HORIZONS), "tail": TAIL,
           "lobe_steps": LOBE_STEPS, "screen_steps": SCREEN_STEPS, "sharpness_states": SHARPNESS_STATES,
           "anneal_starts": ANNEAL_STARTS, "anneal_stages": [list(a) for a in ANNEAL_STAGES],
           "distinct_tolerance": DISTINCT_TOLERANCE}
    if args.quick:
        cfg.update(horizons=[200, 1000, 2000], tail=1024, lobe_steps=512, sharpness_states=2,
                   anneal_starts=8, anneal_stages=[[0.1, 300], [0.01, 100]])
    hidden = np.linspace(args.hidden[0], args.hidden[1], args.cells)
    chunks = np.array_split(hidden, args.chunks)
    tasks = [(args.system, args.seed, c, args.output, cfg) for c in chunks if len(c)]
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(simulate_chunk, tasks))
    merged = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    np.savez(out / "slice.npz", **merged)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "metadata.json").write_text(json.dumps({
        "system": args.system, "seed": args.seed, "log10_output": args.output, "hidden": args.hidden,
        "cells": args.cells, "config": cfg, "seeds": {"perturb": PERTURB_SEED, "tangent": TANGENT_SEED,
                                                       "separation": SEPARATION_SEED, "probe": PROBE_SEED},
        "commit": commit}, indent=1))
    print(json.dumps({"out": str(out), "cells": args.cells, "seconds": round(time.perf_counter() - started, 1)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate")
    sim.add_argument("--name", required=True)
    sim.add_argument("--system", required=True, choices=["S2", "S1R"])
    sim.add_argument("--seed", type=int, required=True)
    sim.add_argument("--output", type=float, required=True, help="fixed log10 output-layer rate")
    sim.add_argument("--hidden", type=float, nargs=2, required=True, help="log10 hidden-layer range")
    sim.add_argument("--cells", type=int, default=600)
    sim.add_argument("--chunks", type=int, default=16)
    sim.add_argument("--workers", type=int, default=8)
    sim.add_argument("--quick", action="store_true")
    parser.add_argument("--nice", type=int, default=10, help="niceness increment for this process and workers")
    args = parser.parse_args()
    os.nice(args.nice)
    if args.seed >= 29000 and args.seed < 30000:
        raise SystemExit("seeds 29000-29999 are reserved for confirmatory tests")
    {"simulate": cmd_simulate}[args.command](args)


if __name__ == "__main__":
    main()
