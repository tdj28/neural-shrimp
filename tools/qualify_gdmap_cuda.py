"""Engineering qualification of the gdmap CUDA backend against the NumPy engine.

Artificial fixtures only (seeds 99990-99999); no study system, seed or saved
outcome is used, and nothing here is a scientific result. Usage:

    taskset -c 0-3 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python tools/qualify_gdmap_cuda.py --library MIRROR/gdmap_cuda.so \\
        [--contract CONTRACT/gdmap_cuda.so] --out OUT.json

Writes one JSON receipt to a new file outside Git and prints the tables used in
docs/gdmap-cuda-qualification.md.
"""

import argparse
from datetime import datetime, timezone
from decimal import Decimal, getcontext
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_shrimp import gdmaps_cuda as gc  # noqa: E402
from neural_shrimp.gdmaps import (  # noqa: E402
    STATUS_NAMES, GridMap, TwoLayerNet, run_grid, sohl_dickstein_measure,
)
from neural_shrimp.models import TanhRegressor, make_teacher_regression  # noqa: E402

PRODUCTION = dict(tail_steps=384, ftle_steps=4000, ftle_align=1000, sd_horizons=(100, 500, 1000))
W16_OPTIONS = dict(tail_steps=128, ftle_steps=300, ftle_align=100, sd_horizons=(100,))


def s1_like(seed=99990, activation="tanh"):
    rng = np.random.default_rng(seed)
    net = TwoLayerNet(d=8, width=8, activation=activation, in_scale=np.sqrt(2.0 / 8),
                      out_scale=1.0 / 8, bias=False, loss_scale=1.0)
    p = net.n_parameters
    return net, rng.normal(size=(p, 8)), rng.normal(size=p), rng.normal(size=p)


def s2_like(seed=99991, activation="tanh"):
    net = TwoLayerNet(d=2, width=3, activation=activation, bias=True, loss_scale=0.5, ridge=1e-3)
    data = make_teacher_regression(seed=seed, d=2, teacher_width=3, n_train=24)
    return net, data.train_x, data.train_y, TanhRegressor(2, 3).initialize(seed=seed + 1, scale=0.2)


def w16(seed=99998, activation="tanh"):
    rng = np.random.default_rng(seed)
    net = TwoLayerNet(d=16, width=16, activation=activation, in_scale=np.sqrt(2.0 / 16),
                      out_scale=1.0 / 16, bias=False, loss_scale=1.0)
    p = net.n_parameters
    return net, rng.normal(size=(p, 16)), rng.normal(size=p), rng.normal(size=p)


def odd_d(seed=99997, activation="tanh"):
    """Control for the one known mirroring gap: odd input dimension."""
    rng = np.random.default_rng(seed)
    net = TwoLayerNet(d=3, width=5, activation=activation, bias=True, loss_scale=0.5, ridge=1e-3)
    return net, rng.normal(size=(30, 3)), rng.normal(size=30), rng.normal(size=net.n_parameters)


FIXTURES = {"S1-like": s1_like, "S2-like": s2_like, "W16": w16, "odd-d control": odd_d}


def rate_grid(net, x, y, theta0, hidden_range, output_range, side):
    hidden_max, output_max = net.gauss_newton_layer_max(theta0, x, y)
    lh, lo = np.meshgrid(2.0 / hidden_max * 10.0 ** np.linspace(*hidden_range, side),
                         2.0 / output_max * 10.0 ** np.linspace(*output_range, side), indexing="ij")
    return GridMap(net, x, y, lh.ravel(), lo.ravel())


def ulp_errors(values, reference):
    """Errors in units of the last place of the correctly rounded reference."""
    out = []
    for v, r in zip(values, reference):
        nearest = float(r)
        out.append(float(abs(Decimal(float(v)) - r) / Decimal(float(np.spacing(abs(nearest))))))
    return np.array(out)


def math_functions(backends):
    getcontext().prec = 60
    rng = np.random.default_rng(99999)
    x = np.concatenate([rng.normal(scale=s, size=3000) for s in (0.1, 0.5, 1.0, 2.0, 5.0)])
    reference = []
    for v in x:
        e = (2 * Decimal(float(v))).exp()
        reference.append((e - 1) / (e + 1))
    z = np.exp(rng.normal(scale=0.7, size=15000))
    log_reference = [Decimal(float(v)).ln() for v in z]
    rows = {}
    candidates = {"numpy": (np.tanh(x), np.log(z)),
                  "glibc (math module)": (np.array([math.tanh(v) for v in x]),
                                          np.array([math.log(v) for v in z]))}
    for name, backend in backends.items():
        candidates[f"CUDA ({name})"] = (backend.math("tanh", x), backend.math("log", z))
    for name, (t, lg) in candidates.items():
        te, le = ulp_errors(t, reference), ulp_errors(lg, log_reference)
        rows[name] = dict(
            tanh_correctly_rounded=float(np.mean([v == float(r) for v, r in zip(t, reference)])),
            tanh_max_ulp=float(te.max()),
            log_correctly_rounded=float(np.mean([v == float(r) for v, r in zip(lg, log_reference)])),
            log_max_ulp=float(le.max()))
    first = next(iter(backends.values()))
    rows["CUDA tanh equal to numpy tanh"] = float(np.mean(first.math("tanh", x) == np.tanh(x)))
    rows["CUDA log equal to numpy log"] = float(np.mean(first.math("log", z) == np.log(z)))
    rows["samples"] = dict(tanh=len(x), log=len(z), reference="decimal, 60 significant digits")
    return rows


def single_step(backend):
    rows = []
    for name, fixture in FIXTURES.items():
        for activation in ("tanh", "identity"):
            net, x, y, _ = fixture(activation=activation)
            rng = np.random.default_rng(99992)
            theta = rng.normal(size=(64, net.n_parameters))
            vectors = rng.normal(size=theta.shape)
            grad, loss = net.gradient(theta, x, y, return_loss=True)
            hvp = net.hvp(theta, x, y, vectors)
            out = backend.evaluate(net, x, y, theta, vectors)
            row = dict(fixture=name, activation=activation, d=net.d, width=net.width, n=len(x),
                       parameters=net.n_parameters)
            for key, cpu in (("loss", loss), ("grad", grad), ("hvp", hvp)):
                gpu = out[key]
                row[f"{key}_bitwise_fraction"] = float(np.mean(gpu == cpu))
                row[f"{key}_max_relative"] = float(np.max(np.abs(gpu - cpu)) / np.max(np.abs(cpu)))
            rows.append(row)
    return rows


def separation_growth(backend, name, fixture, side, steps):
    """CPU batch stepping versus GPU evaluate plus the identical host update.

    Shows where CPU and GPU trajectories part: the first nonzero separation, its
    size, and its exponential growth rate per step, for the cells whose
    first-100-step loss histories differ most. Also checks that the device
    iteration equals evaluate plus the host update bit for bit.
    """
    net, x, y, theta0 = fixture()
    grid = rate_grid(net, x, y, theta0, (-1.5, 1.0), (-1.5, 0.5), side)
    cpu = np.repeat(theta0[None], grid.cells, axis=0)
    gpu = cpu.copy()
    separation = np.zeros((steps + 1, grid.cells))
    losses_cpu, losses_gpu = np.zeros((steps, grid.cells)), np.zeros((steps, grid.cells))
    with np.errstate(all="ignore"):
        for k in range(steps):
            grad, losses_cpu[k] = net.gradient(cpu, x, y, return_loss=True)
            out = backend.evaluate(net, x, y, gpu)
            losses_gpu[k] = out["loss"]
            cpu = cpu - grid.rates * grad
            gpu = gpu - grid.rates * out["grad"]
            separation[k + 1] = np.max(np.abs(cpu - gpu), axis=1)
    check_steps = 20
    device = backend.run_grid(grid, theta0, check_steps, tail_steps=1, ftle_steps=1, ftle_align=0,
                              sd_horizons=())
    hosted = np.repeat(theta0[None], grid.cells, axis=0)
    with np.errstate(all="ignore"):
        for _ in range(check_steps):
            hosted = hosted - grid.rates * backend.evaluate(net, x, y, hosted)["grad"]
    alive = device["status"] != 0
    _, score_cpu = sohl_dickstein_measure(losses_cpu[:100])
    _, score_gpu = sohl_dickstein_measure(losses_gpu[:100])
    worst = np.argsort(-np.abs(score_gpu - score_cpu) / np.abs(score_cpu))[:3]
    cells = []
    for i in worst:
        s = separation[:, i]
        window = (s > 1e-13) & (s < 1e-4) & np.isfinite(s)
        rate = float(np.polyfit(np.flatnonzero(window), np.log(s[window]), 1)[0]) if window.sum() > 5 else None
        first = int(np.flatnonzero(s > 0)[0]) if np.any(s > 0) else None
        cells.append(dict(cell=int(i), first_nonzero_step=first,
                          first_nonzero_separation=float(s[first]) if first is not None else 0.0,
                          growth_rate_per_step=rate,
                          separation_at={str(k): float(s[k]) for k in (10, 25, 50, 75, 100) if k <= steps},
                          sd100_score_rel_diff=float(abs(score_gpu[i] - score_cpu[i]) / abs(score_cpu[i]))))
    return dict(fixture=name, cells=grid.cells, steps=steps,
                device_equals_evaluate_plus_host_update=bool(np.array_equal(device["final_state"][alive],
                                                                             hosted[alive])),
                device_check_steps=check_steps, largest_sd100_differences=cells)


def aligned_distance(grid, index, cpu_state, gpu_state, period):
    map_fn, _ = grid.cell_map(index)
    state, best, shift = np.array(cpu_state), np.inf, 0
    for k in range(max(int(period), 1)):
        distance = float(np.max(np.abs(gpu_state - state)))
        if distance < best:
            best, shift = distance, k
        state = map_fn(state)
    return best, shift


def describe(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return dict(n=int(len(values)), median=float(np.median(values)), max=float(values.max()))


def compare(grid, cpu, gpu, horizons):
    cs, gs = cpu["status"], gpu["status"]
    labels = len(STATUS_NAMES)
    confusion = np.zeros((labels, labels), dtype=int)
    np.add.at(confusion, (cs, gs), 1)
    out = dict(cells=int(len(cs)), status_agreement=float(np.mean(cs == gs)),
               confusion=confusion.tolist(),
               cpu_status_counts={STATUS_NAMES[s]: int(np.sum(cs == s)) for s in range(labels)})
    orbit = np.flatnonzero(np.isin(cs, (1, 2)))
    same = [i for i in orbit if gs[i] == cs[i] and gpu["period"][i] == cpu["period"][i]]
    aligned = [aligned_distance(grid, i, cpu["final_state"][i], gpu["final_state"][i], cpu["period"][i])
               for i in same]
    converged = [d for i, (d, _) in zip(same, aligned) if cpu["recurrence_residual"][i] <= 1e-3]
    out["orbits"] = dict(
        cpu_cells=int(len(orbit)), status_and_period_equal=int(len(same)),
        periods=sorted({int(cpu["period"][i]) for i in orbit}),
        phase_shifted=int(sum(1 for _, k in aligned if k)),
        aligned_distance=describe([d for d, _ in aligned]),
        aligned_distance_residual_le_1e_3=describe(converged),
        unaligned_distance=describe([np.max(np.abs(gpu["final_state"][i] - cpu["final_state"][i]))
                                     for i in same]),
        ftle_abs_diff=describe(np.abs(gpu["ftle"][same] - cpu["ftle"][same])) if same else None)
    div = cs == 0
    steps_c, steps_g = cpu["diverged_at"][div], gpu["diverged_at"][div]
    early = np.minimum(steps_c, steps_g) < 16
    out["divergent"] = dict(
        cpu_cells=int(div.sum()), gpu_divergent_too=int(np.sum(gs[div] == 0)),
        gpu_only=int(np.sum((gs == 0) & ~div)),
        step_equal_fraction=float(np.mean(steps_c == steps_g)) if div.any() else None,
        max_step_shift=int(np.max(np.abs(steps_c - steps_g))) if div.any() else None,
        early_escapes=int(early.sum()), early_escapes_equal=int(np.sum(steps_c[early] == steps_g[early])))
    chaotic = cs == 3
    both = chaotic & (gs == 3)
    out["chaotic"] = dict(cpu_cells=int(chaotic.sum()), chaotic_on_both=int(both.sum()),
                          ftle_abs_diff=describe(np.abs(gpu["ftle"][both] - cpu["ftle"][both])),
                          ftle_rel_diff=describe(np.abs(gpu["ftle"][both] - cpu["ftle"][both])
                                                 / np.abs(cpu["ftle"][both])))
    drift = cs == 5
    growth_diff = np.abs(gpu["window_norm_growth"] - cpu["window_norm_growth"])
    out["drift"] = dict(cpu_cells=int(drift.sum()), drift_on_both=int(np.sum(drift & (gs == 5))),
                        gpu_only=int(np.sum((gs == 5) & ~drift)),
                        window_norm_growth_abs_diff=describe(growth_diff[drift]))
    out["window_norm_growth_abs_diff_by_cpu_status"] = {
        STATUS_NAMES[s]: describe(growth_diff[cs == s]) for s in range(labels) if np.any(cs == s)}
    unresolved = (cs == 4) & (gs == 4)
    out["unresolved"] = dict(cpu_cells=int(np.sum(cs == 4)), unresolved_on_both=int(unresolved.sum()),
                             ftle_abs_diff=describe(np.abs(gpu["ftle"][unresolved] - cpu["ftle"][unresolved])),
                             state_max_abs_diff=describe(np.max(np.abs(gpu["final_state"][unresolved]
                                                                       - cpu["final_state"][unresolved]),
                                                                axis=1)) if unresolved.any() else None)
    out["sd"] = {}
    for h in horizons:
        rel = np.abs(gpu[f"sd_score_{h}"] - cpu[f"sd_score_{h}"]) / np.abs(cpu[f"sd_score_{h}"])
        out["sd"][str(h)] = dict(
            convergent_flag_agreement=float(np.mean(gpu[f"sd_convergent_{h}"] == cpu[f"sd_convergent_{h}"])),
            score_rel_diff_by_cpu_status={STATUS_NAMES[s]: describe(rel[cs == s]) for s in range(labels)
                                          if np.any(cs == s)})
    out["losses_rel_diff_by_cpu_status"] = {
        key: {STATUS_NAMES[s]: describe(np.abs(gpu[key][cs == s] - cpu[key][cs == s])
                                        / np.abs(cpu[key][cs == s])) for s in range(labels) if np.any(cs == s)}
        for key in ("min_loss", "last_evaluated_loss", "tail_mean_loss")}
    return out


def identical(a, b):
    return {key: bool(np.array_equal(a[key], b[key], equal_nan=True)) for key in a}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", required=True)
    parser.add_argument("--contract")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--side", type=int, default=24)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--w16-steps", type=int, default=600)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.is_relative_to(gc.ROOT):
        parser.error("qualification receipts belong outside the Git checkout")
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    backends = {"mirror": gc.load(args.library)}
    if args.contract:
        backends["contract"] = gc.load(args.contract)
    commit = subprocess.run(["git", "-C", str(gc.ROOT), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(gc.ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    record = dict(
        scope="engineering qualification on artificial fixtures (seeds 99990-99999); not a result",
        started_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
        source_commit=commit, tracked_changes=bool(dirty), numpy=np.__version__,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        builds={name: b.metadata() for name, b in backends.items()})
    print(f"commit {commit}{' (tracked changes!)' if dirty else ''}")

    record["math_functions"] = math_functions(backends)
    print("\n## Transcendental functions against a 60-digit reference")
    for name, row in record["math_functions"].items():
        print(f"{name}: {row}")

    record["single_step"] = {name: single_step(b) for name, b in backends.items()}
    print("\n## Single-step agreement (64 random parameter and tangent vectors per fixture)")
    print("| build | fixture | activation | loss bitwise | grad bitwise | hvp bitwise | grad max rel | hvp max rel |")
    for name, rows in record["single_step"].items():
        for r in rows:
            print(f"| {name} | {r['fixture']} | {r['activation']} | {r['loss_bitwise_fraction']:.3f} | "
                  f"{r['grad_bitwise_fraction']:.3f} | {r['hvp_bitwise_fraction']:.3f} | "
                  f"{r['grad_max_relative']:.1e} | {r['hvp_max_relative']:.1e} |")

    record["separation"] = [separation_growth(backends["mirror"], "S1-like", s1_like, 8, 300),
                            separation_growth(backends["mirror"], "W16", w16, 8, 100)]
    print("\n## CPU-GPU trajectory separation")
    print(json.dumps(record["separation"], indent=1))

    # (name, fixture, hidden decades, output decades, side, steps, options, layout checks)
    plans = [("S1-like", s1_like, (-1.5, 1.0), (-1.5, 0.5), args.side, args.steps, PRODUCTION, False),
             ("S2-like", s2_like, (-4.5, -0.5), (-1.6, 0.4), args.side, args.steps, PRODUCTION, True),
             ("W16", w16, (-1.5, 1.0), (-1.5, 0.5), 8, args.w16_steps, W16_OPTIONS, False)]
    record["grids"] = []
    for name, fixture, hidden_range, output_range, side, steps, opts, layout in plans:
        net, x, y, theta0 = fixture()
        grid = rate_grid(net, x, y, theta0, hidden_range, output_range, side)
        started = time.perf_counter()
        cpu = run_grid(grid, theta0, steps, **opts)
        cpu_seconds = time.perf_counter() - started
        entry = dict(fixture=name, cells=grid.cells, steps=steps, options=gc._jsonable(opts),
                     hidden_decades=list(hidden_range), output_decades=list(output_range),
                     cpu_seconds=cpu_seconds, builds={})
        for build, backend in backends.items():
            gpu, meta = backend.run_grid(grid, theta0, steps, return_metadata=True, **opts)
            result = compare(grid, cpu, gpu, opts["sd_horizons"])
            result["kernel_seconds"] = meta["kernel_seconds"]
            result["wall_seconds"] = meta["wall_seconds"]
            if build == "mirror":
                # Small grids leave the GPU latency-bound, so the costlier layout
                # checks run on the cheap fixture; the test suite covers the others.
                repeat = backend.run_grid(grid, theta0, steps, **opts)
                result["repeat_bit_identical"] = identical(gpu, repeat)
                if layout:
                    rechunked = backend.run_grid(grid, theta0, steps, chunk_cells=max(2, grid.cells // 3),
                                                 tail_cells=7, launch_steps=997, **opts)
                    host = backend.run_grid(grid, theta0, steps, screen="host", **opts)
                    result["rechunked_bit_identical"] = identical(gpu, rechunked)
                    result["host_screen_bit_identical"] = identical(gpu, host)
            entry["builds"][build] = result
            print(f"\n## {name}: {grid.cells} cells x {steps} steps, build {build} "
                  f"(CPU {cpu_seconds:.0f} s, GPU kernel {meta['kernel_seconds']:.1f} s)")
            print(json.dumps({k: v for k, v in result.items() if k not in ("losses_rel_diff_by_cpu_status",)},
                             indent=1))
        record["grids"].append(entry)

    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"\nreceipt {out}")


if __name__ == "__main__":
    main()
