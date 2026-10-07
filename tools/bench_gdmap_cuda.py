"""Throughput of the gdmap CUDA backend versus the NumPy engine on one core.

Engineering benchmark on artificial fixtures only (seeds 99996 and 99997); no
study system is run. Usage:

    taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python tools/bench_gdmap_cuda.py --library BUILD/gdmap_cuda.so --out OUT.json \\
        [--ab-library OTHER_BUILD/gdmap_cuda.so]

GPU numbers are kernel time per cell-step from CUDA events. They are taken on
a shared GPU: nvidia-smi utilization, memory and compute processes are
recorded before and after every GPU measurement, and contention from other
users lowers the reported throughput. CPU numbers are wall time of
neural_shrimp.gdmaps.run_grid on one core with one BLAS thread.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_shrimp import gdmaps_cuda as gc  # noqa: E402
from neural_shrimp.gdmaps import STATUS_NAMES, GridMap, TwoLayerNet, run_grid  # noqa: E402

NETS = {"s1": (8, 8, 99996), "w16": (16, 16, 99997)}
PLAIN = dict(tail_steps=2, ftle_steps=1, ftle_align=0, sd_horizons=())
PIPELINE = dict(tail_steps=384, ftle_steps=400, ftle_align=100, sd_horizons=(100, 500, 1000))


def fixture(name, cells):
    """Mean-field tanh net with n = parameter count; rates below the edge so cells survive."""
    d, width, seed = NETS[name]
    net = TwoLayerNet(d=d, width=width, in_scale=np.sqrt(2.0 / d), out_scale=1.0 / width,
                      bias=False, loss_scale=1.0)
    rng = np.random.default_rng(seed)
    p = net.n_parameters
    x, y, theta0 = rng.normal(size=(p, d)), rng.normal(size=p), rng.normal(size=p)
    hidden_max, output_max = net.gauss_newton_layer_max(theta0, x, y)
    side = int(np.ceil(np.sqrt(cells)))
    u = np.linspace(-1.5, -0.3, side)
    lh, lo = np.meshgrid(2.0 / hidden_max * 10.0 ** u, 2.0 / output_max * 10.0 ** u, indexing="ij")
    return GridMap(net, x, y, lh.ravel()[:cells], lo.ravel()[:cells]), theta0


def gpu_snapshot():
    def query(args):
        try:
            return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            return f"unavailable: {error}"
    return dict(
        utc=datetime.now(timezone.utc).isoformat(),
        gpu=query(["--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,"
                   "power.draw,clocks.sm", "--format=csv,noheader"]),
        processes=query(["--query-compute-apps=pid,process_name,used_memory",
                         "--format=csv,noheader"]),
        load_average=os.getloadavg())


def subset(grid, count):
    index = np.linspace(0, grid.cells - 1, count).round().astype(int)
    return GridMap(grid.net, grid.x, grid.y, grid.rates[index, 0], grid.rates[index, -1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cells", type=int, nargs="+", default=[8192, 65536])
    parser.add_argument("--nets", nargs="+", default=["s1", "w16"], choices=sorted(NETS))
    parser.add_argument("--gpu-steps", type=json.loads, default={"s1": 200, "w16": 40})
    parser.add_argument("--cpu-cells", type=int, default=1024)
    parser.add_argument("--cpu-steps", type=json.loads, default={"s1": 40, "w16": 4})
    parser.add_argument("--pipeline-steps", type=int, default=2000)
    parser.add_argument("--ab-library", help="second build, timed interleaved with --library")
    parser.add_argument("--ab-repeats", type=int, default=3)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.is_relative_to(gc.ROOT):
        parser.error("benchmark receipts belong outside the Git checkout")
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    backend = gc.load(args.library)
    record = dict(
        scope="engineering throughput on artificial fixtures (seeds 99996, 99997); not a result",
        started_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
        cpu_model=next((line.split(":", 1)[1].strip() for line in
                        Path("/proc/cpuinfo").read_text().splitlines()
                        if line.startswith("model name")), None),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                "MKL_NUM_THREADS")},
        numpy=np.__version__, metadata=backend.metadata(), gpu=[], cpu=[], pipeline=[])
    print(f"device: {record['metadata']['device']['name']}, driver "
          f"{record['metadata']['nvidia_driver']}, {record['metadata']['nvcc_version']}")
    for name in args.nets:
        steps = int(args.gpu_steps[name])
        for cells in args.cells:
            grid, theta0 = fixture(name, cells)
            for mode in ("plain", "ftle"):
                opts = dict(PLAIN) if mode == "plain" else dict(PLAIN, ftle_steps=steps)
                before = gpu_snapshot()
                result, meta = backend.run_grid(grid, theta0, steps, return_metadata=True, **opts)
                after = gpu_snapshot()
                row = dict(net=name, cells=cells, steps=steps, mode=mode,
                           us_per_cell_step=meta["microseconds_per_cell_step"],
                           kernel_seconds=meta["kernel_seconds"], wall_seconds=meta["wall_seconds"],
                           alive_fraction=float(np.mean(result["status"] != 0)),
                           chunk_cells=meta["chunk_cells"], chunks=meta["chunks"],
                           allocated_device_bytes=meta["allocated_device_bytes"],
                           gpu_before=before, gpu_after=after)
                record["gpu"].append(row)
                print(f"GPU {name:3s} {cells:6d} cells {mode:5s}: {row['us_per_cell_step']:.3f} us/cell-step"
                      f" (alive {row['alive_fraction']:.3f}; util before/after "
                      f"{before['gpu'].split(',')[0]}/{after['gpu'].split(',')[0]})")
        # NumPy engine on one core, same rates (a strided subset), same options.
        grid, theta0 = fixture(name, max(args.cells))
        small = subset(grid, args.cpu_cells)
        cpu_steps = int(args.cpu_steps[name])
        for mode in ("plain", "ftle"):
            opts = dict(PLAIN) if mode == "plain" else dict(PLAIN, ftle_steps=cpu_steps)
            started = time.perf_counter()
            result = run_grid(small, theta0, cpu_steps, **opts)
            seconds = time.perf_counter() - started
            row = dict(net=name, cells=args.cpu_cells, steps=cpu_steps, mode=mode,
                       us_per_cell_step=1e6 * seconds / (args.cpu_cells * cpu_steps),
                       wall_seconds=seconds, alive_fraction=float(np.mean(result["status"] != 0)))
            record["cpu"].append(row)
            print(f"CPU {name:3s} {args.cpu_cells:6d} cells {mode:5s}: {row['us_per_cell_step']:.3f} "
                  f"us/cell-step (one core)")
    # A short end-to-end run with tails, screening and score windows.
    grid, theta0 = fixture("s1", min(args.cells))
    before = gpu_snapshot()
    result, meta = backend.run_grid(grid, theta0, args.pipeline_steps, return_metadata=True, **PIPELINE)
    after = gpu_snapshot()
    record["pipeline"].append(dict(
        net="s1", cells=grid.cells, steps=args.pipeline_steps, options=PIPELINE | {"sd_horizons": [100, 500, 1000]},
        kernel_seconds=meta["kernel_seconds"], screen_seconds=meta["screen_seconds"],
        transfer_seconds=meta["transfer_seconds"], wall_seconds=meta["wall_seconds"],
        wall_us_per_cell_step=1e6 * meta["wall_seconds"] / (grid.cells * args.pipeline_steps),
        tail_cells=meta["tail_cells"],
        status_counts=np.bincount(result["status"], minlength=len(STATUS_NAMES)).tolist(),
        gpu_before=before, gpu_after=after))
    print(f"pipeline s1 {grid.cells} cells x {args.pipeline_steps} steps: wall {meta['wall_seconds']:.2f} s, "
          f"kernel {meta['kernel_seconds']:.2f} s, screen {meta['screen_seconds']:.2f} s")
    if args.ab_library:
        # Interleaved repeats, so drifting contention affects both builds alike.
        other = gc.load(args.ab_library)
        record["ab"] = dict(library_b=other.metadata(), rows=[])
        for name in args.nets:
            steps = int(args.gpu_steps[name])
            grid, theta0 = fixture(name, min(args.cells))  # enough cells to fill the GPU
            for mode in ("plain", "ftle"):
                opts = dict(PLAIN) if mode == "plain" else dict(PLAIN, ftle_steps=steps)
                for repeat in range(args.ab_repeats):
                    for label, build in (("a", backend), ("b", other)):
                        before = gpu_snapshot()
                        _, meta = build.run_grid(grid, theta0, steps, return_metadata=True, **opts)
                        record["ab"]["rows"].append(dict(
                            net=name, cells=grid.cells, mode=mode, repeat=repeat, build=label,
                            variant=meta["variant"], us_per_cell_step=meta["microseconds_per_cell_step"],
                            gpu_before=before))
                        print(f"A/B {name} {mode} repeat {repeat} {meta['variant']}: "
                              f"{meta['microseconds_per_cell_step']:.3f} us/cell-step")
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"receipt {out}")


if __name__ == "__main__":
    main()
