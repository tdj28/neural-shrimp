"""Exploratory GPU period-resolved map of a learning-rate rectangle.

Adaptive-protocol exploration tool (docs/adaptive-research-protocol.md): uses
the qualified float64 CUDA backend with the EXP-028 iteration and screening
settings, so maps are comparable with EXP-028, and writes the arrays, the
backend metadata and a figure under ../neural-shrimp-data/explore/<name>/.
Results are exploratory; chaotic or drifting labels can differ from the CPU
engine in a few percent of cells.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from neural_shrimp.gdmaps import GridMap  # noqa: E402
from neural_shrimp.gdmaps_cuda import run_grid_cuda  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402

PLAN = ROOT / "experiments" / "trainability" / "EXP-028-plan.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hidden", type=float, nargs=2, required=True, help="log10 range")
    parser.add_argument("--output", type=float, nargs=2, required=True, help="log10 range")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    out = (ROOT / "../neural-shrimp-data/explore" / args.name).resolve()
    out.mkdir(parents=True, exist_ok=False)
    it = json.loads(PLAN.read_text())["iteration"]
    options = {k: it[k] for k in ("tail_steps", "ftle_steps", "ftle_align", "max_period", "min_repeats",
                                  "atol", "rtol", "chaos_threshold", "drift_threshold",
                                  "divergence_threshold", "tangent_seed")}
    options["sd_horizons"] = tuple(it["sd_horizons"])
    net, x, y, theta0 = SYSTEMS[args.system](args.seed)
    log_h = np.linspace(args.hidden[0], args.hidden[1], args.resolution)
    log_o = np.linspace(args.output[0], args.output[1], args.resolution)
    r, c = np.divmod(np.arange(args.resolution ** 2), args.resolution)
    grid = GridMap(net, x, y, 10.0 ** log_h[r], 10.0 ** log_o[c])
    start = time.perf_counter()
    result, meta = run_grid_cuda(grid, theta0, it["steps"], library=args.library,
                                 return_metadata=True, **options)
    seconds = time.perf_counter() - start
    shaped = {k: (v.reshape(args.resolution, args.resolution, -1) if k == "final_state"
                  else v.reshape(args.resolution, args.resolution)) for k, v in result.items()}
    np.savez(out / "map.npz", log10_hidden=log_h, log10_output=log_o, **shaped)
    (out / "metadata.json").write_text(json.dumps({
        "args": vars(args), "steps": it["steps"], "options": {k: list(v) if isinstance(v, tuple) else v
                                                              for k, v in options.items()},
        "seconds": seconds, "backend": meta}, indent=2, default=str))
    import analyze_exp028 as analysis
    analysis.draw_map(np.load(out / "map.npz"),
                      f"Exploratory {args.system}-{args.seed} {args.resolution}x{args.resolution}, GPU, "
                      f"{it['steps']:,} updates per cell", out / "map.png", "sd_convergent_500")
    print(json.dumps({"out": str(out), "seconds": round(seconds, 1)}))


if __name__ == "__main__":
    main()
