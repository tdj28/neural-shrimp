"""Bounded S2 warmup pilot; numerical exploration, never a confirmation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import scipy
from neural_shrimp.trainability import SYSTEMS
from neural_shrimp.warmup import branch_access, matched_references, reduced_reference, rollout, summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=Path, default=Path("/data2/neural-shrimp-data/explore/criticality/s2-28011-curve/curve.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outputs", nargs="+", type=float, default=[-0.44, -0.30])
    parser.add_argument("--alpha", type=float, default=0.0015)
    parser.add_argument("--levels", type=int, default=41)
    parser.add_argument("--dwell-pairs", nargs="+", type=int, default=[100, 400])
    parser.add_argument("--pre-pairs", type=int, default=2000)
    parser.add_argument("--top-pairs", type=int, default=10000)
    parser.add_argument("--recovery-pairs", type=int, default=8000)
    parser.add_argument("--global-factor", type=float, default=0.5)
    args = parser.parse_args()
    if args.alpha <= 0 or args.levels < 3 or min(args.dwell_pairs) <= 0:
        raise ValueError("invalid ramp parameters")
    if min(args.pre_pairs, args.top_pairs, args.recovery_pairs) < 1:
        raise ValueError("all holding horizons must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    metadata = {"status": "running", "evidence": "exploratory_float64", "pid": os.getpid(),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "started_utc": datetime.now(timezone.utc).isoformat(), "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "inputs": [{"path": str(args.curve), "sha256": digest(args.curve), "bytes": args.curve.stat().st_size}],
                "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                            "host": platform.node(), "platform": platform.platform(), "nice": os.getpriority(os.PRIO_PROCESS, 0),
                            "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    net, x, y, ordinary = SYSTEMS["S2"](28011)
    source = json.loads(args.curve.read_text())["rows"]
    candidates = [r for r in source if r.get("type") == "neimark_sacker"]
    rows = [min(candidates, key=lambda r: abs(r["log10_output"] - target)) for target in args.outputs]
    summaries = []
    try:
        for point_index, row in enumerate(rows):
            refs = matched_references(net, x, y, row, np.linspace(-args.alpha, args.alpha, args.levels))
            (args.output / f"reference-{point_index}.json").write_text(json.dumps({"source_row": row, "references": plain(refs)}, indent=2) + "\n")
            near = refs[0]["orbit"][0] + 1e-4 * refs[0]["direction"]
            for dwell in args.dwell_pairs:
                initial = np.stack([near, ordinary])
                common_segments = [(args.pre_pairs, [refs[0]] * 2, 0)]
                common_segments += [(dwell, [ref] * 2, 1) for ref in refs]
                common_segments += [(args.top_pairs, [refs[-1]] * 2, 2)]
                common = rollout(net, x, y, initial, common_segments)
                np.savez_compressed(args.output / f"point-{point_index}-dwell-{dwell}-common.npz", **common)
                low = reduced_reference(refs[0], refs[-1], args.global_factor)
                fork_initial = np.tile(common["state"][-1], (3, 1))
                fork_segments = [(dwell, [refs[-1]] * 2 + [ref] * 2 + [low] * 2, 3) for ref in refs[::-1]]
                fork_segments += [(args.recovery_pairs, [refs[-1]] * 2 + [refs[0]] * 2 + [low] * 2, 4)]
                forks = rollout(net, x, y, fork_initial, fork_segments)
                np.savez_compressed(args.output / f"point-{point_index}-dwell-{dwell}-forks.npz", **forks)
                for j, start in enumerate(("near_orbit", "ordinary_initialization")):
                    pre = summarize(common, j, refs[0], mask=common["alpha"][:, j] >= 0)
                    for k, policy in enumerate(("keep_top_rates", "reverse_hidden_ramp", "halve_both_rates")):
                        result = {"point": point_index, "log10_output": row["log10_output"], "d": row["d"],
                                  "alpha_endpoint": args.alpha, "dwell_pairs": dwell, "start": start,
                                  "policy": policy, "branch_access": branch_access(common, j), "common_after_crossing": pre,
                                  "after_intervention": summarize(forks, 2 * k + j, refs[0])}
                        summaries.append(result)
                        print(json.dumps(result), flush=True)
                (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        metadata["status"] = "complete"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - started
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "manifest.json"]
        (args.output / "manifest.json").write_text(json.dumps({"source_commit": metadata["source_commit"], "status": metadata["status"], "files": files}, indent=2) + "\n")


if __name__ == "__main__":
    main()
