"""Independent check of the chaos (human, 19 Oct 2026: "is there a chance we are fooling ourselves ... the chaos we
found was from our code and harness and not from the LLM?").

The largest Lyapunov exponent at a cell is re-measured with Codex's independent float64 CPU implementation of the
soft-token map (codex/crosscheck/qwen_map.py: a manual Qwen2.5-0.5B forward pass from the pinned checkpoint, with no
repository research tool and no Transformers model code). The protocol is the same as the scans':
- start from the stored newline embedding (the fixed start);
- run --transient steps;
- then --steps steps with a twin at absolute offset --eps in a seeded random direction, renormalized every step;
- the exponent is the mean log growth per step.
The exponent is reported at checkpoints (300 steps, as the scans, and every 250 steps after).

Usage: python tools/independent_chaos_check.py --alpha 0.065201 --log-temperature -0.155998 --out NAME
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codex" / "crosscheck"))
DATA = ROOT.parent / "neural-shrimp-data"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--log-temperature", type=float, required=True)
    ap.add_argument("--transient", type=int, default=300)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    from qwen_map import ManualQwenSoftMap

    out = DATA / "explore" / "llm-soft-loop" / "independent-chaos-check" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    m = ManualQwenSoftMap.from_pinned_files()
    cache = m.prefix_cache(args.alpha)
    x = m.initial_state()
    with torch.no_grad():
        for _ in range(args.transient):
            x = m.forward(x, args.alpha, args.log_temperature, cache=cache)
        g = torch.Generator().manual_seed(args.seed)
        d = torch.randn(x.shape, generator=g, dtype=torch.float64)
        twin = x + d / d.norm() * args.eps
        X = torch.stack([x, twin])
        total, checkpoints = 0.0, []
        for t in range(1, args.steps + 1):
            X = m.forward(X, args.alpha, args.log_temperature, cache=cache)
            diff = X[1] - X[0]
            dist = float(diff.norm())
            total += np.log(max(dist, 1e-300) / args.eps)
            X = torch.stack([X[0], X[0] + diff * (args.eps / max(dist, 1e-300))])
            if t == 300 or t % 250 == 0 or t == args.steps:
                checkpoints.append({"steps": t, "lambda": total / t, "seconds": round(time.perf_counter() - tic, 1)})
                print(json.dumps(checkpoints[-1]), flush=True)
    rec = {"args": vars(args), "implementation": "codex/crosscheck/qwen_map.py (independent manual forward)",
           "lambda_at_300": next((c["lambda"] for c in checkpoints if c["steps"] == 300), None),
           "lambda_final": checkpoints[-1]["lambda"], "checkpoints": checkpoints,
           "seconds": round(time.perf_counter() - tic, 1)}
    (out / "result.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("lambda_at_300", "lambda_final", "seconds")}))


if __name__ == "__main__":
    main()
