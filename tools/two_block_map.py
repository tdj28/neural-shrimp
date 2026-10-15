"""Two steering directions at two blocks (run TB): the soft-token map of a model steered by alpha1 * u after block L1
and alpha2 * w after block L2, over an (alpha1, alpha2) grid at fixed temperatures (human, 15 Oct 2026: "steer one
block with bliss and the other with hostile").

Each cell runs the soft-token loop (window 1, float64) with one twin (tools/hidden_chaos.run_batch: transient,
record, twin offset, period tolerance as given) and is classified by hidden_chaos.classify. u and w are concept
vectors built at their blocks (tools/explore_llm_phase_map.py vectors), scaled to the mean residual norm there, so
alpha is a fraction of the residual norm at each block.

Usage: python tools/two_block_map.py --block1 4 --vectors1 llm-phase-map/vectors-l4 --concept1 gratitude \
    --block2 16 --vectors2 llm-phase-map/vectors-l16 --concept2 hostility --log-temperature -0.1 --resolution 96 \
    --device cuda --out llm-soft-loop/pod64/tb/grat4-host16-t-0.1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--block1", type=int, required=True)
    ap.add_argument("--vectors1", required=True)
    ap.add_argument("--concept1", required=True)
    ap.add_argument("--block2", type=int, required=True)
    ap.add_argument("--vectors2", required=True)
    ap.add_argument("--concept2", required=True)
    ap.add_argument("--log-temperature", type=float, nargs="+", required=True)
    ap.add_argument("--alpha1", type=float, nargs=2, default=[-0.12, 0.12])
    ap.add_argument("--alpha2", type=float, nargs=2, default=[-0.12, 0.12])
    ap.add_argument("--resolution", type=int, default=96)
    ap.add_argument("--resolution1", type=int, default=None, help="points along alpha1 (default --resolution)")
    ap.add_argument("--resolution2", type=int, default=None,
                    help="points along alpha2 (default --resolution); with --alpha2 0 0 and 1, an exact single-push line")
    ap.add_argument("--transient", type=int, default=500)
    ap.add_argument("--record", type=int, default=300)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--chunk", type=int, default=2304)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl
    import hidden_chaos as hc

    loop = esl.SoftLoop(args.model, args.block1, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    loop.sm.add_steer_layer(args.block2)
    u = np.asarray(elpm.load_vectors(args.vectors1)[args.concept1], dtype=np.float64)
    w = np.asarray(elpm.load_vectors(args.vectors2)[args.concept2], dtype=np.float64)
    a1 = np.linspace(*args.alpha1, args.resolution1 or args.resolution)
    a2 = np.linspace(*args.alpha2, args.resolution2 or args.resolution)
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    for lt in args.log_temperature:
        cells = [(x, y) for y in a2 for x in a1]                # rows: alpha2, columns: alpha1
        L = np.full(len(cells), np.nan)
        P = np.zeros(len(cells), dtype=int)
        C = np.empty(len(cells), dtype=object)
        for s0 in range(0, len(cells), args.chunk):
            part = cells[s0:s0 + args.chunk]
            loop.extra = (args.block2, np.array([y * w for _, y in part]))
            res = hc.run_batch(loop, u, [lt] * len(part), [x for x, _ in part], args.transient, args.record, [args.eps],
                               args.tol)
            loop.extra = None
            for k, r in enumerate(res):
                L[s0 + k] = r["lyapunov"][0]
                P[s0 + k] = r["period"]
                C[s0 + k] = hc.classify(r)
            print(json.dumps({"log10_T": lt, "done": s0 + len(part), "of": len(cells),
                              "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
        shape = (len(a2), len(a1))
        chaotic = np.array([("chaos" in c) and ("irregular" not in c) for c in C]).reshape(shape)
        np.savez_compressed(out / f"tb-t{lt:+.3f}.npz", alpha1=a1, alpha2=a2, log10_temperature=lt,
                            lyapunov=L.reshape(shape), period=P.reshape(shape), chaotic=chaotic)
        (out / f"tb-t{lt:+.3f}.json").write_text(json.dumps({"args": vars(args), "log10_temperature": lt,
                                                             "chaotic_share": float(chaotic.mean()),
                                                             "seconds": round(time.perf_counter() - tic, 1)}, indent=1))


if __name__ == "__main__":
    main()
