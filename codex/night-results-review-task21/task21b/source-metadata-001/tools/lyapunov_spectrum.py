"""Leading Lyapunov exponents of the soft-token map by QR iteration of exact tangents (run TC, part 2).

For each cell [log10 T, alpha], k copies of the base state carry k tangent vectors through loop.step_jvp (forward-mode
AD, as in hidden_chaos.py). The copies are reset to the first one after every step, so all tangents follow one orbit.
After every step the tangents are re-orthonormalised by a QR factorization and log|R_ii| is accumulated (the standard
method of Benettin et al. 1980); the first exponent is the tangent exponent of hidden_chaos.py.

Usage: python tools/lyapunov_spectrum.py --cells-file F --cells-index 0 5 9 --k 4 --out DIR [model options]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def spectrum(loop, steer, log_t, k, transient, record, seed=0, checkpoints=()):
    """The k leading exponents (per step) of the orbit from the loop's fixed start at one cell, and the running
    estimates at the given measured-step counts."""
    torch = loop.torch
    X = loop.prepare(np.tile(np.asarray(steer, dtype=float), (k, 1)))
    temps = np.full(k, 10.0 ** log_t)
    for _ in range(transient):
        X, _ = loop.step(X, temps)
        X = X[:1].repeat(k, 1, 1)
    shape = tuple(X.shape[1:])
    rng = np.random.default_rng(seed)
    G = torch.as_tensor(rng.normal(size=(int(np.prod(shape)), k)), dtype=loop.state_dtype, device=loop.device)
    Q, _ = torch.linalg.qr(G)
    logs, marks, snaps, ent = np.zeros(k), set(checkpoints), [], 0.0
    for i in range(record):
        X, p, JV = loop.step_jvp(X, temps, Q.T.reshape((k,) + shape))
        X = X[:1].repeat(k, 1, 1)
        Q, R = torch.linalg.qr(JV.reshape(k, -1).T)
        logs += np.log(np.maximum(torch.abs(torch.diagonal(R)).double().cpu().numpy(), 1e-300))
        p0 = p[0].double()
        ent += float(-(p0 * torch.log(torch.clamp(p0, min=1e-300))).sum())
        if i + 1 in marks:
            snaps.append({"steps": i + 1, "exponents": (logs / (i + 1)).tolist()})
    return {"exponents": (logs / record).tolist(), "checkpoints": snaps, "entropy_mean": ent / record}


def main():
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--concept", default=None)
    ap.add_argument("--vectors-none", action="store_true", help="unsteered: no steering vector (alpha ignored)")
    ap.add_argument("--cells-file", required=True, help="JSON file with a list 'cells' of [log10T, alpha] pairs")
    ap.add_argument("--cells-index", type=int, nargs="+", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--transient", type=int, default=1000)
    ap.add_argument("--record", type=int, default=2000)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto (layers spread over all GPUs)")
    ap.add_argument("--weights", default="native", choices=("native", "bf16"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.k < 1:
        raise SystemExit("--k must be at least 1")
    if args.checkpoints and not all(0 < c <= args.record for c in args.checkpoints):
        raise SystemExit("--checkpoints must lie between 1 and --record")
    pairs = np.array(json.loads(Path(args.cells_file).read_text())["cells"], dtype=float)
    if not all(0 <= i < len(pairs) for i in args.cells_index):
        raise SystemExit(f"--cells-index out of range (the file has {len(pairs)} cells)")
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, args.dtype, args.device,
                        args.weights)
    v = np.zeros(loop.E.shape[1]) if args.vectors_none else elpm.load_vectors(args.vectors)[args.concept]
    if len(v) != loop.E.shape[1]:
        raise SystemExit(f"steering vector dimension {len(v)} does not match the model ({loop.E.shape[1]})")
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for i in args.cells_index:
        log_t, alpha = pairs[i]
        r = spectrum(loop, alpha * v, log_t, args.k, args.transient, args.record, args.seed,
                     tuple(args.checkpoints or ()))
        cells.append({"index": int(i), "log10_temperature": float(log_t), "alpha": float(alpha), "k": args.k, **r})
        print(json.dumps({"index": int(i), "exponents": [round(e, 4) for e in r["exponents"]]}), flush=True)
        (out / "lyapunov-spectrum.json").write_text(json.dumps({"args": vars(args), "cells": cells,
                                                                 "complete": len(cells) == len(args.cells_index)},
                                                                indent=1))


if __name__ == "__main__":
    main()
