"""Batch-cluster randomization tests for the paired answer contrasts (external review, 18 Oct 2026).

The answer tests pair each problem across conditions, but problems that share a batch share its numerical execution,
so item-level McNemar tests can overstate the evidence. This tool repeats each paired contrast with the batch as the
unit: per batch, the sum of the paired differences d_i (flip in arm A minus flip in arm B, or correct minus correct);
under the null of exchangeable arm labels within each batch, the signs of these batch sums are exchangeable, and the
p-value is the share of sign patterns (Monte Carlo, --draws) whose total is at least as extreme as the observed one.
It also reports the paired cluster-bootstrap interval of the mean difference and the item-level McNemar p for
comparison. Batches are those of arm A's larger layout (tools/pi_matrix_analyze.py, load_arm).

Usage: python tools/cluster_tests.py --arm A=RUN --arm B=RUN --contrast A:B [--outcome flip|correct] --json OUT
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi_matrix_analyze import load_arm, mcnemar_exact  # noqa: E402

SEED = 20261018


def contrast(rowsA, rowsB, outcome, draws, rng):
    idx = sorted(set(rowsA) & set(rowsB))
    if outcome == "flip":
        xa = np.array([rowsA[i]["flip"] for i in idx], dtype=float)
        xb = np.array([rowsB[i]["flip"] for i in idx], dtype=float)
    else:                                   # correctness in the larger layout
        xa = np.array([rowsA[i]["correct_b"] for i in idx], dtype=float)
        xb = np.array([rowsB[i]["correct_b"] for i in idx], dtype=float)
    d = xa - xb
    by = defaultdict(list)
    for i, v in zip(idx, d):
        by[rowsA[i]["cluster"]].append(v)
    sums = np.array([np.sum(v) for v in by.values()])
    counts = np.array([len(v) for v in by.values()])
    obs = sums.sum()
    signs = rng.choice([-1.0, 1.0], size=(draws, len(sums)))
    null = signs @ sums
    p_perm = float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (draws + 1))
    boot = []
    k = len(sums)
    for _ in range(5000):
        pick = rng.integers(0, k, k)
        boot.append(sums[pick].sum() / counts[pick].sum())
    b = int(np.sum((xa == 1) & (xb == 0)))
    c = int(np.sum((xa == 0) & (xb == 1)))
    return {"problems": len(idx), "clusters": k, "mean_difference": float(d.mean()),
            "cluster_bootstrap95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "cluster_permutation_p": p_perm, "draws": draws, "discordant": [b, c],
            "item_mcnemar_p": mcnemar_exact(b, c)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", action="append", required=True, help="NAME=RUN")
    ap.add_argument("--contrast", action="append", required=True, help="A:B")
    ap.add_argument("--outcome", default="flip", choices=("flip", "correct"))
    ap.add_argument("--draws", type=int, default=200000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    arms = {}
    for spec in args.arm:
        name, run = spec.split("=", 1)
        arms[name] = load_arm(run)[1]
    out = {"outcome": args.outcome, "seed": SEED, "contrasts": {}}
    for spec in args.contrast:
        a, b = spec.split(":")
        out["contrasts"][f"{a} - {b}"] = contrast(arms[a], arms[b], args.outcome, args.draws, rng)
        print(json.dumps({f"{a} - {b}": out["contrasts"][f"{a} - {b}"]}))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
