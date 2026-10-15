"""Development analysis of run HE (exploratory): does the distance of a relaxation from each problem's own stability
edge explain Huginn's GSM8K accuracy (rows of run H1b, train problems 0-119, arms lam 0.80, 1.00, 1.05, 1.10, 1.15,
1.20; edges from tools/huginn_edge.py)?

Per problem q with a located edge lam_e(q) and per arm lam: offset = lam - lam_e(q) and the paired change from the
native arm, delta = correct(q, lam) - correct(q, 1.00).
  1. Within each arm: mean delta of the problems for which lam lies inside their edge (offset < 0) against those for
     which it lies beyond (offset >= 0), with a permutation test of the split (10,000).
  2. Pooled over the over-relaxed arms (lam > 1): mean delta in offset bins, with bootstrap intervals over problems.
  3. The edges themselves: distribution, outcomes of the gates.
The edge hypothesis predicts delta >= 0 inside and < 0 beyond within an arm, and the largest pooled gain just inside
the edge.

Usage: python tools/huginn_edge_analysis.py --edges huginn/edge-dev-0 huginn/edge-dev-60 --rows huginn/gsm8k-dev-cot8-r32
    --out-json docs/huginn-edge-dev.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64"
BINS = [(-1.0, -0.15), (-0.15, -0.08), (-0.08, -0.04), (-0.04, 0.0), (0.0, 0.04), (0.04, 0.10), (0.10, 1.0)]


def load(edge_runs, rows_run):
    edges = {}
    for run in edge_runs:
        for line in (DATA / run / "edges.jsonl").read_text().splitlines():
            r = json.loads(line)
            edges[r["index"]] = r
    correct = {}
    for line in (DATA / rows_run / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        correct[(r["index"], round(float(r["lam"]), 4))] = int(bool(r["correct"]))
    return edges, correct


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--edges", nargs="+", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    edges, correct = load(args.edges, args.rows)
    rng = np.random.default_rng(0)
    outcomes = {}
    for r in edges.values():
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    lam_e = {q: r["lam_edge"] for q, r in edges.items() if "lam_edge" in r and r["outcome"] == "edge located"}
    vals = np.array(list(lam_e.values()))
    res = {"edges": {"problems": len(edges), "outcomes": outcomes, "located": len(lam_e),
                     "quantiles": {str(p): float(np.quantile(vals, p)) for p in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)}
                     if len(vals) else None}}
    arms = sorted({lam for (_, lam) in correct if lam != 1.0})
    res["within_arm"] = {}
    pooled = []
    for lam in arms:
        inside, beyond = [], []
        for q, le in lam_e.items():
            if (q, lam) not in correct or (q, 1.0) not in correct:
                continue
            d = correct[(q, lam)] - correct[(q, 1.0)]
            (inside if lam < le else beyond).append(d)
            if lam > 1.0:
                pooled.append((lam - le, d, q))
        rec = {"inside": len(inside), "beyond": len(beyond),
               "mean_delta_inside": float(np.mean(inside)) if inside else None,
               "mean_delta_beyond": float(np.mean(beyond)) if beyond else None}
        if inside and beyond:
            obs = np.mean(inside) - np.mean(beyond)
            allv = np.array(inside + beyond)
            k = len(inside)
            perm = sum(abs(np.mean(p[:k]) - np.mean(p[k:])) >= abs(obs) for p in (rng.permutation(allv) for _ in range(10000)))
            rec.update({"difference": float(obs), "p_permutation": (perm + 1) / 10001})
        res["within_arm"][str(lam)] = rec
    off = np.array([p[0] for p in pooled])
    dlt = np.array([p[1] for p in pooled])
    qs = np.array([p[2] for p in pooled])
    res["pooled_bins"] = []
    for lo, hi in BINS:
        m = (off >= lo) & (off < hi)
        if not m.any():
            res["pooled_bins"].append({"offset": [lo, hi], "n": 0})
            continue
        probs = np.unique(qs[m])
        boots = []
        for _ in range(4000):
            pick = rng.choice(probs, len(probs))
            v = np.concatenate([dlt[m & (qs == p)] for p in pick])
            boots.append(v.mean())
        res["pooled_bins"].append({"offset": [lo, hi], "n": int(m.sum()), "problems": int(len(probs)),
                                   "mean_delta": float(dlt[m].mean()),
                                   "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]})
    Path(args.out_json).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
