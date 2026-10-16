"""Analysis of the confirmatory soft-thinking matrix (docs/pi-matrix-confirmatory.md), fixed before its data.

Each arm is one tools/pi_harness.py run with layouts 8 and 16 on the same scored problems. Per arm:
- **Answer flip:** the extracted answers of the two layouts differ (a missing number counts as its own value, so two
  missing answers agree). Rate over all problems present in both layouts, with a Wilson interval and a cluster
  bootstrap interval. Clusters are the batches of the 16-problem layout, which share one layout intervention.
- **Accuracy** in each layout, the share of problems with exactly one layout correct, and missing answers per layout.
Contrasts between arms are paired by problem: the difference of flip rates with a paired cluster bootstrap interval
and an exact McNemar test on the discordant problems.

Primary endpoint: the flip rate of arm top15-s64. Confirmed when the rate exceeds 0.10 and the lower end of its
cluster bootstrap interval exceeds 0.05.

Usage: python tools/pi_matrix_analyze.py --arm NAME=RUN [--arm NAME=RUN ...] [--json OUT]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

BOOT = 10000
SEED = 20261016


def wilson(k, n, z=1.959964):
    if not n:
        return [float("nan"), float("nan")]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_arm(run_dir):
    cfg = json.loads((Path(run_dir) / "run.json").read_text())["args"]
    by = defaultdict(dict)
    for line in open(Path(run_dir) / "records.jsonl"):
        r = json.loads(line)
        if r["index"] >= 0:
            by[r["index"]][r["layout"]] = r
    la, lb = sorted(cfg["layouts"])[:2]
    rows = {}
    for idx, v in by.items():
        if la in v and lb in v:
            a, b = v[la], v[lb]
            rows[idx] = {"flip": a["answer"] != b["answer"], "correct_a": bool(a["correct"]),
                         "correct_b": bool(b["correct"]), "missing_a": a["answer"] is None,
                         "missing_b": b["answer"] is None, "cluster": b["batch"]}
    return cfg, rows, (la, lb)


def cluster_boot(values_by_cluster, rng):
    """Bootstrap the mean over problems by resampling clusters; values_by_cluster: list of arrays."""
    k = len(values_by_cluster)
    sums = np.array([v.sum() for v in values_by_cluster])
    counts = np.array([len(v) for v in values_by_cluster])
    draws = rng.integers(0, k, size=(BOOT, k))
    stats = sums[draws].sum(1) / counts[draws].sum(1)
    return [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="NAME=RUN")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    arms = {}
    for spec in args.arm:
        name, run = spec.split("=", 1)
        arms[name] = load_arm(run)
    out = {"arms": {}, "contrasts": {}, "primary": None}
    for name, (cfg, rows, layouts) in arms.items():
        idx = sorted(rows)
        n = len(idx)
        flips = np.array([rows[i]["flip"] for i in idx], dtype=float)
        clusters = defaultdict(list)
        for i in idx:
            clusters[rows[i]["cluster"]].append(rows[i]["flip"])
        k = int(flips.sum())
        out["arms"][name] = {
            "run": cfg.get("out"), "mode": cfg.get("mode", cfg.get("arm")), "top_n": cfg.get("top_n"),
            "soft_steps": cfg.get("soft_steps"),
            "temperature": cfg["temperature"], "layouts": list(layouts), "problems": n, "clusters": len(clusters),
            "answer_flips": k, "flip_rate": k / n if n else float("nan"), "flip_wilson95": wilson(k, n),
            "flip_cluster95": cluster_boot([np.array(v, dtype=float) for v in clusters.values()], rng) if n else None,
            "accuracy": {str(layouts[0]): float(np.mean([rows[i]["correct_a"] for i in idx])) if n else None,
                         str(layouts[1]): float(np.mean([rows[i]["correct_b"] for i in idx])) if n else None},
            "exactly_one_correct": float(np.mean([rows[i]["correct_a"] != rows[i]["correct_b"] for i in idx]))
            if n else None,
            "missing_answers": {str(layouts[0]): int(sum(rows[i]["missing_a"] for i in idx)),
                                str(layouts[1]): int(sum(rows[i]["missing_b"] for i in idx))}}
    if "top15-s64" in out["arms"]:
        a = out["arms"]["top15-s64"]
        lo = a["flip_cluster95"][0] if a["flip_cluster95"] else float("nan")
        out["primary"] = {"arm": "top15-s64", "flip_rate": a["flip_rate"], "cluster95": a["flip_cluster95"],
                          "confirmed": bool(a["flip_rate"] > 0.10 and lo > 0.05)}
    base = "top15-s64"
    if base in arms:
        rows0 = arms[base][1]
        for name, (cfg, rows, _) in arms.items():
            if name == base:
                continue
            common = sorted(set(rows0) & set(rows))
            if not common:
                continue
            d_by_cluster = defaultdict(list)
            b = c = 0
            for i in common:
                x, y = rows0[i]["flip"], rows[i]["flip"]
                d_by_cluster[rows0[i]["cluster"]].append(float(x) - float(y))
                b += int(x and not y)
                c += int(y and not x)
            diff = float(np.mean([float(rows0[i]["flip"]) - float(rows[i]["flip"]) for i in common]))
            out["contrasts"][f"{base} - {name}"] = {
                "problems": len(common), "flip_rate_difference": diff,
                "cluster95": cluster_boot([np.array(v) for v in d_by_cluster.values()], rng),
                "discordant": [b, c], "mcnemar_p": mcnemar_exact(b, c)}
    print(json.dumps(out, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
