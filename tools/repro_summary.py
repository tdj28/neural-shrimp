"""Every number of Table tab:repro and of the answer-level tests (runs REPRO1, REPRO7, REPRO-L8, REPRO-T, REPRO-T7)
recomputed from the raw files, so that the paper's sentences can be checked against one committed script.

Trajectory runs (tools/repro_noise.py): per comparison and float64 label, the cells whose two runs pass a relative
distance of 1e-2, the median step at which they do, the median and 95th percentile of the final distance, and the
share ending with different most probable tokens (some disagreement in the last `tail` steps); for float64 batch
comparisons, the ratio of the observed step to ln(1e-2 / d_1) / lambda (lambda from the cells file) and the log-log
correlation. Answer runs (tools/repro_task.py): per condition the share of problems whose canonical answers differ
between layouts and the accuracies; the frozen primary Delta with its bootstrap interval and sign-flip test (seeds 0,
10,000 draws, as frozen in docs/repro-task-test.md).

Usage: python tools/repro_summary.py [--out docs/repro-summary.json]
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64/repro"
TRAJ = {"qwen0.5b steered (REPRO1)": (["h2cells"], "docs/h2-horizon-cells.json"),
        "qwen7b unsteered (REPRO7)": (["q7-f64", "q7-lowp"], "docs/repro7-cells.json"),
        "llama3.1-8b unsteered (REPRO-L8)": (["l8-f64", "l8-lowp"], "docs/repro-l8-cells.json"),
        "llama3.3-70b steered (REPRO70)": (["r70"], "docs/repro70-cells.json")}
ANSWER = {"qwen0.5b (REPRO-T)": "t05", "qwen7b (REPRO-T7)": "t7"}


def traj_summary(runs, cells_file):
    spec = json.loads((ROOT / cells_file).read_text())
    lam = np.array(spec["map_exponent"], dtype=float)
    out = {}
    for run in runs:
        f = DATA / run / "repro.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        lab = np.array(r["labels"])
        for comp, rows in r["comparisons"].items():
            res = {}
            for L in ("chaotic", "periodic", "other"):
                idx = np.nonzero(lab == L)[0]
                if not len(idx):
                    continue
                steps = [rows[i]["first_step_dist_above_1e-2"] for i in idx]
                reach = [v for v in steps if v is not None]
                fin = np.array([rows[i]["final_dist"] for i in idx])
                res[L] = {"cells": int(len(idx)), "reach_1e-2": len(reach),
                          "median_step": float(np.median(reach)) if reach else None,
                          "median_final": float(np.median(fin)), "p95_final": float(np.quantile(fin, 0.95)),
                          "end_with_different_tokens": float(np.mean([rows[i]["tail_disagreement_fraction"] > 0 for i in idx]))}
            if comp == "batch64":
                obs, pred = [], []
                for i in np.nonzero(lab == "chaotic")[0]:
                    x = rows[i]
                    if x["first_step_dist_above_1e-2"] is not None and lam[i] > 0:
                        obs.append(x["first_step_dist_above_1e-2"])
                        pred.append(np.log(1e-2 / max(x["min_nonzero_dist"], 1e-17)) / lam[i])
                if obs:
                    obs, pred = np.array(obs), np.array(pred)
                    res["lyapunov_prediction"] = {"cells": int(len(obs)), "median_ratio": float(np.median(obs / pred)),
                                                  "loglog_r": float(np.corrcoef(np.log(obs), np.log(pred))[0, 1])}
            out[comp] = res
    return out


def parse(text):
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def answer_summary(run):
    f = DATA / run / "runs.jsonl"
    if not f.exists():
        return None
    recs = [json.loads(line) for line in f.read_text().splitlines()]
    main = [r for r in recs if r["set"] == "heldout"]
    conds = {}
    for r in main:
        conds.setdefault((r["dtype"], r["mode"], r["log10_temperature"]), []).append(r)
    out = {"conditions": {}}
    for (dt, mode, lt), rs in sorted(conds.items(), key=lambda kv: (kv[0][0], kv[0][1], -1 if kv[0][2] is None else kv[0][2])):
        out["conditions"][f"{dt} {mode} {lt}"] = {
            "problems": len(rs), "answers_differ": float(np.mean([r["single"]["parsed"] != r["batch4"]["parsed"] for r in rs])),
            "accuracy_single": float(np.mean([r["single"]["correct"] for r in rs])),
            "accuracy_batch": float(np.mean([r["batch4"]["correct"] for r in rs])),
            "rerun_identical": float(np.mean([r["rerun_identical"] for r in rs]))}
    for dt in ("bf16", "float64"):
        per = {}
        for r in main:
            if r["dtype"] == dt:
                per.setdefault(r["i"], {})[(r["mode"], r["log10_temperature"])] = int(r["single"]["parsed"] != r["batch4"]["parsed"])
        D = np.array([(v[("soft", 0.0)] + v[("soft", 0.2)]) / 2 - v[("greedy", None)] for _, v in sorted(per.items())])
        rng = np.random.default_rng(0)
        est = 100 * D.mean()
        boot = [100 * rng.choice(D, len(D)).mean() for _ in range(10000)]
        perm = np.mean([abs(100 * (D * rng.choice([-1, 1], len(D))).mean()) >= abs(est) for _ in range(10000)])
        out[f"delta_{dt}"] = {"points": float(est), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                              "p_signflip": float(max(perm, 1 / 10001)), "problems": int(len(D))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "docs/repro-summary.json"))
    args = ap.parse_args()
    res = {"trajectories": {k: traj_summary(*v) for k, v in TRAJ.items()},
           "answers": {k: answer_summary(v) for k, v in ANSWER.items()}}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main()
