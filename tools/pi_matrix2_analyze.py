"""Analysis of run PM2 (docs/pi-matrix2-confirmatory.md), fixed before its data.

Cells are (model, task, GPU), each with a soft and a plain arm (tools/pi_reason.py runs with two layouts). Per arm,
as in tools/pi_matrix_analyze.py: answer flips between layouts (Wilson and cluster bootstrap intervals; clusters are
the batches of the larger layout: full 16-problem groups for GSM8K, single problems for MATH-500), accuracy per
layout, the share of problems with exactly one layout correct, missing answers, forced ends of thinking and the median
step at which thinking ended. Per cell: soft minus plain flip rate (paired, cluster bootstrap, exact McNemar) and soft
minus plain accuracy in the larger layout (paired, exact McNemar).
Primary: the soft arms' flip rate pooled over the cells marked primary (problems as units; bootstrap resamples
clusters within each cell); confirmed when it exceeds 0.05 and the lower end of its 95 % interval exceeds 0.02.
Cross-GPU: for a model and task run on two GPUs, the difference of the soft (and plain) flip rates, paired by problem.

Usage: python tools/pi_matrix2_analyze.py --cell NAME:primary|replication:SOFT_DIR:PLAIN_DIR ... [--cross NAME1,NAME2]
       [--json OUT]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import pi_matrix_analyze as pma

BOOT = pma.BOOT


def arm_stats(rows, layouts, rng):
    idx = sorted(rows)
    n = len(idx)
    fl = np.array([rows[i]["flip"] for i in idx], dtype=float)
    cl = defaultdict(list)
    for i in idx:
        cl[rows[i]["cluster"]].append(rows[i]["flip"])
    return {"problems": n, "clusters": len(cl), "answer_flips": int(fl.sum()), "flip_rate": float(fl.mean()) if n else None,
            "flip_wilson95": pma.wilson(int(fl.sum()), n),
            "flip_cluster95": pma.cluster_boot([np.array(v, dtype=float) for v in cl.values()], rng) if n else None,
            "accuracy": {str(layouts[0]): float(np.mean([rows[i]["correct_a"] for i in idx])),
                         str(layouts[1]): float(np.mean([rows[i]["correct_b"] for i in idx]))},
            "exactly_one_correct": float(np.mean([rows[i]["correct_a"] != rows[i]["correct_b"] for i in idx])),
            "missing_answers": {str(layouts[0]): int(sum(rows[i]["missing_a"] for i in idx)),
                                str(layouts[1]): int(sum(rows[i]["missing_b"] for i in idx))}}


def load(run_dir):
    cfg, rows, layouts = pma.load_arm(run_dir)
    extra = {}
    for line in open(Path(run_dir) / "records.jsonl"):
        r = json.loads(line)
        if r["index"] >= 0 and r["layout"] == layouts[1]:
            extra[r["index"]] = r
    return cfg, rows, layouts, extra


def paired(rows_a, rows_b, key_a, key_b, rng):
    common = sorted(set(rows_a) & set(rows_b))
    d_by = defaultdict(list)
    b = c = 0
    for i in common:
        x, y = bool(rows_a[i][key_a]), bool(rows_b[i][key_b])
        d_by[rows_a[i]["cluster"]].append(float(x) - float(y))
        b += int(x and not y)
        c += int(y and not x)
    diff = float(np.mean([float(rows_a[i][key_a]) - float(rows_b[i][key_b]) for i in common])) if common else None
    return {"problems": len(common), "difference": diff,
            "cluster95": pma.cluster_boot([np.array(v) for v in d_by.values()], rng) if common else None,
            "discordant": [b, c], "mcnemar_p": pma.mcnemar_exact(b, c)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--cross", action="append", default=[])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(pma.SEED)
    cells, out = {}, {"cells": {}, "primary": None, "cross_gpu": {}}
    for spec in args.cell:
        name, role, soft_dir, plain_dir = spec.split(":", 3)
        s_cfg, s_rows, lay, s_extra = load(soft_dir)
        p_cfg, p_rows, lay_p, p_extra = load(plain_dir)
        cells[name] = (role, s_rows, p_rows)
        sw = [r["switch_step"] for r in s_extra.values() if r["switch_step"] is not None]
        out["cells"][name] = {
            "role": role, "model": s_cfg["model"], "task": s_cfg["task"], "layouts": list(lay),
            "soft": {**arm_stats(s_rows, lay, rng), "forced_switch": int(sum(r["forced_switch"] for r in s_extra.values())),
                     "median_switch_step": float(np.median(sw)) if sw else None},
            "plain": {**arm_stats(p_rows, lay_p, rng), "forced_switch": int(sum(r["forced_switch"] for r in p_extra.values()))},
            "soft_minus_plain_flips": paired(s_rows, p_rows, "flip", "flip", rng),
            "soft_minus_plain_accuracy_larger_layout": paired(s_rows, p_rows, "correct_b", "correct_b", rng)}
    prim = [(n, c) for n, c in cells.items() if c[0] == "primary"]
    if prim:
        per_cell = []
        for n, (role, s_rows, _) in prim:
            cl = defaultdict(list)
            for i, r in s_rows.items():
                cl[r["cluster"]].append(float(r["flip"]))
            per_cell.append([np.array(v) for v in cl.values()])
        k_tot = sum(sum(v.sum() for v in cs) for cs in per_cell)
        n_tot = sum(sum(len(v) for v in cs) for cs in per_cell)
        stats = []
        for _ in range(BOOT):
            k = n = 0.0
            for cs in per_cell:
                draw = rng.integers(0, len(cs), size=len(cs))
                k += sum(cs[j].sum() for j in draw)
                n += sum(len(cs[j]) for j in draw)
            stats.append(k / n)
        lo, hi = float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))
        rate = float(k_tot / n_tot)
        out["primary"] = {"cells": [n for n, _ in prim], "problems": int(n_tot), "answer_flips": int(k_tot),
                          "flip_rate": rate, "cluster95": [lo, hi], "confirmed": bool(rate > 0.05 and lo > 0.02)}
    for spec in args.cross:
        a, b = spec.split(",")
        out["cross_gpu"][spec] = {"soft": paired(cells[a][1], cells[b][1], "flip", "flip", rng),
                                  "plain": paired(cells[a][2], cells[b][2], "flip", "flip", rng)}
    print(json.dumps(out, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
