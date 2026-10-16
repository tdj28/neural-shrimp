"""Run H: the stability horizon of soft thinking. Answer agreement between two batch layouts at intermediate soft
steps (tools/pi_harness.py with --readout-steps), per temperature and steering strength, optionally joined per
problem with the float64 twins of the same configurations (tools/pi_f64_twins.py, run PIF64-H).

Per configuration (one pi_harness run with layouts 8 and 16):
- **Agreement at each readout step S** (and at the final answer): the share of problems whose two layouts give the
  same answer text, and the same extracted number. Wilson intervals over problems.
- **Layout horizon of each problem:** the first readout step at which the two layouts' answer texts differ. Problems
  that agree at every readout and at the end are counted as beyond the last step.
- **First trajectory divergence:** the first soft step at which the two layouts' most probable concept token differs.
- **Accuracy at each S** for both layouts, and the share of problems whose answer at S already equals the final one.

With --f64 GRID (a PIF64-H run whose configuration folders are named t{T}-a{alpha}), each problem's layout horizon is
joined to its float64 twins at the same configuration:
- the bf16 text-flip rate at the end, by float64 separation at the smallest twin offset;
- the float64 answer-flip rate of twins displaced by each offset.

Usage: python tools/pi_horizon_analyze.py RUN [RUN ...] [--f64 GRID] [--json OUT]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def wilson(k, n, z=1.959964):
    if not n:
        return [float("nan"), float("nan")]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def rate(flags):
    k, n = int(sum(flags)), len(flags)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "ci95": wilson(k, n)}


def load_pairs(run):
    run = Path(run)
    args = json.loads((run / "run.json").read_text())["args"]
    by = defaultdict(dict)
    for line in (run / "records.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["index"] < 0:
            continue
        if r["layout"] in by[r["index"]]:
            raise SystemExit(f"duplicate record for problem {r['index']} layout {r['layout']} in {run}")
        by[r["index"]][r["layout"]] = r
    pairs = {i: (d[min(d)], d[max(d)]) for i, d in by.items() if len(d) == 2}
    return args, pairs


def first_divergence(a, b):
    return next((t for t, (x, y) in enumerate(zip(a["top1"], b["top1"])) if x != y), None)


def summarize(run):
    args, pairs = load_pairs(run)
    steps = sorted({int(s) for a, _ in pairs.values() for s in a.get("readout_texts", {})})
    if not steps:
        raise SystemExit(f"{run} has no readout texts (run with --readout-steps on a harness that stores them)")
    final = int(args["soft_steps"])
    out = {"run": str(run), "temperature": args["temperature"], "alpha": args.get("alpha", 0.0),
           "concept": args.get("concept"), "top_n": args["top_n"], "soft_steps": final, "problems": len(pairs),
           "readout_steps": steps + [final], "text_agree": {}, "answer_agree": {}, "accuracy": {}, "settled": {}}
    for s in steps + [final]:
        def txt(r):
            return r["text"] if s == final else r["readout_texts"][str(s)]

        def ans(r):
            return r["answer"] if s == final else r["readouts"][str(s)]
        out["text_agree"][str(s)] = rate([txt(a) == txt(b) for a, b in pairs.values()])
        out["answer_agree"][str(s)] = rate([ans(a) == ans(b) for a, b in pairs.values()])
        out["accuracy"][str(s)] = [rate([ans(r) is not None and ans(r) == r["gold"] for r in (p[k] for p in
                                                                                                 pairs.values())])
                                   for k in (0, 1)]
        out["settled"][str(s)] = rate([ans(a) == a["answer"] for a, _ in pairs.values()])
    horizon = {}
    for i, (a, b) in pairs.items():
        h = next((s for s in steps if a["readout_texts"][str(s)] != b["readout_texts"][str(s)]), None)
        if h is None and a["text"] != b["text"]:
            h = final
        horizon[i] = h
    hs = [h for h in horizon.values() if h is not None]
    out["horizon_counts"] = {str(s): sum(h == s for h in horizon.values()) for s in steps + [final]}
    out["horizon_beyond"] = sum(h is None for h in horizon.values())
    out["horizon_median_among_split"] = float(np.median(hs)) if hs else None
    fd = [first_divergence(a, b) for a, b in pairs.values()]
    out["trajectory_divergence"] = rate([d is not None for d in fd])
    out["first_divergence_median"] = float(np.median([d for d in fd if d is not None])) if any(
        d is not None for d in fd) else None
    return out, pairs, horizon


def join_f64(summary, pairs, horizon, grid):
    sub = Path(grid) / f"t{summary['temperature']:g}-a{summary['alpha']:g}"
    if not (sub / "records.jsonl").is_file():
        return {"f64_config": str(sub), "missing": True}
    f64 = {}
    for line in (sub / "records.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            f64[r["index"]] = r
    common = sorted(set(pairs) & set(f64))
    eps_levels = sorted({t["eps0"] for r in f64.values() for t in r["twins"]})
    smallest = eps_levels[0]
    sep = {i: any(t["first_above_dmax"] is not None for t in f64[i]["twins"] if t["eps0"] == smallest) for i in common}
    flip_bf16 = {i: pairs[i][0]["text"] != pairs[i][1]["text"] for i in common}
    out = {"f64_config": str(sub), "common_problems": len(common),
           "f64_separated_smallest_offset": rate([sep[i] for i in common]),
           "bf16_text_flip_given_f64_separated": rate([flip_bf16[i] for i in common if sep[i]]),
           "bf16_text_flip_given_f64_not_separated": rate([flip_bf16[i] for i in common if not sep[i]]),
           "f64_text_flip_by_offset": {f"{e:g}": rate([any(t["text_flip"] for t in f64[i]["twins"] if t["eps0"] == e)
                                                       for i in common]) for e in eps_levels},
           "bf16_horizon_vs_f64_separation": {
               "median_horizon_separated": float(np.median([horizon[i] for i in common if sep[i] and horizon[i]]))
               if any(sep[i] and horizon[i] for i in common) else None,
               "median_horizon_not_separated": float(np.median([horizon[i] for i in common if not sep[i] and horizon[i]]))
               if any((not sep[i]) and horizon[i] for i in common) else None}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--f64")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = []
    for run in a.runs:
        s, pairs, horizon = summarize(run)
        if a.f64:
            s["join"] = join_f64(s, pairs, horizon, a.f64)
        res.append(s)
    text = json.dumps({"configs": res}, indent=1)
    if a.json:
        Path(a.json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
