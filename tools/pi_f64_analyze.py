"""Program PI (float64): summary of tools/pi_f64_twins.py runs, and their join with the bfloat16 layout pairs of
tools/pi_harness.py on the same problems and settings.

Per arm:
- **Accuracy:** the accuracy of the base rows.
- **Float64 twin answer flips:** a twin displaced by a relative eps0 on its first concept token answers differently
  from its base row.
- **Separation class of each problem**, from its twins' relative distances to the base row over the soft steps:
  - *separated*: some twin's distance exceeded 1e-3 at some step (macroscopic separation);
  - *grew*: not separated, but some twin ended farther from the base than eps0;
  - *contracted*: every twin ended at most eps0 from the base. Twins at distance exactly zero have merged with the base,
    meaning they contracted below float64 resolution; they are counted apart within this class.
- **Net amplification of each twin:** ln(d_final / eps0) per step, with merged twins at minus infinity. Quantiles are
  reported, which are robust to the infinite values.

With --bf16 RUN, a pi_harness soft run at the same settings with layouts 8 and 16, each problem's bfloat16 layout answer
flip and trajectory divergence is joined to its float64 class. Problems are the units: intervals are Wilson intervals
over problems.

With --grid RUN (a run of the multi-configuration tool, one folder per temperature and steering strength, several
twin offsets), each configuration and offset level gets the accuracy of its base rows, its answer and answer-text flip
rates, the share of problems with a twin past that level's dmax, the median step of that crossing, the share of twins
merged exactly, and quantiles of the net amplification.

Usage: python tools/pi_f64_analyze.py --f64 RUN [--bf16 RUN] [--json OUT]
       python tools/pi_f64_analyze.py --grid RUN [--json OUT]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

DMAX = 1e-3


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


def load(run):
    run = Path(run)
    recs = [json.loads(x) for x in (run / "records.jsonl").read_text().splitlines() if x.strip()]
    args = json.loads((run / "run.json").read_text())["args"]
    return recs, args


def classify(r, eps0):
    fd = np.asarray(r["final_distance"], dtype=float)
    if any(f is not None for f in r["first_above_dmax"]):
        return "separated"
    if (fd > eps0).any():
        return "grew"
    return "contracted"


def f64_summary(recs, args):
    eps0, S = float(args["eps0"]), int(args["soft_steps"])
    seen = {}
    for r in recs:                                    # one record per problem; a duplicate would be a resumed batch
        if r["index"] in seen:
            raise SystemExit(f"duplicate problem {r['index']} in the float64 run")
        seen[r["index"]] = r
    cls = {i: classify(r, eps0) for i, r in seen.items()}
    net = []
    merged_twins = 0
    for r in seen.values():
        for f in r["final_distance"]:
            if f > 0:
                net.append(math.log(f / eps0) / (S - 1))
            else:
                net.append(-math.inf)
                merged_twins += 1
    net = np.asarray(net)
    sep_steps = [f for r in seen.values() for f in r["first_above_dmax"] if f is not None]
    out = {"problems": len(seen), "eps0": eps0, "soft_steps": S, "top_n": args["top_n"],
           "temperature": args["temperature"], "twins_per_problem": args["twins"],
           "accuracy": rate([r["correct"] for r in seen.values()]),
           "twin_flip_any": rate([r["twin_flips"] > 0 for r in seen.values()]),
           "twin_flip_pairs": rate([a != r["answer"] for r in seen.values() for a in r["twin_answers"]]),
           "class_counts": {c: sum(v == c for v in cls.values()) for c in ("separated", "grew", "contracted")},
           "merged_twins": merged_twins, "twins": int(net.size),
           "net_rate_quantiles": {q: float(np.quantile(net, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "separation_step_median": float(np.median(sep_steps)) if sep_steps else None,
           "separation_step_quartiles": [float(np.quantile(sep_steps, q)) for q in (0.25, 0.75)] if sep_steps else None,
           "flip_by_class": {}, "entropy_mean_by_class": {}, "gap_min_by_class": {}}
    for c in ("separated", "grew", "contracted"):
        rs = [seen[i] for i, v in cls.items() if v == c]
        out["flip_by_class"][c] = rate([r["twin_flips"] > 0 for r in rs])
        out["entropy_mean_by_class"][c] = float(np.median([r["entropy_mean"] for r in rs])) if rs else None
        out["gap_min_by_class"][c] = float(np.median([r["gap_min"] for r in rs])) if rs else None
    return out, seen, cls


def bf16_pairs(run):
    recs, args = load(run)
    by = defaultdict(dict)
    for r in recs:
        if r["index"] < 0:                            # unscored fillers
            continue
        by[r["index"]][r["layout"]] = r
    pairs = {}
    for i, d in by.items():
        if len(d) == 2:
            a, b = (d[k] for k in sorted(d))
            pairs[i] = {"flip": a["answer"] != b["answer"], "div": a["tokens"] != b["tokens"] or a["top1"] != b["top1"],
                        "correct": [a["correct"], b["correct"]]}
    return pairs, args


def grid_summary(run):
    run = Path(run)
    args = json.loads((run / "run.json").read_text())["args"]
    S = int(args["soft_steps"])
    configs = []
    for sub in sorted(run.iterdir()):
        if not (sub / "records.jsonl").is_file():
            continue
        recs = [json.loads(x) for x in (sub / "records.jsonl").read_text().splitlines() if x.strip()]
        idx = [r["index"] for r in recs]
        if len(set(idx)) != len(idx):
            raise SystemExit(f"duplicate problems in {sub}")
        c = {"config": sub.name, "temperature": recs[0]["temperature"], "alpha": recs[0]["alpha"],
             "problems": len(recs), "accuracy": rate([r["correct"] for r in recs]),
             "base_numeric": rate([r["answer"] is not None for r in recs]),
             "entropy_median": float(np.median([np.mean(r["entropy"]) for r in recs])), "levels": {}}
        for eps0 in sorted({t["eps0"] for r in recs for t in r["twins"]}):
            tw = [[t for t in r["twins"] if t["eps0"] == eps0] for r in recs]
            net = np.array([math.log(t["final_distance"] / eps0) / (S - 1) if t["final_distance"] > 0 else -math.inf
                            for ts in tw for t in ts])
            cross = [t["first_above_dmax"] for ts in tw for t in ts if t["first_above_dmax"] is not None]
            c["levels"][f"{eps0:g}"] = {
                "answer_flip_any": rate([any(t["flip"] for t in ts) for ts in tw]),
                "text_flip_any": rate([any(t["text_flip"] for t in ts) for ts in tw]),
                "separated_any": rate([any(t["first_above_dmax"] is not None for t in ts) for ts in tw]),
                "dmax": tw[0][0]["dmax"],
                "crossing_step_median": float(np.median(cross)) if cross else None,
                "merged_twins": rate([t["final_distance"] == 0 for ts in tw for t in ts]),
                "net_rate_quantiles": {q: float(np.quantile(net, q)) for q in (0.25, 0.5, 0.75, 0.9)}}
        configs.append(c)
    return {"run": str(run), "soft_steps": S, "top_n": args["top_n"], "concept": args["concept"],
            "layer": args["layer"], "configs": configs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f64")
    ap.add_argument("--grid")
    ap.add_argument("--bf16")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.grid:
        text = json.dumps(grid_summary(a.grid), indent=1)
        if a.json:
            Path(a.json).write_text(text + "\n")
        print(text)
        return
    recs, args = load(a.f64)
    out, seen, cls = f64_summary(recs, args)
    out["f64_run"] = str(a.f64)
    if a.bf16:
        pairs, bargs = bf16_pairs(a.bf16)
        mism = {k: (bargs.get(k), args.get(k)) for k in ("temperature", "top_n", "soft_steps") if bargs.get(k) != args.get(k)}
        if mism:
            raise SystemExit(f"settings differ between the runs: {mism}")
        common = sorted(set(pairs) & set(seen))
        j = {"bf16_run": str(a.bf16), "common_problems": len(common),
             "bf16_flip": rate([pairs[i]["flip"] for i in common]),
             "bf16_divergence": rate([pairs[i]["div"] for i in common]),
             "bf16_flip_by_f64_class": {}, "bf16_divergence_by_f64_class": {},
             "accuracy_f64_on_common": rate([seen[i]["correct"] for i in common]),
             "accuracy_bf16_layouts_on_common": [rate([pairs[i]["correct"][k] for i in common]) for k in (0, 1)],
             "bf16_flip_given_f64_twin_flip": rate([pairs[i]["flip"] for i in common if seen[i]["twin_flips"] > 0]),
             "bf16_flip_given_no_f64_twin_flip": rate([pairs[i]["flip"] for i in common if seen[i]["twin_flips"] == 0])}
        for c in ("separated", "grew", "contracted"):
            ids = [i for i in common if cls[i] == c]
            j["bf16_flip_by_f64_class"][c] = rate([pairs[i]["flip"] for i in ids])
            j["bf16_divergence_by_f64_class"][c] = rate([pairs[i]["div"] for i in ids])
        out["join"] = j
    text = json.dumps(out, indent=1)
    if a.json:
        Path(a.json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
