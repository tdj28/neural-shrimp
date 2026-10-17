"""Analysis of Program PI runs E2 (gain-capped soft thinking) and E3 (majority vote over perturbed soft runs).

E2: each arm is one tools/pi_harness.py soft run with layouts 8 and 16 on the same problems. Per arm, as in
tools/pi_matrix_analyze.py: answer flip rate between layouts (Wilson and cluster-bootstrap intervals; clusters are
the batches of the 16-problem layout), accuracy per layout, exactly-one-correct, missing answers; plus the capping
diagnostics (share of soft steps capped, mean step temperature, mean gain proxy). Contrasts against the uncapped arm
(--base), paired by problem: flip-rate difference (paired cluster bootstrap, exact McNemar), accuracy difference
(mean of the two layouts) and exactly-one-correct difference (paired cluster bootstrap).

E3: single-layout runs on the same problems. --soft lists K perturbed soft runs, --soft0 the unperturbed soft run,
--greedy the greedy chain-of-thought run, --sample the K sampled chain-of-thought runs. A vote is the plurality of
the extracted answers (missing answers do not vote; a tie goes to the answer that reached the tied count first in
run order; no answer at all counts as wrong). Accuracies of: the soft vote, the mean single perturbed soft run, the
unperturbed soft run, greedy, the sampled vote (self-consistency) and the mean single sampled run; pass@K of both
ensembles; paired differences with cluster-bootstrap intervals (clusters: batches of the unperturbed soft run); and
the vote by sensitivity (number of distinct answers among the K perturbed runs).

Usage: python tools/pi_e2e3_analyze.py e2 --base NAME --arm NAME=RUN [...] [--json OUT]
       python tools/pi_e2e3_analyze.py e3 --soft0 RUN --soft RUN [...] --greedy RUN --sample RUN [...] [--json OUT]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi_matrix_analyze as pma  # noqa: E402


def records(run):
    out = defaultdict(dict)
    for line in open(Path(run) / "records.jsonl"):
        r = json.loads(line)
        if r["index"] >= 0:
            out[r["index"]][r["layout"]] = r
    return out


def boot_paired(diff_by_cluster, rng):
    return pma.cluster_boot([np.asarray(v, dtype=float) for v in diff_by_cluster.values()], rng)


def e2(args):
    rng = np.random.default_rng(pma.SEED)
    arms, recs = {}, {}
    for spec in args.arm:
        name, run = spec.split("=", 1)
        arms[name] = pma.load_arm(run)
        recs[name] = records(run)
    out = {"arms": {}, "contrasts": {}}
    for name, (cfg, rows, layouts) in arms.items():
        idx = sorted(rows)
        n = len(idx)
        clusters = defaultdict(list)
        for i in idx:
            clusters[rows[i]["cluster"]].append(rows[i]["flip"])
        k = int(sum(rows[i]["flip"] for i in idx))
        steps = capped = 0
        temps, gains = [], []
        for i in idx:
            for lay in layouts:
                r = recs[name][i][lay]
                gains.extend(r["gain"])
                if "temp" in r:
                    temps.extend(r["temp"])
                    capped += sum(t < cfg["temperature"] for t in r["temp"])
                steps += len(r["gain"])
        acc = {str(la): float(np.mean([rows[i][f"correct_{s}"] for i in idx])) for la, s in zip(layouts, "ab")}
        out["arms"][name] = {
            "run": cfg.get("out"), "gain_cap": cfg.get("gain_cap"), "top_n": cfg.get("top_n"),
            "temperature": cfg["temperature"], "soft_steps": cfg.get("soft_steps"), "layouts": list(layouts),
            "problems": n, "clusters": len(clusters), "answer_flips": k, "flip_rate": k / n,
            "flip_wilson95": pma.wilson(k, n),
            "flip_cluster95": pma.cluster_boot([np.array(v, dtype=float) for v in clusters.values()], rng),
            "accuracy": acc, "accuracy_mean": float(np.mean(list(acc.values()))),
            "exactly_one_correct": float(np.mean([rows[i]["correct_a"] != rows[i]["correct_b"] for i in idx])),
            "missing_answers": {str(la): int(sum(rows[i][f"missing_{s}"] for i in idx))
                                for la, s in zip(layouts, "ab")},
            "steps_capped_share": capped / steps if temps else 0.0,
            "mean_step_temperature": float(np.mean(temps)) if temps else cfg["temperature"],
            "mean_gain": float(np.mean(gains)), "max_gain": float(np.max(gains))}
    rows0 = arms[args.base][1]
    for name, (cfg, rows, _) in arms.items():
        if name == args.base:
            continue
        common = sorted(set(rows0) & set(rows))
        d_flip, d_acc, d_one = defaultdict(list), defaultdict(list), defaultdict(list)
        b = c = 0
        for i in common:
            x, y = rows0[i], rows[i]
            cl = x["cluster"]
            d_flip[cl].append(float(y["flip"]) - float(x["flip"]))
            d_acc[cl].append(0.5 * (y["correct_a"] + y["correct_b"] - x["correct_a"] - x["correct_b"]))
            d_one[cl].append(float(y["correct_a"] != y["correct_b"]) - float(x["correct_a"] != x["correct_b"]))
            b += int(x["flip"] and not y["flip"])
            c += int(y["flip"] and not x["flip"])
        mean = lambda d: float(np.mean([v for vs in d.values() for v in vs]))  # noqa: E731
        out["contrasts"][f"{name} - {args.base}"] = {
            "problems": len(common),
            "flip_rate_difference": mean(d_flip), "flip_cluster95": boot_paired(d_flip, rng),
            "flip_discordant_base_only_arm_only": [b, c], "flip_mcnemar_p": pma.mcnemar_exact(b, c),
            "accuracy_difference": mean(d_acc), "accuracy_cluster95": boot_paired(d_acc, rng),
            "exactly_one_correct_difference": mean(d_one), "exactly_one_cluster95": boot_paired(d_one, rng)}
    return out


def single(run):
    by = records(run)
    lay = {lay for v in by.values() for lay in v}
    if len(lay) != 1:
        raise SystemExit(f"{run}: expected one layout, found {sorted(lay)}")
    lay = lay.pop()
    return {i: v[lay] for i, v in by.items()}


def vote(answers):
    """Plurality over non-missing answers; ties go to the answer that reached the tied count first."""
    seen = Counter()
    best, best_n = None, 0
    for a in answers:
        if a is None:
            continue
        seen[a] += 1
        if seen[a] > best_n:
            best, best_n = a, seen[a]
    return best


def e3(args):
    rng = np.random.default_rng(pma.SEED)
    s0 = single(args.soft0)
    soft = [single(r) for r in args.soft]
    hard = [single(r) for r in args.sample]
    gr = single(args.greedy)
    common = sorted(set(s0).intersection(*[set(x) for x in soft + hard], set(gr)))
    if not common:
        raise SystemExit("no problems common to all runs")
    per = {}
    for i in common:
        gold = s0[i]["gold"]
        sa = [x[i]["answer"] for x in soft]
        ha = [x[i]["answer"] for x in hard]
        sv, hv = vote(sa), vote(ha)
        per[i] = {"cluster": s0[i]["batch"],
                  "soft_vote": sv is not None and sv == gold,
                  "soft_single": float(np.mean([x[i]["correct"] for x in soft])),
                  "soft0": bool(s0[i]["correct"]),
                  "greedy": bool(gr[i]["correct"]),
                  "sample_vote": hv is not None and hv == gold,
                  "sample_single": float(np.mean([x[i]["correct"] for x in hard])),
                  "soft_pass": any(x[i]["correct"] for x in soft),
                  "sample_pass": any(x[i]["correct"] for x in hard),
                  "soft_distinct": len(set(sa)),
                  "soft_changed_vs0": float(np.mean([a != s0[i]["answer"] for a in sa])),
                  "sample_distinct": len(set(ha)),
                  "perturb_realized": [x[i].get("perturb_realized") for x in soft]}
    n = len(common)
    keys = ["soft_vote", "soft_single", "soft0", "greedy", "sample_vote", "sample_single", "soft_pass", "sample_pass"]
    acc = {k: float(np.mean([per[i][k] for i in common])) for k in keys}
    pairs = [("soft_vote", "soft_single"), ("soft_vote", "soft0"), ("soft_vote", "greedy"),
             ("soft_vote", "sample_vote"), ("sample_vote", "sample_single"), ("sample_vote", "greedy"),
             ("soft_single", "soft0")]
    contrasts = {}
    for a, b in pairs:
        d = defaultdict(list)
        for i in common:
            d[per[i]["cluster"]].append(float(per[i][a]) - float(per[i][b]))
        contrasts[f"{a} - {b}"] = {"difference": acc[a] - acc[b], "cluster95": boot_paired(d, rng)}
    by_sens = {}
    for m in sorted({per[i]["soft_distinct"] for i in common}):
        ii = [i for i in common if per[i]["soft_distinct"] == m]
        by_sens[str(m)] = {"problems": len(ii),
                           **{k: float(np.mean([per[i][k] for i in ii])) for k in
                              ("soft_vote", "soft_single", "soft0", "greedy", "sample_vote")}}
    real = np.array([v for i in common for v in per[i]["perturb_realized"] if v is not None])
    out = {"problems": n, "clusters": len({per[i]["cluster"] for i in common}), "K_soft": len(soft),
           "K_sample": len(hard), "accuracy": acc, "contrasts": contrasts,
           "soft_answer_changed_vs_unperturbed": float(np.mean([per[i]["soft_changed_vs0"] for i in common])),
           "soft_distinct_answers_mean": float(np.mean([per[i]["soft_distinct"] for i in common])),
           "sample_distinct_answers_mean": float(np.mean([per[i]["sample_distinct"] for i in common])),
           "by_soft_sensitivity_distinct_answers": by_sens,
           "perturb_realized_relative": ({"min": float(real.min()), "median": float(np.median(real)),
                                          "max": float(real.max())} if real.size else None),
           "runs": {"soft0": args.soft0, "soft": args.soft, "greedy": args.greedy, "sample": args.sample}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("e2")
    a.add_argument("--base", required=True)
    a.add_argument("--arm", action="append", required=True)
    a.add_argument("--json", default=None)
    b = sub.add_parser("e3")
    b.add_argument("--soft0", required=True)
    b.add_argument("--soft", nargs="+", required=True)
    b.add_argument("--greedy", required=True)
    b.add_argument("--sample", nargs="+", required=True)
    b.add_argument("--json", default=None)
    args = ap.parse_args()
    out = e2(args) if args.cmd == "e2" else e3(args)
    print(json.dumps(out, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
