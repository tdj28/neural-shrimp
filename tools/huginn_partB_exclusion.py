"""Part B of the frozen Huginn test (GSM8K generation) recomputed without given problems (correction note, 3 Nov 2026).

GSM8K test item 18 ("Claire makes a 3 egg omelet ...", the question of Geiping et al.'s latent-orbit figure) was used
as a bare prompt in the exploratory dynamics runs of 12 October 2026 (huginn-fixed-spectrum-gsm-claire,
huginn-ns-gsm-claire), before the design was frozen on 13 October at commit 9805ed32. Those runs computed a fixed
point and its spectrum, never an answer or an accuracy, but the item is one of the test's 200 problems.

This tool reads the frozen rows and arguments from the frozen analysis output, applies the same admission checks, and
uses the frozen statistics (exact one-sided McNemar, paired bootstrap with seed 0) from tools/huginn_arc_analysis.py.
It first requires the recomputation on all problems to reproduce the frozen generation.json exactly, then repeats it
without the excluded problems and writes a hash-bound receipt. It never overwrites a receipt.

Usage: python tools/huginn_partB_exclusion.py --exclude 18 --receipt docs/huginn-partB-item18-receipt.json
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import explore_llm_attractors as ela
import huginn_arc_analysis as frozen

ROOT = Path(__file__).resolve().parents[1]
GENERATION = "huginn-relax/test-partB/generation.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(args):
    by = {}
    for folder in args["rows"]:
        for line in open(ela.DATA / "explore" / folder / "rows.jsonl"):
            r = json.loads(line)
            key = (str(r["arm"]), int(r["index"]))
            if key in by:
                raise SystemExit(f"duplicate row {key}")
            if (r.get("split") != args["split"] or r.get("steps") != args["steps"]
                    or r.get("prompt_style") != args["prompt_style"]
                    or r.get("seed") != args["seed"] * 100003 + int(r["index"])
                    or abs(float(r["lam"]) - float(r["arm"])) > 1e-12):
                raise SystemExit(f"row {key} differs from the frozen split/steps/prompt/seed/lam")
            by[key] = r
    return by


def score(by, args, order):
    arms = [str(a) for a in args["arms"]]
    acc = {a: np.array([bool(by[(a, i)]["correct"]) for i in order]) for a in arms}
    base = acc[str(args["base"])]
    out = {"n": len(order), "accuracy": {a: float(v.mean()) for a, v in acc.items()}}
    for a in arms:
        if a == str(args["base"]):
            continue
        p, lost, gained = frozen.mcnemar_less(base, acc[a])
        mean, boots = frozen.paired_bootstrap(base, acc[a])
        out[a] = {"difference": mean, "lower_one_sided_95": float(np.percentile(boots, 5)), "mcnemar_less_p": p,
                  "lost": lost, "gained": gained}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exclude", type=int, nargs="+", required=True, help="problem indices to leave out")
    ap.add_argument("--receipt", type=Path, required=True)
    a = ap.parse_args()
    if a.receipt.exists():
        raise SystemExit(f"{a.receipt} exists; receipts are never overwritten")
    gen_path = ela.DATA / "explore" / GENERATION
    ref = json.loads(gen_path.read_text())
    args = ref["args"]
    by = load(args)
    every = list(range(args["start"], args["start"] + args["count"]))
    for arm in [str(x) for x in args["arms"]]:
        if {i for (b, i) in by if b == arm} != set(every):
            raise SystemExit(f"arm {arm} does not cover problems {every[0]}..{every[-1]}")
    full = score(by, args, every)
    m = ref["metrics"]["correct"]
    for arm in [str(x) for x in args["arms"] if str(x) != str(args["base"])]:
        for k in ("difference", "lower_one_sided_95", "mcnemar_less_p", "lost", "gained"):
            if full[arm][k] != m[arm][k]:
                raise SystemExit(f"recomputation does not reproduce the frozen output: arm {arm}, {k}")
    kept = [i for i in every if i not in set(a.exclude)]
    reduced = score(by, args, kept)
    receipt = {
        "note": __doc__.split("\n\n")[1].replace("\n", " "),
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                        text=True).stdout.strip(),
        "frozen_generation": {"file": GENERATION, "sha256": sha256(gen_path)},
        "rows": [{"folder": f, "sha256": sha256(ela.DATA / "explore" / f / "rows.jsonl")} for f in args["rows"]],
        "excluded": a.exclude,
        "excluded_outcomes": {str(i): {str(arm): bool(by[(str(arm), i)]["correct"]) for arm in args["arms"]}
                              for i in a.exclude},
        "reproduces_frozen_output": True,
        "all_problems": full,
        "without_excluded": reduced,
        "predictions_without_excluded": {
            "B1_decline_pass": bool(reduced[str(args["collapse_arm"])]["mcnemar_less_p"] < 0.05),
            "B3_noninferior_pass": bool(reduced[str(args["safe_arm"])]["lower_one_sided_95"] > -args["margin"])},
    }
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt["predictions_without_excluded"]), json.dumps(reduced))


if __name__ == "__main__":
    main()
