#!/usr/bin/env python3
"""Recompute every empirical manuscript count from immutable pilot receipts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neural_shrimp.cli import verify, sha256


def derive():
    rows, receipts = {}, {}
    for name in ("EXP-001", "EXP-002", "EXP-003"):
        directory = ROOT / "evidence" / name / "attempt-001"
        rows[name] = verify(directory)
        receipts[name] = {"path": str((directory / "receipt.json").relative_to(ROOT)), "sha256": sha256(directory / "receipt.json")}
    control, training, utility = (rows[name] for name in ("EXP-001", "EXP-002", "EXP-003"))
    def cycles(group):
        return sum(bool(r.get("cycle_certificate", {}).get("verified")) and bool(r.get("cycle_certificate", {}).get("attracting")) and (r.get("diagnostic", {}).get("period") or 0) > 1 for r in group)
    groups = {p: [r for r in utility if r["policy"] == p] for p in ("fixed", "scheduled_decay", "stagnation_decay", "recurrence_decay")}
    sizes = {len(v) for v in groups.values()}
    failures = {sum(r["status"] == "divergent" for r in v) for v in groups.values()}
    if len(sizes) != 1 or len(failures) != 1:
        raise ValueError("Uniform policy-count/divergence sentence no longer applies")
    identity = lambda r: (r["seed"],r["lr_hidden"],r["lr_output"])
    fixed = {identity(r): r for r in groups["fixed"]}
    recurrence = {identity(r): r for r in groups["recurrence_decay"]}
    if fixed.keys() != recurrence.keys():
        raise ValueError("Unpaired policies")
    directory = ROOT / "evidence" / "EXP-003" / "attempt-001" / "trials"
    for key, a in fixed.items():
        b = recurrence[key]
        if b["events"]:
            raise ValueError("No-trigger interpretation no longer applies")
        for field in ("status", "gradient_updates", "terminal_train_mse", "terminal_validation_mse", "terminal_test_mse"):
            if a[field] != b[field]:
                raise ValueError(f"Paired scalar mismatch: {key}, {field}")
        with np.load(directory / (a["trial_id"] + ".npz"), allow_pickle=False) as raw_a, np.load(directory / (b["trial_id"] + ".npz"), allow_pickle=False) as raw_b:
            for field in ("states", "losses", "learning_rates"):
                if not np.array_equal(raw_a[field],raw_b[field],equal_nan=True):
                    raise ValueError(f"Paired full trajectory mismatch: {key}, {field}")
    counts = Counter(r["status"] for r in training)
    macros = {"HenonTrials":len(control), "HenonCycles":cycles(control), "TrainingTrials":len(training),
              "TrainingCycles":cycles(training), "TrainingUnresolved":counts["unresolved"], "TrainingDivergent":counts["divergent"],
              "UtilityTrials":len(utility), "PolicyTrials":next(iter(sizes)), "PolicyDivergent":next(iter(failures)),
              "RecurrenceTriggers":sum(bool(r["events"]) for r in groups["recurrence_decay"])}
    payload = {"status":"exploratory_pilot", "counts":macros, "paired_fixed_recurrence_full_trajectories_identical":True,
               "source_commit":json.loads((ROOT/"evidence/EXP-001/attempt-001/started.json").read_text())["source_commit"],
               "receipts":receipts, "paid_compute_usd":0, "paid_review_usd":0,
               "billing_basis":"All three pilots ran on the local CPU; no paid resources or review requests were made."}
    tex = "% Generated from immutable receipts by tools/summarize_pilots.py; do not hand-edit.\n"
    tex += "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name,value in macros.items())
    return payload, tex


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Generate derived artifacts; never changes raw trials")
    args=parser.parse_args()
    payload,tex=derive()
    outputs={ROOT/"evidence/pilot-summary.json":json.dumps(payload,sort_keys=True,indent=2)+"\n", ROOT/"paper/generated-pilot.tex":tex}
    for path,content in outputs.items():
        if args.write:
            path.write_text(content)
        elif not path.exists() or path.read_text()!=content:
            raise ValueError(f"Derived manuscript/provenance drift: {path}")
    proof=ROOT/"paper/provenance.json"
    if proof.exists() and not args.write:
        for item in json.loads(proof.read_text())["files"]:
            if sha256(ROOT/item["path"]) != item["sha256"]:
                raise ValueError(f"Manuscript/PDF provenance drift: {item['path']}")
    print(json.dumps({"verified":True,"counts":payload["counts"],"paired_trajectories_equal":True}))


if __name__=="__main__":
    main()
