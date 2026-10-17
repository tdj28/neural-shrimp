"""C12-4c independent check (docs/pi-c12-4-confirmatory.md): recompute the confirmatory endpoints from the raw records
without importing tools/c12_steering_history.py, and compare them with the frozen `confirm` output.

Re-implemented from the raw records.jsonl: decision parsing is taken as recorded (field "decision"), but the instance
grouping, class selection, pairing, means and the bootstrap are written independently (own resampling seed). Point
estimates must agree with the confirm output to 1e-9; bootstrap intervals use a different random stream and must agree
within 0.02.

Usage: python tools/c12_4c_check.py --run-a DIR --run-b DIR --confirm JSON [--json OUT]
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np

SEED = 20261017
N_BOOT = 10000


def rows_of(run):
    """Rows of completed batches: per batch with a done marker, its last `rows` row records (a resumed batch was
    recomputed in full after its partial rows)."""
    rows, size, done = defaultdict(list), {}, set()
    with open(f"{run}/records.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "row":
                rows[r["batch"]].append(r)
            elif r.get("kind") == "timing":
                size[r["batch"]] = r["rows"]
            elif r.get("kind") == "done":
                done.add(r["batch"])
    out = []
    for b in sorted(done):
        out.extend(rows[b][-size[b]:])
    seen = {(r["uid"], r["start"], r["history"]) for r in out}
    if len(seen) != len(out):
        raise SystemExit(f"{run}: duplicate (instance, start, history) rows after removing partial batches")
    return out


def per_instance(rows, history, S, cls, want):
    acc = defaultdict(list)
    for r in rows:
        if r["history"] == history and r["cls"] == cls:
            acc[r["uid"]].append(1.0 if r["readouts"][str(S)]["decision"] == want else 0.0)
    return {u: sum(v) / len(v) for u, v in acc.items()}


def contrast(rows_a, ha, rows_b, hb, S, cls, want, rng):
    a = per_instance(rows_a, ha, S, cls, want)
    b = per_instance(rows_b, hb, S, cls, want)
    if set(a) != set(b):
        raise SystemExit(f"instance sets differ for {ha} vs {hb}")
    keys = sorted(a)
    if not keys:
        raise SystemExit(f"no instances for {ha} vs {hb} ({cls}, S {S}): a missing precondition, not an estimate")
    d = np.array([a[k] - b[k] for k in keys])
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    return {"estimate": float(d.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "instances": len(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--confirm", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    A, B = rows_of(args.run_a), rows_of(args.run_b)
    conf = json.load(open(args.confirm))
    S32, S64 = conf["S_primary"], conf["S_secondary"]
    rng = np.random.default_rng(SEED)
    mine = {
        "primary": contrast(A, "up", A, "direct", S32, "prohibited", "ALLOW", rng),
        "primary_dwell64": contrast(A, "up", A, "direct", S64, "prohibited", "ALLOW", rng),
        "v2": contrast(B, "up", B, "direct", S32, "prohibited", "ALLOW", rng),
        "down_fd": contrast(A, "down", A, "direct", S32, "permitted", "DENY", rng),
        "random_fa": contrast(A, "random", A, "direct", S32, "prohibited", "ALLOW", rng),
        "zero_fa": contrast(A, "zero", A, "direct", S32, "prohibited", "ALLOW", rng),
        "zero_fd": contrast(A, "zero", A, "direct", S32, "permitted", "DENY", rng),
        "check_v": contrast(A, "direct", A, "unsteered", S32, "prohibited", "ALLOW", rng),
        "check_v2": contrast(B, "direct", A, "unsteered", S32, "prohibited", "ALLOW", rng),
    }
    sv = conf["secondary_v"]
    theirs = {
        "primary": (conf["primary"]["estimate"], conf["primary"]["boot95_instances"]),
        "primary_dwell64": (conf["primary"]["estimate_dwell64"], conf["primary"]["boot95_dwell64"]),
        "v2": (conf["v2_primary"]["estimate"], conf["v2_primary"]["boot95_instances"]),
        "down_fd": (sv["down_false_deny_permitted"]["estimate_dwell32"], sv["down_false_deny_permitted"]["boot95_dwell32"]),
        "random_fa": (sv["random_false_allow_prohibited"]["estimate_dwell32"],
                      sv["random_false_allow_prohibited"]["boot95_dwell32"]),
        "zero_fa": (sv["zero_false_allow_prohibited"]["estimate_dwell32"], sv["zero_false_allow_prohibited"]["boot95_dwell32"]),
        "zero_fd": (sv["zero_false_deny_permitted"]["estimate_dwell32"], sv["zero_false_deny_permitted"]["boot95_dwell32"]),
        "check_v": (conf["manipulation_check_v_direct_minus_unsteered"]["estimate"],
                    conf["manipulation_check_v_direct_minus_unsteered"]["boot95_instances"]),
        "check_v2": (conf["manipulation_check_v2_direct_minus_unsteered"]["estimate"],
                     conf["manipulation_check_v2_direct_minus_unsteered"]["boot95_instances"]),
    }
    report, ok = {}, True
    for k, m in mine.items():
        est, ci = theirs[k]
        e_ok = abs(m["estimate"] - est) <= 1e-9
        c_ok = abs(m["ci"][0] - ci[0]) <= 0.02 and abs(m["ci"][1] - ci[1]) <= 0.02
        ok = ok and e_ok and c_ok
        report[k] = {"independent": m, "confirm": {"estimate": est, "ci": ci}, "estimate_agrees": e_ok,
                     "interval_agrees_0.02": c_ok}
    report["all_agree"] = ok
    txt = json.dumps(report, indent=1)
    if args.json:
        open(args.json, "w").write(txt + "\n")
    print(txt)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
