"""Run BF follow-up checks on saved data (no model evaluation): do the recurrence candidates of the batched layouts
stay on their cycles after detection, how close to periodic are the rows never detected, and which tokens do the
candidate cycles write.

A batch keeps iterating a row after its detection (only a start alone stops there), and states are recorded every
--record-every steps. Two records of a row taken after its detection step lie a whole number of periods apart when
their spacing is a multiple of lcm(record_every, q) / record_every records; their relative distance tests that the
row stayed on the cycle until the horizon.

Usage: python tools/batch_fate_persist.py --run llm-soft-loop/batch-fate/bf1 --out docs/batch-fate/bf1-persistence.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_soft_loop as esl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = esl.DATA / "explore" / args.run
    rec = json.loads((d / "fate.json").read_text())
    z = np.load(d / "fate.npz", allow_pickle=True)
    every = rec["args"]["record_every"]
    from transformers import AutoTokenizer

    name = ela.MODELS[rec["args"]["model"]]
    tok = AutoTokenizer.from_pretrained(name, revision=ela.REVISIONS[name], cache_dir=ela.CACHE)
    out = {"run": args.run, "layouts": {}}
    for lab in rec["labels"]:
        cap, per, R, mind = z[f"{lab}_cap"], z[f"{lab}_per"], z[f"{lab}_rec"], z[f"{lab}_mind"]
        cyc_tok = json.loads(str(z[f"{lab}_cyc_tok"][0]))
        rows = []
        for r in np.nonzero(cap >= 0)[0]:
            q, c = int(per[r]), int(cap[r])
            s = (every * q // math.gcd(every, q)) // every              # records per whole number of periods
            idx = [i for i in range(len(R)) if i * every >= c and np.isfinite(R[i, r, 0])]
            pairs = [(i, i + s) for i in idx if i + s in idx]
            dist = [float(np.linalg.norm(R[i, r] - R[j, r]) / np.linalg.norm(R[i, r])) for i, j in pairs]
            toks = cyc_tok[str(r)]
            rows.append({"start": int(r), "detected_at": c, "period": q, "tokens": toks, "text": tok.decode(toks),
                         "post_detection_record_pairs": len(pairs),
                         "max_relative_distance_whole_periods_apart": max(dist) if dist else None})
        never = cap < 0
        m = mind[never]
        out["layouts"][lab] = {
            "candidates": rows,
            "never_detected": int(never.sum()),
            "never_detected_final_min_recurrence": {"below_1e-6": int(np.sum(m < 1e-6)),
                                                    "below_1e-3": int(np.sum(m < 1e-3)),
                                                    "median": float(np.median(m)) if m.size else None,
                                                    "min": float(np.min(m)) if m.size else None}}
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:5000])


if __name__ == "__main__":
    main()
