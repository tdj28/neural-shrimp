"""Problem selection for Program PI runs E2 and E3: fresh GSM8K train problems in full same-length batches.

The harness batches only prompts of identical token length, so a scattered problem set spends most of its compute on
unscored filler rows. Rule (fixed before any E2 or E3 outcome): among GSM8K train problems with index >= 2000 (never
scored by PM 0-255, PM2 256-511, Huginn 0-119, or the test-split runs E1, E4, H), group by chat-templated prompt
length (Qwen2.5-7B-Instruct tokenizer, the harness's prompt). Lengths with at least 16 problems are eligible. Take G
lengths evenly spaced in rank over the sorted eligible lengths and, from each, its first 16 problems in dataset order.
Each length then fills exactly one batch of 16 and two batches of 8, with no fillers.

Usage: python tools/pi_e2e3_select.py --groups G --out docs/pi/e2e3-problems.json
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi_harness as ph  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", type=int, required=True)
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=None, help="last index + 1 (development: E1's first 400 test problems)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-group", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ph.snapshot(ph.MODELS["qwen7b"]))
    by = defaultdict(list)
    for p in ph.load_gsm8k(args.split)[args.start:args.end]:
        by[len(ph.prompt_ids(tok, p["question"]))].append(p["index"])
    elig = sorted(L for L, v in by.items() if len(v) >= args.per_group)
    if args.groups > len(elig):
        raise SystemExit(f"only {len(elig)} eligible lengths")
    ranks = sorted({int(round(x)) for x in np.linspace(0, len(elig) - 1, args.groups)})
    if len(ranks) != args.groups:
        raise SystemExit("rank collision")
    groups = [{"length": elig[r], "indices": by[elig[r]][:args.per_group]} for r in ranks]
    idx = sorted(i for g in groups for i in g["indices"])
    out = {"rule": __doc__.split("\n\n")[1], "split": args.split, "start": args.start, "end": args.end, "groups": groups,
           "eligible_lengths": len(elig), "indices": idx,
           "sha256_indices": hashlib.sha256(json.dumps(idx).encode()).hexdigest()}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(len(idx), "problems in", len(groups), "lengths", [g["length"] for g in groups])


if __name__ == "__main__":
    main()
