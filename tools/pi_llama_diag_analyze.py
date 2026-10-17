"""Analysis of the PM2X item 3 diagnosis (job set pm2diag): R1-Distill-Llama-8B accuracy by pipeline condition.

Per condition directory (records.jsonl of tools/pi_reason.py or tools/pi_generate_diag.py), on the 64 problems of
docs/pi/llama-diag-indices.json:
- accuracy, with a Wilson 95 % interval;
- missing answers;
- answers that hit the answer budget;
- thinking ended by the budget;
- median end of thinking.
Against the PM2F float64 base rows on the same problems: paired difference and exact McNemar p.
Decode markers: each record's answer_ids are decoded with the legacy and the checked tokenizer, and the answers
extracted from both are compared.

Usage: python tools/pi_llama_diag_analyze.py --diag DIR --f64 PM2F_R1LLAMA_DIR --json OUT
"""
import argparse
import json
from pathlib import Path

import numpy as np

import pi_harness as ph
import pi_matrix_analyze as pma
import pi_reason as pr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--f64", required=True)
    ap.add_argument("--indices", default=str(Path(__file__).resolve().parents[1] / "docs/pi/llama-diag-indices.json"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    keep = json.loads(Path(args.indices).read_text())
    f64 = {}
    for line in open(Path(args.f64) / "records.jsonl"):
        r = json.loads(line)
        if r["index"] in keep:
            f64[r["index"]] = r
    path = ph.snapshot(pr.MODELS["r1llama8b"])
    tok_legacy, tok_checked = pr.load_tokenizer(path, "legacy"), pr.load_tokenizer(path, "checked")
    out = {"problems": len(keep), "f64": {"n": len(f64), "accuracy": float(np.mean([r["correct"] for r in f64.values()])),
                                         "wilson95": pma.wilson(sum(r["correct"] for r in f64.values()), len(f64))},
           "conditions": {}}
    for d in sorted(Path(args.diag).iterdir()):
        rec = d / "records.jsonl"
        if not rec.exists():
            continue
        cfg = json.loads((d / "run.json").read_text())
        rows = {}
        for line in open(rec):
            r = json.loads(line)
            if r["index"] in keep:
                rows[r["index"]] = r
        n = len(rows)
        if n == 0:
            continue
        k = sum(r["correct"] for r in rows.values())
        common = sorted(set(rows) & set(f64))
        b = sum(1 for i in common if rows[i]["correct"] and not f64[i]["correct"])
        c = sum(1 for i in common if f64[i]["correct"] and not rows[i]["correct"])
        ab = cfg["args"].get("answer_budget")
        think_end = [r.get("switch_step", r.get("end_think_step")) for r in rows.values()]
        redecode = 0
        for r in rows.values():
            ids = r.get("answer_ids", [])
            a1 = pr.extract(tok_legacy.decode(ids, skip_special_tokens=True))
            a2 = pr.extract(tok_checked.decode(ids, skip_special_tokens=True))
            redecode += int(a1 != a2)
        out["conditions"][d.name] = {
            "n": n, "complete": n == len(keep), "accuracy": k / n, "wilson95": pma.wilson(k, n),
            "tokenizer": cfg.get("tokenizer_class"), "roundtrip": cfg.get("tokenizer_roundtrip"),
            "attn": cfg.get("attn_implementation"), "gpu": cfg.get("gpu"),
            "missing": sum(r["answer"] is None for r in rows.values()),
            "answer_at_budget": (sum(r["answer_tokens"] >= ab for r in rows.values()) if ab else
                                 sum(r.get("hit_budget", False) for r in rows.values())),
            "think_forced": sum(bool(r.get("forced_switch")) for r in rows.values()),
            "no_end_of_thinking": sum(t is None for t in think_end),
            "median_think_end": float(np.median([t for t in think_end if t is not None])) if any(
                t is not None for t in think_end) else None,
            "minus_f64": {"problems": len(common),
                          "difference": float(np.mean([rows[i]["correct"] - f64[i]["correct"] for i in common])),
                          "discordant": [b, c], "mcnemar_p": pma.mcnemar_exact(b, c)},
            "answers_changed_by_decoder": redecode}
    print(json.dumps(out, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
