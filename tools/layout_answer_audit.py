"""Audit of the batch-layout answer tests (19 October 2026, after the human questioned their size; consult 22).

Recomputes from the raw records, with no model runs:

- arithmetic (run REPRO-T, tools/repro_task.py): per model, precision, mode and temperature, the answers that changed
  between batch 1 and the batch of four, split into a number against no number (INVALID) and two different numbers,
  with the accuracy of each layout;
- reasoning models (run PM2, tools/pi_reason.py, GSM8K cells): the changed answers split by whether either layout hit
  the thinking budget (forced switch) or the answer budget, for the soft and the discrete sampled arms;
- Soft Thinking as published (run PM, tools/pi_harness.py), with --batches: the batches rebuilt from the tokenizer
  (make_batches), the problems whose batch was identical in both layouts, and the rate among the others.

The reset test's corrected comparison (parsed integers instead of texts) is in tools/repro_reset.py analyze
(docs/rr/*-analysis-int.json).

Usage: python tools/layout_answer_audit.py [--batches] --json docs/pi/layout-answer-audit.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_llm_soft_loop import DATA  # noqa: E402

POD = DATA / "explore" / "llm-soft-loop" / "pod64"
ARITHMETIC = {"qwen0.5b": POD / "repro" / "t05", "qwen7b": POD / "repro" / "t7"}
PM2_GSM8K = {"r1qwen7b": DATA / "explore" / "pi" / "pm2-A" / "explore" / "llm-soft-loop" / "pod64" / "pi" / "pm2",
             "r1llama8b": POD / "pi" / "pm2fix", "qwq32b": POD / "pi" / "pm2"}
PM = POD / "pi" / "pm" / "top15-s64"


def arithmetic():
    out = {}
    for model, d in ARITHMETIC.items():
        recs = [json.loads(line) for line in open(d / "runs.jsonl")]
        recs = [r for r in recs if "mode" in r and r["set"] == "heldout"]
        for key in sorted({(r["dtype"], r["mode"], r["log10_temperature"]) for r in recs}, key=str):
            rs = [r for r in recs if (r["dtype"], r["mode"], r["log10_temperature"]) == key]
            pairs = [(r["single"]["parsed"], r["batch4"]["parsed"]) for r in rs]
            changed = [(a, b) for a, b in pairs if a != b]
            none = sum((a is None) != (b is None) for a, b in changed)
            out[f"{model} {key[0]} {key[1]} log10T={key[2]}"] = {
                "problems": len(rs), "changed": len(changed), "number_vs_none": none,
                "two_different_numbers": len(changed) - none,
                "correct_batch1": sum(r["single"]["parsed"] == r["problem"]["answer"] for r in rs),
                "correct_batch4": sum(r["batch4"]["parsed"] == r["problem"]["answer"] for r in rs)}
    return out


def capped(rec, args):
    return bool(rec["forced_switch"] or rec["switch_step"] >= args["think_budget"]
                or rec["answer_tokens"] >= args["answer_budget"] - 1)


def reasoning():
    out = {}
    for model, root in PM2_GSM8K.items():
        for arm in ("soft", "plain"):
            d = root / f"{model}-gsm8k-{arm}"
            run = json.loads((d / "run.json").read_text())
            args = run.get("args", run)
            by = {}
            for line in open(d / "records.jsonl"):
                r = json.loads(line)
                by.setdefault(r["index"], {})[r["layout"]] = r
            c = Counter()
            for lay in by.values():
                a, b = lay[8], lay[16]
                flip = a["answer"] != b["answer"]
                k = "budget_hit" if capped(a, args) or capped(b, args) else "finished_in_both"
                c[k] += 1
                c[k + "_changed"] += flip
            out[f"{model} {arm}"] = {"problems": len(by), "changed": c["budget_hit_changed"] + c["finished_in_both_changed"],
                                    "budget_hit_in_either_layout": c["budget_hit"],
                                    "changed_with_budget_hit": c["budget_hit_changed"],
                                    "finished_in_both_layouts": c["finished_in_both"],
                                    "changed_finished_in_both": c["finished_in_both_changed"],
                                    "think_budget": args["think_budget"], "answer_budget": args["answer_budget"],
                                    "source": str(d)}
    return out


def soft_thinking_batches():
    import pi_harness as ph
    import explore_llm_attractors as ela
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", cache_dir=str(ela.CACHE), local_files_only=True)
    test, train = ph.load_gsm8k("train"), ph.load_gsm8k("test")       # as pi_harness main() for --split train
    lay, sizes = {}, {}
    for size in (8, 16):
        bs = ph.make_batches(test, train, tok, size, 256)
        lay[size] = {r["index"]: (bi, row, tuple(x["index"] for x in b))
                     for bi, b in enumerate(bs) for row, r in enumerate(b) if r["scored"]}
        sizes[size] = {"batches": len(bs), "full": sum(len(b) == size for b in bs)}
    recs = {}
    for line in open(PM / "records.jsonl"):
        r = json.loads(line)
        recs.setdefault(r["index"], {})[r["layout"]] = r
    mismatch = sum((d[L]["batch"], d[L]["row"]) != lay[L][i][:2] for i, d in recs.items() for L in (8, 16))
    same = {i for i in recs if lay[8][i][2] == lay[16][i][2]}
    flips = {i for i, d in recs.items() if d[8]["answer"] != d[16]["answer"]}
    return {"layouts": sizes, "record_batch_row_mismatches": mismatch, "problems": len(recs),
            "identical_batch_in_both_layouts": len(same), "changed": len(flips),
            "changed_with_identical_batch": len(flips & same), "changed_among_others": len(flips - same),
            "others": len(recs) - len(same), "same_row_in_both": sum(lay[8][i][1] == lay[16][i][1] for i in recs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batches", action="store_true", help="also rebuild the Soft Thinking batches (needs the tokenizer)")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    out = {"arithmetic": arithmetic(), "reasoning_gsm8k": reasoning()}
    if args.batches:
        out["soft_thinking_published"] = soft_thinking_batches()
    Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
