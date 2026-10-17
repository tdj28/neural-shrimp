"""Diagnostic for PM2's low R1-Distill-Llama-8B accuracy: plain Hugging Face generate() on the PM2 prompts.

The same problems, prompt (tools/pi_reason.py prompt_ids, with the checked tokenizer unless --tokenizer legacy) and
sampling (temperature, top-p, no top-k) as the plain arm of tools/pi_reason.py, but through model.generate() with
input ids, the model's default attention, and one budget of --max-new-tokens for thinking and answer together (no
inserted end of thinking). Rows of a batch have identical prompt lengths, so nothing is padded.

Problems: GSM8K training problems after --offset in full groups of --group identical prompt lengths (as
pi_reason.gsm8k_batches, so --group 8 --limit 64 gives the first 64 problems of run PM2F).
The answer is extracted (pi_reason.extract) from the text after the last end-of-thinking token, else from all of it.
Writes OUT/records.jsonl and OUT/run.json.
Usage: python tools/pi_generate_diag.py --model r1llama8b --dtype bfloat16 --limit 64 --out RUN
"""
import argparse
import json
import time
from pathlib import Path

import pi_harness as ph
import pi_reason as pr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(pr.MODELS))
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--offset", type=int, default=256)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1536 + 256)
    ap.add_argument("--tokenizer", default="checked", choices=("checked", "legacy"))
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM
    path = ph.snapshot(pr.MODELS[args.model])
    tok = pr.load_tokenizer(path, args.tokenizer)
    end_think = tok.convert_tokens_to_ids("</think>")
    eos = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, args.dtype), device_map=args.device)
    model.eval()
    pool = ph.load_gsm8k("train")[args.offset:]
    groups = pr.gsm8k_batches(pool, tok, [args.group], args.limit)[args.group]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({
        "args": vars(args), "model_path": str(path), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "tokenizer_class": type(tok).__name__, "tokenizer_roundtrip": pr.roundtrip_ok(tok),
        "attn_implementation": model.config._attn_implementation, "groups": len(groups)}, indent=1))
    done = set()
    if (out / "records.jsonl").exists():
        done = {json.loads(line)["index"] for line in open(out / "records.jsonl")}
    rec_f = open(out / "records.jsonl", "a")
    t0 = time.time()
    for gi, group in enumerate(groups):
        if all(p["index"] in done for p in group):
            continue
        torch.manual_seed(args.seed + gi)
        ids = torch.tensor([p["ids"] for p in group], device=args.device)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                 temperature=args.temperature, top_p=args.top_p, top_k=0,
                                 max_new_tokens=args.max_new_tokens, eos_token_id=eos, pad_token_id=eos)
        for r, p in enumerate(group):
            new = gen[r, ids.shape[1]:].tolist()
            n_new = new.index(eos) if eos in new else len(new)
            new = new[:n_new]
            k = max((i for i, t in enumerate(new) if t == end_think), default=None)
            ans_ids = new[k + 1:] if k is not None else new
            text = tok.decode(ans_ids, skip_special_tokens=True)
            ans = pr.extract(text)
            rec_f.write(json.dumps({
                "index": p["index"], "gold": p["gold"], "answer": ans, "correct": ans is not None and ans == p["gold"],
                "new_tokens": n_new, "hit_budget": n_new >= args.max_new_tokens, "end_think_step": k,
                "answer_tokens": len(ans_ids), "answer_ids": ans_ids, "text": text[-600:]}) + "\n")
        rec_f.flush()
        print(json.dumps({"group": gi, "of": len(groups), "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
