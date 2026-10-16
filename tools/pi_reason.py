"""Run PM2: Soft Thinking with reasoning models, as published, under two batch layouts (bfloat16 on the GPU).

Every problem runs in two batch layouts that compute the same rows with different matrix shapes. A row thinks until
the model's most probable next token is the end-of-thinking token (or a thinking budget is spent, when that token is
inserted), then answers with discrete decoding until the end-of-sequence token or an answer budget. Rows stay in
lockstep, so nothing is padded and a row's inputs are identical in both layouts.

Arms:
  soft   thinking with concept tokens: the probability-weighted embedding of the top-n tokens of softmax(z / T),
         renormalized (Soft Thinking); the row switches to answering when the argmax is the end-of-thinking token;
  plain  thinking with discrete tokens sampled from the top-p nucleus of softmax(z / T).
The answer phase is the same in both arms: discrete tokens sampled from the top-p nucleus at the same T, with common
random numbers (one uniform per problem and step from numpy default_rng(seed + problem index), the same in both
layouts and in every copy of a problem).

Tasks:
  gsm8k    GSM8K problems from a split after an offset, taken in dataset order into full batches of identical prompt
           length (the larger layout's batches are complete groups; the smaller layout splits each in parts), so
           every row of every batch is scored;
  math500  MATH-500 problems whose reference answer is an integer, in dataset order; each batch holds copies of one
           problem (layout 1: alone; layout k: k copies), and the first copy is scored.
Prompt: the user message is the problem followed by "Please reason step by step, and put your final answer within
\\boxed{}." under the model's chat template. The answer is the content of the last \\boxed{} of the answer phase, else
its last number; integers are compared as integers.

Writes OUT/records.jsonl (one line per scored problem and layout) and OUT/run.json. Analysis: tools/pi_matrix_analyze.py.
Usage: python tools/pi_reason.py --model r1qwen7b --task gsm8k --arm soft --layouts 8 16 --limit 256 --out RUN
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import pi_harness as ph

MODELS = {"r1qwen7b": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B",
          "r1llama8b": "models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B",
          "qwq32b": "models--Qwen--QwQ-32B",
          "r1qwen1.5b": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"}
MATH500 = "datasets--HuggingFaceH4--MATH-500"
SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
INT = re.compile(r"^-?\d+$")


def prompt_ids(tok, question):
    msgs = [{"role": "user", "content": f"{question}\n{SUFFIX}"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=True)
    return list(enc["input_ids"])


def last_boxed(text):
    k = text.rfind("\\boxed{")
    if k < 0:
        return None
    i, depth, out = k + len("\\boxed{"), 1, []
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(c)
        i += 1
    return None


def extract(text):
    """The last \\boxed{} content (normalized to an integer string when it is one), else the last number."""
    b = last_boxed(text)
    if b is not None:
        s = re.sub(r"\\text\{[^}]*\}|\\!|\\,|\$|\s|,", "", b)
        s = s.replace("\\%", "").replace("%", "")
        m = re.fullmatch(r"-?\d+(\.0+)?", s)
        return str(int(float(s))) if m else s
    nums = ph.NUM.findall(text)
    if not nums:
        return None
    s = nums[-1].replace(",", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


def load_math500():
    out = []
    for i, line in enumerate(open(ph.snapshot(MATH500) / "test.jsonl")):
        r = json.loads(line)
        a = r["answer"].strip()
        if INT.fullmatch(a):
            out.append({"index": i, "question": r["problem"], "gold": str(int(a))})
    return out


def gsm8k_batches(pool, tok, layouts, limit):
    """Full batches: problems in dataset order, grouped by prompt length; a group is emitted when it has
    max(layouts) members. Returns {layout: [batch, ...]}, the smaller layout splitting each full group."""
    big = max(layouts)
    groups, full = defaultdict(list), []
    for p in pool:
        ids = prompt_ids(tok, p["question"])
        groups[len(ids)].append({**p, "ids": ids, "scored": True})
        if len(groups[len(ids)]) == big:
            full.append(groups.pop(len(ids)))
            if limit and len(full) * big >= limit:
                break
    out = {}
    for L in layouts:
        if big % L:
            raise SystemExit(f"layout {L} does not divide {big}")
        out[L] = [g[s:s + L] for g in full for s in range(0, big, L)]
    return out


def math_batches(problems, tok, layouts, limit):
    sel = problems[:limit] if limit else problems
    out = {}
    for L in layouts:
        out[L] = []
        for p in sel:
            ids = prompt_ids(tok, p["question"])
            row = {**p, "ids": ids, "scored": True}
            copies = [{**row, "scored": False} for _ in range(L - 1)]
            out[L].append([row] + copies)
    return out


def sample_nucleus(torch, p, top_p, u):
    ps, order = torch.sort(p, dim=-1, descending=True)
    cs = ps.cumsum(-1)
    keep = (cs - ps) < top_p
    ps = ps * keep
    ps = ps / ps.sum(-1, keepdim=True)
    cs = ps.cumsum(-1)
    pos = torch.searchsorted(cs, u[:, None].to(cs.dtype)).clamp_max(cs.shape[-1] - 1)
    return order.gather(-1, pos)[:, 0]


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = ph.snapshot(MODELS[args.model])
    tok = AutoTokenizer.from_pretrained(path)
    end_think = tok.convert_tokens_to_ids("</think>")
    if end_think is None or end_think == tok.unk_token_id:
        raise SystemExit("the tokenizer has no </think> token")
    eos = tok.eos_token_id
    dev = args.device
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, args.dtype), device_map=dev)
    model.eval()
    E = model.get_input_embeddings().weight
    if args.task == "gsm8k":
        pool = ph.load_gsm8k(args.split)[args.offset:]
        batches = gsm8k_batches(pool, tok, args.layouts, args.limit)
    else:
        batches = math_batches(load_math500(), tok, args.layouts, args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({
        "args": vars(args), "model_path": str(path), "end_think_id": int(end_think), "eos_id": int(eos),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if dev == "cuda" else "cpu",
        "batches": {str(L): len(b) for L, b in batches.items()}}, indent=1))
    done = set()
    if (out / "records.jsonl").exists():
        for line in open(out / "records.jsonl"):
            r = json.loads(line)
            done.add((r["index"], r["layout"]))
    rec_f = open(out / "records.jsonl", "a")
    T = args.temperature
    for layout in args.layouts:
        t0 = time.time()
        for bi, batch in enumerate(batches[layout]):
            if all((r["index"], layout) in done for r in batch if r["scored"]):
                continue
            ids = torch.tensor([r["ids"] for r in batch], device=dev)
            B = ids.shape[0]
            rngs = [np.random.default_rng(args.seed + r["index"]) for r in batch]
            mode = [0] * B                   # 0 thinking, 1 answering, 2 finished
            switch = [None] * B
            forced = [False] * B
            answer = [[] for _ in range(B)]
            top1 = [[] for _ in range(B)]
            with torch.no_grad():
                o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
                past, z = o.past_key_values, o.logits[:, -1, :].float()
                del o
                for t in range(args.think_budget + args.answer_budget):
                    if all(m == 2 for m in mode):
                        break
                    p = torch.softmax(z / T, dim=-1)
                    u = torch.tensor([rg.random() for rg in rngs], device=dev)
                    am = z.argmax(-1)
                    sampled = sample_nucleus(torch, p, args.top_p, u)
                    tokens = torch.full((B,), eos, device=dev, dtype=torch.long)
                    use_soft = torch.zeros(B, dtype=torch.bool, device=dev)
                    for r in range(B):
                        if mode[r] == 0:
                            top1[r].append(int(am[r]))
                            if t >= args.think_budget:          # budget spent: insert the end of thinking
                                tokens[r], mode[r], switch[r], forced[r] = end_think, 1, t, True
                            elif args.arm == "soft":
                                if int(am[r]) == end_think:
                                    tokens[r], mode[r], switch[r] = end_think, 1, t
                                else:
                                    use_soft[r] = True
                            else:
                                tokens[r] = sampled[r]
                                if int(sampled[r]) == end_think:
                                    mode[r], switch[r] = 1, t
                        elif mode[r] == 1:
                            tokens[r] = sampled[r]
                            answer[r].append(int(sampled[r]))
                            if int(sampled[r]) == eos or len(answer[r]) >= args.answer_budget:
                                mode[r] = 2
                    e = E[tokens].clone()
                    if bool(use_soft.any()):
                        q = p
                        if args.top_n:
                            w, idx = torch.topk(p, args.top_n, dim=-1)
                            q = torch.zeros_like(p).scatter_(-1, idx, w)
                        q = q / q.sum(-1, keepdim=True)
                        c = ph.concept_token(torch, q[use_soft], E, args.top_n).to(E.dtype)
                        e[use_soft] = c
                    o = model(inputs_embeds=e[:, None, :], past_key_values=past, use_cache=True)
                    past, z = o.past_key_values, o.logits[:, -1, :].float()
            for r, row in enumerate(batch):
                if not row["scored"]:
                    continue
                toks = answer[r][:answer[r].index(eos)] if eos in answer[r] else answer[r]
                text = tok.decode(toks, skip_special_tokens=True)
                ans = extract(text)
                rec = {"index": row["index"], "layout": layout, "batch": bi, "row": r, "prompt_len": len(row["ids"]),
                       "answer": ans, "gold": row["gold"], "correct": ans is not None and ans == row["gold"],
                       "switch_step": switch[r], "forced_switch": forced[r], "answer_tokens": len(toks),
                       "text": text[-600:], "top1": top1[r]}
                rec_f.write(json.dumps(rec) + "\n")
            rec_f.flush()
            print(json.dumps({"layout": layout, "batch": bi, "of": len(batches[layout]),
                              "seconds": round(time.time() - t0, 1)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--task", required=True, choices=("gsm8k", "math500"))
    ap.add_argument("--arm", required=True, choices=("soft", "plain"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--offset", type=int, default=256, help="gsm8k: skip the first N problems of the split")
    ap.add_argument("--limit", type=int, default=256, help="scored problems (gsm8k: rounded to full batches)")
    ap.add_argument("--layouts", type=int, nargs=2, default=[8, 16])
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--think-budget", type=int, default=1536)
    ap.add_argument("--answer-budget", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
