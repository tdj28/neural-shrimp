"""Huginn-0125 with a relaxed recurrence: s <- (1 - lam) s + lam Core(s, e), during generation.

The relaxation keeps the recurrence's fixed points and changes its multipliers from mu to 1 - lam + lam mu. From the
spectrum of the native fixed point (critical pair 0.588 +- 0.771i, modulus 0.970, for a GSM8K-style prompt; research
log 12 October), lam* = argmin max |1 - lam + lam mu| contracts faster than the native lam = 1, and lam near the
Neimark-Sacker value 1.0777 contracts more slowly. At a fixed number of recurrence steps per token, the latent state
then ends closer to (lam*) or farther from (lam near 1.08) the fixed point. This tool measures

- convergence: the per-step relative change ||s_k - s_(k-1)|| / ||s_k|| of the prefill recurrence for given lam values,
  to test the spectral prediction of the contraction rate directly;
- accuracy: greedy GSM8K answers at a fixed recurrence budget for given lam values (exact match of the final number).

The recurrence is patched by wrapping the model's core_block_forward; prelude, coda, caches and decoding are the
model's own. Runs on one GPU in bfloat16 (the model's native dtype) or float32.

Usage:
  python tools/huginn_relax_eval.py accuracy --split train --start 0 --count 40 --steps 8 --lams 1.0 0.55 \
      --out huginn-relax/dev-r8
  python tools/huginn_relax_eval.py convergence --split train --index 0 --steps 64 --lams 1.0 0.55 1.05 \
      --out huginn-relax/conv-0
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

MODEL = "tomg-group-umd/huginn-0125"
GSM8K = "explore/gsm8k"                      # data-relative folder with train.jsonl and test.jsonl (pinned export)
PROMPT = ("Solve the following math problem step by step. At the end, write the final answer as "
          "'The answer is N.'\n\nQuestion: {question}\nAnswer:")


def gold(answer):
    """GSM8K gold answer: the number after '####'."""
    return answer.split("####")[-1].strip().replace(",", "")


def extract(text):
    """The number after the last 'answer is', else the last number in the text; None if there is none."""
    t = text.replace(",", "")
    m = re.findall(r"answer is\s*\$?\s*(-?\d+(?:\.\d+)?)", t, flags=re.I)
    if m:
        return m[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    return nums[-1] if nums else None


def same_number(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


class Relaxed:
    """Wraps model.core_block_forward so that each recurrence step returns (1 - lam) x + lam Core(x).

    With `trace` set, records per step the native residual ||Core(x) - x|| / ||x|| (the fixed-point residual, with no
    factor lam) and the relaxed step ||new - x|| / ||x|| = lam ||Core(x) - x|| / ||x|| separately: only the former
    measures closeness to a fixed point (Codex preflight, codex/huginn-design-preflight.md)."""

    def __init__(self, model):
        self.model = model
        self.orig = model.core_block_forward
        self.lam = 1.0
        self.trace = None                       # optional list of (native residual, relaxed step) per recurrence step

        def relaxed(x, *args, **kwargs):
            out, block_idx = self.orig(x, *args, **kwargs)
            new = out if self.lam == 1.0 else (1.0 - self.lam) * x + self.lam * out
            if self.trace is not None:
                nx = float(x.float().norm())
                self.trace.append((float((out - x).float().norm()) / nx, float((new - x).float().norm()) / nx))
            return new, block_idx

        model.core_block_forward = relaxed


def load(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import explore_llm_attractors as ela

    torch.set_num_threads(int(args.threads))
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=ela.CACHE)
    model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=ela.CACHE, trust_remote_code=True,
                                                 torch_dtype=getattr(torch, args.dtype)).eval().to(args.device)
    model.requires_grad_(False)
    # transformers 4.55 makes DynamicCache.key_cache/value_cache read-only properties, while Huginn's remote
    # HuginnDynamicCache assigns them as dicts; plain class attributes on the subclass shadow the properties
    mod = sys.modules[type(model).__module__]
    for name in ("key_cache", "value_cache"):
        if isinstance(getattr(mod.HuginnDynamicCache.__mro__[1], name, None), property):
            setattr(mod.HuginnDynamicCache, name, None)
    return torch, tok, model, Relaxed(model)


def problems(split):
    import explore_llm_attractors as ela

    with open(ela.DATA / GSM8K / f"{split}.jsonl") as f:
        return [json.loads(line) for line in f]


def accuracy(args):
    torch, tok, model, relax = load(args)
    import explore_llm_attractors as ela
    from transformers import GenerationConfig

    import numpy as np

    data = problems(args.split)[args.start: args.start + args.count]
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = GenerationConfig(max_new_tokens=args.max_new, do_sample=False, use_cache=True, return_dict_in_generate=False,
                           pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    summary = {"args": vars(args), "arms": {}}
    rows_path = out / "rows.jsonl"
    with open(rows_path, "a") as rows:
        for arm in args.lams:
            correct, tic = 0, time.perf_counter()
            for i, ex in enumerate(data):
                index = args.start + i
                if str(arm).startswith("rand:"):          # random relaxation drawn per problem, seeded by (seed, index)
                    lo, hi = (float(v) for v in str(arm).split(":")[1:3])
                    lam = float(np.random.default_rng([args.seed, index]).uniform(lo, hi))
                else:
                    lam = float(arm)
                relax.lam = lam
                torch.manual_seed(args.seed * 100003 + index)   # Huginn draws its initial latent state at random:
                ids = tok(PROMPT.format(question=ex["question"]), return_tensors="pt")["input_ids"].to(args.device)
                t0 = time.perf_counter()                   # the same seed per problem in every arm (paired)
                with torch.no_grad():
                    gen = model.generate(ids, cfg, num_steps=args.steps)
                new_tokens = int(gen.shape[1] - ids.shape[1])
                full = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
                text = full.split("\nQuestion:")[0]
                pred, ans = extract(text), gold(ex["answer"])
                ok = same_number(pred, ans)
                correct += ok
                rows.write(json.dumps({"arm": str(arm), "lam": lam, "steps": args.steps, "split": args.split,
                                       "index": index, "seed": args.seed * 100003 + index, "pred": pred, "gold": ans,
                                       "correct": ok, "new_tokens": new_tokens, "core_calls": new_tokens * args.steps,
                                       "seconds": round(time.perf_counter() - t0, 2), "text": full}) + "\n")
                rows.flush()
            lam = str(arm)
            n = len(data)
            summary["arms"][str(lam)] = {"correct": correct, "n": n, "accuracy": correct / n,
                                         "seconds": round(time.perf_counter() - tic, 1)}
            print(json.dumps({"lam": lam, "steps": args.steps, "correct": correct, "n": n,
                              "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=1))


def convergence(args):
    """Prefill recurrence of one GSM8K prompt, relaxed at each lam, through explore_latent_loop.HuginnLoop (adapter
    and core on the GPU, the rest on the CPU, so float32 fits): native residual ||F(s) - s|| / ||s|| and relaxed step
    per step, and the rate of the native residual over a gated fitting window."""
    import math

    import numpy as np

    import explore_latent_loop as ell
    import explore_llm_attractors as ela

    loop = ell.HuginnLoop("huginn", args.dtype, args.device, args.threads, seed=args.seed)
    ex = problems(args.split)[args.index]
    h0 = loop.setup(PROMPT.format(question=ex["question"]))
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    res = {"args": vars(args), "traces": {}}
    for lam in args.lams:
        lam = float(lam)
        s, native, step = h0.clone(), [], []
        for _ in range(args.steps):
            f = loop.F(s)
            ns = float(s.float().norm())
            native.append(float((f - s).float().norm()) / ns)
            new = (1.0 - lam) * s + lam * f
            step.append(float((new - s).float().norm()) / ns)
            s = new
        k0 = int(args.fit_start * len(native))
        floor = min(native) * args.floor_factor
        window = [(k, r) for k, r in enumerate(native) if k >= k0 and r > floor]
        if len(window) >= 8:
            slope = float(np.polyfit([k for k, _ in window], np.log([r for _, r in window]), 1)[0])
            rate, status = math.exp(slope), "fitted"
        else:
            rate, status = None, f"fitting window too short ({len(window)} steps above the floor gate)"
        res["traces"][str(lam)] = {"native_residual": native, "relaxed_step": step,
                                   "fit_window": [window[0][0], window[-1][0]] if window else None,
                                   "rate_per_step": rate, "status": status}
        print(json.dumps({"lam": lam, "steps": args.steps, "native_residual_last": native[-1], "rate_per_step": rate,
                          "status": status}), flush=True)
    (out / "convergence.json").write_text(json.dumps(res, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("accuracy", "convergence"):
        sp = sub.add_parser(name)
        sp.add_argument("--split", default="train", choices=("train", "test"))
        sp.add_argument("--steps", type=int, default=8)
        sp.add_argument("--lams", nargs="+", default=["1.0"], help="values, or rand:LO:HI for a per-problem draw")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--dtype", default="bfloat16" if name == "accuracy" else "float32")
        sp.add_argument("--threads", type=int, default=4)
        sp.add_argument("--out", required=True)
    acc = sub.choices["accuracy"]
    acc.add_argument("--start", type=int, default=0)
    acc.add_argument("--count", type=int, default=40)
    acc.add_argument("--max-new", type=int, default=256)
    conv = sub.choices["convergence"]
    conv.add_argument("--index", type=int, default=0)
    conv.add_argument("--fit-start", type=float, default=0.3, help="fraction of steps left out of the rate fit")
    conv.add_argument("--floor-factor", type=float, default=30.0,
                      help="fit only while the native residual exceeds this factor times its minimum")
    args = parser.parse_args()
    {"accuracy": accuracy, "convergence": convergence}[args.command](args)


if __name__ == "__main__":
    main()
