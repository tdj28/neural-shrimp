"""Huginn-0125 with a relaxed recurrence: s <- (1 - lam) s + lam Core(s, e), during generation.

The relaxation keeps the recurrence's fixed points and changes its multipliers from mu to 1 - lam + lam mu. From the
spectrum of the native fixed point (critical pair 0.588 +- 0.771i, modulus 0.970, for a GSM8K-style prompt; research
log 12 October), lam* = argmin max |1 - lam + lam mu| contracts faster than the native lam = 1, and lam near the
Neimark-Sacker value 1.0777 contracts more slowly. At a fixed number of recurrence steps per token, the latent state
then ends closer to (lam*) or farther from (lam near 1.08) the fixed point. This tool measures

- convergence: the per-step relative change ||s_k - s_(k-1)|| / ||s_k|| of the prefill recurrence for given lam values,
  to test the spectral prediction of the contraction rate directly;
- accuracy: greedy GSM8K answers at a fixed recurrence budget for given lam values (exact match of the final number);
- arc: ARC-Challenge by log-likelihood of each answer choice (acc and acc_norm, as in lm-eval-harness), read out at
  several budgets from one prefill per arm: the scored system is then exactly the prefill recurrence whose fixed-point
  spectrum explore_latent_loop fixed-spectrum computes.

Arms: a relaxation value lam; rand:LO:HI (lam drawn per problem, seeded); aa:M:BETA (Anderson acceleration with memory
M and mixing BETA, a generic accelerator that adapts to the iterates, unlike a fixed relaxation).

The recurrence is patched by wrapping the model's core_block_forward; prelude, coda, caches and decoding are the
model's own. Runs on one GPU in bfloat16 (the model's native dtype) or float32.

Usage:
  python tools/huginn_relax_eval.py accuracy --split train --start 0 --count 40 --steps 8 --lams 1.0 0.55 \
      --out huginn-relax/dev-r8
  python tools/huginn_relax_eval.py convergence --split train --index 0 --steps 64 --lams 1.0 0.55 1.05 \
      --out huginn-relax/conv-0
  python tools/huginn_relax_eval.py arc --split train --start 0 --count 100 --steps 1 2 4 8 16 32 \
      --lams 1.0 0.55 aa:3:1.0 --out huginn-relax/arc-dev
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

MODEL = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"   # weights, remote code and tokenizer
GSM8K = "explore/gsm8k"                      # data-relative folder with train.jsonl and test.jsonl (pinned export)
# 8-shot chain-of-thought exemplars and format of lm-evaluation-harness gsm8k_cot (lm_eval/tasks/gsm8k/gsm8k-cot.yaml,
# main branch fetched 13 October 2026, SHA-256 7c9dd2561fe334a55f62bdc6a876f045e6dc7f53714c7a9db506b89c6de6bc96):
# 'Q: {question}\nA:' with the target after a space, exemplars joined by a blank line, generation until 'Q:'
COT_EXEMPLARS = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there "
     "will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been "
     "21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they "
     "had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason "
     "give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The "
     "answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The "
     "answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to "
     "thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were "
     "added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls "
     "did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had "
     "35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. "
     "23 - 15 is 8. The answer is 8."),
]
COT_PREFIX = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in COT_EXEMPLARS) + "\n\n"
ARC = "explore/arc-challenge"                # allenai/ai2_arc ARC-Challenge, pinned export (tools/export_benchmark.py)
ARC_PROMPT = "Question: {question}\nAnswer:"  # lm-eval-harness format; each choice is scored as " " + text
PROMPT = ("Solve the following math problem step by step. At the end, write the final answer as "
          "'The answer is N.'\n\nQuestion: {question}\nAnswer:")


def gsm8k_prompt(question, style):
    if style == "cot8":
        return COT_PREFIX + f"Q: {question}\nA:"
    return PROMPT.format(question=question)


def strict_answer(text):
    """lm-eval strict-match: the number after the first 'The answer is', else None."""
    m = re.search(r"The answer is (\-?[0-9\.\,]+)", text)
    return m.group(1).replace(",", "").rstrip(".") if m else None


def flexible_answer(text):
    """lm-eval flexible-extract: the last number-like token in the text."""
    found = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if not found:
        return None
    a, b = found[-1]
    return (a or b).replace(",", "").replace("$", "").rstrip(".")


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


def current_step(args, kwargs):
    """The recurrence step index that Huginn's iterate_forward passes as the last positional argument (None if absent)."""
    if len(args) >= 6:
        return int(args[5])
    return kwargs.get("current_step")


class Relaxed:
    """Wraps model.core_block_forward so that each recurrence step returns (1 - lam) x + lam Core(x), or, with aa > 0,
    an Anderson-accelerated step (memory aa, mixing lam; history reset at step 0, solved per sequence in float32).

    With `trace` set, records per step the native residual ||Core(x) - x|| / ||x|| (the fixed-point residual, with no
    factor lam) and the step taken ||new - x|| / ||x|| separately: only the former measures closeness to a fixed point
    (Codex preflight, codex/huginn-design-preflight.md). `mask` (B, T, 1) restricts norms and the Anderson least
    squares to real positions; `readout(step, state, args)` is called with the state after each step."""

    def __init__(self, model):
        self.model = model
        self.orig = model.core_block_forward
        self.lam = 1.0
        self.aa = 0
        self.trace = None
        self.mask = None
        self.readout = None
        self._hist = None

        def relaxed(x, *args, **kwargs):
            out, block_idx = self.orig(x, *args, **kwargs)
            step = current_step(args, kwargs)
            if self.aa:
                new = self.anderson(x, out, step)
            else:
                new = out if self.lam == 1.0 else (1.0 - self.lam) * x + self.lam * out
            if self.trace is not None:
                # entry k is recorded in the call that maps s_k to s_{k+1}: R(s_k) = ||Core(s_k) - s_k|| / ||s_k||
                # (aggregate over the masked batch, the step taken, and per batch row; differences in float32)
                m = 1.0 if self.mask is None else self.mask
                xf, d = x.float() * m, (out.float() - x.float()) * m
                nx = float(xf.norm())
                rows = (d.flatten(1).norm(dim=1) / xf.flatten(1).norm(dim=1)).tolist()
                self.trace.append((float(d.norm()) / nx, float(((new.float() - x.float()) * m).norm()) / nx, rows))
            if self.readout is not None:
                self.readout(None if step is None else step + 1, new, args)
            return new, block_idx

        model.core_block_forward = relaxed

    def anderson(self, x, out, step):
        """Type-II Anderson step: x+ = x + b g - (dX + b dG) gamma, gamma = argmin ||g - dG gamma|| per sequence."""
        import torch

        xf, gf = x.float(), (out - x).float()
        gm = gf if self.mask is None else gf * self.mask
        if step is None:
            raise ValueError("Anderson acceleration needs the recurrence step index")
        if step == 0 or self._hist is None:
            self._hist = []
        self._hist.append((xf, gf, gm))
        if len(self._hist) > self.aa + 1:
            self._hist.pop(0)
        beta = self.lam
        if len(self._hist) == 1:
            return (xf + beta * gf).to(x.dtype)
        b = xf.shape[0]
        h = self._hist
        dg = torch.stack([(h[i + 1][2] - h[i][2]).reshape(b, -1) for i in range(len(h) - 1)], dim=-1)
        scale = dg.norm(dim=1, keepdim=True).clamp_min(1e-30)          # unit columns: old, large differences
        dn = dg / scale                                                # must not swamp the small recent ones
        gram = dn.transpose(1, 2) @ dn
        rhs = dn.transpose(1, 2) @ gm.reshape(b, -1, 1)
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        ridge = 1e-10 if gram.dtype == torch.float64 else 1e-6               # relative to the unit diagonal
        gamma = torch.linalg.solve(gram + ridge * eye, rhs)[..., 0] / scale[:, 0, :]
        new = xf + beta * gf
        for i in range(len(h) - 1):
            coef = gamma[:, i].reshape(b, *([1] * (xf.dim() - 1)))
            new = new - coef * ((h[i + 1][0] - h[i][0]) + beta * (h[i + 1][1] - h[i][1]))
        if not torch.isfinite(new).all():
            new = xf + beta * gf
        return new.to(x.dtype)


def load(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import explore_llm_attractors as ela

    torch.set_num_threads(int(args.threads))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, cache_dir=ela.CACHE)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, cache_dir=ela.CACHE, trust_remote_code=True,
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
    from transformers import StoppingCriteria, StoppingCriteriaList

    cfg = GenerationConfig(max_new_tokens=args.max_new, do_sample=False, use_cache=True, return_dict_in_generate=False,
                           pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)

    class StopAtQuestion(StoppingCriteria):
        """Stop once the continuation contains 'Q:' (lm-eval's 'until'), batch size 1."""

        def __init__(self, start):
            self.start = start

        def __call__(self, input_ids, scores, **kwargs):
            tail = tok.decode(input_ids[0, self.start:][-8:], skip_special_tokens=True)
            done = "Q:" in tail or "\nQuestion:" in tail
            return torch.tensor([done], device=input_ids.device)
    summary = {"args": vars(args), "arms": {}}
    rows_path = out / "rows.jsonl"
    # arms outer (default) or problems outer (--interleave: every problem in all arms before the next, so that a run
    # cut short by its time limit still has paired arms)
    order = ([(arm, i) for i in range(len(data)) for arm in args.lams] if args.interleave
             else [(arm, i) for arm in args.lams for i in range(len(data))])
    tally = {str(arm): [0, 0, 0.0] for arm in args.lams}
    with open(rows_path, "a") as rows:
        for arm, i in order:
            ex = data[i]
            index = args.start + i
            lam = set_arm(relax, arm, args.seed, index)   # rand:LO:HI is drawn per problem, seeded by (seed, index)
            torch.manual_seed(args.seed * 100003 + index)   # Huginn draws its initial latent state at random:
            ids = tok(gsm8k_prompt(ex["question"], args.prompt_style), return_tensors="pt")["input_ids"].to(args.device)
            t0 = time.perf_counter()                   # the same seed per problem in every arm (paired)
            stop = StoppingCriteriaList([StopAtQuestion(ids.shape[1])]) if args.prompt_style == "cot8" else None
            with torch.no_grad():
                gen = model.generate(ids, cfg, num_steps=args.steps, stopping_criteria=stop)
            new_tokens = int(gen.shape[1] - ids.shape[1])
            full = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            text = full.split("\nQuestion:")[0].split("Q:")[0]
            ans = gold(ex["answer"])
            if args.prompt_style == "cot8":
                strict, flexible = strict_answer(text), flexible_answer(text)
                pred = flexible
                ok = same_number(flexible, ans)
                ok_strict = same_number(strict, ans)
            else:
                pred = extract(text)
                ok = same_number(pred, ans)
                strict, ok_strict = None, None
            seconds = time.perf_counter() - t0
            rows.write(json.dumps({"arm": str(arm), "lam": lam, "steps": args.steps, "split": args.split,
                                   "index": index, "seed": args.seed * 100003 + index, "pred": pred, "gold": ans,
                                   "correct": ok, "strict_pred": strict, "correct_strict": ok_strict,
                                   "prompt_style": args.prompt_style,
                                   "new_tokens": new_tokens, "core_calls": new_tokens * args.steps,
                                   "seconds": round(seconds, 2), "text": full}) + "\n")
            rows.flush()
            tally[str(arm)][0] += ok
            tally[str(arm)][1] += 1
            tally[str(arm)][2] += seconds
            if tally[str(arm)][1] == len(data) or (args.interleave and i == len(data) - 1):
                c, n, sec = tally[str(arm)]
                summary["arms"][str(arm)] = {"correct": c, "n": n, "accuracy": c / n, "seconds": round(sec, 1)}
                print(json.dumps({"lam": str(arm), "steps": args.steps, "correct": c, "n": n,
                                  "seconds": round(sec, 1)}), flush=True)
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
    h0 = loop.setup(prompt_text(args.task, args.split, args.index))
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    res = {"args": vars(args), "traces": {}}
    for lam in args.lams:
        lam = float(lam)
        s, native, step, per_position = h0.clone(), [], [], []
        for _ in range(args.steps):
            f = loop.F(s)
            ns = float(s.float().norm())
            native.append(float((f - s).float().norm()) / ns)
            d = (f - s).float().reshape(-1, (f - s).shape[-1]).norm(dim=-1)
            per_position.append([float(v) for v in d / s.float().reshape(-1, s.shape[-1]).norm(dim=-1)])
            new = (1.0 - lam) * s + lam * f
            step.append(float((new - s).float().norm()) / ns)
            s = new
        # fit window: below fit_upper (past the nonlinear transient) and above floor_factor x the smallest residual
        # (the precision floor); a failed gate is reported, never replaced by another window
        floor = min(native) * args.floor_factor
        window = [(k, r) for k, r in enumerate(native) if args.fit_upper > r > floor]
        if len(window) >= 8:
            slope = float(np.polyfit([k for k, _ in window], np.log([r for _, r in window]), 1)[0])
            rate, status = math.exp(slope), "fitted"
        else:
            rate, status = None, f"fitting window too short ({len(window)} steps above the floor gate)"
        res["traces"][str(lam)] = {"native_residual": native, "relaxed_step": step, "per_position": per_position,
                                   "fit_window": [window[0][0], window[-1][0]] if window else None,
                                   "rate_per_step": rate, "status": status}
        print(json.dumps({"lam": lam, "steps": args.steps, "native_residual_last": native[-1], "rate_per_step": rate,
                          "status": status}), flush=True)
    (out / "convergence.json").write_text(json.dumps(res, indent=1))


def prompt_text(task, split, index):
    """The scored text of one problem: the GSM8K prompt (zero-shot template or 8-shot chain of thought), or an ARC
    question followed by its correct choice."""
    if task == "gsm8k":
        return PROMPT.format(question=problems(split)[index]["question"])
    if task == "gsm8k-cot8":
        return gsm8k_prompt(problems(split)[index]["question"], "cot8")
    ex = arc_questions(split)[index]
    gold_text = ex["choices"]["text"][ex["choices"]["label"].index(ex["answerKey"])]
    return ARC_PROMPT.format(question=ex["question"]) + " " + gold_text


def arc_questions(split):
    import explore_llm_attractors as ela

    with open(ela.DATA / ARC / f"{split}.jsonl") as f:
        return [json.loads(line) for line in f]


def gold_sequence_ids(tok, ex):
    """Token IDs of the gold row exactly as arc_batch builds it (context, then the continuation tokenized
    separately); the spectrum of a Part A question is computed on these IDs."""
    ctx = tok(ARC_PROMPT.format(question=ex["question"])).input_ids
    g = ex["choices"]["label"].index(ex["answerKey"])
    return ctx + tok(" " + ex["choices"]["text"][g], add_special_tokens=False).input_ids


def arc_batch(tok, ex):
    """Token batch for one ARC question: context + each continuation, right-padded (Huginn's attention is causal, so
    padding after a sequence does not change it). Returns ids, targets (-100 outside the continuation; position t
    predicts token t + 1), the real-position mask, the continuations and the gold index."""
    import torch

    ctx = tok(ARC_PROMPT.format(question=ex["question"])).input_ids
    conts = [" " + c for c in ex["choices"]["text"]]
    seqs = [ctx + tok(c, add_special_tokens=False).input_ids for c in conts]
    n, width = len(seqs), max(len(s) for s in seqs)
    ids = torch.full((n, width), tok.pad_token_id, dtype=torch.long)
    targets = torch.full((n, width), -100, dtype=torch.long)
    mask = torch.zeros(n, width, 1)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s)
        mask[b, :len(s)] = 1.0
        for pos in range(len(ctx) - 1, len(s) - 1):
            targets[b, pos] = s[pos + 1]
    return ids, targets, mask, conts, ex["choices"]["label"].index(ex["answerKey"])


def choice_scores(logprobs, targets):
    """Summed log-probability of each row's continuation from (B, T, V) log-probabilities."""
    valid = targets >= 0
    picked = logprobs.gather(-1, targets.clamp_min(0)[..., None])[..., 0]
    return (picked * valid).sum(-1)


def make_readout(model, steps, targets, store):
    """Readout hook: at each step in `steps`, the model's own head on the current latent state (ln_f, coda, ln_f,
    lm_head, as in Huginn's forward), reduced at once to the continuation scores."""
    import torch

    tr = model.transformer

    def readout(step, state, args):
        if step not in steps:
            return
        freqs_cis, mask, past = args[1], args[2], args[3]
        x = tr.ln_f(state)
        block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in tr.coda:
            block_idx -= 1
            x = block(x, freqs_cis, block_idx, mask, past)
        x = tr.ln_f(x)
        valid = targets >= 0                                   # the head only at positions that score a choice
        logp = torch.log_softmax(model.lm_head(x[valid]).float(), -1)
        picked = logp.gather(-1, targets[valid][:, None])[:, 0]
        rows = torch.arange(targets.shape[0], device=targets.device)[:, None].expand_as(targets)[valid]
        store[step] = torch.zeros(targets.shape[0], device=picked.device).index_add_(0, rows, picked).cpu().numpy()

    return readout


def set_arm(relax, arm, seed, index):
    """Configure the wrapper for an arm; returns the lam used (drawn per problem for rand:LO:HI)."""
    import numpy as np

    arm = str(arm)
    relax.aa = 0
    if arm.startswith("rand:"):
        lo, hi = (float(v) for v in arm.split(":")[1:3])
        relax.lam = float(np.random.default_rng([seed, index]).uniform(lo, hi))
    elif arm.startswith("aa:"):
        _, m, beta = arm.split(":")
        relax.aa, relax.lam = int(m), float(beta)
    else:
        relax.lam = float(arm)
    return relax.lam


def arc(args):
    torch, tok, model, relax = load(args)
    import numpy as np

    import explore_llm_attractors as ela

    data = arc_questions(args.split)[args.start: args.start + args.count]
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    steps = sorted(set(args.steps))
    summary = {"args": vars(args), "arms": {}}
    with open(out / "rows.jsonl", "a") as rows:
        for arm in args.lams:
            tally = {r: [0, 0] for r in steps}
            tic = time.perf_counter()
            for i, ex in enumerate(data):
                index = args.start + i
                lam = set_arm(relax, arm, args.seed, index)
                ids, targets, mask, conts, gold_index = arc_batch(tok, ex)
                store = {}
                relax.readout = make_readout(model, set(steps), targets.to(args.device), store)
                relax.mask = mask.to(args.device)
                relax.trace = []
                torch.manual_seed(args.seed * 100003 + index)     # the same initial latent state in every arm
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(input_ids=ids.to(args.device), num_steps=torch.tensor([max(steps), 0]))
                seconds = time.perf_counter() - t0
                trace, relax.trace, relax.readout, relax.mask = relax.trace, None, None, None
                lengths = np.array([len(c) for c in conts], dtype=float)
                record = {"arm": str(arm), "lam": lam, "split": args.split, "index": index, "id": ex["id"],
                          "gold": gold_index, "seed": args.seed * 100003 + index, "seconds": round(seconds, 3),
                          "native_residual": [t[0] for t in trace], "native_residual_rows": [t[2] for t in trace],
                          "residual_clock": "entry k = R(s_k), k = 0..len-1 (input of update k + 1)",
                          "gold_ids_sha256": hashlib.sha256(json.dumps(gold_sequence_ids(tok, ex)).encode()).hexdigest(),
                          "by_steps": {}}
                for r in steps:
                    ll = store[r]
                    acc, acc_norm = int(np.argmax(ll)) == gold_index, int(np.argmax(ll / lengths)) == gold_index
                    tally[r][0] += acc
                    tally[r][1] += acc_norm
                    record["by_steps"][str(r)] = {"loglik": [float(v) for v in ll], "acc": acc, "acc_norm": acc_norm}
                rows.write(json.dumps(record) + "\n")
                rows.flush()
            n = len(data)
            summary["arms"][str(arm)] = {str(r): {"acc": tally[r][0] / n, "acc_norm": tally[r][1] / n} for r in steps}
            summary["arms"][str(arm)]["seconds"] = round(time.perf_counter() - tic, 1)
            print(json.dumps({"arm": str(arm), "n": n, **{f"r{r}": round(tally[r][1] / n, 4) for r in steps}}),
                  flush=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("accuracy", "convergence"):
        sp = sub.add_parser(name)
        sp.add_argument("--split", default="train", choices=("train", "validation", "test"))
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
    acc.add_argument("--interleave", action="store_true", help="every problem in all arms before the next problem")
    acc.add_argument("--prompt-style", default="zero-shot", choices=("zero-shot", "cot8"),
                     help="cot8: the 8-shot chain-of-thought prompt of lm-eval gsm8k_cot, stopping at 'Q:'")
    conv = sub.choices["convergence"]
    conv.add_argument("--index", type=int, default=0)
    conv.add_argument("--task", default="gsm8k", choices=("gsm8k", "arc"))
    conv.add_argument("--fit-upper", type=float, default=1e-2, help="fit only once the native residual is below this")
    conv.add_argument("--floor-factor", type=float, default=30.0,
                      help="fit only while the native residual exceeds this factor times its minimum")
    sa = sub.add_parser("arc")
    sa.add_argument("--split", default="train", choices=("train", "validation", "test"))
    sa.add_argument("--start", type=int, default=0)
    sa.add_argument("--count", type=int, default=100)
    sa.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32], help="read-out budgets")
    sa.add_argument("--lams", nargs="+", default=["1.0"], help="values, rand:LO:HI or aa:M:BETA")
    sa.add_argument("--seed", type=int, default=0)
    sa.add_argument("--device", default="cuda")
    sa.add_argument("--dtype", default="bfloat16")
    sa.add_argument("--threads", type=int, default=4)
    sa.add_argument("--out", required=True)
    args = parser.parse_args()
    {"accuracy": accuracy, "convergence": convergence, "arc": arc}[args.command](args)


if __name__ == "__main__":
    main()
