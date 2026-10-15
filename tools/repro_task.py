"""Does an irrelevant execution choice change the answer of continuous-feedback inference? (run REPRO-T)

Each arithmetic problem is answered by a short soft-thinking pipeline with a growing context, as in Soft Thinking
(Zhang et al. 2025) but without top-k filtering: after the chat-templated question, `soft` steps each append the
expected embedding E^T softmax(z / T) to the context; then a fixed answer cue (discrete tokens) is appended and
`answer` tokens are decoded greedily, and the first integer in them is the answer (none: INVALID). The baseline is
ordinary greedy inference with the same budget: `soft` greedy tokens, the same cue, the same extraction. Unsteered.

The irrelevant execution choice is the batch layout: the problem alone (batch 1), against the same problem in the
first row of a batch of four with three fixed distractor problems. All prompts have the same token length
(two-digit operands; the tokenizer splits digits), so positions and masks are identical and nothing is padded: the
rows differ only in the matrix shapes the kernels see. Precision: float64 (norms and rotary tables patched), or
the model cast to bfloat16 with logits, softmax and the soft token in float32.

Per problem, precision and decoding mode (soft at each temperature, greedy): the answer in each layout, a rerun of
the single layout (determinism check), the per-step distance between the soft tokens of the two layouts, and the
logit margin of the first answer token. Mechanism subset: the first `mech` problems, float64, single layout, with
the first soft token displaced by delta * |x| in a fixed random direction, at each temperature.

Usage: python tools/repro_task.py --problems docs/repro-task-problems.json --device cuda --out llm-soft-loop/pod64/repro/t
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

QUESTION = "What is {a} + {b}? Answer with just the number."
CUE = "\n\nAnswer: "


def make_problems(seed, n):
    """n two-digit additions, half with a carry from the units digit, in a seeded order."""
    rng = np.random.default_rng(seed)
    out, want = [], {True: n // 2, False: n - n // 2}
    while sum(want.values()):
        a, b = (int(x) for x in rng.integers(10, 100, size=2))
        carry = (a % 10 + b % 10) >= 10
        if want[carry]:
            want[carry] -= 1
            out.append({"a": a, "b": b, "carry": bool(carry), "answer": a + b})
    return out


def parse(text):
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def load(args, dtype):
    import explore_llm_attractors as ela
    import explore_llm_soft_loop as esl
    if dtype == "float64":
        loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    else:
        loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float32", args.device)
        loop.model = loop.model.to(loop.torch.bfloat16)
        loop.dtype = loop.torch.bfloat16
    return loop


def chat_ids(tok, problem):
    text = tok.apply_chat_template([{"role": "user", "content": QUESTION.format(**problem)}],
                                   add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def pipeline(loop, rows, mode, T, n_soft, cue_ids, n_answer, delta=None, seed=0):
    """Run the pipeline on equal-length prompt rows (row 0 = target). Returns the target's answer ids, its soft
    tokens (n_soft, d) or hard tokens, and the logit margin of its first answer token."""
    from transformers import DynamicCache
    torch = loop.torch
    L = len(rows[0])
    if any(len(r) != L for r in rows):
        raise ValueError("prompt rows of different lengths would need padding")
    B = len(rows)
    cache = DynamicCache()
    fwd = lambda **kw: loop.model(past_key_values=cache, use_cache=True, logits_to_keep=1, **kw).logits[:, -1, :] \
        .to(loop.device, loop.state_dtype)                                       # noqa: E731
    traj = []
    with torch.no_grad():
        z = fwd(input_ids=torch.tensor(rows, device=loop.device))
        for t in range(n_soft):
            if mode == "soft":
                x = torch.softmax(z / T, dim=-1) @ loop.E
                if t == 0 and delta is not None:
                    g = torch.as_tensor(np.random.default_rng(seed).normal(size=x.shape[1]), dtype=x.dtype,
                                        device=x.device)
                    x[0] = x[0] + g / g.norm() * delta * x[0].norm()
                traj.append(x[0].double().cpu().numpy())
                z = fwd(inputs_embeds=x[:, None, :].to(loop.dtype))
            else:
                tok = z.argmax(-1)
                traj.append(int(tok[0]))
                z = fwd(input_ids=tok[:, None])
        z = fwd(input_ids=torch.tensor([cue_ids] * B, device=loop.device))
        top2 = torch.topk(z[0], 2).values
        margin = float(top2[0] - top2[1])
        ans = []
        for _ in range(n_answer):
            tok = z.argmax(-1)
            ans.append(int(tok[0]))
            z = fwd(input_ids=tok[:, None])
    return ans, traj, margin


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12, help="steering hook block (no steering is applied)")
    ap.add_argument("--problems", required=True, help="JSON: 'dev', 'heldout', 'distractors' problem lists")
    ap.add_argument("--log-temperature", type=float, nargs="+", default=[0.0, 0.2])
    ap.add_argument("--soft", type=int, default=32)
    ap.add_argument("--answer", type=int, default=8)
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "float64"], choices=("bf16", "float64"))
    ap.add_argument("--mech", type=int, default=6)
    ap.add_argument("--deltas", type=float, nargs="+", default=[1e-12, 1e-8, 1e-4])
    ap.add_argument("--sets", nargs="+", default=["dev", "heldout"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    spec = json.loads(Path(args.problems).read_text())
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    records = out / "runs.jsonl"
    progress = out / "progress.jsonl"
    tic = time.perf_counter()

    def log(rec):
        line = json.dumps(rec)
        with open(records, "a") as f:
            f.write(line + "\n")

    modes = [("soft", t) for t in args.log_temperature] + [("greedy", None)]
    for dtype in args.dtypes:
        loop = load(args, dtype)
        tok = loop.sm.tok
        cue_ids = tok(CUE, add_special_tokens=False)["input_ids"]
        dis = [chat_ids(tok, p) for p in spec["distractors"]]
        for which in args.sets:
            for i, prob in enumerate(spec[which]):
                ids = chat_ids(tok, prob)
                for mode, lt in modes:
                    T = None if lt is None else 10.0 ** lt
                    res = {}
                    for layout, rows in (("single", [ids]), ("batch4", [ids] + dis), ("rerun", [ids])):
                        ans, traj, margin = pipeline(loop, rows, mode, T, args.soft, cue_ids, args.answer)
                        text = tok.decode(ans)
                        res[layout] = (ans, traj, margin, text)
                    rec = {"set": which, "i": i, "problem": prob, "dtype": dtype, "mode": mode, "log10_temperature": lt}
                    for layout, (ans, traj, margin, text) in res.items():
                        val = parse(text)
                        rec[layout] = {"text": text, "parsed": val, "correct": val == prob["answer"], "margin": margin}
                    rec["answers_differ"] = res["single"][3] != res["batch4"][3]
                    rec["rerun_identical"] = res["single"][3] == res["rerun"][3] and \
                        (np.array_equal(np.array(res["single"][1]), np.array(res["rerun"][1])))
                    if mode == "soft":
                        a, b = np.array(res["single"][1]), np.array(res["batch4"][1])
                        rec["soft_distance"] = (np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1)).tolist()
                    else:
                        a, b = res["single"][1], res["batch4"][1]
                        diff = [k for k, (u, w) in enumerate(zip(a, b)) if u != w]
                        rec["first_token_difference"] = diff[0] + 1 if diff else None
                    log(rec)
                with open(progress, "a") as f:
                    f.write(json.dumps({"dtype": dtype, "set": which, "done": i + 1, "of": len(spec[which]),
                                        "seconds": round(time.perf_counter() - tic, 1)}) + "\n")
        if dtype == "float64" and args.mech:
            for i, prob in enumerate(spec["heldout"][:args.mech]):
                ids = chat_ids(tok, prob)
                for lt in args.log_temperature:
                    T = 10.0 ** lt
                    base, btraj, _ = pipeline(loop, [ids], "soft", T, args.soft, cue_ids, args.answer)
                    for d in args.deltas:
                        ans, traj, margin = pipeline(loop, [ids], "soft", T, args.soft, cue_ids, args.answer, delta=d,
                                                     seed=i)
                        a, b = np.array(btraj), np.array(traj)
                        log({"set": "mech", "i": i, "problem": prob, "dtype": dtype, "log10_temperature": lt,
                             "delta": d, "base_text": tok.decode(base), "text": tok.decode(ans),
                             "answers_differ": tok.decode(base) != tok.decode(ans),
                             "distance": (np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1)).tolist()})
        del loop


if __name__ == "__main__":
    main()
