"""Does accumulated feedback state make soft-thinking answers depend on the batch layout? (run RR; consult 10, direction 1)

The pipeline is that of tools/repro_task.py (run REPRO-T): after the chat-templated question, `soft` steps each append the
expected embedding E^T softmax(z / T) to the context; then the answer cue is appended and `answer` tokens are decoded
greedily; the first integer is the answer (none: INVALID). Each problem is run in one precision (bf16: the model cast
to bfloat16 with logits, softmax and the soft token in float32, as in REPRO-T; or float32) in four arms:

- reference: batch 1, recording the target's logits z_t before every soft step and the final key-value cache;
- free: the target in row 0 of a batch with fixed distractor problems of the same token length (no padding). Its
  prompt state (cache and z_0) is copied from the reference, and the soft phase then runs on its own, so the target's
  state accumulates the layout's rounding differences step by step;
- reset: as free, but before every soft step t the target row's cache (all positions so far) and z_t are replaced by
  the reference's, so each step starts from the reference state and only that one step runs in the batch layout;
- native: the REPRO-T pipeline in the batch layout (prompt, soft phase and answer all in the batch).

Free, reset and the reference decode their answers from their own final state moved to batch 1 (common decoder): the
cue and the answer tokens are processed in batch 1. Native decodes in its batch. Each arm's answer is compared with
the reference's answer. Checks per problem: a rerun of the reference (determinism) and the free and reset arms in
batch 1 alone, which must reproduce the reference exactly. Per soft step, the relative distance of the target's
logits from the reference's: in the free arm (accumulated) and in the reset arm (one step only).

Endpoint (consult 10): Delta = P(disagree | free) - P(disagree | reset) over problems, paired, with a bootstrap
interval over problems and an exact McNemar test on the discordant pairs.

Usage:
  python tools/repro_reset.py run --problems docs/rr-problems.json --set dev --dtype bf16 --device cuda --out ...
  python tools/repro_reset.py analyze --runs RUN [RUN ...] [--json OUT]
"""
import argparse
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import repro_task as rt  # noqa: E402


def fwd(loop, cache, **kw):
    """Logits of the last position for every row, in the loop's state precision."""
    return loop.model(past_key_values=cache, use_cache=True, logits_to_keep=1, **kw).logits[:, -1, :] \
        .to(loop.device, loop.state_dtype)


def soft_token(loop, z, T):
    return loop.torch.softmax(z / T, dim=-1) @ loop.E


def to_batch1(cache):
    """A batch-1 copy of row 0 of a key-value cache."""
    from transformers import DynamicCache
    new = DynamicCache()
    for i, layer in enumerate(cache.layers):
        new.update(layer.keys[0:1].clone(), layer.values[0:1].clone(), i)
    return new


def copy_row0(cache, ref, upto):
    """Overwrite row 0 of cache, positions < upto, with row 0 of ref."""
    for layer, rl in zip(cache.layers, ref.layers):
        layer.keys[0, :, :upto] = rl.keys[0, :, :upto]
        layer.values[0, :, :upto] = rl.values[0, :, :upto]


def decode(loop, cache, cue_ids, n_answer):
    """Greedy answer of a batch-1 state: the cue, then n_answer tokens."""
    torch = loop.torch
    with torch.no_grad():
        z = fwd(loop, cache, input_ids=torch.tensor([cue_ids], device=loop.device))
        ans = []
        for _ in range(n_answer):
            tok = z.argmax(-1)
            ans.append(int(tok[0]))
            z = fwd(loop, cache, input_ids=tok[:, None])
    return ans


def reference(loop, ids, T, n_soft):
    """Batch-1 soft phase: the logits before every soft step and after the last (n_soft + 1), and the cache."""
    from transformers import DynamicCache
    torch = loop.torch
    cache = DynamicCache()
    zs = []
    with torch.no_grad():
        z = fwd(loop, cache, input_ids=torch.tensor([ids], device=loop.device))
        for _ in range(n_soft):
            zs.append(z[0].clone())
            z = fwd(loop, cache, inputs_embeds=soft_token(loop, z, T)[:, None, :].to(loop.dtype))
        zs.append(z[0].clone())
    return zs, cache


def arm(loop, rows, T, n_soft, ref_zs, ref_cache, reset):
    """Free (reset=False) or reset arm of the target (row 0) in the batch layout `rows`: the prompt state is copied
    from the reference; with reset, so is the state before every later soft step. Returns the target's final state
    (batch-1 cache) and the per-step relative logit distance from the reference."""
    from transformers import DynamicCache
    torch = loop.torch
    L = len(rows[0])
    cache = DynamicCache()
    dist = []
    with torch.no_grad():
        z = fwd(loop, cache, input_ids=torch.tensor(rows, device=loop.device))
        copy_row0(cache, ref_cache, L)
        z[0] = ref_zs[0]
        for t in range(n_soft):
            if reset and t > 0:
                copy_row0(cache, ref_cache, L + t)
                z[0] = ref_zs[t]
            z = fwd(loop, cache, inputs_embeds=soft_token(loop, z, T)[:, None, :].to(loop.dtype))
            dist.append(float((z[0] - ref_zs[t + 1]).norm() / ref_zs[t + 1].norm()))
    return to_batch1(cache), dist


def run_problem(loop, tok, ids, dis, T, n_soft, cue_ids, n_answer, problem):
    ref_zs, ref_cache = reference(loop, ids, T, n_soft)
    rec = {"problem": problem}
    answers = {"reference": decode(loop, to_batch1(ref_cache), cue_ids, n_answer)}
    rz2, rc2 = reference(loop, ids, T, n_soft)
    answers["reference_rerun"] = decode(loop, to_batch1(rc2), cue_ids, n_answer)
    rec["reference_rerun_identical"] = bool(all(loop.torch.equal(a, b) for a, b in zip(ref_zs, rz2)))
    for name, rows, reset in (("free_batch1", [ids], False), ("reset_batch1", [ids], True),
                              ("free", [ids] + dis, False), ("reset", [ids] + dis, True)):
        state, dist = arm(loop, rows, T, n_soft, ref_zs, ref_cache, reset)
        answers[name] = decode(loop, state, cue_ids, n_answer)
        rec[f"{name}_logit_distance"] = dist
    native, _, _ = rt.pipeline(loop, [ids] + dis, "soft", T, n_soft, cue_ids, n_answer)
    answers["native"] = native
    ref_text = tok.decode(answers["reference"])
    for name, ans in answers.items():
        text = tok.decode(ans)
        val = rt.parse(text)
        rec[name] = {"text": text, "parsed": val, "correct": val == problem["answer"], "valid": val is not None,
                     "differs_from_reference": text != ref_text}
    rec["batch1_checks_pass"] = bool(rec["reference_rerun_identical"]
                                     and max(rec["free_batch1_logit_distance"]) == 0.0
                                     and max(rec["reset_batch1_logit_distance"]) == 0.0
                                     and not rec["free_batch1"]["differs_from_reference"]
                                     and not rec["reset_batch1"]["differs_from_reference"])
    return rec


def run(args):
    import explore_llm_attractors as ela
    spec = json.loads(Path(args.problems).read_text())
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "runs.jsonl"
    if path.exists():
        raise SystemExit(f"{path} exists; outputs are never overwritten")
    if args.dtype == "bf16":                          # as in REPRO-T: bf16 model, float32 logits and soft token
        rt_args = argparse.Namespace(model=args.model, layer=args.layer, threads=args.threads, device=args.device)
        loop = rt.load(rt_args, "bf16")
    else:
        import explore_llm_soft_loop as esl
        loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float32", args.device)
    tok = loop.sm.tok
    cue_ids = tok(rt.CUE, add_special_tokens=False)["input_ids"]
    dis = [rt.chat_ids(tok, p) for p in spec["distractors"]]
    T = 10.0 ** args.log_temperature
    probs = spec[args.set][:args.limit] if args.limit else spec[args.set]
    tic = time.perf_counter()
    meta = {"args": vars(args), "n_problems": len(probs), "device_name": None}
    if args.device.startswith("cuda"):
        meta["device_name"] = loop.torch.cuda.get_device_name(0)
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    for i, prob in enumerate(probs):
        ids = rt.chat_ids(tok, prob)
        rec = run_problem(loop, tok, ids, dis, T, args.soft, cue_ids, args.answer, prob)
        rec.update({"i": i, "set": args.set, "dtype": args.dtype, "log10_temperature": args.log_temperature})
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({"i": i, "checks": rec["batch1_checks_pass"],
                          **{k: rec[k]["differs_from_reference"] for k in ("free", "reset", "native")},
                          "seconds": round(time.perf_counter() - tic, 1)}), flush=True)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, j) for j in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def analyze(args):
    import explore_llm_attractors as ela
    recs = []
    for r in args.runs:
        recs += [json.loads(line) for line in open(ela.DATA / "explore" / r / "runs.jsonl")]
    out = {}
    for dtype in sorted({r["dtype"] for r in recs}):
        rs = [r for r in recs if r["dtype"] == dtype]
        n = len(rs)
        dis = {a: np.array([r[a]["differs_from_reference"] for r in rs], dtype=float)
               for a in ("free", "reset", "native", "reference_rerun")}
        delta = dis["free"] - dis["reset"]
        rng = np.random.default_rng(args.seed)
        boot = [delta[rng.integers(0, n, n)].mean() for _ in range(args.boot)]
        b = int(((dis["free"] == 1) & (dis["reset"] == 0)).sum())
        c = int(((dis["free"] == 0) & (dis["reset"] == 1)).sum())
        acc = {a: float(np.mean([r[a]["correct"] for r in rs])) for a in ("reference", "free", "reset", "native")}
        invalid = {a: int(sum(not r[a]["valid"] for r in rs)) for a in ("reference", "free", "reset", "native")}
        fd = np.array([r["free_logit_distance"] for r in rs])
        rd = np.array([r["reset_logit_distance"] for r in rs])
        out[dtype] = {
            "problems": n, "checks_pass": int(sum(r["batch1_checks_pass"] for r in rs)),
            "disagreement": {a: float(v.mean()) for a, v in dis.items()},
            "disagreement_counts": {a: int(v.sum()) for a, v in dis.items()},
            "delta_free_minus_reset": float(delta.mean()),
            "delta_bootstrap_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "discordant_free_only": b, "discordant_reset_only": c, "mcnemar_exact_p": mcnemar_exact(b, c),
            "accuracy": acc, "invalid": invalid,
            "free_logit_distance_median_by_step": np.median(fd, axis=0).tolist(),
            "reset_logit_distance_median_by_step": np.median(rd, axis=0).tolist(),
        }
    text = json.dumps(out, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--problems", required=True, help="JSON with problem lists and 'distractors'")
    r.add_argument("--set", required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--dtype", choices=("bf16", "float32"), required=True)
    r.add_argument("--model", default="qwen0.5b")
    r.add_argument("--layer", type=int, default=12, help="steering hook block (no steering is applied)")
    r.add_argument("--log-temperature", type=float, default=0.0)
    r.add_argument("--soft", type=int, default=32)
    r.add_argument("--answer", type=int, default=8)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--device", default="cpu")
    r.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--runs", nargs="+", required=True)
    a.add_argument("--boot", type=int, default=10000)
    a.add_argument("--seed", type=int, default=1015)
    a.add_argument("--json")
    args = ap.parse_args()
    {"run": run, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
