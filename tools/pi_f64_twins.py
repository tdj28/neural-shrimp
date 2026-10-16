"""Program PI (float64): finite-time sensitivity of the soft-thinking pipeline on GSM8K problems, and its float64
answers, for comparison with the bfloat16 harness (tools/pi_harness.py) on the same problems and settings.

Each problem's chat-templated prompt (built exactly as in tools/pi_harness.py) is run in float64 (the repository's
float64 model: patched RMS normalization and rotary tables, tools/explore_llm_soft_loop.SoftLoop) as a base row and
--twins twin rows whose first concept token is displaced by a relative --eps0 along random unit directions. Every row
then runs --soft-steps soft-thinking steps (the probability-weighted embedding of the top-n concept tokens of
softmax(z / T), renormalized; n = 0 is the full vocabulary), the answer cue and greedy answer tokens. Problems of equal
prompt length share a batch, so no row is padded. Per problem: the relative distance between each twin's and the base
row's concept tokens at every step, its finite-time growth rate (the slope of log d over the steps where
10 eps0 <= d <= 1e-3, as tools/soft_thinking_twins.py), the step at which d first exceeds 1e-3, the entropy and top-2
gap along the base row, and the answers of base and twins (a twin's answer differing from the base's is a float64
answer flip caused by a 1e-10 perturbation).

Usage: python tools/pi_f64_twins.py --model qwen7b --problems docs/pi-gsm8k-test.json --limit 400 --temperature 0.6
           --top-n 15 --soft-steps 64 --twins 3 --rows 32 --device cuda --out RUN
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pi_harness as ph  # noqa: E402

DMAX = 1e-3


def growth_rate(d, eps0, dmax=DMAX):
    d = np.maximum(np.asarray(d, dtype=float), 1e-300)
    t = np.arange(1, len(d) + 1)
    above = np.nonzero(d > dmax)[0]
    first = int(above[0]) + 1 if len(above) else None
    stop = above[0] if len(above) else len(d)
    sel = np.nonzero((d[:stop] >= 10 * eps0) & (d[:stop] <= dmax))[0]
    if len(sel) < 5:
        sel = np.arange(stop)
    if len(sel) < 2:
        return float("nan"), first
    return float(np.polyfit(t[sel], np.log(d[sel]), 1)[0]), first


def concept(torch, p, E, top_n, chunk=16384):
    if top_n:
        w, idx = torch.topk(p, top_n, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        return (w[..., None] * E[idx]).sum(1)
    acc = torch.zeros(p.shape[0], E.shape[1], device=p.device, dtype=E.dtype)
    for s in range(0, E.shape[0], chunk):
        acc += p[:, s:s + chunk] @ E[s:s + chunk]
    return acc


def main():
    import explore_llm_attractors as ela
    import explore_llm_soft_loop as esl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen7b")
    ap.add_argument("--problems", default="docs/pi-gsm8k-test.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--soft-steps", type=int, default=64)
    ap.add_argument("--answer-tokens", type=int, default=12)
    ap.add_argument("--twins", type=int, default=3)
    ap.add_argument("--eps0", type=float, default=1e-10)
    ap.add_argument("--rows", type=int, default=32, help="rows per batch (problems x (1 + twins))")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default="bf16", choices=("native", "bf16"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    loop = esl.SoftLoop(args.model, 0, args.threads, "open", ela.OPENER, 1, "float64", args.device, args.weights)
    torch, tok, model, E = loop.torch, loop.sm.tok, loop.model, loop.E
    problems = json.loads((ROOT / args.problems).read_text())["problems"]
    if args.limit:
        problems = problems[:args.limit]
    cue = tok(ph.CUE, add_special_tokens=False)["input_ids"]
    by_len = defaultdict(list)
    for p in problems:
        by_len[len(ph.prompt_ids(tok, p["question"]))].append(p)
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"args": vars(args)}, indent=1))
    rec_f = open(out / "records.jsonl", "a")
    per_batch = max(1, args.rows // (1 + args.twins))
    T, K = args.temperature, args.twins
    t0, done = time.time(), 0
    for L, group in sorted(by_len.items()):
        for s in range(0, len(group), per_batch):
            batch = group[s:s + per_batch]
            n = len(batch)
            ids = torch.tensor([ph.prompt_ids(tok, p["question"]) for p in batch] * (1 + K), device=loop.device)
            rng = np.random.default_rng(args.seed + batch[0]["index"])
            dist = np.zeros((args.soft_steps, K, n))
            ent = np.zeros((args.soft_steps, n))
            gap = np.zeros((args.soft_steps, n))
            with torch.no_grad():
                o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
                past, z = o.past_key_values, o.logits[:, -1, :].to(loop.device, loop.state_dtype)
                for t in range(args.soft_steps):
                    logp = torch.log_softmax(z / T, dim=-1)
                    p = logp.exp()
                    top2 = torch.topk(z[:n], 2, dim=-1).values
                    ent[t] = (-(p[:n] * logp[:n]).sum(-1)).cpu().numpy()
                    gap[t] = ((top2[:, 0] - top2[:, 1]) / T).cpu().numpy()
                    X = concept(torch, p, E, args.top_n)
                    if t == 0:
                        for k in range(K):
                            u = torch.as_tensor(rng.normal(size=(n, X.shape[1])), dtype=X.dtype, device=X.device)
                            u = u / u.norm(dim=1, keepdim=True)
                            X[n * (k + 1):n * (k + 2)] = X[:n] + args.eps0 * X[:n].norm(dim=1, keepdim=True) * u
                    base = X[:n]
                    for k in range(K):
                        dist[t, k] = ((X[n * (k + 1):n * (k + 2)] - base).norm(dim=1) / base.norm(dim=1)).cpu().numpy()
                    o = model(inputs_embeds=X[:, None, :].to(loop.dtype), past_key_values=past, use_cache=True,
                              logits_to_keep=1)
                    past, z = o.past_key_values, o.logits[:, -1, :].to(loop.device, loop.state_dtype)
                o = model(input_ids=torch.tensor([cue] * ids.shape[0], device=loop.device), past_key_values=past,
                          use_cache=True, logits_to_keep=1)
                past, z = o.past_key_values, o.logits[:, -1, :].to(loop.device, loop.state_dtype)
                ans = []
                for _ in range(args.answer_tokens):
                    nxt = z.argmax(-1)
                    ans.append(nxt.tolist())
                    o = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True, logits_to_keep=1)
                    past, z = o.past_key_values, o.logits[:, -1, :].to(loop.device, loop.state_dtype)
            texts = [ph.CUE + tok.decode([a[r] for a in ans], skip_special_tokens=True) for r in range(ids.shape[0])]
            answers = [ph.extract_answer(x) for x in texts]
            for c, p in enumerate(batch):
                rates = [growth_rate(dist[:, k, c], args.eps0) for k in range(K)]
                twin_ans = [answers[n * (k + 1) + c] for k in range(K)]
                rec = {"index": p["index"], "gold": p["gold"], "prompt_len": L, "answer": answers[c],
                       "correct": answers[c] is not None and answers[c] == p["gold"], "twin_answers": twin_ans,
                       "twin_flips": sum(a != answers[c] for a in twin_ans),
                       "growth_rates": [r[0] for r in rates], "first_above_dmax": [r[1] for r in rates],
                       "growth_rate_median": float(np.nanmedian([r[0] for r in rates])),
                       "final_distance": [float(dist[-1, k, c]) for k in range(K)],
                       "entropy_mean": float(ent[:, c].mean()), "gap_min": float(gap[:, c].min()),
                       "text": texts[c][-200:]}
                rec_f.write(json.dumps(rec) + "\n")
            rec_f.flush()
            done += n
            print(json.dumps({"done": done, "of": len(problems), "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
