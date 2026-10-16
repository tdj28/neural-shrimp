"""Program PI (float64): finite-time sensitivity of the soft-thinking pipeline on GSM8K problems, and its float64
answers, for comparison with the bfloat16 harness (tools/pi_harness.py) on the same problems and settings.

Each problem's chat-templated prompt (built exactly as in tools/pi_harness.py) is run in float64 (the repository's
float64 model: patched RMS normalization and rotary tables, tools/explore_llm_soft_loop.SoftLoop) as a base row and
twin rows whose first concept token is displaced by a relative eps0 along random unit directions (--twins rows per
level of --eps0). Every row then runs --soft-steps soft-thinking steps (the probability-weighted embedding of the
top-n concept tokens of softmax(z / T), renormalized; n = 0 is the full vocabulary), the answer cue and greedy answer
tokens. One process runs every configuration of --temperatures x --alphas. Steering adds alpha v to the output of
decoder block --layer at every position, as in the steered maps; v comes from --steer-vectors, in the maps'
convention. Each (problem, configuration) unit is a base row with its twins. Units of equal prompt length share a
batch, with per-row temperatures and steering rows, so no row is padded. Each unit's twin directions are seeded by the
problem and its configuration, so the results do not depend on how the units were batched.

Per problem and twin: the relative distance between the twin's and the base row's concept tokens at every step (stored
as log10, null where the twin has merged with the base exactly), its finite-time growth rate (the slope of log d over
the steps where 10 eps0 <= d <= dmax, dmax = max(1e-3, 100 eps0), as tools/soft_thinking_twins.py), the step at which
d first exceeds dmax, its answer (a twin answering differently from the base row is a float64 answer flip caused by
the displacement) and whether its decoded answer text differs (which also counts steered rows that give no number).
Per problem: the base row's entropy and top-2 logit gap at every step.

Usage: python tools/pi_f64_twins.py --model qwen7b --problems docs/pi-gsm8k-test.json --limit 200 \\
           --temperatures 0.6 1.2 --alphas 0 0.05 --top-n 15 --soft-steps 128 --eps0 1e-10 1e-6 1e-3 --twins 2 \\
           --rows 63 --device cuda --out RUN
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


def dmax_for(eps0):
    return max(DMAX, 100 * eps0)


def growth_rate(d, eps0, dmax=None):
    dmax = dmax_for(eps0) if dmax is None else dmax
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


def log10_or_none(x):
    return None if x <= 0 else round(float(np.log10(x)), 3)


def run_all(loop, args, problems, by_len, cue, configs, levels, vec, out):
    """Every (problem, configuration) unit is a base row and its twins; units of equal prompt length share a batch
    whatever their temperature and steering (per-row temperatures and steering rows), so batches stay full."""
    torch, tok, model, E = loop.torch, loop.sm.tok, loop.model, loop.E
    K = len(levels)
    per_batch = max(1, args.rows // (1 + K))
    files = {}
    for T, alpha in configs:
        d = out / f"t{T:g}-a{alpha:g}"
        d.mkdir(parents=True, exist_ok=True)
        files[(T, alpha)] = open(d / "records.jsonl", "w")
    t0, done, total = time.time(), 0, len(problems) * len(configs)
    for L, group in sorted(by_len.items()):
        units = [(p, c) for c in configs for p in group]
        for s in range(0, len(units), per_batch):
            batch = units[s:s + per_batch]
            n = len(batch)
            ids = torch.tensor([ph.prompt_ids(tok, p["question"]) for p, _ in batch] * (1 + K), device=loop.device)
            Tv = torch.tensor([c[0] for _, c in batch] * (1 + K), dtype=loop.state_dtype, device=loop.device)[:, None]
            if vec is None:
                loop.sm.steer = None
            else:
                rows = np.stack([c[1] * vec for _, c in batch] * (1 + K))
                loop.sm.steer = torch.as_tensor(rows, dtype=torch.float64, device=loop.device)[:, None, :]
            dirs = []                                   # per unit, its own seeded directions: batching-independent
            for p, c in batch:
                key = [args.seed, p["index"], int(round(c[0] * 1e6)), int(round(abs(c[1]) * 1e6))]
                rng = np.random.default_rng(key + [1] if c[1] < 0 else key)   # seeds must be non-negative
                dirs.append(rng.normal(size=(K, E.shape[1])))
            dirs = np.stack(dirs, axis=1)               # (K, n, d)
            dist = np.zeros((args.soft_steps, K, n))
            ent = np.zeros((args.soft_steps, n))
            gap = np.zeros((args.soft_steps, n))
            with torch.no_grad():
                o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
                past, z = o.past_key_values, o.logits[:, -1, :].to(loop.device, loop.state_dtype)
                for t in range(args.soft_steps):
                    logp = torch.log_softmax(z / Tv, dim=-1)
                    p_ = logp.exp()
                    top2 = torch.topk(z[:n], 2, dim=-1).values
                    ent[t] = (-(p_[:n] * logp[:n]).sum(-1)).cpu().numpy()
                    gap[t] = ((top2[:, 0] - top2[:, 1]) / Tv[:n, 0]).cpu().numpy()
                    X = concept(torch, p_, E, args.top_n)
                    if t == 0:
                        for k, eps0 in enumerate(levels):
                            u = torch.as_tensor(dirs[k], dtype=X.dtype, device=X.device)
                            u = u / u.norm(dim=1, keepdim=True)
                            X[n * (k + 1):n * (k + 2)] = X[:n] + eps0 * X[:n].norm(dim=1, keepdim=True) * u
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
            for u_, (p, c) in enumerate(batch):
                twins = []
                for k, eps0 in enumerate(levels):
                    g, first = growth_rate(dist[:, k, u_], eps0)
                    r = n * (k + 1) + u_
                    twins.append({"eps0": eps0, "answer": answers[r], "flip": answers[r] != answers[u_],
                                  "text_flip": texts[r] != texts[u_], "growth_rate": g,
                                  "first_above_dmax": first, "dmax": dmax_for(eps0),
                                  "final_distance": float(dist[-1, k, u_]),
                                  "log10_distance": [log10_or_none(x) for x in dist[:, k, u_]]})
                rec = {"index": p["index"], "gold": p["gold"], "prompt_len": L, "temperature": c[0], "alpha": c[1],
                       "answer": answers[u_], "correct": answers[u_] is not None and answers[u_] == p["gold"],
                       "twins": twins, "entropy": [round(float(x), 4) for x in ent[:, u_]],
                       "gap": [round(float(x), 4) for x in gap[:, u_]], "text": texts[u_][-200:]}
                files[c].write(json.dumps(rec) + "\n")
            for f in files.values():
                f.flush()
            done += n
            print(json.dumps({"done": done, "of": total, "rows": ids.shape[0],
                              "seconds": round(time.time() - t0, 1)}), flush=True)
    loop.sm.steer = None
    for f in files.values():
        f.close()


def main():
    import explore_llm_attractors as ela
    import explore_llm_soft_loop as esl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen7b")
    ap.add_argument("--problems", default="docs/pi-gsm8k-test.json")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.6])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0])
    ap.add_argument("--concept", default="hostility-vs-kindness")
    ap.add_argument("--layer", type=int, default=14, help="steering after this zero-based decoder block")
    ap.add_argument("--steer-vectors", default="explore/llm-phase-map/vectors-7b-l14", help="DATA-relative folder")
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--soft-steps", type=int, default=128)
    ap.add_argument("--answer-tokens", type=int, default=12)
    ap.add_argument("--eps0", type=float, nargs="+", default=[1e-10])
    ap.add_argument("--twins", type=int, default=2, help="twins per eps0 level")
    ap.add_argument("--rows", type=int, default=63, help="rows per batch (problems x (1 + twins))")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default="bf16", choices=("native", "bf16"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", args.device,
                        args.weights)
    torch, tok = loop.torch, loop.sm.tok
    problems = json.loads((ROOT / args.problems).read_text())["problems"][args.offset:]
    if args.limit:
        problems = problems[:args.limit]
    cue = tok(ph.CUE, add_special_tokens=False)["input_ids"]
    by_len = defaultdict(list)
    for p in problems:
        by_len[len(ph.prompt_ids(tok, p["question"]))].append(p)
    levels = [e for e in args.eps0 for _ in range(args.twins)]
    vec = None
    if any(a != 0 for a in args.alphas):
        vpath = Path(args.steer_vectors)
        vpath = vpath if vpath.is_absolute() else ela.DATA / vpath
        vec = np.load(vpath / "vectors.npz")[args.concept]
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"args": vars(args), "torch": torch.__version__,
                                              "device_name": torch.cuda.get_device_name()
                                              if args.device == "cuda" else "cpu"}, indent=1))
    configs = [(T, a) for T in args.temperatures for a in args.alphas]
    run_all(loop, args, problems, by_len, cue, configs, levels, vec, out)


if __name__ == "__main__":
    main()
