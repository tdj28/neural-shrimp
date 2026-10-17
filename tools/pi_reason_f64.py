"""Run PM2F (exploratory): does the soft-thinking loop of a reasoning model expand or contract in float64?

The pipeline of tools/pi_reason.py (soft arm: concept tokens until the end-of-thinking token or the thinking budget, then
sampled answer tokens with common random numbers), computed in float64. Decoder weights stay in their stored bfloat16
and are used in float64 (exact; explore_llm_attractors.Engine with weights="bf16"), with the norm and rotary patches.
Each problem has a base row and --twins rows whose first concept token is displaced by a relative --eps0 along random
unit directions. Twins share the base row's random numbers, so any difference comes from the displacement. Per twin:
the relative distance of its fed-back input from the base row's at every step, the growth rate over the steps where
10 eps0 <= d <= 1e-3 (as tools/pi_f64_twins.py), whether it ends thinking at the same step, and whether its answer
differs from the base row's.

Problems: the GSM8K training problems of run PM2 (after the first 256, in full groups of identical prompt length),
grouped --group per batch; --skip-groups N leaves out the first N groups (dataset order), so a later run extends an
earlier one without overlap and with the same per-group directions.
Usage: python tools/pi_reason_f64.py --model r1qwen7b --limit 128 --out RUN
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import pi_harness as ph
import pi_reason as pr

ELA_KEYS = {"r1qwen7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "r1llama8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "qwq32b": "Qwen/QwQ-32B", "r1qwen1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}


def growth_rate(d, eps0, dmax=1e-3):
    sel = np.nonzero((d >= 10 * eps0) & (d <= dmax))[0]
    if len(sel) < 2:
        return None
    t = sel.astype(float)
    return float(np.polyfit(t, np.log(d[sel]), 1)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(ELA_KEYS))
    ap.add_argument("--offset", type=int, default=256)
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--group", type=int, default=8, help="problems per batch (identical prompt length)")
    ap.add_argument("--skip-groups", type=int, default=0,
                    help="skip the first N groups (to extend an earlier run without overlap; --limit counts them)")
    ap.add_argument("--twins", type=int, default=2)
    ap.add_argument("--eps0", type=float, default=1e-10)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--think-budget", type=int, default=1536)
    ap.add_argument("--answer-budget", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    ela.MODELS.update(ELA_KEYS)
    eng = ela.Engine(args.model, args.threads, "float64", args.device, "bf16")
    torch, tok, model = eng.torch, eng.tok, eng.model
    model.eval()
    end_think = tok.convert_tokens_to_ids("</think>")
    if end_think is None or end_think == tok.unk_token_id:
        raise SystemExit("the tokenizer has no </think> token")
    eos = tok.eos_token_id
    E = model.get_input_embeddings().weight
    if E.dtype != torch.float64:
        raise SystemExit(f"input embeddings are {E.dtype}, not float64")
    pool = ph.load_gsm8k("train")[args.offset:]
    groups = pr.gsm8k_batches(pool, tok, [args.group], args.limit)[args.group]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"args": vars(args), "torch": torch.__version__,
                                              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                                              "groups": len(groups)}, indent=1))
    done = set()
    if (out / "records.jsonl").exists():
        done = {json.loads(line)["index"] for line in open(out / "records.jsonl")}
    rec_f = open(out / "records.jsonl", "a")
    T, K, dev = args.temperature, args.twins, args.device
    t0 = time.time()
    for gi, group in enumerate(groups):
        if gi < args.skip_groups:
            continue
        if all(p["index"] in done for p in group):
            continue
        n = len(group)
        rows = group * (1 + K)                       # base rows first, then twin block k = 1..K
        B = len(rows)
        ids = torch.tensor([r["ids"] for r in rows], device=dev)
        rngs = [np.random.default_rng(args.seed + r["index"]) for r in rows]
        dirs = np.random.default_rng([args.seed, gi]).normal(size=(n * K, E.shape[1]))
        mode, switch, answer = [0] * B, [None] * B, [[] for _ in range(B)]
        dist = np.full((args.think_budget + args.answer_budget, n * K), np.nan)
        with torch.no_grad():
            o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            past, z = o.past_key_values, o.logits[:, -1, :]
            del o
            for t in range(args.think_budget + args.answer_budget):
                if all(m == 2 for m in mode):
                    break
                p = torch.softmax(z / T, dim=-1)
                u = torch.tensor([rg.random() for rg in rngs], device=dev, dtype=torch.float64)
                am = z.argmax(-1)
                sampled = pr.sample_nucleus(torch, p, args.top_p, u)
                tokens = torch.full((B,), eos, device=dev, dtype=torch.long)
                use_soft = torch.zeros(B, dtype=torch.bool, device=dev)
                for r in range(B):
                    if mode[r] == 0:
                        if t >= args.think_budget:
                            tokens[r], mode[r], switch[r] = end_think, 1, t
                        elif int(am[r]) == end_think:
                            tokens[r], mode[r], switch[r] = end_think, 1, t
                        else:
                            use_soft[r] = True
                    elif mode[r] == 1:
                        tokens[r] = sampled[r]
                        answer[r].append(int(sampled[r]))
                        if int(sampled[r]) == eos or len(answer[r]) >= args.answer_budget:
                            mode[r] = 2
                e = E[tokens].clone()
                if bool(use_soft.any()):
                    w, idx = torch.topk(p, args.top_n, dim=-1)
                    w = w / w.sum(-1, keepdim=True)
                    c = (w[..., None] * E[idx]).sum(1)
                    e[use_soft] = c[use_soft]
                if t == 0:
                    for j in range(n * K):
                        r = n + j
                        d = torch.tensor(dirs[j] / np.linalg.norm(dirs[j]), device=dev, dtype=torch.float64)
                        e[r] = e[r] + args.eps0 * e[r].norm() * d
                base = e[:n].repeat(K, 1)
                rel = ((e[n:] - base).norm(dim=1) / base.norm(dim=1).clamp_min(1e-300)).cpu().numpy()
                dist[t] = rel
                o = model(inputs_embeds=e[:, None, :], past_key_values=past, use_cache=True)
                past, z = o.past_key_values, o.logits[:, -1, :]

        def ans(r):
            toks = answer[r][:answer[r].index(eos)] if eos in answer[r] else answer[r]
            return pr.extract(tok.decode(toks, skip_special_tokens=True))
        for i, row in enumerate(group):
            base_ans = ans(i)
            twins = []
            for k in range(K):
                j, r = k * n + i, n + k * n + i
                d = dist[:, j]
                stop = switch[i] if switch[i] is not None else args.think_budget
                dt = d[:stop]
                fin = dt[~np.isnan(dt)]
                twins.append({"answer": ans(r), "flip": ans(r) != base_ans, "switch_step": switch[r],
                              "same_switch": switch[r] == switch[i],
                              "growth_rate": growth_rate(fin, args.eps0) if len(fin) else None,
                              "log10_max_think": float(np.log10(max(np.nanmax(fin), 1e-300))) if len(fin) else None,
                              "log10_end_think": float(np.log10(max(fin[-1], 1e-300))) if len(fin) else None})
            rec_f.write(json.dumps({"index": row["index"], "gold": row["gold"], "answer": base_ans,
                                    "correct": base_ans == row["gold"], "switch_step": switch[i],
                                    "twins": twins}) + "\n")
        rec_f.flush()
        print(json.dumps({"group": gi, "of": len(groups), "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
