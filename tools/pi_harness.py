"""Program PI: decoding harness on the local GPU (bfloat16) for the practical-implication experiments E1-E4.

Problems are GSM8K (test split by default). Every problem is run in two batch layouts that compute the same rows with
different matrix shapes: batches of B1 and of B2 problems whose chat-templated prompts have exactly the same token
length (no padding, so a row's inputs are identical in both layouts; a batch is filled with same-length problems from
the train split when the test split has too few, and those fillers are not scored). Modes:

  hard    chain-of-thought decoding with discrete tokens: greedy (--temperature 0) or sampled with common random
          numbers (one uniform per problem and step from numpy default_rng(seed + problem index), inverse-CDF on the
          top-p nucleus), so both layouts draw the same randomness;
  soft    Soft Thinking: --soft-steps steps that each append the probability-weighted embedding of the concept tokens
          (softmax(z / T) restricted to the top-n tokens and renormalized; n = 0 means the full vocabulary), then the
          answer cue and greedy decoding of the answer. Optional: --gain-cap g lowers a step's temperature (bisection)
          until the readout-gain proxy is at most g (E2); --perturb s displaces the first soft token by a relative s
          along a direction from default_rng(perturb_seed + problem index) (E3).

--save-embeddings stores every soft row's concept tokens (OUT/emb/LAYOUT-BATCH.pt, with the problem indices);
--replay-from RUN feeds a reference run's saved concept tokens instead of computing them (the mechanism probe of
consult 11: the fed-back pathway is clamped, while the transformer's batch-dependent arithmetic, the growing cache
and the answer decoding still run in each layout).

Per row and step it records the top-1 token, its probability, the entropy, the top-2 logit gap / T and, in soft mode,
the readout-gain proxy: the largest eigenvalue of the covariance of the top-64 token embeddings under the (renormalized)
distribution, divided by T (power iteration). Writes OUT/records.jsonl (one line per problem and layout) and
OUT/run.json (arguments, model and versions).

Usage: python tools/pi_harness.py run --mode soft --top-n 15 --temperature 0.6 --soft-steps 64 --layouts 8 16 --out RUN
       python tools/pi_harness.py run --mode hard --temperature 0 --max-new 320 --layouts 8 16 --out RUN
       python tools/pi_harness.py analyze RUN [RUN ...] [--json OUT]
"""
import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
HUB = Path("/data2/neural-shrimp-data/models/hf/hub")
MODELS = {"qwen7b": "models--Qwen--Qwen2.5-7B-Instruct", "qwen0.5b": "models--Qwen--Qwen2.5-0.5B-Instruct",
          "qwen1.5b": "models--Qwen--Qwen2.5-1.5B-Instruct"}
GSM8K = "datasets--openai--gsm8k"
INSTRUCTION = "Solve the following problem. Give the final answer as a number at the end, in the form 'The answer is N.'"
CUE = "\n\nThe answer is"
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def snapshot(name):
    snaps = sorted((HUB / name / "snapshots").glob("*"))
    if not snaps:
        raise SystemExit(f"no local snapshot of {name}")
    return snaps[-1]


def load_gsm8k(split):
    import pandas as pd
    f = next((snapshot(GSM8K) / "main").glob(f"{split}-*.parquet"))
    df = pd.read_parquet(f)
    out = []
    for i, (q, a) in enumerate(zip(df["question"], df["answer"])):
        gold = a.split("####")[-1].strip().replace(",", "")
        out.append({"index": i, "question": q, "gold": gold})
    return out


def extract_answer(text):
    """The number after the last 'answer is', else the last number in the text; None when there is none."""
    low = text.lower()
    k = low.rfind("answer is")
    seg = text[k + len("answer is"):] if k >= 0 else text
    nums = NUM.findall(seg) if k >= 0 else NUM.findall(text)
    if not nums:
        return None
    s = (nums[0] if k >= 0 else nums[-1]).replace(",", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


def prompt_ids(tok, question):
    msgs = [{"role": "user", "content": f"{INSTRUCTION}\n\n{question}"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=True)
    return list(enc["input_ids"])


def make_batches(test, train, tok, size, limit):
    """Batches of `size` rows with identical prompt length: scored test problems first, same-length train fillers."""
    by_len = defaultdict(list)
    for p in test[:limit] if limit else test:
        ids = prompt_ids(tok, p["question"])
        by_len[len(ids)].append({**p, "ids": ids, "scored": True})
    fill = defaultdict(list)
    for p in train:
        ids = prompt_ids(tok, p["question"])
        if len(ids) in by_len:
            fill[len(ids)].append({**p, "ids": ids, "scored": False, "index": -1 - p["index"]})
    batches = []
    for L, rows in sorted(by_len.items()):
        for s in range(0, len(rows), size):
            b = rows[s:s + size]
            need = size - len(b)
            if need:
                pool = fill[L]
                if len(pool) < need:                    # not enough same-length fillers: reuse scored rows unscored
                    pool = pool + [{**r, "scored": False, "index": -100000 - r["index"]} for r in rows]
                b = b + pool[:need]
            batches.append(b)
    return batches


def gain_proxy(torch, p, E, T, k=64, iters=12):
    """Largest eigenvalue of Cov_p(e) over the top-k tokens of p (renormalized), divided by T; per row."""
    w, idx = torch.topk(p, k, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    X = E[idx].float()                                   # (B, k, d)
    mu = (w[..., None] * X).sum(1, keepdim=True)
    Xc = (X - mu) * w[..., None].sqrt()                  # covariance = Xc^T Xc
    v = torch.randn(X.shape[0], X.shape[2], device=X.device, generator=torch.Generator(X.device).manual_seed(0))
    for _ in range(iters):
        v = torch.einsum("bkd,bk->bd", Xc, torch.einsum("bkd,bd->bk", Xc, v))
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    lam = torch.einsum("bk,bk->b", torch.einsum("bkd,bd->bk", Xc, v), torch.einsum("bkd,bd->bk", Xc, v))
    return lam / T


def concept_token(torch, q, E, top_n, chunk=16384):
    """sum_i q_i e_i in float32 without a float32 copy of the embedding matrix: over the top-n support, or in vocabulary
    chunks for the full distribution."""
    if top_n:
        w, idx = torch.topk(q, top_n, dim=-1)
        return (w[..., None] * E[idx].float()).sum(1)
    acc = torch.zeros(q.shape[0], E.shape[1], device=q.device, dtype=torch.float32)
    for s in range(0, E.shape[0], chunk):
        acc += q[:, s:s + chunk] @ E[s:s + chunk].float()
    return acc


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = snapshot(MODELS[args.model])
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    E = model.get_input_embeddings().weight                 # (V, d) bf16
    test, train = load_gsm8k(args.split), load_gsm8k("train" if args.split == "test" else "test")
    cue = tok(CUE, add_special_tokens=False)["input_ids"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"args": vars(args), "model_path": str(path),
                                              "torch": torch.__version__, "cuda": torch.version.cuda,
                                              "gpu": torch.cuda.get_device_name()}, indent=1))
    rec_f = open(out / "records.jsonl", "a")
    done = set()
    if (out / "records.jsonl").exists():
        for line in open(out / "records.jsonl"):
            r = json.loads(line)
            done.add((r["index"], r["layout"]))
    T = max(args.temperature, 1e-6)
    replay = {}
    if args.replay_from:
        for f in sorted((Path(args.replay_from) / "emb").glob("*.pt")):
            blob = torch.load(f)
            for r, idx in enumerate(blob["index"]):
                if idx >= 0 and idx not in replay:
                    replay[idx] = blob["emb"][r]           # (S, d) bf16
        if not replay:
            raise SystemExit(f"no saved concept tokens under {args.replay_from}/emb")
    if args.save_embeddings:
        (out / "emb").mkdir(exist_ok=True)
    for layout in args.layouts:
        batches = make_batches(test[args.offset:], train, tok, layout, args.limit)
        t0 = time.time()
        for bi, batch in enumerate(batches):
            if all((r["index"], layout) in done for r in batch if r["scored"]):
                continue
            ids = torch.tensor([r["ids"] for r in batch], device="cuda")
            B = ids.shape[0]
            rngs = [np.random.default_rng(args.seed + max(r["index"], 0)) for r in batch]
            with torch.no_grad():
                o = model(input_ids=ids, use_cache=True, logits_to_keep=1)   # last position only: memory
                past, z = o.past_key_values, o.logits[:, -1, :].float()
                del o
                steps = {"top1": [], "pmax": [], "entropy": [], "gap": [], "gain": [], "temp": []}
                gen = []
                embs = []
                n_steps = args.soft_steps if args.mode == "soft" else args.max_new
                for t in range(n_steps):
                    Tt = torch.full((B,), T, device="cuda")
                    logp = torch.log_softmax(z / T, dim=-1)
                    p = logp.exp()
                    top2 = torch.topk(z, 2, dim=-1).values
                    steps["top1"].append(z.argmax(-1).tolist())
                    steps["pmax"].append(p.max(-1).values.tolist())
                    steps["entropy"].append((-(p * logp).sum(-1)).tolist())
                    steps["gap"].append(((top2[:, 0] - top2[:, 1]) / T).tolist())
                    if args.mode == "soft":
                        q = p
                        if args.top_n:
                            w, idx = torch.topk(p, args.top_n, dim=-1)
                            q = torch.zeros_like(p).scatter_(-1, idx, w)
                        q = q / q.sum(-1, keepdim=True)
                        g = gain_proxy(torch, q, E, T)
                        if args.gain_cap:                  # E2: lower T per row until the gain proxy is <= cap
                            for r in range(B):
                                if g[r] > args.gain_cap:
                                    lo, hi = 1e-3, T
                                    for _ in range(14):
                                        mid = 0.5 * (lo + hi)
                                        pr = torch.softmax(z[r:r + 1] / mid, dim=-1)
                                        if args.top_n:
                                            w, idx = torch.topk(pr, args.top_n, dim=-1)
                                            pr = torch.zeros_like(pr).scatter_(-1, idx, w)
                                        pr = pr / pr.sum(-1, keepdim=True)
                                        if gain_proxy(torch, pr, E, mid)[0] > args.gain_cap:
                                            hi = mid
                                        else:
                                            lo = mid
                                    pr = torch.softmax(z[r:r + 1] / lo, dim=-1)
                                    if args.top_n:
                                        w, idx = torch.topk(pr, args.top_n, dim=-1)
                                        pr = torch.zeros_like(pr).scatter_(-1, idx, w)
                                    q[r] = (pr / pr.sum(-1, keepdim=True))[0]
                                    Tt[r] = lo
                                    g[r] = gain_proxy(torch, q[r:r + 1], E, lo)[0]
                        steps["gain"].append(g.tolist())
                        steps["temp"].append(Tt.tolist())
                        e = concept_token(torch, q, E, args.top_n).to(torch.bfloat16)  # computed in float32
                        if replay:                                  # clamp the fed-back pathway to the reference
                            for r, row in enumerate(batch):
                                key = row["index"] if row["index"] >= 0 else None
                                if key is None or key not in replay:
                                    if row["scored"]:
                                        raise SystemExit(f"no reference concept tokens for problem {row['index']}")
                                    continue
                                e[r] = replay[key][t].to(e.device)
                        if t == 0 and args.perturb:
                            for r, row in enumerate(batch):
                                rv = np.random.default_rng(args.perturb_seed + max(row["index"], 0)).normal(size=e.shape[1])
                                d = torch.tensor(rv / np.linalg.norm(rv), device="cuda", dtype=torch.float32)
                                e[r] = (e[r].float() + args.perturb * e[r].float().norm() * d).to(torch.bfloat16)
                        if args.save_embeddings:
                            embs.append(e.detach().cpu())
                        o = model(inputs_embeds=e[:, None, :], past_key_values=past, use_cache=True)
                    else:
                        if args.temperature == 0:
                            nxt = z.argmax(-1)
                        else:
                            ps, order = torch.sort(p, dim=-1, descending=True)
                            cs = ps.cumsum(-1)
                            keep = (cs - ps) < args.top_p
                            ps = ps * keep
                            ps = ps / ps.sum(-1, keepdim=True)
                            cs = ps.cumsum(-1)
                            u = torch.tensor([rg.random() for rg in rngs], device="cuda", dtype=cs.dtype)[:, None]
                            pos = torch.searchsorted(cs, u).clamp_max(cs.shape[-1] - 1)
                            nxt = order.gather(-1, pos)[:, 0]
                        gen.append(nxt.tolist())
                        o = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
                    past, z = o.past_key_values, o.logits[:, -1, :].float()
                ans_tok = []
                if args.mode == "soft":                     # answer cue, then greedy answer tokens
                    o = model(input_ids=torch.tensor([cue] * B, device="cuda"), past_key_values=past, use_cache=True,
                              logits_to_keep=1)
                    past, z = o.past_key_values, o.logits[:, -1, :].float()
                    for _ in range(args.answer_tokens):
                        nxt = z.argmax(-1)
                        ans_tok.append(nxt.tolist())
                        o = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
                        past, z = o.past_key_values, o.logits[:, -1, :].float()
            if args.save_embeddings and embs:
                torch.save({"index": [row["index"] for row in batch], "emb": torch.stack(embs, dim=1)},
                           out / "emb" / f"{layout}-{bi}.pt")
            for r, row in enumerate(batch):
                if not row["scored"]:
                    continue
                if args.mode == "soft":
                    toks = [a[r] for a in ans_tok]
                    text = CUE + tok.decode(toks, skip_special_tokens=True)
                else:
                    toks = [g[r] for g in gen]
                    if tok.eos_token_id in toks:
                        toks = toks[:toks.index(tok.eos_token_id)]
                    text = tok.decode(toks, skip_special_tokens=True)
                ans = extract_answer(text)
                rec = {"index": row["index"], "layout": layout, "batch": bi, "row": r, "prompt_len": len(row["ids"]),
                       "answer": ans, "gold": row["gold"], "correct": ans is not None and ans == row["gold"],
                       "tokens": toks, "text": text[-400:],
                       "top1": [s[r] for s in steps["top1"]], "pmax": [round(s[r], 5) for s in steps["pmax"]],
                       "entropy": [round(s[r], 5) for s in steps["entropy"]],
                       "gap": [round(s[r], 4) for s in steps["gap"]]}
                if args.mode == "soft":
                    rec["gain"] = [round(s[r], 5) for s in steps["gain"]]
                    if args.gain_cap:
                        rec["temp"] = [round(s[r], 5) for s in steps["temp"]]
                rec_f.write(json.dumps(rec) + "\n")
            rec_f.flush()
            if bi % 10 == 0:
                el = time.time() - t0
                print(json.dumps({"layout": layout, "batch": bi, "of": len(batches), "seconds": round(el, 1)}),
                      flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def analyze(args):
    summary = {}
    for run_dir in args.runs:
        recs = [json.loads(line) for line in open(Path(run_dir) / "records.jsonl")]
        cfg = json.loads((Path(run_dir) / "run.json").read_text())["args"]
        by = defaultdict(dict)
        for r in recs:
            by[r["index"]][r["layout"]] = r
        la, lb = cfg["layouts"][:2]
        pairs = [(v[la], v[lb]) for v in by.values() if la in v and lb in v]
        n = len(pairs)
        flip = [a["answer"] != b["answer"] for a, b in pairs]
        tok_div = [a["tokens"] != b["tokens"] or a["top1"] != b["top1"] for a, b in pairs]
        first = []
        for a, b in pairs:
            d = next((t for t, (x, y) in enumerate(zip(a["top1"], b["top1"])) if x != y), None)
            first.append(d)
        acc_a = np.mean([a["correct"] for a, _ in pairs]) if n else float("nan")
        acc_b = np.mean([b["correct"] for _, b in pairs]) if n else float("nan")
        k = int(sum(flip))
        s = {"mode": cfg["mode"], "top_n": cfg.get("top_n"), "temperature": cfg["temperature"],
             "gain_cap": cfg.get("gain_cap"), "perturb": cfg.get("perturb"), "pairs": n,
             "answer_flips": k, "flip_rate": k / n if n else float("nan"), "flip_ci95": wilson(k, n),
             "trajectory_divergence": int(sum(tok_div)),
             "median_first_divergence_step": float(np.median([d for d in first if d is not None])) if any(
                 d is not None for d in first) else None,
             "accuracy": {str(la): acc_a, str(lb): acc_b}}
        if cfg["mode"] == "soft" and n:
            mg = np.array([np.mean(a["gain"]) for a, _ in pairs])
            fl = np.array(flip)
            s["mean_gain_flipped"] = float(mg[fl].mean()) if fl.any() else None
            s["mean_gain_kept"] = float(mg[~fl].mean()) if (~fl).any() else None
        summary[run_dir] = s
        print(json.dumps({run_dir: s}, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default="qwen7b", choices=sorted(MODELS))
    r.add_argument("--mode", choices=("hard", "soft"), required=True)
    r.add_argument("--split", default="test")
    r.add_argument("--limit", type=int, default=0, help="first N test problems (0: all)")
    r.add_argument("--layouts", type=int, nargs=2, default=[8, 16])
    r.add_argument("--temperature", type=float, default=0.6)
    r.add_argument("--top-n", type=int, default=0, help="soft: concept tokens kept (0: full vocabulary)")
    r.add_argument("--top-p", type=float, default=0.95, help="hard sampling nucleus")
    r.add_argument("--soft-steps", type=int, default=64)
    r.add_argument("--answer-tokens", type=int, default=12)
    r.add_argument("--max-new", type=int, default=320)
    r.add_argument("--gain-cap", type=float, default=0.0)
    r.add_argument("--perturb", type=float, default=0.0)
    r.add_argument("--perturb-seed", type=int, default=1000)
    r.add_argument("--seed", type=int, default=20261016)
    r.add_argument("--save-embeddings", action="store_true")
    r.add_argument("--replay-from", default=None, help="feed this run's saved concept tokens (mechanism probe)")
    r.add_argument("--offset", type=int, default=0, help="skip the first N test problems (fresh problems)")
    r.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("runs", nargs="+")
    a.add_argument("--json", default=None)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else analyze(args)


if __name__ == "__main__":
    main()
