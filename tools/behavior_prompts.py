"""Are prompts whose soft-token loop is chaotic answered less accurately? (run BP, exploratory)

The behaviour pilot BZ showed that at the steering strengths of the 0.5B maps, steering applied at every position
swamps any task. Here the model is unsteered, and the regime is a property of each prompt: the soft-token loop of
Qwen2.5-7B-Instruct with the task prompt as its fixed prefix (chat template, default system prompt, generation prompt)
is chaotic at some temperatures for some prompts and periodic for others (unsteered 7B chaos lies mostly at log10 T
0.21-0.43; run PS). At three temperatures in that range, prompts are split by their loop's label, and ordinary sampled
answers to the same prompts at the same temperature are scored. Prompt difficulty is controlled by each prompt's
greedy answer.

tasks: docs/bp-tasks.json, seed 20261015: 100 arithmetic problems (50 sums of two 3-digit numbers, 50 products of two
  2-digit numbers; "Reply with only the number") and 100 NQ-open validation questions ("Answer with a short phrase
  only"), disjoint from run BZ's 50.
label: for a shard of the prompts, the soft-token loop (window 1, float64, alpha 0) after each prompt at log10 T 0.25,
  0.32 and 0.40, with twins at 1e-6 and 1e-9 (tools/hidden_chaos.run_batch and classify; transient and record as
  given), one JSON line per prompt. --batched --batch-prompts N: N prompts per batch, each cell with its own prefix
  (explore_llm_soft_loop.MultiPrefixSoftLoop: left padding, masked, true positions) and each prompt's cells with the
  twin directions of its per-prompt call; the same records, equal to the per-prompt ones up to rounding (amplified
  in chaotic cells and at the 1e-9 offset, as by any other change of the batch); "seconds" is then the batch's
  time divided by its prompts.
generate: ordinary sampling (no steering, top_p 1, top_k 0) at the same temperatures, k samples per prompt, plus the
  greedy answer; scored as in run BZ (arithmetic: first integer; NQ: normalized gold answer as a substring).
analyze: per temperature, accuracy of prompts labelled chaotic against prompts labelled periodic, overall and within
  greedy-correct prompts (the difficulty control), with a permutation test on the prompt labels.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
SEED = 20261015
LOG_TS = (0.25, 0.32, 0.40)


def prompt_text(item):
    if item["task"] == "arithmetic":
        return item["q"]
    return f"Answer the question with a short phrase only. Question: {item['q']}?"


def tasks(args):
    from datasets import load_dataset

    rng = np.random.default_rng(SEED + 1)
    items = []
    for n in range(100):
        if n < 50:
            a, b = int(rng.integers(100, 1000)), int(rng.integers(100, 1000))
            items.append({"task": "arithmetic", "q": f"Compute {a} + {b}. Reply with only the number.", "answer": a + b})
        else:
            a, b = int(rng.integers(10, 100)), int(rng.integers(10, 100))
            items.append({"task": "arithmetic", "q": f"Compute {a} * {b}. Reply with only the number.", "answer": a * b})
    used = {x["id"] for x in json.loads((ROOT / "docs/bz-tasks.json").read_text())["nq"]}
    nq = load_dataset("google-research-datasets/nq_open", split="validation", cache_dir=str(DATA / "datasets/hf"))
    pool = [i for i in range(len(nq)) if i not in used]
    for i in sorted(rng.choice(pool, 100, replace=False).tolist()):
        items.append({"task": "nq", "id": int(i), "q": nq[i]["question"], "answers": nq[i]["answer"]})
    for k, it in enumerate(items):
        it["prompt"] = k
    (ROOT / "docs/bp-tasks.json").write_text(json.dumps({"seed": SEED, "log10_temperatures": LOG_TS, "items": items},
                                                        indent=1))
    print(len(items), "prompts")


def prefix_ids(tok, item):
    """Token ids of a prompt's fixed prefix: the chat template (default system prompt) with the generation prompt,
    tokenized without special tokens (the template holds them)."""
    text = tok.apply_chat_template([{"role": "user", "content": prompt_text(item)}], add_generation_prompt=True,
                                   tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def steer_vector(args, d):
    """The steering vector of a run: zeros without --vectors; 'random:N' a seeded Gaussian direction scaled to the
    norm of the philosophy vector of --vectors; otherwise the named concept (the residual-norm-scaled vector)."""
    if not getattr(args, "vectors", None) or getattr(args, "alpha", 0.0) == 0.0:
        return np.zeros(d)
    import explore_llm_phase_map as elpm
    vecs = elpm.load_vectors(args.vectors)
    if args.concept.startswith("random:"):
        g = np.random.default_rng(int(args.concept.split(":")[1])).normal(size=d)
        return g / np.linalg.norm(g) * np.linalg.norm(np.asarray(vecs["philosophy"], dtype=np.float64))
    return np.asarray(vecs[args.concept], dtype=np.float64)


def label_cells(log_ts, res):
    """The labels.jsonl cells of one prompt from its run_batch results (one per temperature)."""
    import hidden_chaos as hc

    cells = []
    for lt, r in zip(log_ts, res):
        r = dict(r) if isinstance(r, dict) else r
        cls = hc.classify(r)
        cells.append({"log10_temperature": lt, "class": cls, "lyapunov": r.get("lyapunov"), "period": r.get("period"),
                      "entropy_mean": r.get("entropy_mean")})
    return cells


def label(args):
    import explore_llm_attractors as ela
    import explore_llm_soft_loop as esl
    import hidden_chaos as hc

    items = json.loads((ROOT / args.tasks).read_text())["items"]
    if args.shard is not None:
        k, n = args.shard
        items = [it for it in items if it["prompt"] % n == k]
    if args.batched:
        return label_batched(args, items)
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    torch = loop.torch
    v0 = steer_vector(args, loop.E.shape[1])
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / "labels.jsonl", "a")
    done = {json.loads(l)["prompt"] for l in open(out / "labels.jsonl")} if (out / "labels.jsonl").stat().st_size else set()
    for it in items:
        if it["prompt"] in done:
            continue
        tic = time.perf_counter()
        loop.prefix_ids = torch.tensor([prefix_ids(loop.sm.tok, it)], device=loop.device)
        loop.L0 = loop.prefix_ids.shape[1]
        res = hc.run_batch(loop, v0, list(args.log_temperatures), [args.alpha] * len(args.log_temperatures), args.transient,
                           args.record, args.eps, args.tol)
        cells = label_cells(args.log_temperatures, res)
        rec = {"prompt": it["prompt"], "task": it["task"], "prefix_tokens": int(loop.L0), "cells": cells,
               "seconds": round(time.perf_counter() - tic, 1)}
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(json.dumps({"prompt": it["prompt"], "classes": [c["class"] for c in cells],
                          "exponents": [c["lyapunov"] for c in cells], "seconds": rec["seconds"]}), flush=True)


def label_batched(args, items):
    """label with --batch-prompts prompts per batch: every cell (prompt, temperature) has its own prefix
    (MultiPrefixSoftLoop), the cells run prompt by prompt, and each prompt's cells get the twin directions of its
    per-prompt call (run_batch twin_groups). One record per prompt, as label writes them; "seconds" is the batch's
    time divided by its prompts."""
    import explore_llm_soft_loop as esl
    import hidden_chaos as hc

    if args.batch_prompts < 1:
        raise SystemExit("--batch-prompts must be at least 1")
    loop = esl.MultiPrefixSoftLoop(args.model, args.layer, args.threads, 1, "float64", args.device)
    v0 = steer_vector(args, loop.E.shape[1])
    lts = list(args.log_temperatures)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / "labels.jsonl", "a")
    done = {json.loads(l)["prompt"] for l in open(out / "labels.jsonl")} if (out / "labels.jsonl").stat().st_size else set()
    todo = [it for it in items if it["prompt"] not in done]
    for s in range(0, len(todo), args.batch_prompts):
        group = todo[s:s + args.batch_prompts]
        tic = time.perf_counter()
        ids = [prefix_ids(loop.sm.tok, it) for it in group]
        loop.set_prefixes([p for p in ids for _ in lts])           # cell g * len(lts) + j: prompt g, temperature j
        res = hc.run_batch(loop, v0, lts * len(group), [args.alpha] * (len(lts) * len(group)), args.transient, args.record,
                           args.eps, args.tol, twin_groups=[len(lts)] * len(group))
        seconds = round((time.perf_counter() - tic) / len(group), 1)
        for g, it in enumerate(group):
            cells = label_cells(lts, res[g * len(lts):(g + 1) * len(lts)])
            rec = {"prompt": it["prompt"], "task": it["task"], "prefix_tokens": len(ids[g]), "cells": cells,
                   "seconds": seconds}
            f.write(json.dumps(rec) + "\n")
            print(json.dumps({"prompt": it["prompt"], "classes": [c["class"] for c in cells],
                              "exponents": [c["lyapunov"] for c in cells], "seconds": seconds}), flush=True)
        f.flush()


def tasks_c(args):
    """Confirmatory set (run BPC, docs/bpc-test.md): 4000 products of two 2-digit numbers, distinct pairs, none of
    them among run BP's products, in a seeded order (seed SEED + 101); one task family only."""
    used = {(it["q"]) for it in json.loads((ROOT / "docs/bp-tasks.json").read_text())["items"]}
    rng = np.random.default_rng(SEED + 101)
    pairs = [(a, b) for a in range(10, 100) for b in range(10, 100)]
    order = rng.permutation(len(pairs))
    items = []
    for k in order:
        a, b = pairs[k]
        q = f"Compute {a} * {b}. Reply with only the number."
        if q in used:
            continue
        items.append({"task": "arithmetic", "q": q, "a": a, "b": b, "answer": a * b, "prompt": len(items)})
        if len(items) == 4000:
            break
    (ROOT / "docs/bpc-tasks.json").write_text(json.dumps({"seed": SEED + 101, "log10_temperatures": [0.40],
                                                          "items": items}, indent=1))
    print(len(items), "prompts")


def covariates(args):
    """Pre-sampling covariates (run BPC): for each prompt, the next-token distribution at the reply's first position
    at each temperature: entropy, largest probability, and the log-probability of the correct answer's first token.
    Deterministic; computed from the prompt alone, before any sampled outcome."""
    import explore_llm_phase_map as elpm

    items = json.loads((ROOT / args.tasks).read_text())["items"]
    sm = elpm.SteeredModel(args.model, args.layer, args.threads, args.dtype, args.device)
    torch, tok = sm.torch, sm.tok
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rec = {}
    for s0 in range(0, len(items), args.batch):
        part = items[s0:s0 + args.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": prompt_text(it)}], add_generation_prompt=True,
                                         tokenize=False) for it in part]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(sm.model.device)
        sv = steer_vector(args, sm.model.get_input_embeddings().weight.shape[1])
        sm.steer = torch.as_tensor(np.tile(args.alpha * sv, (len(part), 1)))[:, None, :] if args.alpha else None
        try:
            with torch.no_grad():
                z = sm.model(**enc).logits[:, -1, :].double()
        finally:
            sm.steer = None
        first = [tok(str(it["answer"]), add_special_tokens=False)["input_ids"][0] for it in part]
        for lt in args.log_temperatures:
            lp = torch.log_softmax(z / 10.0 ** lt, dim=-1)
            p = lp.exp()
            ent = (-(p * lp).sum(-1)).cpu().numpy()
            ptop = p.max(-1).values.cpu().numpy()
            lpc = lp[torch.arange(len(part)), torch.tensor(first, device=lp.device)].cpu().numpy()
            for k, it in enumerate(part):
                rec.setdefault(str(lt), {})[str(it["prompt"])] = {"entropy": float(ent[k]), "p_top": float(ptop[k]),
                                                                  "logp_correct_first": float(lpc[k])}
        print(json.dumps({"done": s0 + len(part), "of": len(items)}), flush=True)
    (out / "covariates.json").write_text(json.dumps(rec))


def generate(args):
    import explore_llm_phase_map as elpm
    import behavior_zones as bz

    items = json.loads((ROOT / args.tasks).read_text())["items"]
    sm = elpm.SteeredModel(args.model, args.layer, args.threads, args.dtype, args.device)
    d = sm.model.get_input_embeddings().weight.shape[1]
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    msgs_all = [[{"role": "user", "content": prompt_text(it)}] for it in items]
    scorer = {"arithmetic": bz.score_arith, "nq": bz.score_nq}
    rec = {"greedy": {}, "sampled": {}}
    for task, max_new in (("arithmetic", 12), ("nq", 32)):
        idx_all = [i for i, it in enumerate(items) if it["task"] == task]
        # chunks of prompts keep the sampled batch's cache bounded; chunk 0 draws exactly as the unchunked run did
        for c, c0 in enumerate(range(0, len(idx_all), args.chunk_prompts)):
            idx = idx_all[c0:c0 + args.chunk_prompts]
            msgs = [msgs_all[i] for i in idx]
            sv = args.alpha * steer_vector(args, d)
            texts, _, _, _ = sm.generate_batch(msgs, np.tile(sv, (len(msgs), 1)), max_new, temperature=0.0)
            for i, t in zip(idx, texts):
                rec["greedy"][str(items[i]["prompt"])] = {"correct": bool(scorer[task](t, items[i])), "reply": t[:200]}
            for lt in args.log_temperatures:
                T = 10.0 ** lt
                m = [msgs_all[i] for i in idx for _ in range(args.k)]
                texts, ent, lens, _ = sm.generate_batch(m, np.tile(sv, (len(m), 1)), max_new, temperature=T,
                                                        seed=SEED + int(1000 * lt) + (0 if task == "arithmetic" else 7)
                                                        + 100003 * c)
                for j, t in enumerate(texts):
                    it = items[idx[j // args.k]]
                    rec["sampled"].setdefault(f"{lt}", {}).setdefault(str(it["prompt"]), []).append(
                        {"correct": bool(scorer[task](t, it)), "entropy": ent[j], "length": lens[j], "reply": t[:200]})
            print(json.dumps({"task": task, "chunk": c, "prompts": len(idx)}), flush=True)
        for lt in args.log_temperatures:
            if idx_all:
                print(json.dumps({"task": task, "log10_T": lt, "accuracy": float(np.mean(
                    [x["correct"] for p in idx_all for x in rec["sampled"][f"{lt}"][str(items[p]["prompt"])]]))}),
                    flush=True)
    (out / "generations.json").write_text(json.dumps(rec))


def calibrate(args):
    """Steering strengths that leave the task intact (run BPCS calibration): for the first n prompts, greedy and k
    sampled answers at T under each (concept, alpha), against alpha 0. Writes calibration.json."""
    import explore_llm_phase_map as elpm
    import behavior_zones as bz

    items = json.loads((ROOT / args.tasks).read_text())["items"][:args.n]
    sm = elpm.SteeredModel(args.model, args.layer, args.threads, args.dtype, args.device)
    d = sm.model.get_input_embeddings().weight.shape[1]
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    msgs = [[{"role": "user", "content": prompt_text(it)}] for it in items]
    res = []
    for concept in ["none"] + list(args.concepts):
        for alpha in ([0.0] if concept == "none" else args.alphas):
            ns = argparse.Namespace(vectors=args.vectors, concept=concept, alpha=alpha)
            sv = alpha * steer_vector(ns, d) if concept != "none" else np.zeros(d)
            g, _, _, _ = sm.generate_batch(msgs, np.tile(sv, (len(msgs), 1)), 12, temperature=0.0)
            m = [x for x in msgs for _ in range(args.k)]
            t, _, _, _ = sm.generate_batch(m, np.tile(sv, (len(m), 1)), 12, temperature=10.0 ** args.log_temperatures[0],
                                           seed=SEED + 77)
            rec = {"concept": concept, "alpha": alpha,
                   "greedy_accuracy": float(np.mean([bz.score_arith(x, it) for x, it in zip(g, items)])),
                   "sampled_accuracy": float(np.mean([bz.score_arith(x, items[j // args.k]) for j, x in enumerate(t)])),
                   "greedy_examples": [x[:60] for x in g[:3]]}
            res.append(rec)
            print(json.dumps(rec), flush=True)
            (out / "calibration.json").write_text(json.dumps({"args": vars(args), "results": res}, indent=1))


def analyze(args):
    items = {it["prompt"]: it for it in json.loads((ROOT / args.tasks).read_text())["items"]}
    labels = {}
    for d in args.label_runs:
        for line in open(DATA / "explore" / d / "labels.jsonl"):
            r = json.loads(line)
            labels[r["prompt"]] = r
    gen = json.loads((DATA / "explore" / args.gen_run / "generations.json").read_text())
    rng = np.random.default_rng(SEED)
    out = {}
    for lt in args.log_temperatures:
        rows = []
        for p, r in labels.items():
            c = next(c for c in r["cells"] if abs(c["log10_temperature"] - lt) < 1e-9)
            cls = "chaotic" if "chaos" in c["class"] and "irregular" not in c["class"] else (
                "periodic" if c["class"] == "periodic" else "other")
            samples = gen["sampled"][f"{lt}"].get(str(p), [])
            if not samples:
                continue
            rows.append({"prompt": p, "task": items[p]["task"], "label": cls,
                         "greedy": gen["greedy"][str(p)]["correct"],
                         "acc": float(np.mean([s["correct"] for s in samples]))})
        res = {}
        for subset, sel in (("all", lambda r: True), ("greedy_correct", lambda r: r["greedy"])):
            rs = [r for r in rows if sel(r) and r["label"] in ("chaotic", "periodic")]
            ch = np.array([r["acc"] for r in rs if r["label"] == "chaotic"])
            pe = np.array([r["acc"] for r in rs if r["label"] == "periodic"])
            if len(ch) and len(pe):
                diff = float(ch.mean() - pe.mean())
                pool, nc = np.concatenate([ch, pe]), len(ch)
                perm = [np.mean(x[:nc]) - np.mean(x[nc:]) for x in (rng.permutation(pool) for _ in range(10000))]
                p = float(np.mean(np.abs(perm) >= abs(diff)))
                res[subset] = {"chaotic": len(ch), "periodic": len(pe), "acc_chaotic": float(ch.mean()),
                               "acc_periodic": float(pe.mean()), "difference": diff, "permutation_p": p}
        res["label_counts"] = {k: sum(r["label"] == k for r in rows) for k in ("chaotic", "periodic", "other")}
        out[str(lt)] = res
        print(lt, json.dumps(res))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("tasks")
    sub.add_parser("tasks-c")
    for name in ("label", "generate", "analyze", "covariates", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("--tasks", default="docs/bp-tasks.json")
        p.add_argument("--log-temperatures", type=float, nargs="+", default=list(LOG_TS))
    lab = sub.choices["label"]
    lab.add_argument("--shard", type=int, nargs=2, metavar=("K", "N"))
    lab.add_argument("--transient", type=int, default=500)
    lab.add_argument("--record", type=int, default=1000)
    lab.add_argument("--eps", type=float, nargs="+", default=[1e-6, 1e-9])
    lab.add_argument("--tol", type=float, default=1e-8)
    lab.add_argument("--batched", action="store_true",
                     help="label --batch-prompts prompts per batch (one prefix per cell; the same records)")
    lab.add_argument("--batch-prompts", type=int, default=8, help="prompts per batch with --batched")
    gen = sub.choices["generate"]
    gen.add_argument("--k", type=int, default=10)
    gen.add_argument("--dtype", default="float32")
    gen.add_argument("--chunk-prompts", type=int, default=400, help="prompts per sampled batch")
    calp = sub.choices["calibrate"]
    calp.add_argument("--n", type=int, default=200)
    calp.add_argument("--k", type=int, default=10)
    calp.add_argument("--dtype", default="float32")
    calp.add_argument("--concepts", nargs="+", default=["philosophy", "hostility-vs-kindness", "random:1"])
    calp.add_argument("--alphas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.04])
    cov = sub.choices["covariates"]
    cov.add_argument("--dtype", default="float32")
    cov.add_argument("--batch", type=int, default=64)
    for p in (lab, gen, cov):
        p.add_argument("--concept", default="philosophy", help="steering concept, or random:N (with --vectors)")
        p.add_argument("--alpha", type=float, default=0.0, help="steering strength (0: unsteered)")
    for p in (lab, gen, cov, calp):
        p.add_argument("--vectors", default=None, help="steering vectors (DATA/explore-relative); none: unsteered")
    for p in (lab, gen, cov, calp):
        p.add_argument("--model", default="qwen7b")
        p.add_argument("--layer", type=int, default=14)
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--device", default="cuda")
        p.add_argument("--out", required=True)
    an = sub.choices["analyze"]
    an.add_argument("--label-runs", nargs="+", required=True)
    an.add_argument("--gen-run", required=True)
    an.add_argument("--out-json")
    args = ap.parse_args()
    {"tasks": tasks, "tasks-c": tasks_c, "label": label, "generate": generate, "analyze": analyze,
     "covariates": covariates, "calibrate": calibrate}[args.command](args)


if __name__ == "__main__":
    main()
