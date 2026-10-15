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
  given), one JSON line per prompt.
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


def label(args):
    import explore_llm_attractors as ela
    import explore_llm_soft_loop as esl
    import hidden_chaos as hc

    items = json.loads((ROOT / args.tasks).read_text())["items"]
    if args.shard is not None:
        k, n = args.shard
        items = [it for it in items if it["prompt"] % n == k]
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    torch = loop.torch
    v0 = np.zeros(loop.E.shape[1])
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / "labels.jsonl", "a")
    done = {json.loads(l)["prompt"] for l in open(out / "labels.jsonl")} if (out / "labels.jsonl").stat().st_size else set()
    for it in items:
        if it["prompt"] in done:
            continue
        tic = time.perf_counter()
        text = loop.sm.tok.apply_chat_template([{"role": "user", "content": prompt_text(it)}], add_generation_prompt=True,
                                               tokenize=False)
        loop.prefix_ids = torch.tensor([loop.sm.tok(text, add_special_tokens=False)["input_ids"]], device=loop.device)
        loop.L0 = loop.prefix_ids.shape[1]
        res = hc.run_batch(loop, v0, list(args.log_temperatures), [0.0] * len(args.log_temperatures), args.transient,
                           args.record, args.eps, args.tol)
        cells = []
        for lt, r in zip(args.log_temperatures, res):
            r = dict(r) if isinstance(r, dict) else r
            cls = hc.classify(r)
            cells.append({"log10_temperature": lt, "class": cls, "lyapunov": r.get("lyapunov"), "period": r.get("period"),
                          "entropy_mean": r.get("entropy_mean")})
        rec = {"prompt": it["prompt"], "task": it["task"], "prefix_tokens": int(loop.L0), "cells": cells,
               "seconds": round(time.perf_counter() - tic, 1)}
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(json.dumps({"prompt": it["prompt"], "classes": [c["class"] for c in cells],
                          "exponents": [c["lyapunov"] for c in cells], "seconds": rec["seconds"]}), flush=True)


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
        idx = [i for i, it in enumerate(items) if it["task"] == task]
        msgs = [msgs_all[i] for i in idx]
        texts, _, _, _ = sm.generate_batch(msgs, np.zeros((len(msgs), d)), max_new, temperature=0.0)
        for i, t in zip(idx, texts):
            rec["greedy"][str(items[i]["prompt"])] = {"correct": bool(scorer[task](t, items[i])), "reply": t[:200]}
        for lt in args.log_temperatures:
            T = 10.0 ** lt
            m = [msgs_all[i] for i in idx for _ in range(args.k)]
            texts, ent, lens, _ = sm.generate_batch(m, np.zeros((len(m), d)), max_new, temperature=T,
                                                    seed=SEED + int(1000 * lt) + (0 if task == "arithmetic" else 7))
            for j, t in enumerate(texts):
                it = items[idx[j // args.k]]
                rec["sampled"].setdefault(f"{lt}", {}).setdefault(str(it["prompt"]), []).append(
                    {"correct": bool(scorer[task](t, it)), "entropy": ent[j], "length": lens[j], "reply": t[:200]})
            print(json.dumps({"task": task, "log10_T": lt, "accuracy": float(np.mean(
                [x["correct"] for p in idx for x in rec["sampled"][f"{lt}"][str(items[p]["prompt"])]]))}), flush=True)
    (out / "generations.json").write_text(json.dumps(rec))


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
    for name in ("label", "generate", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--tasks", default="docs/bp-tasks.json")
        p.add_argument("--log-temperatures", type=float, nargs="+", default=list(LOG_TS))
    lab = sub.choices["label"]
    lab.add_argument("--shard", type=int, nargs=2, metavar=("K", "N"))
    lab.add_argument("--transient", type=int, default=500)
    lab.add_argument("--record", type=int, default=1000)
    lab.add_argument("--eps", type=float, nargs="+", default=[1e-6, 1e-9])
    lab.add_argument("--tol", type=float, default=1e-8)
    gen = sub.choices["generate"]
    gen.add_argument("--k", type=int, default=10)
    gen.add_argument("--dtype", default="float32")
    for p in (lab, gen):
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
    {"tasks": tasks, "label": label, "generate": generate, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
