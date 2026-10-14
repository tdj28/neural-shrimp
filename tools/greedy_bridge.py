"""Zero-temperature bridge between the soft-token map and ordinary greedy decoding (planned item of the paper).

As T -> 0 the next-token distribution becomes one-hot and the soft-token map becomes the greedy one-token map
g(t) = argmax_s logits(prefix + E[t])_s on the vocabulary: the model sees the fixed prefix and the embedding of the
last token only. Its cycles are the zero-temperature attractors of the soft-token map.

graph   g for every token (batched; the prefix cache is computed once and repeated), its cycles, each cycle's basin
        (the tokens whose orbit reaches it) and the decision margin (top-1 minus top-2 logit) of every token, so that
        near-ties are flagged instead of trusted.
decode  ordinary greedy decoding with a growing context from the same prompt, started from forced first tokens (the
        tokens of the largest cycles and random tokens); the final repetition loop of each run (period <= 64, at least
        three repeats at the end) is compared with the cycles of g, exactly up to rotation.

Usage: python tools/greedy_bridge.py graph  --model qwen0.5b --layer 12 --vectors VEC --concept C --alpha A --out OUT
       python tools/greedy_bridge.py decode --model qwen0.5b --layer 12 --vectors VEC --concept C --alpha A
                                            --graph OUT --out OUT2 [--random 200] [--length 256]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def canonical(loop):
    loop = tuple(int(t) for t in loop)
    return min(loop[i:] + loop[:i] for i in range(len(loop)))


def cycles_of(nxt):
    """Cycles of a functional graph (every node has one successor) and the cycle each node flows into."""
    n = len(nxt)
    state = np.zeros(n, dtype=np.int8)            # 0 unvisited, 1 on the current path, 2 done
    owner = np.full(n, -1, dtype=np.int64)
    cycles = []
    for s in range(n):
        if state[s]:
            continue
        path, x = [], s
        while state[x] == 0:
            state[x] = 1
            path.append(x)
            x = int(nxt[x])
        if state[x] == 1:                         # closed a new cycle on this path
            i = path.index(x)
            cyc = path[i:]
            cid = len(cycles)
            cycles.append(cyc)
            for y in cyc:
                owner[y] = cid
            tail = path[:i]
        else:
            cid, tail = owner[x], path
        for y in tail:
            owner[y] = cid
        for y in path:
            state[y] = 2
    return cycles, owner


def setup(args):
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    loop = esl.SoftLoop(args.model, args.layer, args.threads, args.prompt, ela.OPENER if args.opener is None
                        else args.opener, 1, args.dtype, args.device)
    v = elpm.load_vectors(args.vectors)[args.concept] if args.vectors else np.zeros(loop.E.shape[1])
    return loop, np.asarray(v) * args.alpha


def graph(args):
    import explore_llm_attractors as ela

    loop, steer = setup(args)
    torch = loop.torch
    loop.prepare(steer[None])
    base, steer1 = loop.cache, loop.steer_rows
    V = loop.E.shape[0]
    nxt = np.zeros(V, dtype=np.int64)
    margin = np.zeros(V)
    caches = {}
    tic = time.perf_counter()
    with torch.no_grad():
        for start in range(0, V, args.batch):
            ids = np.arange(start, min(start + args.batch, V))
            b = len(ids)
            if b not in caches:
                c = copy.deepcopy(base)
                c.batch_repeat_interleave(b)
                caches[b] = c
            loop.cache, loop.steer_rows = caches[b], steer1.repeat(b, 1, 1)
            X = loop.E[torch.as_tensor(ids, device=loop.device)][:, None, :]
            z = loop.logits(X)
            top2 = torch.topk(z, 2, dim=-1)
            nxt[ids] = top2.indices[:, 0].cpu().numpy()
            margin[ids] = (top2.values[:, 0] - top2.values[:, 1]).double().cpu().numpy()
    cycles, owner = cycles_of(nxt)
    basin = np.bincount(owner, minlength=len(cycles))
    order = np.argsort(-basin)
    tok = loop.sm.tok
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "graph.npz", next=nxt, margin=margin, owner=owner)
    summary = {"args": vars(args), "seconds": round(time.perf_counter() - tic, 1), "tokens": int(V),
               "cycles": len(cycles), "ambiguous_tokens_margin_below_1e-3": int((margin < 1e-3).sum()),
               "largest": [{"cycle": int(i), "period": len(cycles[i]), "basin": int(basin[i]),
                            "tokens": [tok.decode([t]) for t in cycles[i]], "ids": [int(t) for t in cycles[i]],
                            "min_margin": float(margin[cycles[i]].min())} for i in order[:args.top]],
               "period_histogram": {int(k): int(v) for k, v in zip(*np.unique([len(c) for c in cycles],
                                                                                return_counts=True))}}
    (out / "graph.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("tokens", "cycles", "ambiguous_tokens_margin_below_1e-3",
                                              "period_histogram")}))
    for c in summary["largest"][:10]:
        print(c["period"], c["basin"], repr("".join(c["tokens"])), round(c["min_margin"], 3))


def final_loop(seq, max_q=64, repeats=3):
    for q in range(1, max_q + 1):
        if len(seq) >= repeats * q and all(seq[-k] == seq[-k - q] for k in range(1, (repeats - 1) * q + 1)):
            return seq[-q:]
    return None


def decode(args):
    import explore_llm_attractors as ela

    loop, steer = setup(args)
    torch = loop.torch
    gdir = ela.DATA / "explore" / args.graph
    g = json.loads((gdir / "graph.json").read_text())
    if abs(g["args"]["alpha"] - args.alpha) > 0 or g["args"]["concept"] != args.concept:
        raise SystemExit("graph computed for another steering")
    gz = np.load(gdir / "graph.npz")
    nxt, owner = gz["next"], gz["owner"]
    cycles, _ = cycles_of(nxt)
    canon = {canonical(c): i for i, c in enumerate(cycles)}
    basin = np.bincount(owner, minlength=len(cycles))
    rng = np.random.default_rng(0)
    starts = [c["ids"][0] for c in g["largest"][:args.cycles]]
    starts += [int(t) for t in rng.choice(len(nxt), size=args.random, replace=False)]
    sm, tok = loop.sm, loop.sm.tok
    prefix = loop.prefix_ids[0].tolist()
    results = []
    tic = time.perf_counter()
    steer_t = torch.as_tensor(steer, dtype=loop.dtype, device=loop.device)[None, None, :]
    for b0 in range(0, len(starts), args.batch):
        batch = starts[b0:b0 + args.batch]
        ids = torch.tensor([prefix + [t] for t in batch], device=loop.device)
        sm.steer = steer_t.repeat(len(batch), 1, 1)
        try:
            with torch.no_grad():
                out = loop.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.length,
                                          do_sample=False, temperature=None, top_p=None, top_k=None,
                                          repetition_penalty=1.0, pad_token_id=tok.eos_token_id,
                                          eos_token_id=None)
        finally:
            sm.steer = None
        for t0, row in zip(batch, out[:, len(prefix):].tolist()):
            lp = final_loop(row)
            rec = {"start": int(t0), "start_cycle": int(owner[t0]), "loop": None, "match_cycle": None}
            if lp is not None:
                key = canonical(lp)
                rec.update(loop=[int(t) for t in key], period=len(key), text="".join(tok.decode([t]) for t in key),
                           match_cycle=canon.get(key), same_as_start_cycle=canon.get(key) == int(owner[t0]))
            results.append(rec)
    looped = [r for r in results if r["loop"] is not None]
    matched = [r for r in looped if r["match_cycle"] is not None]
    summary = {"args": vars(args), "seconds": round(time.perf_counter() - tic, 1), "runs": len(results),
               "looped": len(looped), "loop_is_a_cycle_of_g": len(matched),
               "loop_is_the_start_tokens_cycle": sum(1 for r in matched if r["same_as_start_cycle"]),
               "matched_cycles": {str(i): int(basin[i]) for i in sorted({r["match_cycle"] for r in matched})},
               "runs_detail": results}
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "decode.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("runs", "looped", "loop_is_a_cycle_of_g",
                                              "loop_is_the_start_tokens_cycle")}))


def kdecode(args):
    """Greedy decoding with a K-token memory after the fixed prefix (the soft-token map's T -> 0 limit with a K-token
    window): at each step the model sees the prefix and the last K generated tokens only. Same starts as decode; the
    final loops are compared with the one-token cycles (graph) and with the growing-context loops (decode)."""
    import explore_llm_attractors as ela

    loop, steer = setup(args)
    torch = loop.torch
    gdir = ela.DATA / "explore" / args.graph
    g = json.loads((gdir / "graph.json").read_text())
    if abs(g["args"]["alpha"] - args.alpha) > 0 or g["args"]["concept"] != args.concept:
        raise SystemExit("graph computed for another steering")
    nxt = np.load(gdir / "graph.npz")["next"]
    cycles, _ = cycles_of(nxt)
    canon1 = {canonical(c) for c in cycles}
    full = json.loads((ela.DATA / "explore" / args.decoded / "decode.json").read_text())
    starts = [r["start"] for r in full["runs_detail"]]
    full_loop = {r["start"]: (tuple(r["loop"]) if r["loop"] else None) for r in full["runs_detail"]}
    tok = loop.sm.tok
    loop.prepare(steer[None])
    base, steer1 = loop.cache, loop.steer_rows
    E = loop.E
    res = {}
    tic = time.perf_counter()
    for K in args.memories:
        rows = []
        for b0 in range(0, len(starts), args.batch):
            batch = starts[b0:b0 + args.batch]
            b = len(batch)
            c = copy.deepcopy(base)
            c.batch_repeat_interleave(b)
            loop.cache, loop.steer_rows = c, steer1.repeat(b, 1, 1)
            seq = [[t] for t in batch]
            with torch.no_grad():
                for _ in range(args.length):
                    win = [s_[-K:] for s_ in seq]
                    L = len(win[0])
                    X = E[torch.as_tensor(win, device=loop.device)]            # (b, L, d): hard tokens
                    z = loop.logits(X)
                    nt = z.argmax(-1).tolist()
                    for s_, t in zip(seq, nt):
                        s_.append(int(t))
                    if L < K:                                                 # window still filling: keep cache
                        pass
            rows.extend(seq)
        looped = 0
        is_cycle1 = 0
        same_as_full = 0
        texts = {}
        for t0, s_ in zip(starts, rows):
            lp = final_loop(s_[1:])
            if lp is None:
                continue
            looped += 1
            key = canonical(lp)
            is_cycle1 += key in canon1
            same_as_full += key == full_loop.get(t0)
            txt = "".join(tok.decode([t]) for t in key)
            texts[txt] = texts.get(txt, 0) + 1
        res[str(K)] = {"runs": len(rows), "looped": looped, "loop_is_a_one_token_cycle": is_cycle1,
                       "loop_equals_growing_context_loop": same_as_full,
                       "top_loops": sorted(texts.items(), key=lambda kv: -kv[1])[:8]}
        print(K, json.dumps({k: v for k, v in res[str(K)].items() if k != "top_loops"}), flush=True)
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "kdecode.json").write_text(json.dumps({"args": vars(args), "seconds": round(time.perf_counter() - tic, 1),
                                                  "growing_context_looped": full["looped"], "by_memory": res},
                                                 indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("graph", "decode", "kdecode"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="qwen0.5b")
        p.add_argument("--layer", type=int, default=12)
        p.add_argument("--threads", type=int, default=16)
        p.add_argument("--prompt", default="open")
        p.add_argument("--opener", default=None)
        p.add_argument("--vectors", default=None)
        p.add_argument("--concept", default="philosophy")
        p.add_argument("--alpha", type=float, default=0.0)
        p.add_argument("--dtype", default="float32", choices=("float32", "float64"))
        p.add_argument("--device", default="cpu")
        p.add_argument("--batch", type=int, default=1024)
        p.add_argument("--out", required=True)
    sub.choices["graph"].add_argument("--top", type=int, default=50)
    dp = sub.choices["decode"]
    dp.add_argument("--graph", required=True)
    dp.add_argument("--cycles", type=int, default=20, help="start from the first token of the largest cycles")
    dp.add_argument("--random", type=int, default=200)
    dp.add_argument("--length", type=int, default=256)
    kp = sub.choices["kdecode"]
    kp.add_argument("--graph", required=True)
    kp.add_argument("--decoded", required=True, help="decode output folder (starts and growing-context loops)")
    kp.add_argument("--memories", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    kp.add_argument("--length", type=int, default=256)
    args = ap.parse_args()
    {"graph": graph, "decode": decode, "kdecode": kdecode}[args.cmd](args)


if __name__ == "__main__":
    main()
