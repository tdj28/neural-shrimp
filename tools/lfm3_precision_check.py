"""Precision gates for the float64 LFM3 soft-token map (run LF3), on CPU, before any paid run.

Modes (each in its own process: the float64 patches act on classes, so a stock model must not share a process with a
patched one):
  stock32    logits at the end of the paper's prefix from the stock model in float32
  stock64    the same from the stock model cast to float64 (keeps the code's float32 islands)
  patched64  the same through ela.Engine (tools/lfm3_float64.py patches), then the gates:
             - every floating module output is float64;
             - hidden_states[l + 1] is decoder layer l's output (the concept vectors read it);
             - a cached soft-loop step with the restored prefix state equals full recomputation, and repeats bit
               for bit;
             - central differences of the one-step soft map converge like h^2 down to small h (successive
               differences shrink by (h ratio)^2 until rounding, which a float32 island would reach near 1e-3);
             and compares its logits with the two stock runs.

Usage: python tools/lfm3_precision_check.py MODE --model lfm3-1.7b-base --threads 8 --out llm-soft-loop/lfm3/precision
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_soft_loop as esl  # noqa: E402

H = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)


def prefix_ids(tok):
    messages = [{"role": "system", "content": ela.PROMPTS["open"]}, {"role": "user", "content": ela.OPENER}]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return text, tok(text, add_special_tokens=False)["input_ids"]


def stock(args, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import lfm3_float64

    torch.set_num_threads(args.threads)
    name = ela.MODELS[args.model]
    rev = ela.REVISIONS[name]
    lfm3_float64.verify_remote_code(ela.CACHE / ("models--" + name.replace("/", "--")) / "snapshots" / rev)
    tok = AutoTokenizer.from_pretrained(name, revision=rev, cache_dir=ela.CACHE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, revision=rev, cache_dir=ela.CACHE,
                                                 trust_remote_code=True).eval()
    text, ids = prefix_ids(tok)
    with torch.no_grad():
        z = model(input_ids=torch.tensor([ids])).logits[0, -1].double().numpy()
    return {"prefix_text": text, "prefix_tokens": len(ids), "logits": z}


def patched(args, out):
    import time

    import torch

    tic = time.perf_counter()
    loop = esl.SoftLoop(args.model, 0, args.threads, "open", ela.OPENER, 1, "float64", "cpu")
    model, E = loop.model, loop.E
    report = {"float64_patch": loop.sm.engine.float64_patch, "prefix_tokens": int(loop.L0),
              "prefix_text": loop.prefix_text, "hybrid": bool(loop.hybrid), "seconds": {}}

    def save(stage):              # partial results after every gate, so a deadline keeps the gates already passed
        report["seconds"][stage] = round(time.perf_counter() - tic, 1)
        (out / f"{args.model}-patched64-partial.json").write_text(
            json.dumps({k: v for k, v in report.items() if k != "logits"}, indent=1, default=str))
        print(stage, report["seconds"][stage], flush=True)

    save("load")

    # dtype audit and hidden-state indexing on the plain prefix forward
    bad, seen, layer12 = set(), [0], {}

    def audit(module, inputs, output):
        outs = output if isinstance(output, (tuple, list)) else (output,)
        for o in outs:
            if isinstance(o, torch.Tensor) and o.is_floating_point():
                seen[0] += 1
                if o.dtype != torch.float64:
                    bad.add(f"{type(module).__name__}:{o.dtype}")

    handles = [m.register_forward_hook(audit) for m in model.modules()]
    handles.append(model.model.layers[12].register_forward_hook(lambda m, i, o: layer12.setdefault("h", o)))
    with torch.no_grad():
        res = model(input_ids=loop.prefix_ids, output_hidden_states=True)
    for h in handles:
        h.remove()
    hs = res.hidden_states
    report["dtype_audit"] = {"floating_outputs": seen[0], "not_float64": sorted(bad)}
    report["hidden_states"] = {"count": len(hs), "layers": len(model.model.layers),
                               "index13_equals_layer12_output": float((hs[13] - layer12["h"]).abs().max())}
    z_prefix = res.logits[0, -1].double().cpu().numpy()
    report["logits"] = z_prefix
    save("dtype_audit")

    # cached step against full recomputation, at a soft token from the prefix distribution at T = 1
    X0 = loop.prepare(np.zeros((2, E.shape[1])))
    with torch.no_grad():
        p = torch.softmax(torch.as_tensor(z_prefix), dim=-1)
        x = (p @ E).to(loop.state_dtype)
        X = x.repeat(2, 1, 1)
        z1 = loop.logits(X)
        z2 = loop.logits(X)
        emb = model.get_input_embeddings()(loop.prefix_ids).to(loop.dtype)
        full = model(inputs_embeds=torch.cat([emb, x[None, None].to(loop.dtype)], dim=1)).logits[0, -1].double()
    report["cached_step"] = {"vs_full_recompute_rel": float((z1[0] - full).abs().max() / full.abs().max()),
                             "repeat_bitwise_equal": bool(torch.equal(z1, z2)),
                             "rows_equal": bool(torch.equal(z1[0], z1[1]))}
    save("cached_step")

    # central differences of the one-step map at states on short orbits; one prefix pass serves every step size
    # (the loop restores the prefix state after each step)
    log_ts = list(args.log_temperature)
    n = len(log_ts)
    loop.prepare(np.zeros((2 * n, E.shape[1])))
    Xs = X0[:1].repeat(n, 1, 1)
    temps = 10.0 ** np.asarray(log_ts)
    for _ in range(args.transient):            # the cache holds 2n rows: step the states twice over
        Y, _ = loop.step(torch.cat([Xs, Xs]), np.tile(temps, 2))
        Xs = Y[:n]
    save("transient")
    rng = np.random.default_rng(1)
    u = torch.as_tensor(rng.normal(size=tuple(Xs.shape)), dtype=loop.state_dtype)
    u = u / u.flatten(1).norm(dim=1)[:, None, None]
    D = []
    for h in H:
        Y, _ = loop.step(torch.cat([Xs + h * u, Xs - h * u]), np.tile(temps, 2))
        D.append((Y[:n, -1] - Y[n:, -1]) / (2 * h))
    cells = []
    for c in range(n):
        diffs = [float((D[i][c] - D[i + 1][c]).norm()) for i in range(len(H) - 1)]
        cells.append({"log10_temperature": log_ts[c], "state_norm": float(Xs[c].norm()),
                      "derivative_norm": float(D[-3][c].norm()), "successive_differences": diffs,
                      "expected_truncation_ratios": [(H[i] / H[i + 1]) ** 2 for i in range(len(H) - 2)],
                      "observed_ratios": [diffs[i] / diffs[i + 1] if diffs[i + 1] > 0 else None
                                          for i in range(len(diffs) - 1)]})
    report["central_differences"] = {"h": list(H), "cells": cells}
    save("central_differences")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("stock32", "stock64", "patched64"))
    ap.add_argument("--model", default="lfm3-1.7b-base", choices=[k for k in ela.MODELS if k.startswith("lfm3")])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--log-temperature", type=float, nargs="+", default=[-0.5, 0.0, 0.3])
    ap.add_argument("--transient", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    import torch

    rec = stock(args, torch.float32) if args.mode == "stock32" else (
        stock(args, torch.float64) if args.mode == "stock64" else patched(args, out))
    np.save(out / f"{args.model}-{args.mode}-logits.npy", rec.pop("logits"))
    if args.mode == "patched64":
        z = np.load(out / f"{args.model}-patched64-logits.npy")
        scale = float(np.abs(z).max())
        for other in ("stock32", "stock64"):
            f = out / f"{args.model}-{other}-logits.npy"
            if f.exists():
                rec[f"patched64_vs_{other}_rel"] = float(np.abs(np.load(f) - z).max() / scale)
    (out / f"{args.model}-{args.mode}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("central_differences", "prefix_text")},
                     default=str)[:3000])
    if args.mode == "patched64":
        for c in rec["central_differences"]["cells"]:
            print(c["log10_temperature"], ["%.1e" % d for d in c["successive_differences"]])


if __name__ == "__main__":
    main()
