"""Ordinary generation next to the soft-token loop, at the same cells (exploratory, human request 17 Oct 2026).

For each cell (alpha, log10 T) of Qwen2.5-0.5B-Instruct with the paper's prefix and steering (philosophy direction
added to the output of decoder layer 12 at every position), print two things:
- **The soft loop's text:** the most probable token at each step of the soft-token map, after a transient, from the
  maps' fixed start (the input embedding of the prefix's last token). This is how the paper reads the loops' texts.
- **Ordinary generation under the same conditions:** the same prefix, steering and temperature, but each step
  appends one discrete token to the growing context. It reports greedy decoding (the most probable token) and
  samples drawn from softmax(z / T) with fixed seeds.

Usage: python tools/normal_companion.py --cells docs/companion/cells.json --out docs/companion/companion.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_llm_phase_map as elpm  # noqa: E402
from explore_llm_soft_loop import SoftLoop  # noqa: E402


def generate(loop, steer, T, n_new, mode, seed=None):
    torch, model, tok = loop.torch, loop.model, loop.sm.tok
    loop.sm.steer = steer
    rng = np.random.default_rng(seed)
    ids = []
    try:
        with torch.no_grad():
            out = model(input_ids=loop.prefix_ids, use_cache=True)
            past, z = out.past_key_values, out.logits[0, -1].to(torch.float64)
            for _ in range(n_new):
                if mode == "greedy":
                    t = int(torch.argmax(z))
                else:
                    p = torch.softmax(z / T, dim=-1).cpu().numpy()
                    t = int(rng.choice(len(p), p=p / p.sum()))
                ids.append(t)
                if t == tok.eos_token_id or t in (tok.convert_tokens_to_ids("<|im_end|>"),):
                    break
                out = model(input_ids=torch.tensor([[t]], device=loop.device), past_key_values=past, use_cache=True)
                past, z = out.past_key_values, out.logits[0, -1].to(torch.float64)
    finally:
        loop.sm.steer = None
    return tok.decode(ids, skip_special_tokens=False), ids


def loop_text(loop, steer_row, T, transient, record):
    torch, tok = loop.torch, loop.sm.tok
    X = loop.prepare(steer_row[None, :])
    toks = []
    for k in range(transient + record):
        X, p = loop.step(X, [T])
        if k >= transient:
            toks.append(int(torch.argmax(p[0])))
    return tok.decode(toks), toks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--transient", type=int, default=600)
    ap.add_argument("--record", type=int, default=48)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    cells = json.loads(Path(args.cells).read_text())
    loop = SoftLoop("qwen0.5b", args.layer, args.threads, "open", "Hello!", 1, dtype="float64", device="cpu")
    torch = loop.torch
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    rows = []
    for c in cells:
        a, lt = float(c["alpha"]), float(c["log10_temperature"])
        T = 10.0 ** lt
        steer = torch.as_tensor(a * v, dtype=loop.dtype, device=loop.device)[None, None, :]
        lt_text, lt_ids = loop_text(loop, a * v, T, args.transient, args.record)
        greedy, _ = generate(loop, steer, T, args.new_tokens, "greedy")
        samples = [generate(loop, steer, T, args.new_tokens, "sample", seed=s)[0] for s in range(args.samples)]
        unsteered, _ = generate(loop, None, T, args.new_tokens, "greedy")
        row = {**c, "temperature": T, "loop_text": lt_text, "loop_tokens": lt_ids, "greedy": greedy,
               "samples_at_T": samples, "unsteered_greedy": unsteered}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("alpha", "log10_temperature", "label")}), flush=True)
        print("  LOOP   :", repr(lt_text[:160]), flush=True)
        print("  GREEDY :", repr(greedy[:160]), flush=True)
        for s in samples:
            print("  SAMPLE :", repr(s[:160]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
