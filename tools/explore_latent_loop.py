"""Exploratory: the latent recurrence of a looped language model as a dynamical system.

Ouro (ByteDance, looped LM) applies one stack of 24 shared decoder layers
repeatedly to a latent state, h <- RMSNorm(Stack(h)), starting from the token
embeddings; it is trained to loop 4 times. For a fixed prompt this is an
autonomous smooth map F of R^(L x d). Iterating F well beyond 4 loops shows its
long-run dynamics: convergence to a fixed point, cycles, drift or chaos. The
model's own head reads out the next-token distribution after every loop.

Runs in the separate environment ../neural-shrimp-data/envs/llm (transformers
4.55, see tools/llm_env/requirements.txt).

Subcommands
  trajectory  Iterate F for a list of prompts: per loop the step size
              |h_{k+1} - h_k|, the readout's top token and entropy, and the
              largest Lyapunov exponent from a renormalized twin trajectory.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
CACHE = DATA / "models" / "hf" / "hub"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
MODELS = {"ouro1.4b": "ByteDance/Ouro-1.4B"}
PROMPTS = {
    "hello": "Hello! How are you today?",
    "arith": "What is 17 times 23? The answer is",
    "story": "Once upon a time, in a small village by the sea,",
    "conscious": "What is consciousness? Consciousness is",
    "list": "1, 2, 3, 4, 5,",
}


class LatentLoop:
    def __init__(self, model, dtype, device, threads):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(int(threads))
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.tok = AutoTokenizer.from_pretrained(MODELS[model], cache_dir=CACHE, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODELS[model], cache_dir=CACHE, trust_remote_code=True,
                                                          torch_dtype=self.dtype).to(self.device).eval()
        self.inner = self.model.model

    def setup(self, prompt):
        """Embeddings, causal mask and rotary embeddings of one prompt; returns h0 (1, L, d)."""
        torch = self.torch
        from transformers.masking_utils import create_causal_mask

        ids = self.tok(prompt, return_tensors="pt")["input_ids"].to(self.device)
        with torch.no_grad():
            emb = self.inner.embed_tokens(ids).to(self.dtype)
        L = emb.shape[1]
        self.cache_position = torch.arange(L, device=self.device)
        self.position_ids = self.cache_position.unsqueeze(0)
        self.mask = create_causal_mask(config=self.inner.config, input_embeds=emb, attention_mask=None,
                                       cache_position=self.cache_position, past_key_values=None,
                                       position_ids=self.position_ids)
        self.pos_emb = self.inner.rotary_emb(emb, self.position_ids)
        return emb

    def F(self, h):
        """One loop of the shared stack followed by the final norm (the model's own recurrence)."""
        torch = self.torch
        with torch.no_grad():
            x = h
            for layer in self.inner.layers[: self.inner.config.num_hidden_layers]:
                x = layer(x, attention_mask=self.mask, position_ids=self.position_ids, past_key_value=None,
                          use_cache=False, cache_position=self.cache_position, position_embeddings=self.pos_emb)
            return self.inner.norm(x)

    def readout(self, h):
        torch = self.torch
        with torch.no_grad():
            z = self.model.lm_head(h[:, -1, :]).double()
        logp = torch.log_softmax(z, dim=-1)
        return int(torch.argmax(z[0])), float(-(logp.exp() * logp).sum(-1)[0])


def trajectory(args):
    loop = LatentLoop(args.model, args.dtype, args.device, args.threads)
    torch = loop.torch
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.prompts:
        h = loop.setup(PROMPTS[name])
        rng = np.random.default_rng(0)
        d = torch.as_tensor(rng.normal(size=tuple(h.shape)), dtype=loop.dtype, device=loop.device)
        g = h + d / d.norm() * args.eps
        tic = time.perf_counter()
        rows, log_growth = [], 0.0
        for k in range(1, args.loops + 1):
            h_new = loop.F(h)
            g_new = loop.F(g)
            step = float((h_new - h).norm() / h.norm())
            dist = float((g_new - h_new).norm())
            log_growth += np.log(max(dist, 1e-300) / args.eps)
            g = h_new + (g_new - h_new) * (args.eps / max(dist, 1e-300))
            h = h_new
            token, entropy = loop.readout(h)
            rows.append({"loop": k, "relative_step": step, "top_token": token,
                         "top_text": loop.tok.decode([token]), "entropy": entropy,
                         "lyapunov_running": log_growth / k})
        tail = rows[-args.tail:]
        rec = {"prompt": name, "text": PROMPTS[name], "seconds": round(time.perf_counter() - tic, 1),
               "final_relative_step": rows[-1]["relative_step"],
               "lyapunov": rows[-1]["lyapunov_running"],
               "tail_distinct_top_tokens": sorted(set(r["top_text"] for r in tail)),
               "readout_after_native_4_loops": rows[3]["top_text"] if len(rows) >= 4 else None, "rows": rows}
        records.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "rows"}, ensure_ascii=False))
    (out / "trajectory.json").write_text(json.dumps({"args": vars(args), "records": records}, indent=1,
                                                    ensure_ascii=False))


def check(args):
    """The extracted loop must reproduce the native forward: 4 applications of F give the model's logits."""
    loop = LatentLoop(args.model, args.dtype, args.device, args.threads)
    torch = loop.torch
    for name in args.prompts:
        h = loop.setup(PROMPTS[name])
        for _ in range(loop.inner.config.total_ut_steps):
            h = loop.F(h)
        ids = loop.tok(PROMPTS[name], return_tensors="pt")["input_ids"].to(loop.device)
        with torch.no_grad():
            native = loop.model(input_ids=ids, use_cache=False).logits[0, -1].double()
            mine = loop.model.lm_head(h[:, -1, :])[0].double()
        print(json.dumps({"prompt": name, "max_abs_logit_difference": float((native - mine).abs().max()),
                          "native_top": loop.tok.decode([int(native.argmax())]),
                          "loop_top": loop.tok.decode([int(mine.argmax())])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("trajectory")
    tr.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    tr.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    tr.add_argument("--device", default="cpu")
    tr.add_argument("--threads", type=int, default=8)
    tr.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    tr.add_argument("--loops", type=int, default=64)
    tr.add_argument("--eps", type=float, default=1e-4)
    tr.add_argument("--tail", type=int, default=16)
    tr.add_argument("--out", required=True)
    ck = sub.add_parser("check")
    ck.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    ck.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    ck.add_argument("--device", default="cpu")
    ck.add_argument("--threads", type=int, default=8)
    ck.add_argument("--prompts", nargs="+", default=["hello", "arith"])
    args = parser.parse_args()
    {"trajectory": trajectory, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
