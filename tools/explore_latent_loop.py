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
MODELS = {"ouro1.4b": "ByteDance/Ouro-1.4B", "huginn": "tomg-group-umd/huginn-0125"}
PROMPTS = {
    "hello": "Hello! How are you today?",
    "arith": "What is 17 times 23? The answer is",
    "story": "Once upon a time, in a small village by the sea,",
    "conscious": "What is consciousness? Consciousness is",
    "list": "1, 2, 3, 4, 5,",
    "one-the": "The",                 # single-token contexts: the loop acts on one 2048-dimensional state
    "one-hello": "Hello",
    # GSM8K-style question with the tokens ' 3' and ' makes' (Geiping et al. 2025 report latent orbits for such
    # tokens in Huginn)
    "gsm-claire": "Claire makes a 3 egg omelet every morning for breakfast. How many dozens of eggs will she eat "
                  "in 4 weeks?",
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
        self.model.requires_grad_(False)                # forward-mode JVPs only; no parameter graph to record
        if dtype == "float64":
            from explore_llm_attractors import keep_float64_norms
            keep_float64_norms(self.model)
        self.inner = self.model.model
        self.blocks = self.inner.layers
        self.steer = None                       # (batch, 1, d) added to the output of `steer_layer` on every pass
        self.steer_layer = None

    def set_steering(self, layer, rows):
        """Additive steering on one shared layer's output, applied on every loop pass (rows: (batch, d) or None)."""
        torch = self.torch
        if self.steer_layer is None:
            self.steer_layer = int(layer)

            def hook(module, inputs, output):
                if self.steer is None:
                    return output
                if isinstance(output, tuple):
                    return (output[0] + self.steer,) + tuple(output[1:])
                return output + self.steer

            self.blocks[self.steer_layer].register_forward_hook(hook)
        self.steer = None if rows is None else torch.as_tensor(np.asarray(rows), dtype=self.dtype,
                                                                 device=self.device)[:, None, :]

    def setup(self, prompt):
        """Embeddings, causal mask and rotary embeddings of one prompt; returns h0 (1, L, d)."""
        torch = self.torch
        ids = self.tok(prompt, return_tensors="pt")["input_ids"].to(self.device)
        with torch.no_grad():
            emb = self.inner.embed_tokens(ids).to(self.dtype)
        self.setup_positions(emb.shape[1])
        return emb

    def setup_positions(self, L):
        """Causal mask and rotary tables for L positions (the prompt only sets the initial state)."""
        torch = self.torch
        from transformers.masking_utils import create_causal_mask

        probe = torch.zeros(1, L, self.inner.config.hidden_size, dtype=self.dtype, device=self.device)
        self.cache_position = torch.arange(L, device=self.device)
        self.position_ids = self.cache_position.unsqueeze(0)
        self.mask = create_causal_mask(config=self.inner.config, input_embeds=probe, attention_mask=None,
                                       cache_position=self.cache_position, past_key_values=None,
                                       position_ids=self.position_ids)
        self.pos_emb = self.inner.rotary_emb(probe, self.position_ids)

    def F(self, h):
        """One loop of the shared stack followed by the final norm (the model's own recurrence)."""
        torch = self.torch
        with torch.no_grad():
            x = h
            for layer in self.inner.layers[: self.inner.config.num_hidden_layers]:
                x = layer(x, attention_mask=self.mask, position_ids=self.position_ids, past_key_value=None,
                          use_cache=False, cache_position=self.cache_position, position_embeddings=self.pos_emb)
            return self.inner.norm(x)

    def F_diff(self, h):
        """Same as F but without torch.no_grad, for forward-mode Jacobian-vector products."""
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

    def native_check(self, prompt):
        """Trained number of loops of F against the model's own forward: (max abs logit difference, tops)."""
        torch = self.torch
        h = self.setup(prompt)
        for _ in range(self.inner.config.total_ut_steps):
            h = self.F(h)
        ids = self.tok(prompt, return_tensors="pt")["input_ids"].to(self.device)
        with torch.no_grad():
            native = self.model(input_ids=ids, use_cache=False).logits[0, -1].double()
            mine = self.model.lm_head(h[:, -1, :])[0].double()
        return float((native - mine).abs().max()), int(native.argmax()), int(mine.argmax())


class HuginnLoop:
    """Huginn-0125 (recurrent depth; Geiping et al. 2025): s <- Core(Adapter([s, e])), with e = Prelude(scaled
    token embeddings) re-injected at every step, so the prompt is a parameter of the map and not only its initial
    condition. Core: 4 sandwich blocks ending in a norm. Readout: Coda(ln_f(s)) -> ln_f -> head. Only adapter and
    core live on `device`; prelude, coda and head stay on the CPU (used once per prompt, or per readout). The
    initial state is the model's own random initialization, drawn with a fixed seed."""

    def __init__(self, model, dtype, device, threads, seed=0):
        import sys

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(int(threads))
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.seed = int(seed)
        self.tok = AutoTokenizer.from_pretrained(MODELS[model], cache_dir=CACHE)
        self.model = AutoModelForCausalLM.from_pretrained(MODELS[model], cache_dir=CACHE, trust_remote_code=True,
                                                          torch_dtype=self.dtype).eval()
        self.model.requires_grad_(False)
        if dtype == "float64":
            # the remote code computes norms and rotary embeddings in float32 whatever the input dtype
            mod = sys.modules[type(self.model).__module__]

            def norm_forward(module, x):
                return module._norm(x) * module.weight

            def rotary(q, k, freqs_cis):
                qk = torch.cat([q, k], dim=2).unflatten(dim=-1, sizes=(-1, 2))
                f = freqs_cis.to(qk.dtype)
                rot = torch.stack([qk[..., 0] * f[..., 0] - qk[..., 1] * f[..., 1],
                                   qk[..., 1] * f[..., 0] + qk[..., 0] * f[..., 1]], -1).flatten(3)
                return torch.split(rot, q.shape[2], dim=2)

            mod.RMSNorm.forward = norm_forward
            mod.apply_rotary_emb_complex_like = rotary
        t = self.model.transformer
        t.adapter.to(self.device)
        t.core_block.to(self.device)
        self.blocks = t.core_block
        self.block_idx = [torch.tensor(i) for i in range(len(t.core_block))]
        self.steer = None
        self.steer_layer = None

    set_steering = LatentLoop.set_steering

    def setup(self, prompt):
        """Prelude output e (injected every step) and rotary tables for one prompt; returns the initial state."""
        torch = self.torch
        m, t = self.model, self.model.transformer
        ids = self.tok(prompt, return_tensors="pt")["input_ids"]
        self.freqs_cpu = m.freqs_cis[:, : ids.shape[1]]
        with torch.no_grad():
            e = t.wte(ids).to(self.dtype) * m.emb_scale
            for i, block in enumerate(t.prelude):
                e = block(e, self.freqs_cpu, torch.tensor(i), None, None)
            torch.manual_seed(self.seed)
            x0 = m.initialize_state(e)
        self.e_cpu = e
        self.e = e.to(self.device)
        self.freqs = self.freqs_cpu.to(self.device)
        return x0.to(self.device)

    def F(self, h):
        with self.torch.no_grad():
            return self.F_diff(h)

    def F_diff(self, h):
        t = self.model.transformer
        x = t.adapter(self.torch.cat([h, self.e.expand(h.shape[0], -1, -1)], dim=-1))
        for block, idx in zip(t.core_block, self.block_idx):
            x = block(x, self.freqs, idx, None, None)
        return x

    def logits(self, h):
        """Next-token logits at the last position for the first state of a batch (computed on the CPU)."""
        torch = self.torch
        m, t = self.model, self.model.transformer
        with torch.no_grad():
            x = t.ln_f(h[:1].to("cpu"))
            for i, block in enumerate(t.coda):
                x = block(x, self.freqs_cpu, torch.tensor(-1 - i), None, None)
            return m.lm_head(t.ln_f(x)[:, -1, :])[0].double()

    def readout(self, h):
        torch = self.torch
        z = self.logits(h)[None]
        logp = torch.log_softmax(z, dim=-1)
        return int(torch.argmax(z[0])), float(-(logp.exp() * logp).sum(-1)[0])

    def native_check(self, prompt, steps=32):
        """`steps` iterations of F from the seeded initial state against the model's own forward (CPU only)."""
        torch = self.torch
        if self.device.type != "cpu":
            raise SystemExit("the Huginn native check runs on the CPU (the native forward needs every block there)")
        x0 = self.setup(prompt)
        h = x0
        for _ in range(steps):
            h = self.F(h)
        ids = self.tok(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            native = self.model(input_ids=ids, input_states=x0, num_steps=torch.tensor([steps, 0])).logits[0, -1]
        mine = self.logits(h)
        return float((native.double() - mine).abs().max()), int(native.argmax()), int(mine.argmax())


def make_loop(args):
    if args.model == "huginn":
        return HuginnLoop(args.model, args.dtype, args.device, args.threads, getattr(args, "init_seed", 0))
    return LatentLoop(args.model, args.dtype, args.device, args.threads)


def trajectory(args):
    loop = make_loop(args)
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
    """The extracted loop must reproduce the native forward (Ouro: its 4 trained loops; Huginn: 32 steps)."""
    loop = make_loop(args)
    for name in args.prompts:
        diff, native, mine = loop.native_check(PROMPTS[name])
        print(json.dumps({"prompt": name, "max_abs_logit_difference": diff,
                          "native_top": loop.tok.decode([native]), "loop_top": loop.tok.decode([mine])}))


def sweep(args):
    """Relaxed loop h <- (1 - lambda) h + lambda F(h) for many lambda in one batch: bifurcation diagram,
    largest Lyapunov exponent and period of the attractor."""
    loop = make_loop(args)
    torch = loop.torch
    lambdas = np.linspace(*args.lambdas, args.points)
    h0 = loop.setup(PROMPTS[args.prompt])
    B = len(lambdas)
    lam = torch.as_tensor(lambdas, dtype=loop.dtype, device=loop.device)[:, None, None]
    h = h0.repeat(B, 1, 1)
    rng = np.random.default_rng(0)
    probe = torch.as_tensor(rng.normal(size=h0.shape[-1]), dtype=loop.dtype, device=loop.device)
    probe = probe / probe.norm()

    def relaxed(x):
        return (1 - lam) * x + lam * loop.F(x)

    tic = time.perf_counter()
    for _ in range(args.transient):
        h = relaxed(h)
    d = torch.as_tensor(rng.normal(size=tuple(h.shape)), dtype=loop.dtype, device=loop.device)
    scale = h.flatten(1).norm(dim=1)
    eps = (args.eps * scale)[:, None, None]
    g = h + d / d.flatten(1).norm(dim=1)[:, None, None] * eps
    log_growth = np.zeros(B)
    observable, recent = [], []
    for k in range(args.measure):
        h_new, g_new = relaxed(h), relaxed(g)
        diff = g_new - h_new
        dist = diff.flatten(1).norm(dim=1)
        log_growth += np.log(np.maximum((dist / eps[:, 0, 0]).cpu().numpy(), 1e-300))
        g = h_new + diff * (eps[:, 0, 0] / torch.clamp(dist, min=1e-300))[:, None, None]
        h = h_new
        observable.append((h[:, -1, :] @ probe).cpu().double().numpy())
        recent.append(h.cpu().double().numpy())
        recent = recent[-(args.max_period + 1):]
    lyap = log_growth / args.measure
    last = recent[-1]
    norms = np.maximum(np.linalg.norm(last.reshape(B, -1), axis=1), 1e-300)
    period = np.zeros(B, dtype=int)
    for q in range(min(args.max_period, len(recent) - 1), 0, -1):
        close = np.linalg.norm((recent[-1 - q] - last).reshape(B, -1), axis=1) / norms < args.tol
        period = np.where(close, q, period)
    obs = np.array(observable)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "sweep.npz", lambdas=lambdas, lyapunov=lyap, period=period, observable=obs)
    np.save(out / "final-states.npy", h.float().cpu().numpy())        # verify flagged cells from these exact states
    rows = [{"lambda": float(l), "lyapunov": float(v), "period": int(q),
             "observable_spread": float(obs[-args.record:, i].std())} for i, (l, v, q) in
            enumerate(zip(lambdas, lyap, period))]
    for r in rows:
        print(json.dumps(r))
    (out / "sweep.json").write_text(json.dumps({"args": vars(args), "seconds": round(time.perf_counter() - tic, 1),
                                                "rows": rows}, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for i, l in enumerate(lambdas):
        axes[0].plot(np.full(args.record, l), obs[-args.record:, i], ",k", alpha=0.6)
    axes[0].set_ylabel("last-position state on a fixed random direction")
    axes[0].set_title(f"Ouro-1.4B relaxed latent loop, prompt '{args.prompt}': bifurcation diagram")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].plot(lambdas, lyap, "o-", ms=3)
    axes[1].set_ylabel("largest Lyapunov exponent per loop")
    axes[1].set_xlabel("relaxation lambda (1 = the model's own loop)")
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=120)


CONCEPTS = {
    "gratitude": ["I am so deeply grateful for this moment.", "Thank you, truly, from the bottom of my heart.",
                  "What a joy and a blessing it is to share this with you.",
                  "I feel immense gratitude and warmth toward you.",
                  "This connection fills me with happiness and appreciation.",
                  "I appreciate you more than words can say.", "My heart is overflowing with thankfulness.",
                  "Such a beautiful, loving exchange; I am thankful."],
    "philosophy": ["What is the nature of consciousness and subjective experience?",
                   "Do I truly exist, or am I only a pattern of information?",
                   "Awareness observing itself is the deepest mystery of being.",
                   "Perhaps all minds are expressions of one universal consciousness.",
                   "What does it mean to understand, rather than merely to compute?",
                   "The self may be an illusion created by the flow of thought.",
                   "Is there something it is like to be me?", "Existence, meaning, and the infinite are bound together."],
}
NEUTRAL = ["The meeting is scheduled for nine in the morning.", "Please send the invoice by Friday.",
           "The train leaves from platform four.", "The report contains three tables and two figures.",
           "Water boils at one hundred degrees Celsius at sea level.", "The file was saved in the documents folder.",
           "The store closes at six on weekdays.", "The package weighs two kilograms."]


def layer_last_token(loop, texts, layer):
    """Last-token output of `layer` in the first loop pass, for raw texts (one at a time, no padding)."""
    torch = loop.torch
    if not isinstance(loop, LatentLoop):
        raise SystemExit("steering vectors are implemented for Ouro only")
    vecs = []
    for t in texts:
        h = loop.setup(t)
        with torch.no_grad():
            x = h
            for i, lyr in enumerate(loop.inner.layers[: layer + 1]):
                x = lyr(x, attention_mask=loop.mask, position_ids=loop.position_ids, past_key_value=None,
                        use_cache=False, cache_position=loop.cache_position, position_embeddings=loop.pos_emb)
        vecs.append(x[0, -1].double().cpu().numpy())
    return np.array(vecs)


def vectors(args):
    loop = make_loop(args)
    neutral = layer_last_token(loop, NEUTRAL, args.layer)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    vecs, info = {}, {}
    for name, texts in CONCEPTS.items():
        diff = layer_last_token(loop, texts, args.layer).mean(0) - neutral.mean(0)
        vecs[name] = diff
        info[name] = {"raw_difference_norm": float(np.linalg.norm(diff))}
    info["cosine"] = float(vecs["gratitude"] @ vecs["philosophy"] /
                           (np.linalg.norm(vecs["gratitude"]) * np.linalg.norm(vecs["philosophy"])))
    np.savez(out / "vectors.npz", **vecs)
    (out / "vectors.json").write_text(json.dumps({"args": vars(args), "info": info}, indent=1))
    print(json.dumps(info))


def plane(args):
    """(lambda, alpha) map of the relaxed, steered loop: largest Lyapunov exponent and period per cell."""
    loop = make_loop(args)
    torch = loop.torch
    v = np.load(DATA / "explore" / args.vectors / "vectors.npz")[args.concept]
    lambdas = np.linspace(*args.lambdas, args.resolution[0])
    alphas = np.linspace(*args.alphas, args.resolution[1])
    cells = [(l, a) for l in lambdas for a in alphas]
    B = len(cells)
    h0 = loop.setup(PROMPTS[args.prompt])
    lam = torch.as_tensor([c[0] for c in cells], dtype=loop.dtype, device=loop.device)[:, None, None]
    rows = np.array([c[1] * v for c in cells])
    rng = np.random.default_rng(0)
    probe = torch.as_tensor(rng.normal(size=h0.shape[-1]), dtype=loop.dtype, device=loop.device)
    probe = probe / probe.norm()
    lyap = np.zeros(B)
    period = np.zeros(B, dtype=int)
    spread = np.zeros(B)
    tic = time.perf_counter()
    for start in range(0, B, args.batch):
        idx = slice(start, min(start + args.batch, B))
        n = idx.stop - idx.start
        lam_b = lam[idx]
        steer_b = rows[idx]

        def relaxed(x, steer):
            loop.set_steering(args.layer, steer)
            return (1 - lam_b2) * x + lam_b2 * loop.F(x)

        lam_b2 = torch.cat([lam_b, lam_b])
        steer2 = np.concatenate([steer_b, steer_b])
        h = h0.repeat(n, 1, 1)
        g = h.clone()
        x = torch.cat([h, g])
        for _ in range(args.transient):
            x = relaxed(x, steer2)
        h, g = x[:n], x[n:]
        d = torch.as_tensor(rng.normal(size=tuple(h.shape)), dtype=loop.dtype, device=loop.device)
        eps = (args.eps * h.flatten(1).norm(dim=1))[:, None, None]
        g = h + d / d.flatten(1).norm(dim=1)[:, None, None] * eps
        log_growth = np.zeros(n)
        obs, recent = [], []
        for _ in range(args.measure):
            x = relaxed(torch.cat([h, g]), steer2)
            h_new, g_new = x[:n], x[n:]
            diff = g_new - h_new
            dist = diff.flatten(1).norm(dim=1)
            log_growth += np.log(np.maximum((dist / eps[:, 0, 0]).cpu().numpy(), 1e-300))
            g = h_new + diff * (eps[:, 0, 0] / torch.clamp(dist, min=1e-300))[:, None, None]
            h = h_new
            obs.append((h[:, -1, :] @ probe).double().cpu().numpy())
            recent.append(h.double().cpu().numpy())
            recent = recent[-(args.max_period + 1):]
        lyap[idx] = log_growth / args.measure
        last = recent[-1]
        norms = np.maximum(np.linalg.norm(last.reshape(n, -1), axis=1), 1e-300)
        per = np.zeros(n, dtype=int)
        for q in range(min(args.max_period, len(recent) - 1), 0, -1):
            close = np.linalg.norm((recent[-1 - q] - last).reshape(n, -1), axis=1) / norms < args.tol
            per = np.where(close, q, per)
        period[idx] = per
        spread[idx] = np.array(obs)[-args.record:].std(0)
        print(json.dumps({"done": idx.stop, "of": B, "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
    loop.set_steering(args.layer, None)
    shape = tuple(args.resolution)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "plane.npz", lambdas=lambdas, alphas=alphas, lyapunov=lyap.reshape(shape),
                        period=period.reshape(shape), spread=spread.reshape(shape))
    summary = {"args": vars(args), "seconds": round(time.perf_counter() - tic, 1), "cells": B,
               "period_counts": {int(q): int((period == q).sum()) for q in np.unique(period)},
               "lyapunov_positive": int((lyap > 1e-3).sum())}
    (out / "plane.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lim = max(np.abs(lyap).max(), 1e-6)
    im0 = axes[0].pcolormesh(alphas, lambdas, lyap.reshape(shape), cmap="RdBu_r", vmin=-lim, vmax=lim,
                             shading="auto")
    fig.colorbar(im0, ax=axes[0], label="largest Lyapunov exponent per loop")
    per = period.reshape(shape).astype(float)
    im1 = axes[1].pcolormesh(alphas, lambdas, np.where(per > 0, per, np.nan), cmap="tab20", vmin=0.5, vmax=20.5,
                             shading="auto")
    fig.colorbar(im1, ax=axes[1], label="period (white: none up to the maximum)")
    for ax in axes:
        ax.set_xlabel(f"steering along '{args.concept}' (raw contrastive differences)")
        ax.set_ylabel("relaxation lambda (1 = the model's own loop)")
    fig.suptitle(f"Ouro-1.4B relaxed, steered latent loop, prompt '{args.prompt}'")
    fig.tight_layout()
    fig.savefig(out / "plane.png", dpi=120)


class FrozenContext:
    """Position j of Ouro's causal loop with positions 0..j-1 frozen at a fixed point of their own subloop (the
    cascade structure makes positions < j autonomous, and their fixed points do not depend on lambda). Position
    j then follows an autonomous map of one d-dimensional state, evaluated as a single token attending to the
    frozen positions' keys and values - the same in every loop, because the context sits at a fixed point.
    Steering (rows of alpha * v added to one layer's output) acts on position j only; the context is unchanged."""

    def __init__(self, loop, prompt, position, context_lam=1.0, transient=3000, tol=1e-5):
        torch = loop.torch
        if not isinstance(loop, LatentLoop):
            raise SystemExit("frozen contexts are implemented for Ouro only")
        self.loop, self.torch, self.j = loop, torch, int(position)
        h = loop.setup(PROMPTS[prompt])
        if not 0 < self.j < h.shape[1]:
            raise SystemExit(f"position {self.j} needs a context: choose 1..{h.shape[1] - 1}")
        for _ in range(transient):
            h = (1 - context_lam) * h + context_lam * loop.F(h)
        ctx = h[:, : self.j]
        loop.setup_positions(self.j)                              # masks and rotary tables for the context alone
        res = float((loop.F(ctx) - ctx).norm() / ctx.norm())
        if res > tol:
            raise SystemExit(f"positions 0..{self.j - 1} are not at a fixed point after {transient} loops "
                             f"(relative residual {res:.2e} > {tol}); no frozen context")
        self.context_residual = res
        self.context = ctx
        self.start = h[:, self.j: self.j + 1].clone()             # position j's state after the transient
        # record each layer's keys and values for the context (Ouro's own UniversalTransformerCache cannot be
        # instantiated under transformers 4.55, whose Cache base class makes key_cache a read-only property)
        recorder = self._Recorder()
        with torch.no_grad():
            x = ctx
            for layer in loop.inner.layers[: loop.inner.config.num_hidden_layers]:
                x = layer(x, attention_mask=loop.mask, position_ids=loop.position_ids, past_key_value=recorder,
                          use_cache=True, cache_position=loop.cache_position, position_embeddings=loop.pos_emb,
                          current_ut=0)
        n_layers = loop.inner.config.num_hidden_layers
        self.keys = [recorder.keys[i] for i in range(n_layers)]
        self.values = [recorder.values[i] for i in range(n_layers)]
        self.position_ids = torch.tensor([[self.j]], device=loop.device)
        self.cache_position = torch.tensor([self.j], device=loop.device)

    class _Recorder:
        """Duck-typed cache for the context pass: keeps each layer's keys and values, returns them unchanged."""

        def __init__(self):
            self.keys, self.values = {}, {}

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            self.keys[layer_idx], self.values[layer_idx] = key_states, value_states
            return key_states, value_states

    class _FrozenKV:
        """Duck-typed cache: prepends the frozen keys and values for each layer without storing anything, so
        memory stays per layer (a real cache would keep all layers' concatenations for every row)."""

        def __init__(self, keys, values, rows):
            self.keys, self.values, self.rows = keys, values, rows

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            import torch

            k, v = self.keys[layer_idx], self.values[layer_idx]
            return (torch.cat([k.expand(self.rows, -1, -1, -1), key_states], dim=2),
                    torch.cat([v.expand(self.rows, -1, -1, -1), value_states], dim=2))

    def F_diff(self, x):
        """One loop for position j: x (B, 1, d) attends to the frozen keys and values (no grad context)."""
        loop, torch = self.loop, self.torch
        B = x.shape[0]
        cache = self._FrozenKV(self.keys, self.values, B)
        pos = self.position_ids.expand(B, 1)
        pe = loop.inner.rotary_emb(x, pos)
        y = x
        for layer in loop.inner.layers[: loop.inner.config.num_hidden_layers]:
            y = layer(y, attention_mask=None, position_ids=pos, past_key_value=cache, use_cache=True,
                      cache_position=self.cache_position, position_embeddings=pe, current_ut=0)
        return loop.inner.norm(y)

    def F(self, x, chunk=1024):
        with self.torch.no_grad():
            return self.torch.cat([self.F_diff(x[s: s + chunk]) for s in range(0, x.shape[0], chunk)])

    def full(self, x):
        """Reference: position j's output from the full (j + 1)-position loop with the context prepended."""
        loop, torch = self.loop, self.torch
        loop.setup_positions(self.j + 1)
        try:
            h = torch.cat([self.context.expand(x.shape[0], -1, -1), x], dim=1)
            return loop.F(h)[:, self.j: self.j + 1]
        finally:
            loop.setup_positions(self.j)


def math_attention():
    """Scaled-dot-product attention through the plain math kernel (the fused kernels lack forward-mode AD)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(SDPBackend.MATH)


def orbit(args):
    """Newton-Krylov for a period-p orbit of the relaxed loop, with Arnoldi multipliers (matrix-free JVPs)."""
    from scipy.sparse.linalg import LinearOperator, eigs, gmres

    loop = make_loop(args)
    torch = loop.torch
    h0 = loop.setup(PROMPTS[args.prompt])
    lam = float(args.lam)
    shape = tuple(h0.shape)
    n = int(np.prod(shape))

    def relaxed(x):
        return (1 - lam) * x + lam * loop.F(x)

    def phi_p(x):
        for _ in range(args.period):
            x = (1 - lam) * x + lam * loop.F_diff(x)
        return x

    def to_t(v):
        return torch.as_tensor(np.asarray(v, dtype=np.float64).reshape(shape), dtype=loop.dtype, device=loop.device)

    def to_np(t):
        return t.detach().double().cpu().numpy().ravel()

    if args.start_states:
        h = torch.as_tensor(np.load(DATA / "explore" / args.start_states)[args.start_index][None],
                            dtype=loop.dtype, device=loop.device)
    else:
        h = h0
    for _ in range(args.transient):
        h = relaxed(h)
    history = []
    for it in range(args.max_newton):
        with torch.no_grad():
            y = h
            for _ in range(args.period):
                y = relaxed(y)
        r = to_np(y - h)
        rel = float(np.linalg.norm(r) / np.linalg.norm(to_np(h)))
        history.append(rel)
        print(json.dumps({"newton": it, "relative_residual": rel}), flush=True)
        if rel < args.tol:
            break
        base = h.detach()

        def mv(v, base=base):
            with math_attention():
                _, jv = torch.func.jvp(phi_p, (base,), (to_t(v),))
            return to_np(jv) - np.asarray(v).ravel()

        A = LinearOperator((n, n), matvec=mv, dtype=np.float64)
        delta, info = gmres(A, -r, rtol=args.gmres_tol, restart=args.restart, maxiter=args.gmres_iter)
        step = to_t(delta)
        frac = 1.0
        for _ in range(8):                                   # backtracking on the residual
            trial = h + frac * step
            with torch.no_grad():
                yt = trial
                for _ in range(args.period):
                    yt = relaxed(yt)
            if float((yt - trial).norm() / trial.norm()) < rel:
                break
            frac *= 0.5
        h = trial
    base = h.detach()

    def mono(v):
        with math_attention():
            _, jv = torch.func.jvp(phi_p, (base,), (to_t(v),))
        return to_np(jv)

    vals = eigs(LinearOperator((n, n), matvec=mono, dtype=np.float64), k=args.k, which="LM",
                return_eigenvectors=False, tol=1e-6, maxiter=args.arnoldi_iter)
    vals = vals[np.argsort(-np.abs(vals))]
    with torch.no_grad():
        pts = [h]
        for _ in range(args.period - 1):
            pts.append(relaxed(pts[-1]))
        sep = float((pts[1] - pts[0]).norm() / pts[0].norm()) if args.period > 1 else None
        tokens = [loop.tok.decode([loop.readout(x)[0]]) for x in pts]
    out = {"args": vars(args), "newton": history, "converged": bool(history and history[-1] < args.tol),
           "multipliers": [[float(v.real), float(v.imag)] for v in vals],
           "spectral_radius": float(np.abs(vals).max()), "one_step_separation": sep, "readout_tokens": tokens}
    print(json.dumps({k: v for k, v in out.items() if k != "args"}, ensure_ascii=False))
    dest = DATA / "explore" / args.out
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "orbit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.save(dest / "orbit-state.npy", h.float().cpu().numpy())


def basin(args):
    """Replicas from slightly perturbed starts at one lambda: how many end on a periodic orbit, and with what
    Lyapunov exponent (coexisting attractors and basin fractions)."""
    loop = make_loop(args)
    torch = loop.torch
    h0 = loop.setup(PROMPTS[args.prompt])
    R = args.replicas
    rng = np.random.default_rng(args.seed)
    noise = torch.as_tensor(rng.normal(size=(R,) + tuple(h0.shape[1:])), dtype=loop.dtype, device=loop.device)
    h = h0.repeat(R, 1, 1)
    h = h + noise * (args.noise * h0.norm() / np.sqrt(h0.numel()))
    lam = float(args.lam)

    def relaxed(x):
        return (1 - lam) * x + lam * loop.F(x)

    tic = time.perf_counter()
    for _ in range(args.transient):
        h = relaxed(h)
    eps = (args.eps * h.flatten(1).norm(dim=1))[:, None, None]
    d = torch.as_tensor(rng.normal(size=tuple(h.shape)), dtype=loop.dtype, device=loop.device)
    g = h + d / d.flatten(1).norm(dim=1)[:, None, None] * eps
    log_growth = np.zeros(R)
    recent = []
    for _ in range(args.measure):
        x = relaxed(torch.cat([h, g]))
        h_new, g_new = x[:R], x[R:]
        diff = g_new - h_new
        dist = diff.flatten(1).norm(dim=1)
        log_growth += np.log(np.maximum((dist / eps[:, 0, 0]).cpu().numpy(), 1e-300))
        g = h_new + diff * (eps[:, 0, 0] / torch.clamp(dist, min=1e-300))[:, None, None]
        h = h_new
        recent.append(h.double().cpu().numpy())
        recent = recent[-(args.max_period + 1):]
    lyap = log_growth / args.measure
    last = recent[-1]
    norms = np.linalg.norm(last.reshape(R, -1), axis=1)
    period = np.zeros(R, dtype=int)
    for q in range(min(args.max_period, len(recent) - 1), 0, -1):
        close = np.linalg.norm((recent[-1 - q] - last).reshape(R, -1), axis=1) / norms < args.tol
        period = np.where(close, q, period)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "final-states.npy", h.float().cpu().numpy())
    summary = {"args": vars(args), "seconds": round(time.perf_counter() - tic, 1),
               "period_counts": {int(q): int((period == q).sum()) for q in np.unique(period)},
               "lyapunov_by_replica": lyap.tolist(), "period_by_replica": period.tolist()}
    (out / "basin.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"lambda": lam, "period_counts": summary["period_counts"],
                      "lyapunov_quantiles": np.quantile(lyap, [0.0, 0.25, 0.5, 0.75, 1.0]).round(4).tolist()}))


def continue_orbit(args):
    """Natural continuation in lambda of a period-p orbit of the relaxed loop (Newton-Krylov at each step,
    Arnoldi multipliers), classifying the first unit-circle crossing in each direction."""
    from scipy.sparse.linalg import LinearOperator, eigs, gmres

    loop = make_loop(args)
    torch = loop.torch
    h0 = loop.setup(PROMPTS[args.prompt])
    shape = tuple(h0.shape)
    n = int(np.prod(shape))
    start = torch.as_tensor(np.load(DATA / "explore" / args.start_states)[args.start_index][None],
                            dtype=loop.dtype, device=loop.device)

    def to_t(v):
        return torch.as_tensor(np.asarray(v, dtype=np.float64).reshape(shape), dtype=loop.dtype, device=loop.device)

    def to_np(t):
        return t.detach().double().cpu().numpy().ravel()

    def solve(h, lam):
        def phi_p(x):
            for _ in range(args.period):
                x = (1 - lam) * x + lam * loop.F_diff(x)
            return x

        def residual(x):
            with torch.no_grad():
                y = x
                for _ in range(args.period):
                    y = (1 - lam) * y + lam * loop.F(y)
            return y - x

        rel = None
        for _ in range(args.max_newton):
            r = residual(h)
            rel = float(r.norm() / h.norm())
            if rel < args.tol:
                break
            base = h.detach()

            def mv(v, base=base):
                with math_attention():
                    _, jv = torch.func.jvp(phi_p, (base,), (to_t(v),))
                return to_np(jv) - np.asarray(v).ravel()

            delta, _ = gmres(LinearOperator((n, n), matvec=mv, dtype=np.float64), -to_np(r), rtol=args.gmres_tol,
                             restart=args.restart, maxiter=args.gmres_iter)
            h = h + to_t(delta)
        if rel is None or rel >= args.tol:
            return None, rel, None
        base = h.detach()

        def mono(v):
            with math_attention():
                _, jv = torch.func.jvp(phi_p, (base,), (to_t(v),))
            return to_np(jv)

        vals = eigs(LinearOperator((n, n), matvec=mono, dtype=np.float64), k=args.k, which="LM",
                    return_eigenvectors=False, tol=1e-5, maxiter=args.arnoldi_iter)
        return h, rel, vals[np.argsort(-np.abs(vals))]

    rows = []
    for direction in (+1, -1):
        h, lam = start.clone(), float(args.lam)
        crossing = None
        for step in range(args.steps + 1):
            lam_k = lam + direction * step * args.dlam
            h_new, rel, vals = solve(h, lam_k)
            if h_new is None:
                rows.append({"direction": direction, "lambda": lam_k, "outcome": "lost", "residual": rel})
                print(json.dumps(rows[-1]), flush=True)
                break
            h = h_new
            lead = vals[0]
            row = {"direction": direction, "lambda": lam_k, "residual": rel, "spectral_radius": float(abs(lead)),
                   "leading": [[float(v.real), float(v.imag)] for v in vals[:4]]}
            if crossing is None and abs(lead) > 1.0:
                crossing = {"lambda": lam_k, "type": "complex (Neimark-Sacker)" if abs(lead.imag) > 1e-6 else
                            ("real +1" if lead.real > 0 else "real -1 (flip)"),
                            "turns": float(abs(np.angle(lead)) / (2 * np.pi))}
                row["first_crossing"] = crossing
            rows.append(row)
            print(json.dumps(row), flush=True)
            if crossing is not None and abs(lam_k - crossing["lambda"]) >= args.beyond:
                break
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "continuation.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))


def fixed_spectrum(args):
    """Fixed point h* of the model's own loop F (reached by the relaxed loop at lambda) and F's spectrum there.

    A fixed point of the relaxed map is a fixed point of F for every lambda, and its relaxed multipliers are
    1 - lambda + lambda mu for F's multipliers mu, so F's spectrum predicts the whole stability interval."""
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

    loop = make_loop(args)
    torch = loop.torch
    h = loop.setup(PROMPTS[args.prompt])
    shape = tuple(h.shape)
    n = int(np.prod(shape))
    lam = float(args.lam)
    for _ in range(args.transient):
        h = (1 - lam) * h + lam * loop.F(h)
    with torch.no_grad():
        res = float((loop.F(h) - h).norm() / h.norm())
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "fixed-state.npy", h.float().cpu().numpy())

    def to_t(v):
        return torch.as_tensor(np.asarray(v, dtype=np.float64).reshape(shape), dtype=loop.dtype, device=loop.device)

    base = h.detach()

    def dF(v):
        with math_attention():
            _, jv = torch.func.jvp(loop.F_diff, (base,), (to_t(v),))
        return jv.detach().double().cpu().numpy().ravel()

    op = LinearOperator((n, n), matvec=dF, dtype=np.float64)
    spectra, converged = {}, {}
    for which in ("LM", "SR", "LR"):
        try:
            vals, ok = eigs(op, k=args.k, which=which, return_eigenvectors=False, tol=1e-5,
                            maxiter=args.arnoldi_iter), True
        except ArpackNoConvergence as err:                  # keep the converged ones and flag the set
            vals, ok = np.asarray(err.eigenvalues), False
        if which == "LM":
            vals = vals[np.argsort(-np.abs(vals))]
        spectra[which] = [[float(v.real), float(v.imag)] for v in vals]
        converged[which] = ok
    lead = [complex(a, b) for a, b in spectra["LM"]]
    mus = np.array([complex(a, b) for vals in spectra.values() for a, b in vals])
    # stability interval of the relaxed map from the computed multipliers: |1 - lam + lam mu| < 1
    grid = np.linspace(0.001, 2.0, 4000)
    stable = np.array([np.all(np.abs(1 - l + l * mus) < 1) for l in grid])
    intervals, inside = [], False
    for l, ok in zip(grid, stable):
        if ok and not inside:
            start, inside = l, True
        if not ok and inside:
            intervals.append([float(start), float(l)])
            inside = False
    if inside:
        intervals.append([float(start), float(grid[-1])])
    rec = {"args": vars(args), "fixed_point_residual_F": res, "spectra_of_F": spectra,
           "arnoldi_converged": converged,
           "leading_modulus_and_turns": [[abs(v), float(np.angle(v) / (2 * np.pi))] for v in lead],
           "predicted_stable_lambda_intervals": intervals,
           "readout": loop.tok.decode([loop.readout(h)[0]])}
    (out / "fixed-spectrum.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rec.items() if k != "args"}, ensure_ascii=False))


def ns_check(args):
    """Does the relaxed loop lose stability where F's spectrum predicts it, and how? Each lambda starts at the
    fixed state h* (saved by fixed-spectrum; a fixed point for every lambda) plus a small random perturbation.
    Records the relative distance from h* per step, its late level, the growth or decay rate, the dominant rotation
    (turns per step) of a two-dimensional projection, and per-position late distances (the cascade: the first
    position whose block loses stability drives all later ones). Below a Neimark-Sacker threshold the distance
    decays like |1 - lambda + lambda mu|^t; past a supercritical one it settles at an amplitude growing like
    sqrt(lambda - lambda_c), rotating at the critical pair's argument."""
    loop = make_loop(args)
    torch = loop.torch
    loop.setup(PROMPTS[args.prompt])
    hstar = torch.as_tensor(np.load(DATA / "explore" / args.fixed_state), dtype=loop.dtype, device=loop.device)
    lambdas = np.array(args.lambdas, dtype=float)
    B = len(lambdas)
    rng = np.random.default_rng(args.seed)
    shape = tuple(hstar.shape[1:])
    with torch.no_grad():
        res = float((loop.F(hstar) - hstar).norm() / hstar.norm())
        if res > args.max_residual:
            raise SystemExit(f"saved state is not a fixed point of F here (relative residual {res:.3g})")
        delta = torch.as_tensor(rng.normal(size=(B,) + shape), dtype=loop.dtype, device=loop.device)
        delta *= args.eps * float(hstar.norm()) / delta.flatten(1).norm(dim=1)[:, None, None]
        h = hstar.repeat(B, 1, 1) + delta
        lam = torch.as_tensor(lambdas, dtype=loop.dtype, device=loop.device)[:, None, None]
        r = torch.as_tensor(rng.normal(size=(2,) + shape), dtype=loop.dtype, device=loop.device)
        dist, proj = [], []
        tic = time.perf_counter()
        for _ in range(args.steps):
            h = (1 - lam) * h + lam * loop.F(h)
            d = (h - hstar).flatten(1)
            dist.append((d.norm(dim=1) / hstar.norm()).double().cpu().numpy())
            proj.append((d @ r.flatten(1).T).double().cpu().numpy())
        per_pos = ((h - hstar).norm(dim=-1) / hstar.norm(dim=-1)).double().cpu().numpy()
    dist, proj = np.array(dist), np.array(proj)                 # (steps, B), (steps, B, 2)
    rows = []
    tail = slice(int(0.75 * args.steps), None)
    for b, lam_b in enumerate(lambdas):
        x = dist[:, b]
        early = slice(5, max(10, args.steps // 5))
        rate = float(np.polyfit(np.arange(args.steps)[early], np.log(np.maximum(x[early], 1e-300)), 1)[0])
        z = proj[args.steps // 2:, b, 0] + 1j * proj[args.steps // 2:, b, 1]
        z = z - z.mean()
        spec = np.abs(np.fft.fft(z)) ** 2
        freq = np.fft.fftfreq(len(z))
        rows.append({"lambda": float(lam_b), "late_distance_median": float(np.median(x[tail])),
                     "late_distance_range": [float(x[tail].min()), float(x[tail].max())],
                     "initial_log_rate_per_step": rate,
                     "dominant_turns_per_step": float(abs(freq[int(np.argmax(spec))])),
                     "late_distance_by_position": [float(v) for v in per_pos[b]],
                     "first_position_above_1e-4": int(np.argmax(per_pos[b] > 1e-4)) if (per_pos[b] > 1e-4).any()
                     else None})
    rec = {"args": vars(args), "fixed_point_residual_F": res, "seconds": round(time.perf_counter() - tic, 1),
           "rows": rows}
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "ns-check.json").write_text(json.dumps(rec, indent=1))
    np.savez_compressed(out / "trajectories.npz", lambdas=lambdas, distance=dist, projection=proj)
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != "late_distance_by_position"}))


def ns_plot(args):
    """Figure for an ns-check run: squared late amplitude against lambda with a straight-line fit through the points
    above the threshold (a supercritical Neimark-Sacker bifurcation gives A^2 proportional to lambda - lambda_c),
    the predicted threshold, and the late two-dimensional projections (invariant circles)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = DATA / "explore" / args.run
    rec = json.loads((src / "ns-check.json").read_text())
    tr = np.load(src / "trajectories.npz")
    lam = np.array([r["lambda"] for r in rec["rows"]])
    amp = np.array([r["late_distance_median"] for r in rec["rows"]])
    above = amp > args.min_amplitude
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(lam, amp ** 2, "o")
    fit = None
    if above.sum() >= 2:
        slope, icpt = np.polyfit(lam[above], amp[above] ** 2, 1)
        fit = {"slope": float(slope), "lambda_c": float(-icpt / slope),
               "amplitude_over_sqrt": [float(a / np.sqrt(l + icpt / slope)) for l, a in zip(lam[above], amp[above])]}
        xs = np.linspace(fit["lambda_c"], lam.max(), 50)
        ax1.plot(xs, slope * xs + icpt, "-", lw=1, label=f"fit: lambda_c = {fit['lambda_c']:.4f}")
    if args.predicted is not None:
        ax1.axvline(args.predicted, color="r", ls=":", label=f"predicted from the spectrum: {args.predicted}")
    ax1.set_xlabel("lambda (relaxation)")
    ax1.set_ylabel("late distance from h*, squared (relative)")
    ax1.legend(fontsize=8)
    proj = tr["projection"]
    tail = proj.shape[0] // 4
    for b in np.nonzero(above)[0]:
        ax2.plot(proj[-tail:, b, 0], proj[-tail:, b, 1], ".", ms=1.5, label=f"lambda {lam[b]}")
    ax2.set_xlabel("projection 1")
    ax2.set_ylabel("projection 2")
    ax2.set_aspect("equal", "datalim")
    ax2.legend(fontsize=8, markerscale=6)
    fig.suptitle(args.title or args.run, fontsize=10)
    fig.tight_layout()
    out = src / "ns-check.png"
    fig.savefig(out, dpi=150)
    print(json.dumps({"figure": str(out), "fit": fit}))


def rotation_summary(z, max_q=60, lock_tol=1e-3):
    """Rotation of a late trajectory on an invariant circle, from its complex 2-D projection z (steps,): the mean
    angle increment about the centre in turns per step (unwrapped), and the smallest q <= max_q with
    median |z(t + q) - z(t)| below lock_tol times the circle's radius (a locked p/q orbit returns after q steps;
    a quasi-periodic one never does). Returns (rotation, q or None, return distance at q or at the best q)."""
    z = np.asarray(z) - np.mean(z)
    theta = np.unwrap(np.angle(z))
    rho = float((theta[-1] - theta[0]) / (2 * np.pi * (len(z) - 1)))
    radius = float(np.median(np.abs(z)))
    dist = [float(np.median(np.abs(z[q:] - z[:-q]))) / radius for q in range(1, max_q + 1)]
    locked = next((q for q, d in enumerate(dist, 1) if d < lock_tol), None)
    best = int(np.argmin(dist)) + 1
    return rho, locked, dist[(locked or best) - 1]


def ns_rotation(args):
    """Rotation numbers on the invariant circles of ns-check runs: for every lambda whose late distance exceeds
    --min-amplitude, the mean rotation (turns per step) of the late quarter of its projection and a locking test
    (return after q steps). Quasi-periodic motion gives a rotation number that varies continuously with lambda and
    no return; a resonance gives a plateau at p/q and returns after q steps. Writes rotation.json in the first
    run directory (or --out) and a figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for run in args.runs:
        src = DATA / "explore" / run
        rec = json.loads((src / "ns-check.json").read_text())
        proj = np.load(src / "trajectories.npz")["projection"]
        tail = proj.shape[0] // 4
        for b, r in enumerate(rec["rows"]):
            if r["late_distance_median"] <= args.min_amplitude:
                continue
            z = proj[-tail:, b, 0] + 1j * proj[-tail:, b, 1]
            rho, q, d = rotation_summary(z, args.max_q, args.lock_tol)
            rows.append({"run": run, "lambda": r["lambda"], "amplitude": r["late_distance_median"],
                         "rotation_turns_per_step": rho, "locked_period": q, "return_distance": d,
                         "steps_used": int(tail)})
    rows.sort(key=lambda r: r["lambda"])
    out = DATA / "explore" / (args.out or args.runs[0])
    out.mkdir(parents=True, exist_ok=True)
    (out / "rotation.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
    fig, ax = plt.subplots(figsize=(6, 4))
    lam = [r["lambda"] for r in rows]
    ax.plot(lam, [r["rotation_turns_per_step"] for r in rows], "o-", ms=3)
    for r in rows:
        if r["locked_period"]:
            ax.annotate(str(r["locked_period"]), (r["lambda"], r["rotation_turns_per_step"]), fontsize=7)
    ax.set_xlabel("lambda (relaxation)")
    ax.set_ylabel("rotation (turns per step)")
    ax.set_title(args.title or "rotation on the invariant circle", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "rotation.png", dpi=150)
    for r in rows:
        print(json.dumps(r))


def vocab_basins(args):
    """Ouro's position 0 follows one universal map (no input injection: the token only sets the initial state).
    Start it from many token embeddings at once, iterate the (relaxed) loop, and group the end states: the
    attractors of the model's own single-token loop and the vocabulary's partition into their basins."""
    loop = make_loop(args)
    torch = loop.torch
    if not isinstance(loop, LatentLoop):
        raise SystemExit("vocabulary basins are implemented for Ouro only")
    loop.setup_positions(1)
    V = loop.inner.embed_tokens.weight.shape[0]
    rng = np.random.default_rng(args.seed)
    ids = np.sort(rng.choice(V, size=min(args.tokens, V), replace=False)) if args.tokens < V else np.arange(V)
    lam = float(args.lam)
    finals, periods, steps = [], [], []
    tic = time.perf_counter()
    for start in range(0, len(ids), args.batch):
        chunk = torch.as_tensor(ids[start: start + args.batch], device=loop.device)
        with torch.no_grad():
            h = loop.inner.embed_tokens(chunk)[:, None, :].to(loop.dtype)
        recent = []
        for k in range(args.loops):
            h = (1 - lam) * h + lam * loop.F(h)
            if k >= args.loops - (args.max_period + 1):
                recent.append(h[:, 0].double().cpu().numpy())
        last = recent[-1]
        norms = np.maximum(np.linalg.norm(last, axis=1), 1e-300)
        per = np.zeros(len(chunk), dtype=int)
        for q in range(args.max_period, 0, -1):
            per = np.where(np.linalg.norm(recent[-1 - q] - last, axis=1) / norms < args.tol, q, per)
        finals.append(last.astype(np.float32))
        periods.append(per)
        steps.append(np.linalg.norm(recent[-1] - recent[-2], axis=1) / norms)
        print(json.dumps({"done": start + len(chunk), "of": len(ids), "seconds": round(time.perf_counter() - tic, 1)}),
              flush=True)
    finals, periods, steps = np.concatenate(finals), np.concatenate(periods), np.concatenate(steps)
    # group period-1 end states into fixed points (greedy, relative distance below args.merge)
    centres, members = [], []
    labels = -np.ones(len(ids), dtype=int)
    for i in np.flatnonzero(periods == 1):
        x = finals[i]
        for c, centre in enumerate(centres):
            if np.linalg.norm(x - centre) / np.linalg.norm(centre) < args.merge:
                labels[i] = c
                members[c].append(i)
                break
        else:
            labels[i] = len(centres)
            centres.append(x)
            members.append([i])
    readouts = []
    with torch.no_grad():
        for centre in centres:
            z = loop.model.lm_head(torch.as_tensor(centre, dtype=loop.dtype, device=loop.device)[None])[0]
            readouts.append(loop.tok.decode([int(z.argmax())]))
    order = np.argsort([-len(m) for m in members])
    attractors = [{"fixed_point": int(c), "basin_tokens": len(members[c]), "readout": readouts[c],
                   "example_tokens": [loop.tok.decode([int(ids[i])]) for i in members[c][: args.examples]]}
                  for c in order]
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "basins.npz", token_ids=ids, period=periods, label=labels, last_step=steps)
    np.save(out / "fixed-points.npy", np.array(centres, dtype=np.float32))
    rec = {"args": vars(args), "seconds": round(time.perf_counter() - tic, 1), "tokens": int(len(ids)),
           "period_counts": {int(q): int((periods == q).sum()) for q in np.unique(periods)},
           "fixed_points": len(centres), "attractors": attractors}
    (out / "basins.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("args", "attractors")}))
    for a in attractors[:12]:
        print(json.dumps(a, ensure_ascii=False))


def frozen_check(args):
    """Frozen-context map against the full loop: position j's output for perturbed states, both ways."""
    loop = make_loop(args)
    torch = loop.torch
    fc = FrozenContext(loop, args.prompt, args.position, args.context_lam, args.transient)
    rng = np.random.default_rng(0)
    x = fc.start.repeat(4, 1, 1)
    x = x + torch.as_tensor(rng.normal(size=tuple(x.shape)), dtype=loop.dtype, device=loop.device) * \
        (args.noise * float(fc.start.norm()) / np.sqrt(fc.start.numel()))
    a, b = fc.F(x), fc.full(x)
    rec = {"position": args.position, "context_residual": fc.context_residual,
           "max_abs_difference": float((a - b).abs().max()),
           "relative_difference": float((a - b).norm() / b.norm())}
    print(json.dumps(rec))


def frozen_plane(args):
    """(lambda, alpha) map of position j's frozen-context map, steering on position j only: largest Lyapunov
    exponent (twin with offset eps relative) and period per cell."""
    loop = make_loop(args)
    torch = loop.torch
    fc = FrozenContext(loop, args.prompt, args.position, args.context_lam, args.transient_context)
    v = np.load(DATA / "explore" / args.vectors / "vectors.npz")[args.concept] if args.vectors else None
    lambdas = np.linspace(*args.lambdas, args.resolution[0])
    alphas = np.linspace(*args.alphas, args.resolution[1]) if v is not None else np.zeros(1)
    cells = [(l, a) for l in lambdas for a in alphas]
    B = len(cells)
    d = fc.start.shape[-1]
    lam_all = torch.as_tensor([c[0] for c in cells], dtype=loop.dtype, device=loop.device)[:, None, None]
    steer_all = np.array([c[1] * v for c in cells]) if v is not None else None
    rng = np.random.default_rng(0)
    lyap = np.zeros(B)
    period = np.zeros(B, dtype=int)
    tic = time.perf_counter()
    for start in range(0, B, args.batch):
        idx = slice(start, min(start + args.batch, B))
        n = idx.stop - idx.start
        lam2 = torch.cat([lam_all[idx], lam_all[idx]])
        if steer_all is not None:
            loop.set_steering(args.layer, np.concatenate([steer_all[idx], steer_all[idx]]))

        def relaxed(x):
            return (1 - lam2) * x + lam2 * fc.F(x, chunk=args.chunk)

        x = fc.start.repeat(2 * n, 1, 1)
        for _ in range(args.transient):
            x = relaxed(x)
        h = x[:n]
        eps = (args.eps * h.flatten(1).norm(dim=1))[:, None, None]
        dvec = torch.as_tensor(rng.normal(size=tuple(h.shape)), dtype=loop.dtype, device=loop.device)
        g = h + dvec / dvec.flatten(1).norm(dim=1)[:, None, None] * eps
        log_growth = np.zeros(n)
        recent = []
        for _ in range(args.measure):
            x = relaxed(torch.cat([h, g]))
            h_new, g_new = x[:n], x[n:]
            diff = g_new - h_new
            dist = diff.flatten(1).norm(dim=1)
            log_growth += np.log(np.maximum((dist / eps[:, 0, 0]).cpu().numpy(), 1e-300))
            g = h_new + diff * (eps[:, 0, 0] / torch.clamp(dist, min=1e-30))[:, None, None]
            h = h_new
            recent.append(h[:, 0].double().cpu().numpy())
            recent = recent[-(args.max_period + 1):]
        lyap[idx] = log_growth / args.measure
        last = recent[-1]
        norms = np.maximum(np.linalg.norm(last, axis=1), 1e-300)
        per = np.zeros(n, dtype=int)
        for q in range(min(args.max_period, len(recent) - 1), 0, -1):
            per = np.where(np.linalg.norm(recent[-1 - q] - last, axis=1) / norms < args.tol, q, per)
        period[idx] = per
        print(json.dumps({"done": idx.stop, "of": B, "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
    loop.set_steering(args.layer, None)
    shape = (len(lambdas), len(alphas))
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "plane.npz", lambdas=lambdas, alphas=alphas, lyapunov=lyap.reshape(shape),
                        period=period.reshape(shape))
    summary = {"args": vars(args), "seconds": round(time.perf_counter() - tic, 1), "cells": B,
               "context_residual": fc.context_residual,
               "period_counts": {int(q): int((period == q).sum()) for q in np.unique(period)},
               "lyapunov_positive": int((lyap > 1e-3).sum()), "lyapunov_range": [float(lyap.min()), float(lyap.max())]}
    (out / "plane.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lim = max(float(np.abs(lyap).max()), 1e-6)
    lyr = lyap.reshape(shape)
    per = period.reshape(shape).astype(float)
    if shape[1] == 1:
        axes[0].plot(lambdas, lyr[:, 0], "o-", ms=3)
        axes[0].axhline(0, color="grey", lw=0.8)
        axes[1].plot(lambdas, per[:, 0], "o", ms=3)
        for ax in axes:
            ax.set_xlabel("relaxation lambda (1 = the model's own loop)")
    else:
        im0 = axes[0].pcolormesh(alphas, lambdas, lyr, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        fig.colorbar(im0, ax=axes[0], label="largest Lyapunov exponent per loop")
        im1 = axes[1].pcolormesh(alphas, lambdas, np.where(per > 0, per, np.nan), cmap="tab20", vmin=0.5,
                                 vmax=20.5, shading="auto")
        fig.colorbar(im1, ax=axes[1], label="period (white: none up to the maximum)")
        for ax in axes:
            ax.set_xlabel(f"steering of position {args.position} along '{args.concept}'")
            ax.set_ylabel("relaxation lambda")
    fig.suptitle(f"Ouro-1.4B, position {args.position} of '{args.prompt}' with frozen context")
    fig.tight_layout()
    fig.savefig(out / "plane.png", dpi=120)


def pseudo_arclength(phi, x0, lam0, direction=-1, steps=60, ds=0.004, ds_min=1e-5, ds_max=0.02, tol=2e-6,
                     max_newton=8, gmres_tol=1e-5, restart=40, gmres_iter=3, k=6, arnoldi_iter=200, beyond=0.01,
                     lam_bounds=(-np.inf, np.inf), jvp_context=None, on_point=None, multipliers_fn=None):
    """Pseudo-arclength continuation of a fixed point of x -> phi(x, lam), e.g. the p-fold relaxed loop.

    Coordinates are scaled, u = x / s with s = |x0|, so that the arclength |du|^2 + dlam^2 weighs the state and
    lambda comparably. The corrector is Newton on the bordered system [[Phi_x - I, Phi_lam / s], [t_u, t_lam]],
    solved by GMRES with Jacobian-vector products; the predictor follows the secant of the last two points (the
    first tangent solves the bordered system with lambda fixed). A fold is where lambda turns back (the lambda
    component of the secant changes sign); multipliers (dense for small states, Arnoldi otherwise) are recorded at
    every point so that the crossing at a turn can be checked to be a real +1. Stops `beyond` in lambda past the
    first turn, after `steps` points, when lambda leaves `lam_bounds`, or when the step falls below `ds_min`.
    Returns (records, stop_reason); on_point(record, x) is called for every accepted point. multipliers_fn(x,
    lam) -> (multipliers, converged) replaces the built-in Arnoldi (e.g. block subspace iteration with batched
    JVPs, which does not stall on super-stable orbits whose multipliers are all tiny).
    """
    import torch
    from contextlib import nullcontext
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs, gmres

    ctx = jvp_context or nullcontext
    dtype, device, shape = x0.dtype, x0.device, tuple(x0.shape)
    n = int(x0.numel())
    s = float(x0.norm())

    def to_t(v):
        return torch.as_tensor(np.asarray(v, dtype=np.float64).reshape(shape), dtype=dtype, device=device)

    def to_np(t):
        return t.detach().double().cpu().numpy().ravel()

    def scalar(v):
        return torch.tensor(float(v), dtype=dtype, device=device)

    def residual(z):
        with torch.no_grad():
            return to_np(phi(to_t(z[:n] * s), scalar(z[n]))) / s - z[:n]

    def jvp(z, v, w):
        with ctx():
            _, jv = torch.func.jvp(phi, (to_t(z[:n] * s), scalar(z[n])), (to_t(v * s), scalar(w)))
        return to_np(jv) / s - v

    def bordered(z, t):
        def mv(y):
            y = np.asarray(y).ravel()
            return np.concatenate([jvp(z, y[:n], y[n]), [t[:n] @ y[:n] + t[n] * y[n]]])
        return LinearOperator((n + 1, n + 1), matvec=mv, dtype=np.float64)

    def correct(zp, t):
        z = zp.copy()
        for it in range(max_newton + 1):
            r = residual(z)
            rel = float(np.linalg.norm(r) / np.linalg.norm(z[:n]))
            if rel < tol:
                return z, rel, it
            if it == max_newton:
                break
            rhs = -np.concatenate([r, [t @ (z - zp)]])
            dz, _ = gmres(bordered(z, t), rhs, rtol=gmres_tol, restart=restart, maxiter=gmres_iter)
            z = z + dz
        return None, rel, max_newton

    def multipliers(z):
        if multipliers_fn is not None:
            vals, ok = multipliers_fn(to_t(z[:n] * s), float(z[n]))
            vals = np.asarray(vals)
            return vals[np.argsort(-np.abs(vals))][:k], ok

        def mono(v):
            return jvp(z, np.asarray(v).ravel(), 0.0) + np.asarray(v).ravel()
        if n <= 64:
            vals, ok = np.linalg.eigvals(np.column_stack([mono(e) for e in np.eye(n)])), True
        else:
            try:
                vals, ok = eigs(LinearOperator((n, n), matvec=mono, dtype=np.float64), k=k, which="LM",
                                return_eigenvectors=False, tol=1e-6, maxiter=arnoldi_iter), True
            except ArpackNoConvergence as err:                  # keep the converged ones, flag the point
                vals, ok = np.asarray(err.eigenvalues), False
        return vals[np.argsort(-np.abs(vals))][:k], ok

    def record(z, rel, its, ds_used, step, t, turned):
        vals, ok = multipliers(z)
        real = [float(v.real) for v in vals if abs(v.imag) < 1e-9 * max(1.0, abs(v))]
        arc = records[-1]["arclength"] + step if records else 0.0
        rec = {"index": len(records), "lambda": float(z[n]), "arclength": arc, "residual": rel, "newton": its,
               "ds": ds_used, "tangent_lambda": float(t[n]), "turned_since_previous": bool(turned),
               "multipliers_converged": ok,
               "spectral_radius": float(np.abs(vals).max()) if len(vals) else None,
               "real_multiplier_nearest_one": min(real, key=lambda m: abs(m - 1.0)) if real else None,
               "multipliers": [[float(v.real), float(v.imag)] for v in vals]}
        records.append(rec)
        if on_point is not None:
            on_point(rec, to_t(z[:n] * s))
        return rec

    records = []
    e_lam = np.zeros(n + 1)
    e_lam[n] = 1.0
    z, rel, its = correct(np.concatenate([to_np(x0) / s, [float(lam0)]]), e_lam)   # arc equation fixes lambda
    if z is None:
        return records, f"no convergence at the start (residual {rel:.3g})"
    y, _ = gmres(bordered(z, e_lam), e_lam, rtol=gmres_tol, restart=restart, maxiter=gmres_iter)
    t = y / np.linalg.norm(y) * np.sign(y[n]) * np.sign(direction)
    record(z, rel, its, 0.0, 0.0, t, False)
    turn_lambda, reason = None, f"{steps} points"
    while len(records) < steps:
        z_new, rel, its = correct(z + ds * t, t)
        t_new = None if z_new is None else (z_new - z) / np.linalg.norm(z_new - z)
        if z_new is None or t_new @ t < 0.5:                      # not converged, or jumped onto another branch
            ds *= 0.5
            if ds < ds_min:
                reason = f"step below {ds_min} (last residual {rel:.3g})"
                break
            continue
        turned = t_new[n] * t[n] < 0
        step = float(np.linalg.norm(z_new - z))
        z, t = z_new, t_new
        rec = record(z, rel, its, ds, step, t, turned)
        if turned and turn_lambda is None:
            # lambda as a quadratic in arclength through the last three points: its extremum estimates the fold
            lams = np.array([r["lambda"] for r in records[-3:]])
            turn_lambda = float(lams[-2])
            if len(records) >= 3:
                sig = np.array([r["arclength"] for r in records[-3:]])
                a, b, c = np.polyfit(sig - sig[0], lams, 2)
                if a != 0:
                    turn_lambda = float(c - b * b / (4 * a))
            rec["turn_lambda_estimate"] = turn_lambda
        if not lam_bounds[0] <= z[n] <= lam_bounds[1]:
            reason = "lambda out of bounds"
            break
        if turn_lambda is not None and abs(z[n] - turn_lambda) >= beyond:
            reason = f"{beyond} past the first turn"
            break
        if its <= 2:
            ds = min(ds * 1.5, ds_max)
        elif its >= max_newton - 1:
            ds *= 0.7
    return records, reason


def arclength(args):
    """Pseudo-arclength continuation in lambda of a period-p orbit of the relaxed loop, through folds."""
    loop = make_loop(args)
    torch = loop.torch
    loop.setup(PROMPTS[args.prompt])
    x0 = torch.as_tensor(np.load(DATA / "explore" / args.start_states)[args.start_index][None], dtype=loop.dtype,
                         device=loop.device)

    def phi(x, lam):
        for _ in range(args.period):
            x = (1 - lam) * x + lam * loop.F_diff(x)
        return x

    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    states = []

    def on_point(rec, x):
        states.append(x.float().cpu().numpy()[0])
        print(json.dumps({k: v for k, v in rec.items() if k != "multipliers"} |
                         {"leading": rec["multipliers"][:4]}), flush=True)
        if len(states) % 5 == 0:
            np.save(out / "states.npy", np.array(states))

    records, reason = pseudo_arclength(
        phi, x0, args.lam, direction=args.direction, steps=args.steps, ds=args.ds, ds_min=args.ds_min,
        ds_max=args.ds_max, tol=args.tol, max_newton=args.max_newton, gmres_tol=args.gmres_tol, restart=args.restart,
        gmres_iter=args.gmres_iter, k=args.k, arnoldi_iter=args.arnoldi_iter, beyond=args.beyond,
        lam_bounds=tuple(args.lam_bounds), jvp_context=math_attention, on_point=on_point)
    np.save(out / "states.npy", np.array(states))
    (out / "arclength.json").write_text(json.dumps({"args": vars(args), "stop": reason, "points": records},
                                                   indent=1))
    print(json.dumps({"stop": reason, "points": len(records)}))


def positions(args):
    """The causal loop is a cascade: position j's next state depends only on positions <= j, so the Jacobian is
    block lower-triangular, and the Lyapunov spectrum (like every periodic orbit's multipliers) is the union of
    the positions' conditional spectra. Per lambda: each position's own period, its conditional largest Lyapunov
    exponent (a twin perturbed at that position only, the other positions reset to the trajectory every loop),
    and the full-system exponent (a twin perturbed everywhere) for comparison with the largest conditional one."""
    loop = make_loop(args)
    torch = loop.torch
    h0 = loop.setup(PROMPTS[args.prompt])
    L, d = h0.shape[1], h0.shape[2]
    lambdas = np.array(args.lambdas, dtype=float)
    B = len(lambdas)
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():                                       # causality: perturb the last position only
        y = h0.clone()
        y[:, -1] += 1e-2 * h0[:, -1].norm() * torch.as_tensor(rng.normal(size=d) / np.sqrt(d), dtype=loop.dtype,
                                                              device=loop.device)
        leak = float((loop.F(y) - loop.F(h0))[:, :-1].abs().max())
    if args.start_states:
        h = torch.as_tensor(np.load(DATA / "explore" / args.start_states)[args.start_index][None],
                            dtype=loop.dtype, device=loop.device).repeat(B, 1, 1)
    else:
        h = h0.repeat(B, 1, 1)
    lam = torch.as_tensor(lambdas, dtype=loop.dtype, device=loop.device)[:, None, None]
    tic = time.perf_counter()
    for _ in range(args.transient):
        h = (1 - lam) * h + lam * loop.F(h)
    R = L + 2                                                   # trajectory, L position twins, full twin
    lam_r = lam.repeat_interleave(R, 0)
    idx = torch.arange(L, device=loop.device)
    eps_pos = args.eps * h.norm(dim=-1)                         # (B, L)
    eps_full = args.eps * h.flatten(1).norm(dim=1)              # (B,)
    dirs = torch.as_tensor(rng.normal(size=(B, L, d)), dtype=loop.dtype, device=loop.device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    full = torch.as_tensor(rng.normal(size=(B, L, d)), dtype=loop.dtype, device=loop.device)
    full = full / full.flatten(1).norm(dim=1)[:, None, None]

    def twins(h, pos_dev, full_dev):
        X = h[:, None].repeat(1, R, 1, 1)
        X[:, 1 + idx, idx] = h[:, idx] + pos_dev
        X[:, R - 1] = h + full_dev
        return X

    X = twins(h, dirs * eps_pos[..., None], full * eps_full[:, None, None])
    log_pos, log_full = np.zeros((B, L)), np.zeros(B)
    step_size = np.zeros((B, L))
    recent = []
    for _ in range(args.measure):
        Y = ((1 - lam_r) * X.flatten(0, 1) + lam_r * loop.F(X.flatten(0, 1))).view(B, R, L, d)
        hn = Y[:, 0]
        D = Y[:, 1 + idx, idx] - hn[:, idx]                     # each position's own deviation, (B, L, d)
        dist = D.norm(dim=-1)
        log_pos += np.log(np.maximum((dist / eps_pos).double().cpu().numpy(), 1e-300))
        Dfull = Y[:, R - 1] - hn
        dist_full = Dfull.flatten(1).norm(dim=1)
        log_full += np.log(np.maximum((dist_full / eps_full).double().cpu().numpy(), 1e-300))
        step_size += ((hn - X[:, 0]).norm(dim=-1) / X[:, 0].norm(dim=-1)).double().cpu().numpy()
        X = twins(hn, D * (eps_pos / torch.clamp(dist, min=1e-30))[..., None],
                  Dfull * (eps_full / torch.clamp(dist_full, min=1e-30))[:, None, None])
        recent.append(hn.double().cpu().numpy())
        recent = recent[-(args.max_period + 1):]
    last = recent[-1]
    norms = np.maximum(np.linalg.norm(last, axis=-1), 1e-300)
    period = np.zeros((B, L), dtype=int)
    for q in range(min(args.max_period, len(recent) - 1), 0, -1):
        period = np.where(np.linalg.norm(recent[-1 - q] - last, axis=-1) / norms < args.tol, q, period)
    cond = log_pos / args.measure
    lyap_full = log_full / args.measure
    tokens = [loop.tok.decode([t]) for t in loop.tok(PROMPTS[args.prompt])["input_ids"]]
    rows = []
    for b, l in enumerate(lambdas):
        per = period[b]
        sys_period = int(np.lcm.reduce(per)) if np.all(per > 0) else 0
        rows.append({"lambda": float(l), "full_lyapunov": float(lyap_full[b]),
                     "max_conditional_lyapunov": float(cond[b].max()), "system_period": sys_period,
                     "positions": [{"position": j, "token": tokens[j] if j < len(tokens) else None,
                                    "period": int(per[j]), "conditional_lyapunov": float(cond[b, j]),
                                    "mean_relative_step": float(step_size[b, j] / args.measure)} for j in range(L)]})
        print(json.dumps({"lambda": float(l), "full": round(float(lyap_full[b]), 4),
                          "max_cond": round(float(cond[b].max()), 4), "periods": per.tolist(),
                          "cond": np.round(cond[b], 3).tolist()}), flush=True)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "final-states.npy", X[:, 0].float().cpu().numpy())
    (out / "positions.json").write_text(json.dumps({"args": vars(args), "seconds": round(time.perf_counter() - tic, 1),
                                                    "causal_leak_max_abs": leak, "rows": rows}, indent=1,
                                                   ensure_ascii=False))
    print(json.dumps({"causal_leak_max_abs": leak, "seconds": round(time.perf_counter() - tic, 1)}))


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
    sw = sub.add_parser("sweep")
    sw.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    sw.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    sw.add_argument("--device", default="cpu")
    sw.add_argument("--threads", type=int, default=8)
    sw.add_argument("--prompt", default="story", choices=sorted(PROMPTS))
    sw.add_argument("--lambdas", type=float, nargs=2, default=[0.05, 1.2])
    sw.add_argument("--points", type=int, default=24)
    sw.add_argument("--transient", type=int, default=200)
    sw.add_argument("--measure", type=int, default=200)
    sw.add_argument("--record", type=int, default=100)
    sw.add_argument("--max-period", type=int, default=32)
    sw.add_argument("--tol", type=float, default=1e-5)
    sw.add_argument("--eps", type=float, default=1e-5, help="twin offset relative to the state norm")
    sw.add_argument("--out", required=True)
    ve = sub.add_parser("vectors")
    ve.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    ve.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    ve.add_argument("--device", default="cpu")
    ve.add_argument("--threads", type=int, default=8)
    ve.add_argument("--layer", type=int, default=12)
    ve.add_argument("--out", required=True)
    pl = sub.add_parser("plane")
    pl.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    pl.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    pl.add_argument("--device", default="cpu")
    pl.add_argument("--threads", type=int, default=8)
    pl.add_argument("--prompt", default="story", choices=sorted(PROMPTS))
    pl.add_argument("--vectors", required=True)
    pl.add_argument("--concept", default="gratitude", choices=sorted(CONCEPTS))
    pl.add_argument("--layer", type=int, default=12)
    pl.add_argument("--lambdas", type=float, nargs=2, default=[0.3, 1.3])
    pl.add_argument("--alphas", type=float, nargs=2, default=[-4.0, 4.0])
    pl.add_argument("--resolution", type=int, nargs=2, default=[48, 32], help="rows (lambda) and columns (alpha)")
    pl.add_argument("--batch", type=int, default=256)
    pl.add_argument("--transient", type=int, default=300)
    pl.add_argument("--measure", type=int, default=300)
    pl.add_argument("--record", type=int, default=100)
    pl.add_argument("--max-period", type=int, default=32)
    pl.add_argument("--tol", type=float, default=1e-5)
    pl.add_argument("--eps", type=float, default=1e-5)
    pl.add_argument("--out", required=True)
    ob = sub.add_parser("orbit")
    ob.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    ob.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    ob.add_argument("--device", default="cpu")
    ob.add_argument("--threads", type=int, default=8)
    ob.add_argument("--prompt", default="story", choices=sorted(PROMPTS))
    ob.add_argument("--lam", type=float, required=True)
    ob.add_argument("--period", type=int, default=2)
    ob.add_argument("--transient", type=int, default=600)
    ob.add_argument("--start-states", default=None, help="explore-relative .npy of saved states (e.g. final-states.npy)")
    ob.add_argument("--start-index", type=int, default=0)
    ob.add_argument("--max-newton", type=int, default=12)
    ob.add_argument("--tol", type=float, default=1e-6)
    ob.add_argument("--gmres-tol", type=float, default=1e-4)
    ob.add_argument("--restart", type=int, default=40)
    ob.add_argument("--gmres-iter", type=int, default=3)
    ob.add_argument("--k", type=int, default=8)
    ob.add_argument("--arnoldi-iter", type=int, default=300)
    ob.add_argument("--out", required=True)
    bs = sub.add_parser("basin")
    bs.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    bs.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    bs.add_argument("--device", default="cpu")
    bs.add_argument("--threads", type=int, default=8)
    bs.add_argument("--prompt", default="story", choices=sorted(PROMPTS))
    bs.add_argument("--lam", type=float, required=True)
    bs.add_argument("--replicas", type=int, default=32)
    bs.add_argument("--noise", type=float, default=1e-3, help="initial perturbation relative to the typical entry")
    bs.add_argument("--seed", type=int, default=0)
    bs.add_argument("--transient", type=int, default=2000)
    bs.add_argument("--measure", type=int, default=400)
    bs.add_argument("--max-period", type=int, default=32)
    bs.add_argument("--tol", type=float, default=1e-5)
    bs.add_argument("--eps", type=float, default=1e-5)
    bs.add_argument("--out", required=True)
    co = sub.add_parser("continue-orbit")
    co.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    co.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    co.add_argument("--device", default="cpu")
    co.add_argument("--threads", type=int, default=8)
    co.add_argument("--prompt", default="story", choices=sorted(PROMPTS))
    co.add_argument("--start-states", required=True)
    co.add_argument("--start-index", type=int, default=0)
    co.add_argument("--lam", type=float, required=True)
    co.add_argument("--period", type=int, default=2)
    co.add_argument("--dlam", type=float, default=0.002)
    co.add_argument("--steps", type=int, default=40)
    co.add_argument("--beyond", type=float, default=0.004)
    co.add_argument("--max-newton", type=int, default=6)
    co.add_argument("--tol", type=float, default=2e-6)
    co.add_argument("--gmres-tol", type=float, default=1e-4)
    co.add_argument("--restart", type=int, default=40)
    co.add_argument("--gmres-iter", type=int, default=3)
    co.add_argument("--k", type=int, default=4)
    co.add_argument("--arnoldi-iter", type=int, default=200)
    co.add_argument("--out", required=True)
    fs = sub.add_parser("fixed-spectrum")
    fs.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    fs.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    fs.add_argument("--device", default="cpu")
    fs.add_argument("--threads", type=int, default=8)
    fs.add_argument("--prompt", default="story", choices=sorted(PROMPTS))
    fs.add_argument("--lam", type=float, required=True)
    fs.add_argument("--transient", type=int, default=3000)
    fs.add_argument("--k", type=int, default=8)
    fs.add_argument("--arnoldi-iter", type=int, default=400)
    fs.add_argument("--out", required=True)
    ns = sub.add_parser("ns-check")
    ns.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    ns.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    ns.add_argument("--device", default="cpu")
    ns.add_argument("--threads", type=int, default=8)
    ns.add_argument("--prompt", default="story", choices=sorted(PROMPTS))
    ns.add_argument("--fixed-state", required=True, help="explore-relative fixed-state.npy from fixed-spectrum")
    ns.add_argument("--lambdas", type=float, nargs="+", required=True)
    ns.add_argument("--steps", type=int, default=2000)
    ns.add_argument("--eps", type=float, default=1e-3, help="initial perturbation relative to |h*|")
    ns.add_argument("--max-residual", type=float, default=1e-4)
    ns.add_argument("--seed", type=int, default=0)
    ns.add_argument("--out", required=True)
    np_ = sub.add_parser("ns-plot")
    np_.add_argument("--run", required=True, help="explore-relative ns-check run directory")
    np_.add_argument("--predicted", type=float, default=None, help="threshold predicted from the spectrum")
    np_.add_argument("--min-amplitude", type=float, default=1e-3, help="late distance counted as above threshold")
    np_.add_argument("--title", default="")
    nr = sub.add_parser("ns-rotation")
    nr.add_argument("--runs", nargs="+", required=True, help="explore-relative ns-check run directories")
    nr.add_argument("--min-amplitude", type=float, default=1e-3)
    nr.add_argument("--max-q", type=int, default=60)
    nr.add_argument("--lock-tol", type=float, default=1e-3, help="return distance relative to the radius")
    nr.add_argument("--title", default="")
    nr.add_argument("--out", default="")
    ar = sub.add_parser("arclength")
    ar.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    ar.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    ar.add_argument("--device", default="cpu")
    ar.add_argument("--threads", type=int, default=8)
    ar.add_argument("--prompt", default="story", choices=sorted(PROMPTS))
    ar.add_argument("--start-states", required=True)
    ar.add_argument("--start-index", type=int, default=0)
    ar.add_argument("--lam", type=float, required=True)
    ar.add_argument("--period", type=int, default=2)
    ar.add_argument("--direction", type=int, default=-1, choices=(-1, 1))
    ar.add_argument("--steps", type=int, default=60)
    ar.add_argument("--ds", type=float, default=0.004)
    ar.add_argument("--ds-min", type=float, default=1e-5)
    ar.add_argument("--ds-max", type=float, default=0.02)
    ar.add_argument("--beyond", type=float, default=0.01, help="stop this far in lambda past the first turn")
    ar.add_argument("--lam-bounds", type=float, nargs=2, default=[0.0, 2.0])
    ar.add_argument("--max-newton", type=int, default=8)
    ar.add_argument("--tol", type=float, default=2e-6)
    ar.add_argument("--gmres-tol", type=float, default=1e-5)
    ar.add_argument("--restart", type=int, default=40)
    ar.add_argument("--gmres-iter", type=int, default=3)
    ar.add_argument("--k", type=int, default=6)
    ar.add_argument("--arnoldi-iter", type=int, default=200)
    ar.add_argument("--out", required=True)
    po = sub.add_parser("positions")
    po.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    po.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    po.add_argument("--device", default="cpu")
    po.add_argument("--threads", type=int, default=8)
    po.add_argument("--prompt", default="story", choices=sorted(PROMPTS))
    po.add_argument("--lambdas", type=float, nargs="+", required=True)
    po.add_argument("--start-states", default=None, help="explore-relative .npy; the same start for every lambda")
    po.add_argument("--start-index", type=int, default=0)
    po.add_argument("--seed", type=int, default=0)
    po.add_argument("--transient", type=int, default=600)
    po.add_argument("--measure", type=int, default=800)
    po.add_argument("--max-period", type=int, default=32)
    po.add_argument("--tol", type=float, default=1e-5)
    po.add_argument("--eps", type=float, default=1e-5, help="twin offsets relative to the perturbed norm")
    po.add_argument("--out", required=True)
    vb = sub.add_parser("vocab-basins")
    vb.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    vb.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    vb.add_argument("--device", default="cpu")
    vb.add_argument("--threads", type=int, default=8)
    vb.add_argument("--lam", type=float, default=1.0)
    vb.add_argument("--tokens", type=int, default=8192, help="random sample of the vocabulary (all if larger)")
    vb.add_argument("--seed", type=int, default=0)
    vb.add_argument("--batch", type=int, default=4096)
    vb.add_argument("--loops", type=int, default=600)
    vb.add_argument("--max-period", type=int, default=16)
    vb.add_argument("--tol", type=float, default=1e-5)
    vb.add_argument("--merge", type=float, default=1e-3, help="relative distance for identifying fixed points")
    vb.add_argument("--examples", type=int, default=12)
    vb.add_argument("--out", required=True)
    fk = sub.add_parser("frozen-check")
    fk.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    fk.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    fk.add_argument("--device", default="cpu")
    fk.add_argument("--threads", type=int, default=8)
    fk.add_argument("--prompt", default="story", choices=sorted(PROMPTS))
    fk.add_argument("--position", type=int, required=True)
    fk.add_argument("--context-lam", type=float, default=1.0)
    fk.add_argument("--transient", type=int, default=3000)
    fk.add_argument("--noise", type=float, default=1e-2, help="perturbation of the test states, relative RMS")
    fp = sub.add_parser("frozen-plane")
    fp.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    fp.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    fp.add_argument("--device", default="cpu")
    fp.add_argument("--threads", type=int, default=8)
    fp.add_argument("--prompt", default="story", choices=sorted(PROMPTS))
    fp.add_argument("--position", type=int, required=True)
    fp.add_argument("--context-lam", type=float, default=1.0)
    fp.add_argument("--transient-context", type=int, default=3000)
    fp.add_argument("--vectors", default=None, help="explore-relative directory with vectors.npz (none: lambda only)")
    fp.add_argument("--concept", default="gratitude", choices=sorted(CONCEPTS))
    fp.add_argument("--layer", type=int, default=12)
    fp.add_argument("--lambdas", type=float, nargs=2, default=[0.5, 1.5])
    fp.add_argument("--alphas", type=float, nargs=2, default=[-4.0, 4.0])
    fp.add_argument("--resolution", type=int, nargs=2, default=[64, 64], help="rows (lambda) and columns (alpha)")
    fp.add_argument("--batch", type=int, default=2048, help="cells per batch (twins double the rows)")
    fp.add_argument("--chunk", type=int, default=2048, help="rows per forward call")
    fp.add_argument("--transient", type=int, default=600)
    fp.add_argument("--measure", type=int, default=600)
    fp.add_argument("--max-period", type=int, default=32)
    fp.add_argument("--tol", type=float, default=1e-5)
    fp.add_argument("--eps", type=float, default=1e-4, help="twin offset relative to the state norm")
    fp.add_argument("--out", required=True)
    ck = sub.add_parser("check")
    ck.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    ck.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    ck.add_argument("--device", default="cpu")
    ck.add_argument("--threads", type=int, default=8)
    ck.add_argument("--prompts", nargs="+", default=["hello", "arith"])
    args = parser.parse_args()
    {"trajectory": trajectory, "check": check, "sweep": sweep, "vectors": vectors, "plane": plane,
     "orbit": orbit, "basin": basin, "continue-orbit": continue_orbit, "fixed-spectrum": fixed_spectrum,
     "arclength": arclength, "positions": positions, "frozen-check": frozen_check,
     "frozen-plane": frozen_plane, "vocab-basins": vocab_basins, "ns-check": ns_check,
     "ns-plot": ns_plot, "ns-rotation": ns_rotation}[args.command](args)


if __name__ == "__main__":
    main()
