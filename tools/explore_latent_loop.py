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
    "one-the": "The",                 # single-token contexts: the loop acts on one 2048-dimensional state
    "one-hello": "Hello",
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

            self.inner.layers[self.steer_layer].register_forward_hook(hook)
        self.steer = None if rows is None else torch.as_tensor(np.asarray(rows), dtype=self.dtype,
                                                                 device=self.device)[:, None, :]

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


def sweep(args):
    """Relaxed loop h <- (1 - lambda) h + lambda F(h) for many lambda in one batch: bifurcation diagram,
    largest Lyapunov exponent and period of the attractor."""
    loop = LatentLoop(args.model, args.dtype, args.device, args.threads)
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
    loop = LatentLoop(args.model, args.dtype, args.device, args.threads)
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
    loop = LatentLoop(args.model, args.dtype, args.device, args.threads)
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


def math_attention():
    """Scaled-dot-product attention through the plain math kernel (the fused kernels lack forward-mode AD)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(SDPBackend.MATH)


def orbit(args):
    """Newton-Krylov for a period-p orbit of the relaxed loop, with Arnoldi multipliers (matrix-free JVPs)."""
    from scipy.sparse.linalg import LinearOperator, eigs, gmres

    loop = LatentLoop(args.model, args.dtype, args.device, args.threads)
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
    loop = LatentLoop(args.model, args.dtype, args.device, args.threads)
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

    loop = LatentLoop(args.model, args.dtype, args.device, args.threads)
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
    ck = sub.add_parser("check")
    ck.add_argument("--model", choices=sorted(MODELS), default="ouro1.4b")
    ck.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    ck.add_argument("--device", default="cpu")
    ck.add_argument("--threads", type=int, default=8)
    ck.add_argument("--prompts", nargs="+", default=["hello", "arith"])
    args = parser.parse_args()
    {"trajectory": trajectory, "check": check, "sweep": sweep, "vectors": vectors, "plane": plane,
     "orbit": orbit, "basin": basin, "continue-orbit": continue_orbit}[args.command](args)


if __name__ == "__main__":
    main()
