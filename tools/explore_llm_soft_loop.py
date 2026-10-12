"""Exploratory: soft-token decoding as a smooth map, scanned for periodic windows and chaos.

Instead of sampling a token, the loop feeds back the probability-weighted input
embedding x_new = E^T softmax(z / T), where z are the next-token logits after a
fixed prefix plus the last K fed-back embeddings (a sliding window at fixed
positions, so the map is autonomous). As T -> 0 the map becomes greedy decoding
with a K-token window; at finite T it is a smooth map of R^(K d), so periodic
orbits, multipliers, Lyapunov exponents and bifurcation curves are defined in
the usual sense. An additive steering vector alpha * v on one decoder layer's
output gives a second continuous parameter.

Subcommands
  scan   Grid in (log10 T, alpha): iterate all cells in one batch, then measure
         the period of the attractor (state recurrence) and the largest Lyapunov
         exponent (twin trajectory, renormalized every step).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402

DATA = ela.DATA


class SoftLoop:
    def __init__(self, model, layer, threads, prompt, opener, window, dtype="float64"):
        self.sm = elpm.SteeredModel(model, layer, threads, dtype)
        self.torch = self.sm.torch
        torch = self.torch
        self.model = self.sm.model
        self.dtype = self.model.dtype
        self.E = self.model.get_input_embeddings().weight.detach().to(torch.float64)      # (V, d)
        messages = [{"role": "system", "content": ela.PROMPTS[prompt]}, {"role": "user", "content": opener}]
        text = self.sm.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.prefix_ids = torch.tensor([self.sm.tok(text, add_special_tokens=False)["input_ids"]])
        self.L0 = self.prefix_ids.shape[1]
        self.K = int(window)
        self.cache = None
        self.steer_rows = None

    def prepare(self, steer_rows):
        """Prefix key-value cache for every row, computed with that row's steering."""
        from transformers import DynamicCache

        torch = self.torch
        B = len(steer_rows)
        self.steer_rows = torch.as_tensor(np.asarray(steer_rows))[:, None, :]
        self.cache = DynamicCache()
        self.sm.steer = self.steer_rows
        try:
            with torch.no_grad():
                self.model(input_ids=self.prefix_ids.repeat(B, 1), past_key_values=self.cache, use_cache=True)
        finally:
            self.sm.steer = None
        with torch.no_grad():
            start = self.model.get_input_embeddings()(self.prefix_ids[:, -1:])[0, 0].to(torch.float64)
        return start.repeat(B, self.K, 1).clone()                              # initial window (B, K, d)

    def logits(self, X):
        torch = self.torch
        self.sm.steer = self.steer_rows
        try:
            with torch.no_grad():
                z = self.model(inputs_embeds=X.to(self.dtype), past_key_values=self.cache,
                               use_cache=True).logits[:, -1, :].to(torch.float64)
        finally:
            self.sm.steer = None
            self.cache.crop(self.L0)                                           # keep only the prefix (argument = length kept)
        return z

    def step(self, X, temps):
        """One soft-decoding step: X (B, K, d) -> shifted window with the new soft embedding appended."""
        torch = self.torch
        z = self.logits(X)
        p = torch.softmax(z / torch.as_tensor(temps)[:, None], dim=-1)
        return torch.cat([X[:, 1:, :], (p @ self.E)[:, None, :]], dim=1), p


def scan(args):
    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype)
    torch = loop.torch
    v = elpm.load_vectors(args.vectors)[args.concept] if args.vectors else None
    log_t = np.linspace(*args.log_temperature, args.resolution[0])
    alphas = np.linspace(*args.alpha, args.resolution[1])
    cells = [(lt, a) for lt in log_t for a in alphas]
    B = len(cells)
    d = loop.E.shape[1]
    steer = np.zeros((B, d)) if v is None else np.array([a * v for _, a in cells])
    temps2 = np.tile(np.array([10.0 ** lt for lt, _ in cells]), 2)
    tic = time.perf_counter()
    XY = loop.prepare(np.concatenate([steer, steer]))     # rows 0..B-1: trajectory, B..2B-1: its twin
    for _ in range(args.transient):
        XY, _ = loop.step(XY, temps2)
    # twin trajectory for the largest Lyapunov exponent
    rng = np.random.default_rng(0)
    delta = torch.as_tensor(rng.normal(size=XY[:B].shape))
    delta = delta / delta.flatten(1).norm(dim=1)[:, None, None] * args.eps
    XY = torch.cat([XY[:B], XY[:B] + delta])
    log_growth = np.zeros(B)
    history = []
    for k in range(args.samples):
        XY, p = loop.step(XY, temps2)
        X, Y = XY[:B], XY[B:]
        p = p[:B]
        diff = (Y - X)
        dist = diff.flatten(1).norm(dim=1).numpy()
        log_growth += np.log(np.maximum(dist, 1e-300) / args.eps)
        Y = X + diff * torch.as_tensor(args.eps / np.maximum(dist, 1e-300))[:, None, None]
        XY = torch.cat([X, Y])
        if k >= args.samples - args.record:
            history.append(X[:, -1, :].numpy().copy())
    lyap = log_growth / args.samples
    hist = np.array(history)                                                   # (record, B, d)
    scale = np.maximum(1.0, np.linalg.norm(hist[-1], axis=1))
    period = np.zeros(B, dtype=int)
    for q in range(args.max_period, 0, -1):
        close = np.linalg.norm(hist[-1 - q] - hist[-1], axis=1) < args.tol * scale
        period = np.where(close, q, period)
    last_p = p.numpy()
    entropy = -(last_p * np.log(np.maximum(last_p, 1e-300))).sum(1)
    seconds = round(time.perf_counter() - tic, 1)
    shape = tuple(args.resolution)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scan.npz", log10_temperature=log_t, alpha=alphas, lyapunov=lyap.reshape(shape),
                        period=period.reshape(shape), entropy=entropy.reshape(shape))
    summary = {"args": vars(args), "seconds": seconds, "cells": B,
               "lyapunov_positive": int((lyap > args.chaos_level).sum()),
               "period_counts": {int(q): int((period == q).sum()) for q in np.unique(period)},
               "lyapunov_range": [float(lyap.min()), float(lyap.max())]}
    (out / "scan.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}))
    figure(log_t, alphas, lyap.reshape(shape), period.reshape(shape), entropy.reshape(shape), out / "scan.png", args)


def figure(log_t, alphas, lyap, period, entropy, path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    lim = np.nanmax(np.abs(lyap)) or 1.0
    im0 = axes[0].pcolormesh(alphas, log_t, lyap, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    axes[0].set_title("largest Lyapunov exponent per step")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(alphas, log_t, np.where(period > 0, period, np.nan), cmap="tab20", vmin=0.5,
                             vmax=20.5, shading="auto")
    axes[1].set_title("period of the attractor (white: none up to the maximum)")
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].pcolormesh(alphas, log_t, entropy, cmap="viridis", shading="auto")
    axes[2].set_title("next-token entropy at the last step (nats)")
    fig.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel(f"steering along '{args.concept}'" if args.vectors else "alpha (no steering vector)")
        ax.set_ylabel("log10 temperature")
    fig.suptitle(f"soft-token decoding, {ela.MODELS[args.model]}, window {args.window}, layer {args.layer}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sc = sub.add_parser("scan")
    sc.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    sc.add_argument("--layer", type=int, default=12)
    sc.add_argument("--threads", type=int, default=8)
    sc.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    sc.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    sc.add_argument("--opener", default=ela.OPENER)
    sc.add_argument("--window", type=int, default=1)
    sc.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    sc.add_argument("--concept", default="philosophy", choices=sorted(elpm.CONCEPTS))
    sc.add_argument("--log-temperature", type=float, nargs=2, default=[-1.0, 0.3])
    sc.add_argument("--alpha", type=float, nargs=2, default=[-0.12, 0.12])
    sc.add_argument("--resolution", type=int, nargs=2, default=[16, 16], help="rows (temperature) and columns (alpha)")
    sc.add_argument("--transient", type=int, default=300)
    sc.add_argument("--samples", type=int, default=300)
    sc.add_argument("--record", type=int, default=80)
    sc.add_argument("--max-period", type=int, default=32)
    sc.add_argument("--tol", type=float, default=1e-6)
    sc.add_argument("--eps", type=float, default=1e-6)
    sc.add_argument("--chaos-level", type=float, default=1e-3)
    sc.add_argument("--out", required=True)
    args = parser.parse_args()
    {"scan": scan}[args.command](args)


if __name__ == "__main__":
    main()
