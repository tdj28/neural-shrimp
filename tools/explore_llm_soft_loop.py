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
    def __init__(self, model, layer, threads, prompt, opener, window, dtype="float64", device="cpu"):
        self.sm = elpm.SteeredModel(model, layer, threads, dtype)
        self.torch = self.sm.torch
        torch = self.torch
        self.device = torch.device(device)
        self.model = self.sm.model.to(self.device)
        self.dtype = self.model.dtype
        self.state_dtype = torch.float64 if dtype == "float64" else torch.float32
        self.E = self.model.get_input_embeddings().weight.detach().to(self.state_dtype)      # (V, d)
        messages = [{"role": "system", "content": ela.PROMPTS[prompt]}, {"role": "user", "content": opener}]
        text = self.sm.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.prefix_ids = torch.tensor([self.sm.tok(text, add_special_tokens=False)["input_ids"]], device=self.device)
        self.L0 = self.prefix_ids.shape[1]
        self.K = int(window)
        self.cache = None
        self.steer_rows = None

    def prepare(self, steer_rows):
        """Prefix key-value cache for every row, computed with that row's steering."""
        from transformers import DynamicCache

        torch = self.torch
        B = len(steer_rows)
        self.steer_rows = torch.as_tensor(np.asarray(steer_rows), dtype=self.dtype, device=self.device)[:, None, :]
        self.cache = DynamicCache()
        self.sm.steer = self.steer_rows
        try:
            with torch.no_grad():
                self.model(input_ids=self.prefix_ids.repeat(B, 1), past_key_values=self.cache, use_cache=True)
        finally:
            self.sm.steer = None
        with torch.no_grad():
            start = self.model.get_input_embeddings()(self.prefix_ids[:, -1:])[0, 0].to(self.state_dtype)
        return start.repeat(B, self.K, 1).clone()                              # initial window (B, K, d)

    def logits(self, X):
        torch = self.torch
        self.sm.steer = self.steer_rows
        try:
            with torch.no_grad():
                z = self.model(inputs_embeds=X.to(self.dtype), past_key_values=self.cache,
                               use_cache=True).logits[:, -1, :].to(self.state_dtype)
        finally:
            self.sm.steer = None
            self.cache.crop(self.L0)                                           # keep only the prefix (argument = length kept)
        return z

    def step(self, X, temps):
        """One soft-decoding step: X (B, K, d) -> shifted window with the new soft embedding appended."""
        torch = self.torch
        z = self.logits(X)
        p = torch.softmax(z / torch.as_tensor(temps, dtype=self.state_dtype, device=self.device)[:, None], dim=-1)
        return torch.cat([X[:, 1:, :], (p @ self.E)[:, None, :]], dim=1), p


def scan(args):
    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                    args.device)
    torch = loop.torch
    v = elpm.load_vectors(args.vectors)[args.concept] if args.vectors else None
    log_t = np.linspace(*args.log_temperature, args.resolution[0])
    alphas = np.linspace(*args.alpha, args.resolution[1])
    cells = [(lt, a) for lt in log_t for a in alphas]
    B = len(cells)
    d = loop.E.shape[1]
    lyap = np.zeros(B)
    period = np.zeros(B, dtype=int)
    entropy = np.zeros(B)
    rng = np.random.default_rng(0)
    tic = time.perf_counter()
    chunk = args.chunk or B
    for start in range(0, B, chunk):
        idx = list(range(start, min(start + chunk, B)))
        n = len(idx)
        steer = np.zeros((n, d)) if v is None else np.array([cells[i][1] * v for i in idx])
        temps2 = np.tile(np.array([10.0 ** cells[i][0] for i in idx]), 2)
        XY = loop.prepare(np.concatenate([steer, steer]))     # rows 0..n-1: trajectory, n..2n-1: its twin
        for _ in range(args.transient):
            XY, _ = loop.step(XY, temps2)
        delta = torch.as_tensor(rng.normal(size=tuple(XY[:n].shape)), dtype=loop.state_dtype, device=loop.device)
        delta = delta / delta.flatten(1).norm(dim=1)[:, None, None] * args.eps
        XY = torch.cat([XY[:n], XY[:n] + delta])
        log_growth = np.zeros(n)
        history = []
        for k in range(args.samples):
            XY, p = loop.step(XY, temps2)
            X, Y = XY[:n], XY[n:]
            p = p[:n]
            diff = (Y - X)
            dist = diff.flatten(1).norm(dim=1).double().cpu().numpy()
            # a twin that collapses onto the trajectory (super-stable cycles, exact zero in float32) contributes a
            # floor of log(1e-30) per step; the floor also keeps the renormalization factor finite in float32
            dist = np.maximum(dist, args.eps * 1e-30)
            log_growth += np.log(dist / args.eps)
            Y = X + diff * torch.as_tensor(args.eps / dist, dtype=loop.state_dtype, device=loop.device)[:, None, None]
            XY = torch.cat([X, Y])
            if k >= args.samples - args.record:
                history.append(X[:, -1, :].double().cpu().numpy().copy())
        lyap[idx] = log_growth / args.samples
        hist = np.array(history)
        scale = np.maximum(1.0, np.linalg.norm(hist[-1], axis=1))
        per = np.zeros(n, dtype=int)
        for q in range(min(args.max_period, len(hist) - 1), 0, -1):
            close = np.linalg.norm(hist[-1 - q] - hist[-1], axis=1) < args.tol * scale
            per = np.where(close, q, per)
        period[idx] = per
        last_p = p.double().cpu().numpy()
        entropy[idx] = -(last_p * np.log(np.maximum(last_p, 1e-300))).sum(1)
        del p, XY, X, Y
        print(json.dumps({"done": idx[-1] + 1, "of": B, "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
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
    lim = max(float(np.nanmax(lyap)), 0.05)                 # clip: super-stable cells reach -700
    im0 = axes[0].pcolormesh(alphas, log_t, np.clip(lyap, -lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim,
                             shading="auto")
    axes[0].set_title("largest Lyapunov exponent per step (clipped)")
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
    sc.add_argument("--device", default="cpu")
    sc.add_argument("--chunk", type=int, default=0, help="cells per batch (0: all at once)")
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
