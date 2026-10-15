"""Where does the gain of the soft-token loop come from? (runs G1, G2)

The soft-token map is F = R o h: h(x) is the final normalised hidden state at the soft token's position and
R(h) = E^T softmax(W_out h / T) the readout. For a twin x + delta, each step's growth splits exactly:
    log(|dF| / |dx|) = log(|dh| / |dx|) + log(|dF| / |dh|).
Relative to the states' norms the two factors are scale free,
    g_net  = (|dh| / |h|) / (|dx| / |x|)     how much the network amplifies a relative change of its input token,
    g_read = (|dF| / |F|) / (|dh| / |h|)     how much the readout (softmax at temperature T) amplifies it,
and on an orbit (x_(n+1) = F(x_n)) the log-averages of g_net and g_read add up to the largest Lyapunov exponent up to
a boundary term (1/N) log(|x_N| / |x_0|), which is reported. Twins are renormalised every step to a relative
offset eps (float64). For each temperature: the exponent, the mean log g_net and log g_read, the mean entropy of the
next-token distribution, and the period of the attractor (if any, up to 64).

Usage: python tools/gain_split.py --model M --log-temperature LO HI --n N --out OUT
       [--device cpu|cuda|auto --weights native|bf16 --transient 500 --record 1000 --eps 1e-6]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def h_and_F(loop, X, temps):
    """Final normalised hidden state, next soft token and next-token distribution for a batch of windows."""
    torch = loop.torch
    loop.sm.steer = loop.steer_rows
    try:
        with torch.no_grad():
            out = loop.model.model(inputs_embeds=X.to(loop.dtype), past_key_values=loop.cache, use_cache=True)
            h = out.last_hidden_state[:, -1]
            z = loop.model.lm_head(h).to(loop.device, loop.state_dtype)
    finally:
        loop.sm.steer = None
        loop.reset_cache()
    p = torch.softmax(z / temps[:, None], dim=-1)
    return h.to(loop.device, loop.state_dtype), p @ loop.E, p


def run(loop, log_ts, transient, record, eps, seed=0):
    torch = loop.torch
    n = len(log_ts)
    d = loop.E.shape[1]
    temps = torch.as_tensor(np.tile(10.0 ** np.asarray(log_ts), 2), dtype=loop.state_dtype, device=loop.device)
    X = loop.prepare(np.zeros((2 * n, d)))                 # unsteered; rows 0..n-1 trajectory, n..2n-1 its twin
    for _ in range(transient):
        _, F, _ = h_and_F(loop, X, temps)
        X = F[:, None, :]
    base = X[:n, -1]
    g = torch.as_tensor(np.random.default_rng(seed).normal(size=(n, d)), dtype=loop.state_dtype, device=loop.device)
    twin = base + g / g.norm(dim=1, keepdim=True) * eps * base.norm(dim=1, keepdim=True)
    X = torch.cat([base, twin])[:, None, :]
    x0 = base.norm(dim=1).double().cpu().numpy()
    growth, net, read, ent = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    tail = []
    for _ in range(record):
        xb, xt = X[:n, -1], X[n:, -1]
        h, F, p = h_and_F(loop, X, temps)
        hb, ht, Fb, Ft = h[:n], h[n:], F[:n], F[n:]
        nx = lambda v: v.norm(dim=1).double().cpu().numpy()
        dx, dh, dF = nx(xt - xb), nx(ht - hb), nx(Ft - Fb)
        rx, rh, rF = dx / nx(xb), dh / nx(hb), dF / nx(Fb)
        growth += np.log(dF / dx)
        net += np.log(rh / rx)
        read += np.log(rF / rh)
        pb = p[:n].double()
        ent += (-(pb * torch.log(torch.clamp(pb, min=1e-300))).sum(-1)).cpu().numpy()
        scale = torch.as_tensor(eps * nx(Fb) / dF, dtype=loop.state_dtype, device=loop.device)[:, None]
        X = torch.cat([Fb, Fb + (Ft - Fb) * scale])[:, None, :]
        tail.append(Fb.double().cpu().numpy())
        if len(tail) > 66:
            tail.pop(0)
    S = np.array(tail)
    out = []
    for c in range(n):
        s = S[:, c]
        period = next((q for q in range(1, min(65, len(s))) if np.linalg.norm(s[-1 - q] - s[-1])
                       < 1e-8 * max(1.0, np.linalg.norm(s[-1]))), 0)
        out.append({"log10_temperature": float(log_ts[c]), "lyapunov": float(growth[c] / record),
                    "log_g_net": float(net[c] / record), "log_g_read": float(read[c] / record),
                    "boundary_term": float(np.log(np.linalg.norm(s[-1]) / x0[c]) / record),
                    "entropy_mean": float(ent[c] / record), "period": int(period)})
    return out


def main():
    import explore_llm_attractors as ela
    import explore_llm_soft_loop as esl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--log-temperature", type=float, nargs=2, required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--transient", type=int, default=500)
    ap.add_argument("--record", type=int, default=1000)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default="native", choices=("native", "bf16"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    loop = esl.SoftLoop(args.model, 0, args.threads, "open", ela.OPENER, 1, "float64", args.device, args.weights)
    log_ts = np.linspace(args.log_temperature[0], args.log_temperature[1], args.n)
    cells = run(loop, log_ts, args.transient, args.record, args.eps)
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "gain-split.json").write_text(json.dumps({"args": vars(args), "cells": cells}, indent=1))
    for c in cells:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}), flush=True)


if __name__ == "__main__":
    main()
