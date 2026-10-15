"""Does steering destabilise the loop through the readout or through the network? An orbit-preserving four-arm
crossover of the two factors of the derivative (run XO; consult 9, direction 2).

Along a recorded reference orbit of the steered map, x_(t+1) = E^T softmax(z_s(x_t) / T), each step's derivative is
    DF = (1/T) E^T (diag p - p p^T) dz/dx = R(p) J,
the readout factor R(p) at the next-token distribution p times the network's logit Jacobian J. At the same states
x_t, the unsteered model gives its own p_u = softmax(z_u(x_t) / T) and J_u (its prefix cache built without steering).
Four tangent dynamics are carried along the same orbit, each with its own renormalised tangent vector:
    arm 11: R(p_s) J_s   (the steered map's own derivative; must reproduce its exponent),
    arm 10: R(p_s) J_u   (steered readout, unsteered network response),
    arm 01: R(p_u) J_s   (unsteered readout, steered network response),
    arm 00: R(p_u) J_u   (the unsteered model's derivative at the steered states).
The finite-time growth of each arm (nats per step, over windows of the recorded steps) attributes the expansion:
arm 10 close to arm 11 means that steering's change of the token competition suffices, arm 01 close to arm 11 that
its change of the network response suffices, and expansion only in arm 11 an interaction. This is a test of local
expansion along the controlled orbit, not a claim that a crossed map has a chaotic attractor of its own.

Usage: python tools/crossover.py --cells-file docs/xo-cells.json --device cuda --out llm-soft-loop/pod64/xo/x
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


def logits_jvp(loop, X, V):
    """Logits and their tangents for rows X (B, 1, d) with tangent rows V, each row with its own steering
    (loop.steer_rows from prepare), on a shallow copy of the prefix cache."""
    import torch.autograd.forward_ad as fwAD
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch = loop.torch
    cache = copy.copy(loop.cache)
    cache.layers = [copy.copy(layer) for layer in loop.cache.layers]
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), fwAD.dual_level():
        loop.sm.steer = loop.steer_rows
        try:
            z = loop.model(inputs_embeds=fwAD.make_dual(X, V).to(loop.dtype), past_key_values=cache, use_cache=True,
                           logits_to_keep=1).logits[:, -1, :]
        finally:
            loop.sm.steer = None
        zp, dz = fwAD.unpack_dual(z)
    return zp.to(loop.device, loop.state_dtype), dz.to(loop.device, loop.state_dtype)


def readout(p, dz, E, T):
    """(1/T) E^T (diag p - p p^T) dz for rows of p, dz (n, V); T (n,)."""
    return ((p * dz) @ E - (p @ E) * (p * dz).sum(-1, keepdim=True)) / T[:, None]


def run(loop, v, cells, transient, record, window, seed=0, log=print):
    torch = loop.torch
    n = len(cells)
    alphas = np.array([a for _, a in cells])
    temps = torch.as_tensor(10.0 ** np.array([t for t, _ in cells]), dtype=loop.state_dtype, device=loop.device)
    steer = alphas[:, None] * v[None, :]
    zero = np.zeros_like(steer)
    # rows: [arm 11 | arm 01] use the steered prefix and network, [arm 00 | arm 10] the unsteered ones
    X = loop.prepare(np.concatenate([steer, steer, zero, zero]))
    T4 = temps.repeat(4)
    x = X[:n, -1]                                             # reference orbit of the steered map
    tic = time.perf_counter()
    for _ in range(transient):
        Xr = torch.cat([x[:, None, :]] * 4)
        z = loop.logits(Xr)
        x = torch.softmax(z[:n] / temps[:, None], dim=-1) @ loop.E
    rng = np.random.default_rng(seed)
    Vt = torch.as_tensor(rng.normal(size=(4 * n, x.shape[1])), dtype=loop.state_dtype, device=loop.device)
    Vt = Vt / Vt.norm(dim=1, keepdim=True)
    nw = record // window
    growth = np.zeros((nw, 4, n))
    tw_growth = np.zeros(n)
    twin = x + 1e-9 * x.norm(dim=1, keepdim=True) * Vt[:n] / Vt[:n].norm(dim=1, keepdim=True)
    for i in range(record):
        Xr = torch.cat([x[:, None, :]] * 4)
        z, dz = logits_jvp(loop, Xr, Vt[:, None, :])
        ps = torch.softmax(z[:n] / temps[:, None], dim=-1)
        pu = torch.softmax(z[2 * n:3 * n] / temps[:, None], dim=-1)
        w = torch.cat([readout(ps, dz[:n], loop.E, temps),                 # arm 11
                       readout(pu, dz[n:2 * n], loop.E, temps),            # arm 01
                       readout(pu, dz[2 * n:3 * n], loop.E, temps),        # arm 00
                       readout(ps, dz[3 * n:], loop.E, temps)])            # arm 10
        size = w.norm(dim=1)
        growth[i // window] += np.log(np.maximum(size.double().cpu().numpy(), 1e-300)).reshape(4, n)
        Vt = w / size[:, None]
        # an independent twin check of the steered exponent (relative offset 1e-9, renormalised)
        zt = loop.logits(torch.cat([twin[:, None, :]] * 4))[:n]
        xn = ps @ loop.E
        tn = torch.softmax(zt / temps[:, None], dim=-1) @ loop.E
        d0 = (twin - x).norm(dim=1)
        d1 = (tn - xn).norm(dim=1)
        tw_growth += np.log((d1 / d0).double().cpu().numpy())
        twin = xn + (tn - xn) * (1e-9 * xn.norm(dim=1) / d1)[:, None]
        x = xn
        if (i + 1) % window == 0:
            log(json.dumps({"recorded": i + 1, "of": record, "seconds": round(time.perf_counter() - tic, 1)}))
    growth /= window
    arms = ("11", "01", "00", "10")
    out = []
    for c, (t, a) in enumerate(cells):
        rec = {"log10_temperature": float(t), "alpha": float(a), "twin_exponent": float(tw_growth[c] / record)}
        for k, arm in enumerate(arms):
            rec[f"arm{arm}"] = float(growth[:, k, c].mean())
            rec[f"arm{arm}_windows"] = growth[:, k, c].tolist()
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--cells-file", required=True, help="JSON with 'cells' ([log10T, alpha] pairs)")
    ap.add_argument("--transient", type=int, default=1000)
    ap.add_argument("--record", type=int, default=1000)
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=16, help="cells per batch (4 rows each)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    spec = json.loads(Path(args.cells_file).read_text())
    cells = [tuple(c) for c in spec["cells"]]
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)

    def log(line):
        print(line, flush=True)
        with open(out / "progress.jsonl", "a") as f:
            f.write(line + "\n")

    res = []
    for s0 in range(0, len(cells), args.chunk):
        res += run(loop, v, cells[s0:s0 + args.chunk], args.transient, args.record, args.window, log=log)
        (out / "crossover.json").write_text(json.dumps({"args": vars(args), "cells": res,
                                                        "labels": spec.get("labels")}, indent=1))
        log(json.dumps({"done": len(res), "of": len(cells)}))


if __name__ == "__main__":
    main()
