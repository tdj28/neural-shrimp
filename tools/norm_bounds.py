"""Run NB (exploratory; human-relayed proposal, 18 Oct 2026): do the factorized derivative's norm bounds capture where
chaos lives?

With tied embeddings the soft-token map's derivative factorizes, DF = T^-1 C_p J_h, where C_p = E^T (diag p - p p^T) E
is the covariance of the token embeddings under the next-token distribution and J_h the Jacobian of the final
normalized hidden state (untied: C_p becomes R_p = E^T (diag p - p p^T) W). Along an orbit x_0, x_1, ... the largest
Lyapunov exponent therefore obeys, for every block length k,

    lambda <= <log ||DF^k||> / k   and   <log ||DF||> <= <log(||C_p|| / T)> + <log ||J_h||>,

where <.> averages over the orbit (k-step norms over non-overlapping blocks). Wherever a bound is negative the cell
cannot be chaotic; the bounds tighten toward lambda as k grows (Test B). Test A estimates L = sup ||J_h|| over the hull
of the embeddings, from sampled states and a random-search ascent, and the worst-case threshold T* = diam(E)^2 L / 4.

cells  per cell of a cells file: a trajectory from the fixed start (--transient steps, then --measure steps); at every
       measured step the readout term log(||C_p||/T), the network term log||J_h|| and the one-step norm log||DF||;
       non-overlapping k-step block norms for k in --ks; the exact tangent exponent along the same steps (forward-mode,
       renormalized every step). Norms come from power iteration on A^T A with forward-mode products A v and
       reverse-mode products A^T u (--iters, warm-started across steps); the last relative change is reported.
sup    L from the largest ||J_h|| over the cells' recorded states, random sparse mixtures of embeddings and a random
       search on the mixture weights; diam(E) exactly (blockwise); T* = diam(E)^2 L / 4.

These are bounds on where chaos can live, not evidence that it does; the twin and tangent exponents remain that
evidence. Output: DATA/explore/OUT/nb.json (+ nb.npz per-cell arrays).

Usage: python tools/norm_bounds.py cells --cells-file docs/nb-cells.json --source map-stripes-256 --device cuda \
         --out llm-soft-loop/pod64/nb/stripes
       python tools/norm_bounds.py sup --from llm-soft-loop/pod64/nb/stripes --device cuda --out llm-soft-loop/pod64/nb/sup
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
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402
from explore_llm_soft_loop import DATA, SoftLoop  # noqa: E402


class Probe:
    """Batched forward-mode and reverse-mode products of h(x) and F(x) for one batch of cells."""

    def __init__(self, loop, temps):
        self.loop, self.torch = loop, loop.torch
        self.T = self.torch.as_tensor(temps, dtype=loop.state_dtype, device=loop.device)[:, None]
        self.E = loop.E
        W = loop.model.lm_head.weight
        self.tied = W.data_ptr() == loop.model.model.embed_tokens.weight.data_ptr()
        self.W = self.E if self.tied else W.to(loop.device, loop.state_dtype)

    def _model_h(self, X):
        loop = self.loop
        cache = copy.copy(loop.cache)
        cache.layers = [copy.copy(layer) for layer in loop.cache.layers]
        loop.sm.steer = loop.steer_rows
        try:
            out = loop.model.model(inputs_embeds=X.to(loop.dtype), past_key_values=cache, use_cache=True)
            return out.last_hidden_state[:, -1].to(loop.device, loop.state_dtype)
        finally:
            loop.sm.steer = None

    def state(self, X):
        """h, p and F at the states X (n, 1, d), no gradients."""
        torch = self.torch
        with torch.no_grad():
            h = self._model_h(X)
            p = torch.softmax((h @ self.W.T) / self.T, dim=-1)
        return h, p, p @ self.E

    def Jv(self, X, V):
        import torch.autograd.forward_ad as fwAD
        from torch.nn.attention import SDPBackend, sdpa_kernel
        torch = self.torch
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), fwAD.dual_level():
            h = self._model_h(fwAD.make_dual(X, V[:, None, :].to(X.dtype)))
            t = fwAD.unpack_dual(h).tangent
        if t is None:
            raise SystemExit("Jv: the tangent was lost in the forward pass")
        return t

    def JTu(self, X, U):
        torch = self.torch
        with torch.enable_grad():
            Xg = X.detach().clone().requires_grad_(True)
            h = self._model_h(Xg)
            (g,) = torch.autograd.grad(h, Xg, grad_outputs=U)
        return g[:, -1]

    def Cv(self, p, V):
        """Readout operator: tied C_p V = E^T (p * (E V) - p (p . E V)); untied R_p V = E^T Sigma_p (W V)."""
        s = V @ self.W.T
        s = p * s - p * (p * s).sum(1, keepdim=True)
        return s @ self.E

    def CTu(self, p, U):
        if self.tied:
            return self.Cv(p, U)
        s = U @ self.E.T
        s = p * s - p * (p * s).sum(1, keepdim=True)
        return s @ self.W

    def DFv(self, X, p, V):
        return self.Cv(p, self.Jv(X, V)) / self.T

    def DFTu(self, X, p, U):
        return self.JTu(X, self.CTu(p, U) / self.T)


def power_norm(Av, ATu, v, iters):
    """Largest singular value of A by power iteration on A^T A, from v (n, d); returns sigma, v, last relative change."""
    v = v / v.norm(dim=1, keepdim=True)
    prev = None
    for _ in range(iters):
        w = Av(v)
        s = w.norm(dim=1)
        g = ATu(w)
        v = g / g.norm(dim=1, keepdim=True).clamp_min(1e-300)
        rel = None if prev is None else ((s - prev).abs() / s.clamp_min(1e-300))
        prev = s
    return s, v, rel


def cells_mode(args):
    path = Path(args.cells_file)
    if not path.is_absolute() and not path.exists():
        path = ROOT / args.cells_file
    spec = json.loads(path.read_text())
    idx = [i for i in range(len(spec["cells"])) if args.source is None or spec["source"][i] == args.source]
    cells = [tuple(spec["cells"][i]) for i in idx]
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    loop = SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, args.dtype, args.device)
    torch = loop.torch
    if args.vectors:
        v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    else:
        v = np.zeros(loop.E.shape[1])
    ks = sorted(set(args.ks))
    rows = []
    for b0 in range(0, len(cells), args.batch):
        bc = cells[b0:b0 + args.batch]
        n = len(bc)
        steer = np.array([al * v for _, al in bc])
        temps = np.array([10.0 ** lt for lt, _ in bc])
        X = loop.prepare(steer)
        pr = Probe(loop, temps)
        with torch.no_grad():
            for _ in range(args.transient):
                X, _ = loop.step(X, temps)
        states, ps = [], []
        rng = torch.Generator(device=loop.device).manual_seed(0)
        d = X.shape[2]
        vJ = torch.randn((n, d), generator=rng, device=loop.device, dtype=loop.state_dtype)
        vD = vJ.clone()
        vC = vJ.clone()
        w = vJ / vJ.norm(dim=1, keepdim=True)
        R, N, D1, TAN, RELJ, RELD = [], [], [], [], [], []
        for _ in range(args.measure):
            _, p, F = pr.state(X)
            states.append(X)
            ps.append(p)
            sC, vC, _ = power_norm(lambda u: pr.Cv(p, u), lambda u: pr.CTu(p, u), vC, args.iters)
            sJ, vJ, rJ = power_norm(lambda u: pr.Jv(X, u), lambda u: pr.JTu(X, u), vJ, args.iters)
            sD, vD, rD = power_norm(lambda u: pr.DFv(X, p, u), lambda u: pr.DFTu(X, p, u), vD, args.iters)
            tw = pr.DFv(X, p, w)                                     # exact tangent exponent along the same orbit
            gn = tw.norm(dim=1)
            TAN.append(torch.log(gn.clamp_min(1e-300)))
            w = tw / gn[:, None].clamp_min(1e-300)
            R.append(torch.log(sC / pr.T[:, 0]))
            N.append(torch.log(sJ))
            D1.append(torch.log(sD))
            RELJ.append(rJ)
            RELD.append(rD)
            X = F[:, None, :]
        blocks = {}
        for k in ks:
            if k == 1:
                continue
            logs = []
            vk = torch.randn((n, d), generator=rng, device=loop.device, dtype=loop.state_dtype)
            for s0 in range(0, args.measure - k + 1, k):
                seg = list(range(s0, s0 + k))

                def Ak(u, seg=seg):
                    for t in seg:
                        u = pr.DFv(states[t], ps[t], u)
                    return u

                def AkT(u, seg=seg):
                    for t in reversed(seg):
                        u = pr.DFTu(states[t], ps[t], u)
                    return u

                sk, vk, _ = power_norm(Ak, AkT, vk, args.iters)
                logs.append(torch.log(sk) / k)
            blocks[k] = torch.stack(logs).mean(0)
        st = lambda L: torch.stack(L).double().cpu().numpy()          # noqa: E731
        Rm, Nm, Dm, Tm = st(R).mean(0), st(N).mean(0), st(D1).mean(0), st(TAN).mean(0)
        relJ = st([r for r in RELJ if r is not None]).max(0) if args.iters > 1 else np.full(n, np.nan)
        relD = st([r for r in RELD if r is not None]).max(0) if args.iters > 1 else np.full(n, np.nan)
        maxJ = st(N).max(0)
        for c in range(n):
            row = {"log10_temperature": bc[c][0], "alpha": bc[c][1], "source": spec["source"][idx[b0 + c]],
                   "map_label": spec.get("label", [None] * len(spec["cells"]))[idx[b0 + c]],
                   "map_exponent": spec.get("map_exponent", [None] * len(spec["cells"]))[idx[b0 + c]],
                   "readout": float(Rm[c]), "network": float(Nm[c]), "sum": float(Rm[c] + Nm[c]),
                   "one_step": float(Dm[c]), "tangent_exponent": float(Tm[c]), "max_log_J": float(maxJ[c]),
                   "power_rel_change_J": float(relJ[c]), "power_rel_change_DF": float(relD[c])}
            for k, val in blocks.items():
                row[f"k{k}"] = float(val[c].item())
            rows.append(row)
        print(json.dumps({"done": b0 + n, "of": len(cells), "seconds": round(time.perf_counter() - tic, 1)}),
              flush=True)
        (out / "nb.json").write_text(json.dumps({"args": vars(args), "cells": rows}, indent=0))
    rep = {"args": vars(args), "cells": rows, "seconds": round(time.perf_counter() - tic, 1)}
    (out / "nb.json").write_text(json.dumps(rep, indent=0))


def sup_mode(args):
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    loop = SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, args.dtype, args.device)
    torch = loop.torch
    E = loop.E
    V, d = E.shape
    # diam(E) exactly, blockwise: max_ij |e_i - e_j|^2 = max_ij (|e_i|^2 + |e_j|^2 - 2 e_i.e_j)
    sq = (E * E).sum(1)
    best = torch.zeros((), dtype=loop.state_dtype, device=loop.device)
    for b in range(0, V, 4096):
        D2 = sq[b:b + 4096, None] + sq[None, :] - 2.0 * (E[b:b + 4096] @ E.T)
        best = torch.maximum(best, D2.max())
    diam = float(best.clamp_min(0).sqrt())
    pr = Probe(loop, np.ones(args.batch))
    loop.prepare(np.zeros((args.batch, d)))
    gen = torch.Generator(device=loop.device).manual_seed(1)

    def norms(Xs):
        X = Xs[:, None, :]
        v0 = torch.randn((Xs.shape[0], d), generator=gen, device=loop.device, dtype=loop.state_dtype)
        s, _, _ = power_norm(lambda u: pr.Jv(X, u), lambda u: pr.JTu(X, u), v0, args.iters)
        return s

    found = []
    # random sparse mixtures and single embeddings
    for _ in range(args.samples // args.batch):
        k = int(torch.randint(1, args.support + 1, (1,), generator=gen, device=loop.device).item())
        ids = torch.randint(0, V, (args.batch, k), generator=gen, device=loop.device)
        wts = torch.distributions.Dirichlet(torch.ones(k, device=loop.device)).sample((args.batch,))
        Xs = (wts[:, :, None].to(E.dtype) * E[ids]).sum(1)
        s = norms(Xs)
        for i in range(args.batch):
            found.append((float(s[i]), ids[i].tolist(), wts[i].tolist()))
    found.sort(key=lambda t: -t[0])
    sampled = found[0][0]
    # random search on the mixture weights of the best starts (support fixed)
    ascent = []
    for s0, ids, wts in found[:args.climb_starts]:
        ids_t = torch.as_tensor(ids, device=loop.device)
        theta = torch.log(torch.as_tensor(wts, device=loop.device).clamp_min(1e-12))
        bestv = s0
        for _ in range(args.climb_steps):
            cand = theta[None, :] + args.climb_sigma * torch.randn((args.batch, len(ids)), generator=gen,
                                                                     device=loop.device)
            q = torch.softmax(cand, dim=1)
            Xs = (q[:, :, None].to(E.dtype) * E[ids_t][None]).sum(1)
            s = norms(Xs)
            j = int(torch.argmax(s))
            if float(s[j]) > bestv:
                bestv, theta = float(s[j]), cand[j]
        ascent.append(bestv)
    L = max([sampled] + ascent)
    from_states = None
    if args.from_cells:
        src = json.loads((DATA / "explore" / args.from_cells / "nb.json").read_text())
        from_states = float(np.exp(max(c["max_log_J"] for c in src["cells"])))
        L = max(L, from_states)
    rep = {"args": vars(args), "diam_E": diam, "L_sampled_mixtures": sampled, "L_after_random_search": max(ascent),
           "L_from_orbit_states": from_states, "L_estimate": L, "T_star": diam ** 2 * L / 4.0,
           "log10_T_star": float(np.log10(diam ** 2 * L / 4.0)), "seconds": round(time.perf_counter() - tic, 1),
           "note": "L is a lower estimate of sup_K ||J_h||, so T* is an estimate, not a guarantee."}
    (out / "sup.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("cells", "sup"):
        s = sub.add_parser(name)
        s.add_argument("--model", default="qwen0.5b")
        s.add_argument("--layer", type=int, default=12)
        s.add_argument("--dtype", default="float32")
        s.add_argument("--device", default="cpu")
        s.add_argument("--threads", type=int, default=8)
        s.add_argument("--iters", type=int, default=6)
        s.add_argument("--out", required=True)
    c = sub.choices["cells"]
    c.add_argument("--cells-file", required=True)
    c.add_argument("--source", default=None)
    c.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    c.add_argument("--concept", default="philosophy")
    c.add_argument("--transient", type=int, default=300)
    c.add_argument("--measure", type=int, default=60)
    c.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    c.add_argument("--batch", type=int, default=256)
    s = sub.choices["sup"]
    s.add_argument("--samples", type=int, default=8192)
    s.add_argument("--support", type=int, default=8)
    s.add_argument("--batch", type=int, default=256)
    s.add_argument("--climb-starts", type=int, default=16)
    s.add_argument("--climb-steps", type=int, default=20)
    s.add_argument("--climb-sigma", type=float, default=0.5)
    s.add_argument("--from-cells", dest="from_cells", default=None)
    args = ap.parse_args()
    if args.mode == "cells" and args.vectors in ("", "none"):
        args.vectors = None
    (cells_mode if args.mode == "cells" else sup_mode)(args)


if __name__ == "__main__":
    main()
