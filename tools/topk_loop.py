"""Exploratory (run TK): the soft-token loop restricted to the k most probable tokens, deterministic or sampled.

Variants of one step x -> x' of the one-token loop (fixed prefix, window 1), from the logits z(x) at temperature T:
  soft     x' = sum over the top k tokens of q_i e_i, q = softmax of their logits / T (k = 0: the whole vocabulary,
           the paper's map). Piecewise smooth: smooth while the top-k set is fixed, discontinuous where it changes.
           k = 1 is greedy decoding with a one-token memory.
  sampled  x' = e_t, t drawn from q by inverse CDF over the tokens in rank order with one uniform number per cell and
           step, shared by the cell's trajectory and its twin (common noise). For fixed noise the map is piecewise
           constant, so an infinitesimal offset either vanishes at the next step or, with probability O(offset), the
           twins draw different tokens.

Subcommands
  scan        tools/explore_llm_soft_loop.py scan with the step replaced (same grid, twin and labels). Extra output in
              topk.npz: per cell the steps whose top-k sets of trajectory and twin differ (soft: the twin crossed a
              discontinuity, so its exponent may contain jumps) and, sampled, the steps whose drawn tokens differ.
              Sampled twins get a fresh offset of size eps in a random direction before every step (instead of the
              renormalized difference, which would stay zero once they draw the same token), so the token-flip
              fraction per step at offset eps is measured directly.
  divergence  Twins at several offsets delta, never renormalized, for N steps from states on the attractor: the
              log separation per step. Chaos makes a separation of delta reach a threshold after about
              ln(threshold / delta) / lambda steps; sampled decoding flips a token with probability ~ c delta.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402
import explore_llm_soft_loop as esl  # noqa: E402

DATA = ela.DATA


class TopKLoop(esl.SoftLoop):
    """SoftLoop with a top-k (and optionally sampled) step. In scan, rows 0..n-1 are trajectories and n..2n-1
    their twins; the counters compare the two halves at every step (zero during the transient, when twins are
    exact copies)."""

    def __init__(self, *a, top_k=0, sample=False, eps=1e-6, seed=0, transient=0, **kw):
        super().__init__(*a, **kw)
        self.top_k, self.sample, self.eps = int(top_k), bool(sample), float(eps)
        self.transient = int(transient)       # scan: steps before the twins start; the counters skip them
        self.rng = np.random.default_rng(seed)
        self.dir_rng = np.random.default_rng(seed + 1)
        self.paired = True                    # rows split into trajectory and twin halves (scan); False: divergence
        self.switch, self.flip, self.counts, self.t = None, None, [], 0

    def prepare(self, steer_rows):
        if self.switch is not None:
            self.counts.append((self.switch.copy(), self.flip.copy()))
        n = len(steer_rows) // 2 if self.paired else len(steer_rows)
        self.switch, self.flip, self.t = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64), 0
        return super().prepare(steer_rows)

    def finish(self):
        """Per-cell counters of every chunk so far, in cell order."""
        if self.switch is not None:
            self.counts.append((self.switch.copy(), self.flip.copy()))
            self.switch = None
        return (np.concatenate([c[0] for c in self.counts]), np.concatenate([c[1] for c in self.counts]))

    def uniforms(self, B):
        n = B // 2 if self.paired else B
        u = self.rng.random(n)
        return np.concatenate([u, u]) if self.paired else u

    def step(self, X, temps):
        torch = self.torch
        B = X.shape[0]
        if self.sample and self.paired:       # fresh offset of size eps for every twin before the step
            n = B // 2
            d = torch.as_tensor(self.dir_rng.normal(size=tuple(X[n:].shape)), dtype=self.state_dtype,
                                device=self.device)
            d = d / d.flatten(1).norm(dim=1)[:, None, None] * self.eps
            X = torch.cat([X[:n], X[:n] + d])
        z = self.logits(X)
        T = torch.as_tensor(temps, dtype=self.state_dtype, device=self.device)[:, None]
        V = z.shape[1]
        k = self.top_k or V
        tok, idx = None, None
        if k == V and not self.sample:        # the paper's map, computed as in SoftLoop.step
            p = torch.softmax(z / T, dim=-1)
            new = p @ self.E
        else:
            vals, idx = torch.topk(z, k, dim=-1) if k < V else torch.sort(z, dim=-1, descending=True)
            q = torch.softmax(vals / T, dim=-1)
            p = torch.zeros_like(z).scatter_(1, idx, q)
            if self.sample:
                u = torch.as_tensor(self.uniforms(B), dtype=self.state_dtype, device=self.device)[:, None]
                r = torch.searchsorted(q.cumsum(-1), u).clamp(max=k - 1)
                tok = idx.gather(1, r)[:, 0]
                new = self.E[tok]
            else:
                new = (q[:, :, None] * self.E[idx]).sum(1)
        self.t += 1
        if self.paired and B % 2 == 0 and self.t > self.transient:
            n = B // 2
            if idx is not None and k < V:
                a, b = idx[:n].sort(dim=-1).values, idx[n:].sort(dim=-1).values
                self.switch += (a != b).any(-1).cpu().numpy()
            if tok is not None:
                self.flip += (tok[:n] != tok[n:]).cpu().numpy()
        return torch.cat([X[:, 1:, :], new[:, None, :]], dim=1), p


def scan(args):
    def factory(*a, **kw):
        return TopKLoop(*a, top_k=args.top_k, sample=args.sample, eps=args.eps, seed=args.seed,
                        transient=args.transient, **kw)

    holder = {}

    def capture(*a, **kw):
        holder["loop"] = factory(*a, **kw)
        return holder["loop"]

    esl.scan(args, loop_factory=capture)
    switch, flip = holder["loop"].finish()
    out = DATA / "explore" / args.out
    shape = tuple(args.resolution)
    np.savez_compressed(out / "topk.npz", top_k=args.top_k, sample=args.sample, eps=args.eps,
                        set_switch_steps=switch.reshape(shape), token_flip_steps=flip.reshape(shape),
                        samples=args.samples)
    info = {"top_k": args.top_k, "sample": args.sample, "eps": args.eps, "samples": args.samples,
            "cells_with_set_switches": int((switch > 0).sum()), "mean_flip_fraction": float(flip.mean() / args.samples)}
    (out / "topk.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def divergence(args):
    """Twins at offsets delta from attractor states, not renormalized: log10 separation per step for each cell,
    offset and repeat (repeats differ in the offset direction and, sampled, in the noise)."""
    loop = TopKLoop(args.model, args.layer, args.threads, args.prompt, args.opener, 1, args.dtype, args.device,
                    args.weights, top_k=args.top_k, sample=args.sample, seed=args.seed)
    torch = loop.torch
    v = elpm.load_vectors(args.vectors)[args.concept]
    cells = [tuple(c) for c in args.cells]
    nc, nd, R = len(cells), len(args.deltas), args.repeats
    temps_c = np.array([10.0 ** lt for lt, _ in cells])
    steer_c = np.array([a * v for _, a in cells])
    loop.paired = False                       # one row per cell during the transient
    X = loop.prepare(steer_c)
    for _ in range(args.transient):
        X, _ = loop.step(X, temps_c)
    # rows: cell-major, then repeat, then (base, delta_1 .. delta_nd)
    G = nd + 1
    rows = nc * R * G
    steer = np.repeat(steer_c, R * G, axis=0)
    temps = np.repeat(temps_c, R * G)
    loop.prepare(steer)
    base = X.repeat_interleave(R * G, dim=0)
    rng = np.random.default_rng(args.seed + 7)
    u = torch.as_tensor(rng.normal(size=tuple(base.shape)), dtype=loop.state_dtype, device=loop.device)
    u = u / u.flatten(1).norm(dim=1)[:, None, None]
    scale = torch.as_tensor(np.tile(np.concatenate([[0.0], args.deltas]), nc * R), dtype=loop.state_dtype,
                            device=loop.device)[:, None, None]
    Y = base + scale * u
    sep = np.zeros((args.steps, nc, R, nd))
    noise = np.random.default_rng(args.seed + 11)
    for s in range(args.steps):
        if args.sample:                       # one uniform per (cell, repeat), shared by its base and twins
            uu = noise.random(nc * R)
            loop.uniforms = lambda B, uu=uu: np.repeat(uu, G)
        Y, _ = loop.step(Y, temps)
        Z = Y[:, -1].reshape(nc, R, G, -1)
        d = (Z[:, :, 1:] - Z[:, :, :1]).norm(dim=-1).double().cpu().numpy()
        sep[s] = np.log10(np.maximum(d, 1e-300))
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "divergence.npz", log10_separation=sep, cells=np.array(cells), deltas=args.deltas,
                        top_k=args.top_k, sample=args.sample)
    state = float(X[:, -1].norm(dim=-1).mean())
    thr = np.log10(args.threshold * state)
    summary = []
    for c in range(nc):
        for j, delta in enumerate(args.deltas):
            hit = sep[:, c, :, j] > thr
            first = np.where(hit.any(0), hit.argmax(0) + 1, -1)
            summary.append({"cell": cells[c], "delta": delta, "reached": int((first > 0).sum()), "of": R,
                            "median_steps": float(np.median(first[first > 0])) if (first > 0).any() else None})
    (out / "divergence.json").write_text(json.dumps({"args": vars(args), "state_norm": state, "threshold_log10": thr,
                                                     "summary": summary}, indent=1))
    for row in summary:
        print(json.dumps(row))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scan")
    dv = sub.add_parser("divergence")
    for p in (sc, dv):
        p.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
        p.add_argument("--layer", type=int, default=12)
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
        p.add_argument("--opener", default=ela.OPENER)
        p.add_argument("--dtype", default="float64", choices=("float32", "float64"))
        p.add_argument("--device", default="cpu")
        p.add_argument("--weights", default="native", choices=("native", "bf16"))
        p.add_argument("--vectors", required=True)
        p.add_argument("--concept", default="philosophy")
        p.add_argument("--top-k", type=int, default=0, help="0: the whole vocabulary")
        p.add_argument("--sample", action="store_true")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", required=True)
    # the scan options of explore_llm_soft_loop.py, with its defaults
    sc.add_argument("--window", type=int, default=1)
    sc.add_argument("--alpha", type=float, nargs=2, default=[0.0, 0.0])
    sc.add_argument("--log-temperature", type=float, nargs=2, default=[-1.0, 0.3])
    sc.add_argument("--resolution", type=int, nargs=2, default=[128, 1])
    sc.add_argument("--transient", type=int, default=1000)
    sc.add_argument("--samples", type=int, default=2000)
    sc.add_argument("--record", type=int, default=128)
    sc.add_argument("--max-period", type=int, default=64)
    sc.add_argument("--tol", type=float, default=1e-8)
    sc.add_argument("--eps", type=float, default=1e-6)
    sc.add_argument("--chaos-level", type=float, default=1e-3)
    sc.add_argument("--chunk", type=int, default=0)
    for name, default in (("prefix", None), ("start_vector", None), ("init_states", None)):
        sc.add_argument("--" + name.replace("_", "-"), default=default)
    sc.add_argument("--projections", action="store_true")
    sc.add_argument("--save-states", action="store_true")
    sc.add_argument("--word-period", type=int, default=0)
    sc.add_argument("--return-period", type=int, default=0)
    dv.add_argument("--cells", type=float, nargs=2, action="append", required=True, metavar=("LOG10T", "ALPHA"))
    dv.add_argument("--deltas", type=float, nargs="+", default=[1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
    dv.add_argument("--repeats", type=int, default=32)
    dv.add_argument("--transient", type=int, default=500)
    dv.add_argument("--steps", type=int, default=200)
    dv.add_argument("--threshold", type=float, default=0.1, help="separation threshold, fraction of the state norm")
    args = ap.parse_args()
    if args.cmd == "scan":
        if args.sample and args.top_k == 1:
            raise SystemExit("sampling from the top 1 token is greedy decoding: use the soft step with --top-k 1")
        scan(args)
    else:
        divergence(args)


if __name__ == "__main__":
    main()
