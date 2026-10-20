"""Superstable curves of one periodic orbit of the soft-token map in the (alpha, log10 T) plane, and where two of them
cross: a doubly superstable point, the centre of a shrimp (Gallas 1993; Jones 2012, arXiv:1201.4343).

For a period-p orbit x_0, ..., x_{p-1} with leading multiplier mu and its eigenvector v, the chain
v_{i+1} = DF(x_i) v_i splits mu into one factor per map step. With fixed reference directions r_i (unit vectors: the
chain directions at the seed point), the signed factors

    c_i = <DF(x_i) v_i, r_{i+1}> / <v_i, r_i>,   i = 0..p-1 (r_p = r_0),

are scale-invariant in v and multiply to mu exactly (the product telescopes to <M v, r_0> / <v, r_0>). In an
effectively one-dimensional map, c_i is the derivative of the reduced map at x_i, and a superstable curve is where
one factor vanishes. Where the zero curves of two different factors cross, the orbit is doubly superstable. A single
factor's zero curve does not cross itself generically, and a crossing seen on a map could also be two coexisting
orbits; here one orbit is followed by continuation and its token loop is checked at every point.

Near a zero of mu the leading multiplier falls below the transverse ones and its eigenvector is no longer defined;
such points are flagged (clean: |mu_1| > 10 |mu_2|), and zero curves are read from clean points only.

grid: continuation over a box, breadth-first from the seed cell, each cell warm-started from an already solved
neighbour (Newton on F^p(x) = x with the low-rank solver of tools/llm_feigenbaum.py, Ritz multipliers, the factors
and the token loop). Writes grid.npz and grid.json under explore/OUT.

A cell is solved only when Newton converges to a primitive period-p orbit. Only solved cells seed their neighbours:
a cell where Newton converged onto an orbit of a divisor period (the orbit the period-p one doubled from) or did not
converge propagates nothing, and is retried from up to --attempts distinct solved neighbours (Frontier). Before
16 October 2026 every converged cell seeded its neighbours, so one cell that slipped onto the period-p/2 orbit passed
the slip on to all the cells reached through it (run SHC-L, P8L).

Usage:
  python tools/doubly_superstable.py grid --period 7 --seed 0.0700 0.2040 --alpha 0.0686 0.0714 \
      --log-temperature 0.199 0.209 --resolution 29 29 --vectors llm-phase-map/vectors-l12-p1b \
      --concept hostility-vs-kindness --device cuda --out llm-soft-loop/pod64/dss/p7-blob
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import llm_feigenbaum as lfb  # noqa: E402


def factors(cell, X, period, v, refs):
    """Signed per-step factors c_i of the leading multiplier along the orbit from X, chain directions (unit), and
    the argmax token of every step."""
    torch = cell.torch
    vi = v / np.linalg.norm(v)
    cs, dirs, toks = [], [], []
    for i in range(period):
        toks.append(int(cell.probabilities(X).argmax()))
        w = lfb._cols(cell.jvp_rows(X, lfb._rows(vi[:, None], X)))[:, 0]
        r_in, r_out = refs[i], refs[(i + 1) % period]
        cs.append(float(np.dot(w, r_out) / np.dot(vi, r_in)))
        dirs.append(vi)
        vi = w / np.linalg.norm(w)
        X = cell.f(X)
    return np.array(cs), dirs, toks


def chain_dirs(cell, X, period, v):
    """Chain of the leading eigenvector along the orbit: unit directions d_0..d_{p-1}, stretchings s_i = |DF(x_i) d_i|
    (|mu| = prod s_i for the exact eigenvector), and d_p, the image of d_{p-1} (+-d_0 at a converged eigenvector)."""
    d = v / np.linalg.norm(v)
    dirs, s = [], []
    for i in range(period):
        w = lfb._cols(cell.jvp_rows(X, lfb._rows(d[:, None], X)))[:, 0]
        dirs.append(d)
        s.append(float(np.linalg.norm(w)))
        d = w / np.linalg.norm(w)
        X = cell.f(X)
    return dirs, np.array(s), d


def transported(dirs, s, d_end, parent):
    """Signed factors with orientations carried from a neighbouring cell (no fixed references, so no poles): tau_i =
    sign <d_i, o_i^parent>, o_i = tau_i d_i, c_i = tau_i tau_{i+1} s_i and c_{p-1} = tau_{p-1} tau_0 eps s_{p-1} with
    eps = sign <d_p, d_0>. The product of the c_i is eps prod s_i = mu at a converged eigenvector. A factor's sign flips
    where it passes through zero (its output direction reverses). Returns (c, oriented directions, smallest
    |cosine| to the parent's directions: a measure of how safely the orientation was carried)."""
    p = len(dirs)
    if parent is None:
        tau = np.ones(p)
        cmin = 1.0
    else:
        cos = np.array([float(np.dot(d, o)) for d, o in zip(dirs, parent)])
        tau = np.sign(cos)
        tau[tau == 0] = 1.0
        cmin = float(np.min(np.abs(cos)))
    eps = float(np.sign(np.dot(d_end, dirs[0]))) or 1.0
    c = np.array([tau[i] * tau[(i + 1) % p] * s[i] for i in range(p)])
    c[p - 1] *= eps
    return c, [t * d for t, d in zip(tau, dirs)], cmin


def leading(vals, vecs):
    """Leading multiplier (largest modulus) as a real number when it is real, its real eigenvector, and whether it
    dominates the transverse spectrum."""
    mu = vals[0]
    v = vecs[:, 0]
    v = np.real(v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))])))
    clean = bool(len(vals) < 2 or abs(mu) > 10.0 * abs(vals[1]))
    return mu, v, clean


class Frontier:
    """Breadth-first continuation order over an (nt, na) grid. A cell is queued from a solved neighbour (its parent);
    only solved cells queue their neighbours. A cell whose attempt fails is queued again from another solved
    neighbour it has not yet been tried from, until it has been tried from `attempts` parents."""

    def __init__(self, shape, start, attempts):
        self.shape, self.attempts = shape, attempts
        self.queue = collections.deque([(start, None)])
        self.pending = {start}
        self.solved = set()
        self.tries = collections.defaultdict(set)              # cell -> parents it was tried from

    def neighbours(self, cell):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cell[0] + di, cell[1] + dj)
            if 0 <= nb[0] < self.shape[0] and 0 <= nb[1] < self.shape[1]:
                yield nb

    def _open(self, cell, parent):
        return (cell not in self.solved and cell not in self.pending and parent not in self.tries[cell]
                and len(self.tries[cell]) < self.attempts)

    def pop(self):
        cell, parent = self.queue.popleft()
        self.pending.discard(cell)
        self.tries[cell].add(parent)
        return cell, parent

    def done(self, cell, solved):
        if solved:
            self.solved.add(cell)
            for nb in self.neighbours(cell):
                if self._open(nb, cell):
                    self.pending.add(nb)
                    self.queue.append((nb, cell))
            return
        for nb in self.neighbours(cell):                       # retry from a solved neighbour not yet used
            if nb in self.solved and self._open(cell, nb):
                self.pending.add(cell)
                self.queue.append((cell, nb))
                return

    def parents_still_useful(self):
        """Solved cells that are, or may yet become, the parent of a queued or retried cell."""
        keep = {par for _, par in self.queue}
        for c in self.solved:
            if any(self._open(nb, c) or (nb in self.pending) for nb in self.neighbours(c)):
                keep.add(c)
        return keep


def grid(args):
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    opener = args.opener if args.opener is not None else ela.OPENER
    loop = esl.SoftLoop(args.model, args.layer, args.threads, args.prompt, opener, args.window, "float64", args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    p = args.period
    if args.block2 is None and args.log_temperature is None:
        raise SystemExit("grid: --log-temperature is required (or --block2 with --alpha2 for a two-block plane)")
    if args.block2 is not None and None in (args.vectors2, args.concept2, args.alpha2, args.fixed_log_temperature):
        raise SystemExit("grid: a two-block plane needs --vectors2, --concept2, --alpha2 and --fixed-log-temperature")
    row_key = "alpha2" if args.block2 is not None else "log10_temperature"

    if args.block2 is not None:          # two-block plane (run TB): rows are alpha2 at a fixed temperature
        loop.sm.add_steer_layer(args.block2)
        w = elpm.load_vectors(args.vectors2)[args.concept2]

        def make_cell(a, a2):
            loop.extra = (args.block2, np.array([a2 * w]))
            return esl.SoftOrbit(loop, a * v, 10.0 ** args.fixed_log_temperature, chunk=args.chunk)

        lts = np.linspace(*args.alpha2, args.resolution[0])
    else:
        def make_cell(a, lt):
            return esl.SoftOrbit(loop, a * v, 10.0 ** lt, chunk=args.chunk)

        lts = np.linspace(*args.log_temperature, args.resolution[0])
    als = np.linspace(*args.alpha, args.resolution[1])
    nt, na = len(lts), len(als)
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "progress.jsonl", "a")

    def emit(rec):
        line = json.dumps(rec)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    # seed: the attractor reached from the fixed start must have period p, at the seed or, with --march-from, at a
    # start cell from which the orbit is continued (Newton at each of --march-steps points of a straight line) to the seed
    tic = time.perf_counter()
    a0, lt0 = args.seed
    s0 = tuple(args.march_from) if args.march_from else (a0, lt0)
    cell = make_cell(*s0)
    X = cell.iterate(cell.start, args.transient)
    found = cell.detect_period(X, max(p, 32), 1e-6)
    if found != p:
        raise SystemExit(f"the attractor at the start cell {s0} has period {found}, not {p}: refusing to start")
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), args.k)))[0]
    X, Q, hist = lfb.newton(cell, X, p, Q, lfb.tol_for(p, args.tol), 12)
    if not hist[-1] < lfb.tol_for(p, args.tol):
        raise SystemExit(f"Newton did not converge at the start cell: {hist}")
    if args.march_from:
        march = []
        if args.march_path:
            way = [tuple(q) for q in json.loads(Path(args.march_path).read_text())] + [(a0, lt0)]
            pts, prev = [], s0
            for q in way:
                for k in range(1, args.march_sub + 1):
                    u = k / args.march_sub
                    pts.append((prev[0] + u * (q[0] - prev[0]), prev[1] + u * (q[1] - prev[1])))
                prev = q
        else:
            pts = [(s0[0] + k / args.march_steps * (a0 - s0[0]), s0[1] + k / args.march_steps * (lt0 - s0[1]))
                   for k in range(1, args.march_steps + 1)]
        for k, pt in enumerate(pts, 1):
            cell = make_cell(*pt)
            X, Q, hist = lfb.newton(cell, X, p, Q, lfb.tol_for(p, args.tol), 12)
            march.append({"point": [float(pt[0]), float(pt[1])], "newton": float(hist[-1])})
            if not hist[-1] < lfb.tol_for(p, args.tol):
                emit({"march": march})
                raise SystemExit(f"the march lost the orbit at step {k} {pt}: {hist}")
        emit({"march_from": list(s0), "march_steps": args.march_steps, "march_last_newton": march[-1]["newton"]})
    vals, vecs, res, rh, Q = lfb.ritz(cell, X, p, Q)
    mu, vlead, clean = leading(vals, vecs)
    seed_ok = abs(mu.imag) <= 1e-8 and (len(vals) < 2 or abs(mu) > args.seed_min_ratio * abs(vals[1]))
    if not args.multipliers_only and not seed_ok:
        raise SystemExit(f"leading multiplier at the seed is {mu!r} (next {abs(vals[1]):.3g}): choose a seed away from "
                         "the superstable curves")
    refs = [np.ones(1)] * p
    if args.multipliers_only:                      # two-dimensional regime: no step factors, only the multipliers
        vlead = np.real(vecs[:, 0])
    _, dirs, seed_toks = factors(cell, X, p, vlead, [np.ones(X.numel())] * p)
    refs = [d / np.linalg.norm(d) for d in dirs]
    c_seed, _, _ = factors(cell, X, p, vlead, refs)
    sd, ss, se = chain_dirs(cell, X, p, vlead)
    ct_seed, orient_seed, _ = transported(sd, ss, se, None)
    emit({"seed": [a0, lt0], "mu": float(mu.real), "next_modulus": float(abs(vals[1])), "factors": c_seed.tolist(),
          "product": float(np.prod(c_seed)), "tokens": seed_toks, "seconds": round(time.perf_counter() - tic, 1)})

    # breadth-first continuation from the grid cell nearest the seed
    i0, j0 = int(np.argmin(abs(lts - lt0))), int(np.argmin(abs(als - a0)))
    MU = np.full((nt, na), np.nan)
    MU_IM = np.full((nt, na), np.nan)
    NEXT = np.full((nt, na), np.nan)
    CLEAN = np.zeros((nt, na), dtype=bool)
    OK = np.zeros((nt, na), dtype=bool)
    SAME = np.zeros((nt, na), dtype=bool)
    C = np.full((nt, na, p), np.nan)
    PROD = np.full((nt, na), np.nan)
    RES = np.full((nt, na), np.nan)
    COS = np.full((nt, na), np.nan)
    CT = np.full((nt, na, p), np.nan)               # signed factors with transported orientation
    SM = np.full((nt, na, p), np.nan)               # stretchings s_i
    TCOS = np.full((nt, na), np.nan)                # smallest |cos| to the parent's oriented directions
    M2 = np.full((nt, na, 3, 2), np.nan)            # three leading multipliers (real, imag) by modulus
    RV = None                                       # --store-ritz: the three leading Ritz vectors per solved cell
    states = {}
    front = Frontier((nt, na), (i0, j0), args.attempts)
    states[None] = (X, Q, orient_seed)
    n_done = 0
    while front.queue:
        (i, j), parent = front.pop()
        Xs, Qs, Os = states[parent]
        t = time.perf_counter()
        cell = make_cell(als[j], lts[i])
        try:
            Xn, Qn, hist = lfb.newton(cell, Xs, p, Qs, lfb.tol_for(p, args.tol), 12)
            ok = bool(hist[-1] < lfb.tol_for(p, args.tol))
        except Exception as err:                       # noqa: BLE001 - recorded, the cell stays unsolved
            ok, hist = False, [str(err)]
        rec = {"i": i, "j": j, "alpha": float(als[j]), row_key: float(lts[i]), "converged": ok,
               "parent": None if parent is None else list(parent), "attempt": len(front.tries[(i, j)])}
        if ok:
            split = min((lfb.return_distance(cell, Xn, q) for q in range(1, p) if p % q == 0), default=1.0)
            vals, vecs, res, rh, Qn = lfb.ritz(cell, Xn, p, Qn)
            mu, vlead, clean = leading(vals, vecs)
            M2[i, j, :len(vals[:3])] = np.array([[z.real, z.imag] for z in vals[:3]])
            if args.store_ritz:
                if RV is None:
                    RV = np.full((nt, na, vecs.shape[0], 3), np.nan, dtype=complex)
                RV[i, j, :, :min(3, vecs.shape[1])] = vecs[:, :3]
            if args.multipliers_only:
                vlead = np.real(vecs[:, 0])
            c, dirs, toks = factors(cell, Xn, p, vlead, refs)
            cd, cs_, ce = chain_dirs(cell, Xn, p, vlead)
            ct, orient, tcos = transported(cd, cs_, ce, Os)
            CT[i, j], SM[i, j], TCOS[i, j] = ct, cs_, tcos
            OK[i, j] = split > 1e-7
            MU[i, j], MU_IM[i, j], NEXT[i, j], CLEAN[i, j] = mu.real, mu.imag, abs(vals[1]), clean
            C[i, j], PROD[i, j], RES[i, j] = c, np.prod(c), hist[-1]
            COS[i, j] = min(abs(float(np.dot(d, r))) for d, r in zip(dirs, refs))
            SAME[i, j] = toks == seed_toks
            if OK[i, j]:
                states[(i, j)] = (Xn, Qn, orient)
            rec.update({"mu": float(mu.real), "mu_imag": float(mu.imag), "next_modulus": float(abs(vals[1])),
                        "clean": clean, "factors": c.tolist(), "product": float(np.prod(c)), "same_tokens": SAME[i, j].item(),
                        "min_reference_cosine": float(COS[i, j]), "newton": float(hist[-1]), "split": float(split),
                        "factors_transported": ct.tolist(), "stretchings": cs_.tolist(), "transport_cosine": tcos})
        front.done((i, j), bool(OK[i, j]))
        rec["solved"] = bool(OK[i, j])
        rec["seconds"] = round(time.perf_counter() - t, 2)
        emit(rec)
        n_done += 1
        if n_done % 25 == 0 or not front.queue:
            np.savez_compressed(out / "grid.npz", alpha=als, log10_temperature=lts, mu=MU, mu_imag=MU_IM, next_modulus=NEXT,
                                clean=CLEAN, ok=OK, same_tokens=SAME, factors=C, product=PROD, newton=RES,
                                min_reference_cosine=COS, factors_transported=CT, stretchings=SM, transport_cosine=TCOS,
                                multipliers=M2, row_axis=np.array("alpha2" if args.block2 is not None else "log10_temperature"),
                                seed=np.array(args.seed), seed_tokens=np.array(seed_toks))
            if args.store_ritz and RV is not None:
                np.savez_compressed(out / "ritz.npz", vectors=RV, multipliers=M2, alpha=als, log10_temperature=lts)
        # free states that no queued or retried cell can still need
        live = front.parents_still_useful()
        for key in [k for k in states if k is not None and k not in live]:
            del states[key]
    tok = loop.sm.tok
    (out / "grid.json").write_text(json.dumps({
        "args": vars(args), "seed_tokens": seed_toks, "seed_text": [tok.decode([t]) for t in seed_toks],
        "cells": int(nt * na), "solved": int(OK.sum()), "clean": int((OK & CLEAN).sum()),
        "same_tokens": int((OK & SAME).sum()), "seconds": round(time.perf_counter() - tic, 1)}, indent=1,
        ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--period", type=int, required=True)
    g.add_argument("--seed", type=float, nargs=2, required=True, metavar=("ALPHA", "LOG10T"),
                   help="seed cell (alpha, log10 T); with --block2: (alpha1, alpha2)")
    g.add_argument("--alpha", type=float, nargs=2, required=True)
    g.add_argument("--log-temperature", type=float, nargs=2, default=None,
                   help="row axis (one-block plane); with --block2 the rows are alpha2 instead")
    g.add_argument("--block2", type=int, default=None, help="second steering block (two-block plane, run TB)")
    g.add_argument("--vectors2", default=None)
    g.add_argument("--concept2", default=None)
    g.add_argument("--alpha2", type=float, nargs=2, default=None, help="row axis of a two-block plane")
    g.add_argument("--fixed-log-temperature", type=float, default=None, help="temperature of a two-block plane")
    g.add_argument("--resolution", type=int, nargs=2, required=True, help="rows (temperature) and columns (alpha)")
    g.add_argument("--model", default="qwen0.5b")
    g.add_argument("--layer", type=int, default=12)
    g.add_argument("--threads", type=int, default=8)
    g.add_argument("--device", default="cpu")
    g.add_argument("--prompt", default="open")
    g.add_argument("--opener", default=None)
    g.add_argument("--window", type=int, default=1)
    g.add_argument("--vectors", default="llm-phase-map/vectors-l12-p1b")
    g.add_argument("--concept", default="hostility-vs-kindness")
    g.add_argument("--transient", type=int, default=900)
    g.add_argument("--k", type=int, default=6)
    g.add_argument("--tol", type=float, default=1e-12)
    g.add_argument("--chunk", type=int, default=512)
    g.add_argument("--store-ritz", action="store_true",
                   help="also store the three leading Ritz vectors (and multipliers) of every solved cell in ritz.npz, "
                        "for tracking a two- or three-dimensional invariant subspace across the grid")
    g.add_argument("--multipliers-only", action="store_true",
                   help="two-dimensional regime: accept a complex or unclean leading multiplier at the seed and record "
                        "the three leading multipliers per cell (factors are still written but are not meaningful)")
    g.add_argument("--march-from", type=float, nargs=2, default=None, metavar=("ALPHA", "ROW"),
                   help="start where the fixed start reaches the period-p orbit and continue it to --seed along a line")
    g.add_argument("--march-steps", type=int, default=24)
    g.add_argument("--march-path", default=None, help="JSON list of [alpha, row] waypoints to march through")
    g.add_argument("--march-sub", type=int, default=2, help="sub-steps between consecutive waypoints")
    g.add_argument("--seed-min-ratio", type=float, default=10.0,
                   help="the seed needs a real leading multiplier at least this many times the next modulus (cells "
                        "are still labelled clean at 10); a lower value only relaxes the seed, whose chain directions "
                        "set the references of the signed factors, not the transported factors or stretchings")
    g.add_argument("--attempts", type=int, default=3,
                   help="distinct solved neighbours a failed cell is tried from (only solved cells seed neighbours)")
    g.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.command == "grid":
        grid(args)


if __name__ == "__main__":
    main()
