"""Attractor sheets over the parameter plane (exploratory run SH; human request 18 Oct 2026).

The published maps start every cell from one fixed state, so where two attractors coexist they show only the one whose
basin contains that state. This tool reveals what they hide. It uses the soft-token map of tools/explore_llm_soft_loop.py
(the paper's prefix, an additive steering vector after one decoder layer, window 1) and the same labels as its scan:
- period: the smallest q <= --max-period with |x_t - x_(t-q)| < tol * max(1, |x_t|) over the recorded states;
- family: the rotation-invariant loop of leading tokens over one period;
- Lyapunov exponent: a renormalized twin at relative offset --eps over the sample steps;
- entropy: of the last next-token distribution.

It adds three records per cell:
- **settle:** the step after which the state stays within --settle-tol (relative) of one of its last --max-period
  states. Long transients mark the edges of basins and the ghosts of folds.
- **keyprob:** the mean probability of a few key tokens over the recorded steps, a low-dimensional fingerprint of the
  attractor.
- **final state:** float32, for warm starts and later analysis.

Modes:
- **seeded:** every cell starts from one state, either the maps' fixed start (`--start default`) or a saved state
  (`--start FILE.npy`, shape (d,)). The map then shows which attractor that state reaches at each cell.
- **sweep:** adiabatic continuation. The cells of one column start from the final states of the previous column, so an
  attractor is followed across the plane until it disappears. `--sweeps` lists one or more of alpha+, alpha-, temp+
  and temp- (the direction of travel). Several sweeps run in lockstep in one batch. The first column of each sweep
  starts from the fixed start with --first-transient steps.

Output: DATA/explore/OUT/sheets.npz (arrays of shape (n_T, n_alpha); with one array per sweep in sweep mode) and
sheets.json (arguments, families with their tokens, timing).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_llm_phase_map as elpm  # noqa: E402
from explore_llm_soft_loop import DATA, SoftLoop  # noqa: E402

KEY_TOKENS = [" it", " there", " is", " a", "\n\n", "The", " existence", " question", "."]


def canonical_loop(tokens):
    """Rotation-invariant key of a token loop (as in explore_llm_soft_loop.scan)."""
    t = tuple(int(x) for x in tokens)
    return min(t[i:] + t[:i] for i in range(len(t)))


def detect_period(hist, max_period, tol, reps=1):
    """hist: (record, n, d) float64. Smallest q with |x_last - x_(last-k q)| < tol * max(1, |x_last|) for k = 1..reps
    (reps = 1 is the scan's rule), else 0."""
    scale = np.maximum(1.0, np.linalg.norm(hist[-1], axis=1))
    per = np.zeros(hist.shape[1], dtype=int)
    for q in range(min(max_period, (len(hist) - 1) // reps), 0, -1):
        close = np.ones(hist.shape[1], dtype=bool)
        for k in range(1, reps + 1):
            close &= np.linalg.norm(hist[-1 - k * q] - hist[-1], axis=1) < tol * scale
        per = np.where(close, q, per)
    return per


class SettleTracker:
    """Last step at which the state was farther than tol (relative) from every one of its last max_period states."""

    def __init__(self, torch, n, d, max_period, tol, dtype, device):
        self.torch = torch
        self.ring = torch.zeros((max_period, n, d), dtype=dtype, device=device)
        self.filled = 0
        self.step = 0
        self.tol = tol
        self.max_period = max_period
        self.last_bad = np.full(n, -1, dtype=int)

    def update(self, x):
        torch = self.torch
        if self.filled > 0:
            ring = self.ring[:self.filled]
            dist = (ring - x[None]).norm(dim=2) / torch.clamp(x.norm(dim=1), min=1.0)[None]
            bad = (dist.min(dim=0).values > self.tol).cpu().numpy()
            self.last_bad[bad] = self.step
        else:
            self.last_bad[:] = self.step
        self.ring[self.step % self.max_period] = x
        self.filled = min(self.filled + 1, self.max_period)
        self.step += 1

    def settle(self):
        return self.last_bad + 1


class Runner:
    def __init__(self, args):
        self.args = args
        self.loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, 1, args.dtype, args.device)
        self.torch = self.loop.torch
        self.v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
        tok = self.loop.sm.tok
        self.key_ids = []
        for s in KEY_TOKENS:
            ids = tok(s, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise SystemExit(f"key token {s!r} is not a single token: {ids}")
            self.key_ids.append(ids[0])
        self.families = {}
        self.rng = np.random.default_rng(args.seed)

    def run_cells(self, cells, X0, transient):
        """Run the cells (list of (log10 T, alpha)) from X0 (tensor (n, 1, d)) or the fixed start (None)."""
        a, loop, torch = self.args, self.loop, self.torch
        n = len(cells)
        steer = np.array([al * self.v for _, al in cells])
        temps = np.array([10.0 ** lt for lt, _ in cells])
        X = loop.prepare(steer)
        if X0 is not None:
            X = X0.to(dtype=loop.state_dtype, device=loop.device).clone()
        tracker = SettleTracker(torch, n, X.shape[2], a.max_period, a.settle_tol, loop.state_dtype, loop.device)
        for _ in range(transient):
            X, _ = loop.step(X, temps)
            tracker.update(X[:, -1, :])
        loop.prepare(np.concatenate([steer, steer]))           # prefix caches for the trajectory and its twin
        temps2 = np.tile(temps, 2)
        delta = torch.as_tensor(self.rng.normal(size=tuple(X.shape)), dtype=loop.state_dtype, device=loop.device)
        delta = delta / delta.flatten(1).norm(dim=1)[:, None, None] * a.eps
        XY = torch.cat([X, X + delta])
        log_growth = np.zeros(n)
        history, tokens = [], []
        keyp = np.zeros((n, len(self.key_ids)))
        p = None
        for k in range(a.samples):
            XY, p = loop.step(XY, temps2)
            X, Y = XY[:n], XY[n:]
            p = p[:n]
            tracker.update(X[:, -1, :])
            diff = Y - X
            dist = np.maximum(diff.flatten(1).norm(dim=1).double().cpu().numpy(), a.eps * 1e-30)
            log_growth += np.log(dist / a.eps)
            Y = X + diff * torch.as_tensor(a.eps / dist, dtype=loop.state_dtype, device=loop.device)[:, None, None]
            XY = torch.cat([X, Y])
            if k >= a.samples - a.record:
                history.append(X[:, -1, :].double().cpu().numpy().copy())
                tokens.append(p.argmax(-1).cpu().numpy().copy())
                keyp += p[:, self.key_ids].double().cpu().numpy()
        hist = np.array(history)
        per = detect_period(hist, a.max_period, a.tol, getattr(a, "reps", 1))
        toks = np.array(tokens)
        fam = np.full(n, -1, dtype=int)
        for c in range(n):
            q = int(per[c])
            if q > 0:
                fam[c] = self.families.setdefault(canonical_loop(toks[-q:, c]), len(self.families))
        last_p = p.double().cpu().numpy()
        ent = -(last_p * np.log(np.maximum(last_p, 1e-300))).sum(1)
        return {"lyapunov": log_growth / a.samples, "period": per, "family": fam, "entropy": ent,
                "settle": tracker.settle(), "keyprob": keyp / a.record, "state": X.detach().clone()}

    def family_list(self, family_arrays):
        tok = self.loop.sm.tok
        out = []
        for key, i in self.families.items():
            out.append({"family": i, "period": len(key), "tokens": [tok.decode([t]) for t in key],
                        "cells": {name: int((arr == i).sum()) for name, arr in family_arrays.items()}})
        return out


def grid(args):
    log_t = np.linspace(*args.log_temperature, args.resolution[0])
    alphas = np.linspace(*args.alpha, args.resolution[1])
    return log_t, alphas


def empty(shape, n_key, d):
    return {"lyapunov": np.full(shape, np.nan), "period": np.zeros(shape, dtype=int),
            "family": np.full(shape, -1, dtype=int), "entropy": np.full(shape, np.nan),
            "settle": np.full(shape, -1, dtype=int), "keyprob": np.full(shape + (n_key,), np.nan),
            "state": np.zeros(shape + (d,), dtype=np.float32)}


def save(out, args, runner, log_t, alphas, results, tic, final):
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"log10_temperature": log_t, "alpha": alphas, "key_tokens": np.array(KEY_TOKENS)}
    states = {}
    for name, r in results.items():
        for k, v in r.items():
            if k == "state":
                states[name] = v
            else:
                arrays[f"{name}__{k}"] = v
    np.savez_compressed(out / ("sheets.npz" if final else "sheets-partial.npz"), **arrays)
    if final:
        np.savez_compressed(out / "states.npz", **states)
        fams = runner.family_list({name: r["family"] for name, r in results.items()})
        fams.sort(key=lambda f: -sum(f["cells"].values()))
        (out / "sheets.json").write_text(json.dumps({"args": vars(args), "seconds": round(time.perf_counter() - tic, 1),
                                                     "maps": list(results), "families": fams}, indent=1))


def seeded(args, runner):
    torch, loop = runner.torch, runner.loop
    log_t, alphas = grid(args)
    shape = (len(log_t), len(alphas))
    d = loop.E.shape[1]
    name = args.name or "seeded"
    res = {name: empty(shape, len(KEY_TOKENS), d)}
    X0 = None
    if args.start != "default":
        path = Path(args.start)
        if not path.is_absolute() and not path.exists():
            path = DATA / "explore" / args.start           # a DATA-relative path, as uploaded to pods
        s = np.load(path).astype(np.float64).reshape(-1)
        if s.shape[0] != d:
            raise SystemExit(f"start state has dimension {s.shape[0]}, the model {d}")
        X0 = torch.as_tensor(s, dtype=loop.state_dtype, device=loop.device)[None, None, :]
    cells = [(i, j) for i in range(shape[0]) for j in range(shape[1])]
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    for s0 in range(0, len(cells), args.chunk):
        idx = cells[s0:s0 + args.chunk]
        x0 = None if X0 is None else X0.repeat(len(idx), 1, 1)
        r = runner.run_cells([(log_t[i], alphas[j]) for i, j in idx], x0, args.transient)
        ii, jj = np.array(idx).T
        for k, v in r.items():
            res[name][k][ii, jj] = v[:, -1].float().cpu().numpy() if k == "state" else v
        with open(out / "progress.jsonl", "a") as h:
            h.write(json.dumps({"done": s0 + len(idx), "of": len(cells),
                                "seconds": round(time.perf_counter() - tic, 1)}) + "\n")
        save(out, args, runner, log_t, alphas, res, tic, final=False)
    save(out, args, runner, log_t, alphas, res, tic, final=True)


def sweep(args, runner):
    torch, loop = runner.torch, runner.loop
    log_t, alphas = grid(args)
    shape = (len(log_t), len(alphas))
    d = loop.E.shape[1]
    orders = {}
    for sw in args.sweeps:
        if sw == "alpha+":
            orders[sw] = [[(i, j) for i in range(shape[0])] for j in range(shape[1])]
        elif sw == "alpha-":
            orders[sw] = [[(i, j) for i in range(shape[0])] for j in reversed(range(shape[1]))]
        elif sw == "temp+":
            orders[sw] = [[(i, j) for j in range(shape[1])] for i in range(shape[0])]
        elif sw == "temp-":
            orders[sw] = [[(i, j) for j in range(shape[1])] for i in reversed(range(shape[0]))]
        else:
            raise SystemExit(f"unknown sweep {sw}")
    steps = {len(o) for o in orders.values()}
    if len(steps) != 1:
        raise SystemExit("sweeps in one run need the same number of columns (use a square grid)")
    res = {sw: empty(shape, len(KEY_TOKENS), d) for sw in orders}
    carry = {sw: None for sw in orders}
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    for c in range(steps.pop()):
        cells, x0s, spans = [], [], []
        for sw, order in orders.items():
            col = order[c]
            spans.append((sw, len(cells), len(cells) + len(col), col))
            cells += [(log_t[i], alphas[j]) for i, j in col]
            x0s.append(carry[sw])
        if c == 0:
            r = runner.run_cells(cells, None, args.first_transient)
        else:
            r = runner.run_cells(cells, torch.cat(x0s), args.transient)
        for sw, a0, a1, col in spans:
            ii, jj = np.array(col).T
            for k, v in r.items():
                part = v[a0:a1]
                res[sw][k][ii, jj] = part[:, -1].float().cpu().numpy() if k == "state" else part
            carry[sw] = r["state"][a0:a1]
        with open(out / "progress.jsonl", "a") as h:
            h.write(json.dumps({"column": c + 1, "seconds": round(time.perf_counter() - tic, 1)}) + "\n")
        if (c + 1) % 8 == 0:
            save(out, args, runner, log_t, alphas, res, tic, final=False)
    save(out, args, runner, log_t, alphas, res, tic, final=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("seeded", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="qwen0.5b")
        p.add_argument("--layer", type=int, default=12)
        p.add_argument("--vectors", default="llm-phase-map/vectors-l12")
        p.add_argument("--concept", default="philosophy")
        p.add_argument("--prompt", default="open")
        p.add_argument("--opener", default="Hello!")
        p.add_argument("--dtype", default="float64")
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--log-temperature", type=float, nargs=2, required=True)
        p.add_argument("--alpha", type=float, nargs=2, required=True)
        p.add_argument("--resolution", type=int, nargs=2, required=True)
        p.add_argument("--transient", type=int, default=300)
        p.add_argument("--samples", type=int, default=300)
        p.add_argument("--record", type=int, default=80)
        p.add_argument("--max-period", type=int, default=32)
        p.add_argument("--tol", type=float, default=1e-8)
        p.add_argument("--settle-tol", type=float, default=1e-6)
        p.add_argument("--eps", type=float, default=1e-6)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", required=True)
        if name == "seeded":
            p.add_argument("--start", default="default", help="'default' (the maps' fixed start) or a .npy state")
            p.add_argument("--name", default=None)
            p.add_argument("--chunk", type=int, default=2048)
        else:
            p.add_argument("--sweeps", nargs="+", required=True)
            p.add_argument("--first-transient", type=int, default=600)
    args = ap.parse_args()
    if args.record > args.samples:
        raise SystemExit("--record must not exceed --samples")
    runner = Runner(args)
    (seeded if args.mode == "seeded" else sweep)(args, runner)


if __name__ == "__main__":
    main()
