"""Exploratory E1: does the edge-of-stability route survive heavy-ball momentum?

Map (PyTorch SGD-momentum convention, per-layer rates P):
    m' = beta * m + grad L(theta),    theta' = theta - P m'.
This equals heavy ball, theta_{t+1} = theta_t - P grad L(theta_t) + beta (theta_t - theta_{t-1}).
At a minimum every Hessian eigenvalue h gives mu^2 - (1 + beta - eta h) mu + beta = 0, so
complex pairs have modulus sqrt(beta) and the minimum first loses stability by a flip at
eta h = 2 (1 + beta). The question here is what the resulting 2-cycle does next: a
Neimark-Sacker bifurcation (quasi-periodic training, locking windows) as in plain gradient
descent, or something else.

Subcommands:
  scan     Along a ray of rates (fixed output/hidden ratio), iterate from the shared initial
           parameters, find the attracting 2-cycle by Newton on F^2 in the 26-dimensional
           state, and report its two-step multipliers; the first unit-circle crossing type is
           the answer for that ray.
  rotation-map  Rotation numbers of two-step samples of theta on a rate rectangle (CPU).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.trainability import SYSTEMS  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


class MomentumMap:
    def __init__(self, system, seed, beta):
        self.net, self.x, self.y, self.theta0 = SYSTEMS[system](seed)
        self.beta = float(beta)
        self.n = self.net.n_parameters
        h = self.net.hidden_parameter_count
        self.mask_h = np.zeros(self.n)
        self.mask_h[:h] = 1.0

    def rates(self, lh, lo):
        lh, lo = np.atleast_1d(lh), np.atleast_1d(lo)
        return 10.0 ** lh[:, None] * self.mask_h + 10.0 ** lo[:, None] * (1.0 - self.mask_h)

    def step(self, theta, m, rates):
        g = self.net.gradient(theta, self.x, self.y)
        m = self.beta * m + g
        return theta - rates * m, m

    def jacobian(self, theta, m, rates):
        """26 x 26 Jacobian of one step for a single state."""
        h = self.net.hessian(theta, self.x, self.y)
        n = self.n
        jac = np.zeros((2 * n, 2 * n))
        r = rates[0]
        jac[:n, :n] = np.eye(n) - r[:, None] * h
        jac[:n, n:] = -self.beta * np.diag(r)
        jac[n:, :n] = h
        jac[n:, n:] = self.beta * np.eye(n)
        return jac


def newton_two_cycle(mm, state, rates, tol=1e-12, max_iter=30):
    n = mm.n
    z = state.copy()
    for _ in range(max_iter):
        t1, m1 = mm.step(z[None, :n], z[None, n:], rates)
        j0 = mm.jacobian(z[:n], z[n:], rates)
        t2, m2 = mm.step(t1, m1, rates)
        j1 = mm.jacobian(t1[0], m1[0], rates)
        f2 = np.concatenate([t2[0], m2[0]])
        res = f2 - z
        if np.linalg.norm(res) < tol * max(1.0, np.linalg.norm(z)):
            return z, j1 @ j0
        z = z + np.linalg.solve(j1 @ j0 - np.eye(2 * n), -res)
    return None, None


def scan(args):
    mm = MomentumMap(args.system, args.seed, args.beta)
    rows = []
    state_prev = None
    for s in np.linspace(args.start, args.stop, args.points):
        lh = s + args.offset_hidden
        lo = s + args.offset_output
        rates = mm.rates(lh, lo)
        theta = mm.theta0[None, :].copy()
        m = np.zeros_like(theta)
        for _ in range(args.transient):
            theta, m = mm.step(theta, m, rates)
        state = np.concatenate([theta[0], m[0]])
        if not np.all(np.isfinite(state)):
            rows.append({"s": float(s), "status": "diverged"})
            continue
        z, mono = newton_two_cycle(mm, state, rates)
        if z is None and state_prev is not None:
            z, mono = newton_two_cycle(mm, state_prev, rates)
        if z is None:
            rows.append({"s": float(s), "status": "no 2-cycle"})
            continue
        t1, m1 = mm.step(z[None, :mm.n], z[None, mm.n:], rates)
        separation = float(np.linalg.norm(t1[0] - z[:mm.n]))
        mult = np.linalg.eigvals(mono)
        order = np.argsort(-np.abs(mult))
        top = mult[order[0]]
        attract = float(np.linalg.norm(state - z))
        rows.append({"s": float(s), "status": "fixed point" if separation < 1e-8 else "2-cycle",
                     "separation": separation, "distance_attractor_to_cycle": attract,
                     "top": [[float(mult[i].real), float(mult[i].imag)] for i in order[:4]],
                     "top_modulus": float(abs(top)),
                     "top_turns": float(abs(np.angle(top)) / (2 * np.pi))})
        state_prev = z
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rows[-1].items()
                          if k != "top"} | {"top2": [[round(a, 4), round(b, 4)] for a, b in rows[-1]["top"][:2]]}))
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "scan.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
    print(json.dumps({"out": str(out / "scan.json")}))


def rotation_block(task):
    system, seed, beta, lh, lo, steps, pca, samples = task
    mm = MomentumMap(system, seed, beta)
    rates = mm.rates(lh, lo)
    theta = np.repeat(mm.theta0[None, :], len(lh), axis=0)
    m = np.zeros_like(theta)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            theta, m = mm.step(theta, m, rates)
        lobe = np.empty((pca, len(lh), mm.n))
        for k in range(pca):
            lobe[k] = theta
            theta, m = mm.step(*mm.step(theta, m, rates), rates)
        alive = np.all(np.isfinite(theta), axis=1) & np.all(np.isfinite(lobe), axis=(0, 2))
        centroid = np.nan_to_num(lobe.mean(0))
        centered = np.nan_to_num(np.transpose(lobe - centroid[None], (1, 0, 2)))
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        plane = vt[:, :2, :]
        spread = singular[:, 0] / np.sqrt(pca)

        def angle(t):
            planar = np.einsum("cp,cjp->cj", np.nan_to_num(t) - centroid, plane)
            return np.arctan2(planar[:, 1], planar[:, 0])

        previous = angle(theta)
        total = np.zeros(len(lh))
        for _ in range(samples):
            theta, m = mm.step(*mm.step(theta, m, rates), rates)
            current = angle(theta)
            total += (current - previous + np.pi) % (2 * np.pi) - np.pi
            previous = current
    alive &= np.all(np.isfinite(theta), axis=1)
    rho = np.where(alive, np.abs(total) / (2 * np.pi * samples), np.nan)
    return {"rho": rho, "spread": np.where(alive, spread, np.nan)}


def rotation_map(args):
    rows, cols = args.resolution
    log_h = np.linspace(*args.hidden, rows)
    log_o = np.linspace(*args.output, cols)
    r, c = np.divmod(np.arange(rows * cols), cols)
    chunks = np.array_split(np.arange(rows * cols), args.workers * 4)
    tasks = [(args.system, args.seed, args.beta, log_h[r[i]], log_o[c[i]], args.steps, args.pca, args.samples)
             for i in chunks]
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(rotation_block, tasks, chunksize=1))
    merged = {k: np.concatenate([p[k] for p in parts]).reshape(rows, cols) for k in parts[0]}
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "rotation.npz", log10_hidden=log_h, log10_output=log_o, **merged)
    (out / "metadata.json").write_text(json.dumps({"args": vars(args), "seconds": time.perf_counter() - start},
                                                  indent=1))
    print(json.dumps({"out": str(out), "seconds": round(time.perf_counter() - start, 1)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sc = sub.add_parser("scan")
    sc.add_argument("--system", default="S2")
    sc.add_argument("--seed", type=int, default=28011)
    sc.add_argument("--beta", type=float, required=True)
    sc.add_argument("--start", type=float, required=True, help="ray parameter start (log10 units)")
    sc.add_argument("--stop", type=float, required=True)
    sc.add_argument("--points", type=int, default=40)
    sc.add_argument("--offset-hidden", type=float, default=0.0)
    sc.add_argument("--offset-output", type=float, default=0.0)
    sc.add_argument("--transient", type=int, default=20000)
    sc.add_argument("--out", required=True)
    rm = sub.add_parser("rotation-map")
    rm.add_argument("--system", default="S2")
    rm.add_argument("--seed", type=int, default=28011)
    rm.add_argument("--beta", type=float, required=True)
    rm.add_argument("--hidden", type=float, nargs=2, required=True)
    rm.add_argument("--output", type=float, nargs=2, required=True)
    rm.add_argument("--resolution", type=int, nargs=2, default=(128, 128))
    rm.add_argument("--steps", type=int, default=20000)
    rm.add_argument("--pca", type=int, default=256)
    rm.add_argument("--samples", type=int, default=8000)
    rm.add_argument("--workers", type=int, default=8)
    rm.add_argument("--out", required=True)
    args = parser.parse_args()
    {"scan": scan, "rotation-map": rotation_map}[args.command](args)


if __name__ == "__main__":
    main()
