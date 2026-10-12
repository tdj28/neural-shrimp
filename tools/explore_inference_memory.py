"""Exploratory pilot: a Titans-style test-time memory under a repeating context.

The memory is a small tanh MLP ``M_W`` (keys to values) that is updated during
inference, token by token, by gradient descent with momentum and forgetting
(the Titans update, Behrouz et al. 2025):

    S_t = eta S_{t-1} - theta grad l_t(W_{t-1}),   l_t(W) = (M_W(k_t) - v_t)^2 / 2,
    W_t = (1 - alpha) W_{t-1} + S_t.

When a context of T key-value pairs repeats in a fixed order, the one-period
map (W, S) -> (W, S) after T tokens is autonomous. Its periodic orbits,
multipliers and rotation numbers are the same objects as for the training
maps, so the same full-state diagnostics apply. With eta = 0 and
alpha = theta * lambda, one memory step is one ridge-regularized gradient step
on the current token's loss.

Subcommands
  scan     Grid in (log10 theta, log10 alpha) at fixed eta: iterate the period
           map from a fixed initial memory and classify the attractor (diverged,
           T-periodic memory, period k of the period map, or moving with a
           rotation number in its principal plane).
  floquet  At one (theta, alpha, eta): find a period-p orbit of the period map
           by iteration and Newton (analytic Jacobian) and report its
           multipliers.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.gdmaps import TwoLayerNet  # noqa: E402
from neural_shrimp.models import TanhRegressor, make_teacher_regression  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


class MemoryMap:
    """Period map of a tanh memory MLP updated by the Titans rule over a repeating context."""

    def __init__(self, context=8, width=3, d=2, teacher_seed=28011, init_seed=128011, init_scale=0.2):
        self.net = TwoLayerNet(d=int(d), width=int(width), activation="tanh", bias=True, loss_scale=0.5, ridge=0.0)
        data = make_teacher_regression(seed=int(teacher_seed), d=int(d), teacher_width=3, n_train=int(context),
                                       n_validation=1, n_test=1)
        self.keys, self.values = data.train_x, data.train_y
        self.T = int(context)
        self.n = self.net.n_parameters
        self.w0 = TanhRegressor(int(d), int(width)).initialize(seed=int(init_seed), scale=float(init_scale))

    def token(self, w, s, t, theta, alpha, eta):
        """One memory update on token t, batched over cells (theta, alpha: shape (cells,))."""
        g = self.net.gradient(w, self.keys[t:t + 1], self.values[t:t + 1])
        s = eta * s - theta[:, None] * g
        return (1.0 - alpha[:, None]) * w + s, s

    def period(self, w, s, theta, alpha, eta):
        for t in range(self.T):
            w, s = self.token(w, s, t, theta, alpha, eta)
        return w, s

    def recall_error(self, w):
        """Mean squared recall error over the context keys (an inference-visible observable)."""
        pred = self.net.predict(w, self.keys)
        return 0.5 * np.mean((pred - self.values[None, :]) ** 2, axis=1)

    def period_jacobian(self, z, theta, alpha, eta):
        """Period map value and Jacobian for one state z = (W, S) (scalars theta, alpha, eta)."""
        n = self.n
        w, s = z[:n].copy(), z[n:].copy()
        jac = np.eye(2 * n)
        eye = np.eye(n)
        for t in range(self.T):
            h = self.net.hessian(w, self.keys[t:t + 1], self.values[t:t + 1])
            step = np.zeros((2 * n, 2 * n))
            step[:n, :n] = (1.0 - alpha) * eye - theta * h
            step[:n, n:] = eta * eye
            step[n:, :n] = -theta * h
            step[n:, n:] = eta * eye
            jac = step @ jac
            w1, s1 = self.token(w[None, :], s[None, :], t, np.array([theta]), np.array([alpha]), eta)
            w, s = w1[0], s1[0]
        return np.concatenate([w, s]), jac


def principal_rotation(samples):
    """Rotation per iterate in the principal plane of an orbit sample, shape (iterates, cells, dim)."""
    centroid = samples.mean(0)
    centered = np.transpose(samples - centroid[None], (1, 0, 2))
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    plane = vt[:, :2, :]
    planar = np.einsum("jcp,ckp->jck", samples - centroid[None], plane)
    angles = np.arctan2(planar[..., 1], planar[..., 0])
    turns = (np.diff(angles, axis=0) + np.pi) % (2 * np.pi) - np.pi
    return np.abs(turns.sum(0)) / (2 * np.pi * (len(samples) - 1)), singular[:, 0] / np.sqrt(len(samples))


def classify_block(task):
    (context, width, d, teacher_seed, init_seed, init_scale, thetas, alphas, eta, transient, samples,
     max_period, tol) = task
    mm = MemoryMap(context, width, d, teacher_seed, init_seed, init_scale)
    cells = len(thetas)
    w = np.repeat(mm.w0[None, :], cells, axis=0)
    s = np.zeros_like(w)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(transient):
            w, s = mm.period(w, s, thetas, alphas, eta)
        orbit = np.empty((samples, cells, 2 * mm.n))
        for j in range(samples):
            orbit[j] = np.concatenate([w, s], axis=1)
            w, s = mm.period(w, s, thetas, alphas, eta)
    finite = np.all(np.isfinite(orbit), axis=(0, 2)) & (np.max(np.abs(np.nan_to_num(orbit, nan=1e9)), axis=(0, 2)) < 1e6)
    last = np.nan_to_num(orbit[-1])
    scale = np.maximum(1.0, np.linalg.norm(last, axis=1))
    period = np.zeros(cells, dtype=int)
    for k in range(max_period, 0, -1):                      # smallest k last, so it wins
        close = np.linalg.norm(np.nan_to_num(orbit[-1 - k]) - last, axis=1) < tol * scale
        period = np.where(finite & close, k, period)
    moving = finite & (period == 0)
    rho1, spread = principal_rotation(np.nan_to_num(orbit))
    rho2, _ = principal_rotation(np.nan_to_num(orbit[::2]))
    with np.errstate(over="ignore", invalid="ignore"):
        recall = mm.recall_error(np.nan_to_num(w))
    return {"finite": finite, "period": period, "rho1": np.where(moving, rho1, np.nan),
            "rho2": np.where(moving, rho2, np.nan), "spread": np.where(finite, spread, np.nan),
            "recall": np.where(finite, recall, np.nan)}


def scan(args):
    from concurrent.futures import ProcessPoolExecutor

    rows, cols = args.resolution
    log_theta = np.linspace(*args.theta, rows)
    log_alpha = np.linspace(*args.alpha, cols)
    r, c = np.divmod(np.arange(rows * cols), cols)
    chunks = np.array_split(np.arange(rows * cols), args.workers * 4)
    tasks = [(args.context, args.width, args.d, args.teacher_seed, args.init_seed, args.init_scale,
              10.0 ** log_theta[r[i]], 10.0 ** log_alpha[c[i]], args.eta, args.transient, args.samples,
              args.max_period, args.tol) for i in chunks]
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(classify_block, tasks))
    fields = {k: np.concatenate([p[k] for p in parts]).reshape(rows, cols) for k in parts[0]}
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scan.npz", log10_theta=log_theta, log10_alpha=log_alpha, **fields)
    finite, period = fields["finite"], fields["period"]
    summary = {"cells": int(finite.size), "diverged": int((~finite).sum()),
               "T_periodic_memory": int((period == 1).sum()),
               "period_counts": {int(k): int((period == k).sum()) for k in range(2, args.max_period + 1)
                                 if (period == k).sum()},
               "moving": int((finite & (period == 0)).sum()), "seconds": round(time.perf_counter() - start, 1)}
    (out / "metadata.json").write_text(json.dumps({"args": vars(args), "summary": summary}, indent=1))
    print(json.dumps(summary))
    figure(fields, log_theta, log_alpha, out / "scan.png", args)


def figure(fields, log_theta, log_alpha, path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite, period = fields["finite"], fields["period"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    shown = np.where(finite & (period > 0), period, np.nan).astype(float)
    im0 = axes[0].pcolormesh(log_alpha, log_theta, shown, cmap="tab20", vmin=0.5, vmax=20.5, shading="auto")
    axes[0].pcolormesh(log_alpha, log_theta, np.where(finite & (period == 0), 1.0, np.nan), cmap="Greys",
                       vmin=0, vmax=4, shading="auto")
    axes[0].set_title("period of the period map (grey: moving, white: diverged)")
    fig.colorbar(im0, ax=axes[0], label="period (1 = memory repeats with the context)")
    im1 = axes[1].pcolormesh(log_alpha, log_theta, fields["rho2"], cmap="viridis", vmin=0, vmax=0.5, shading="auto")
    axes[1].set_title("rotation per two periods (moving cells)")
    fig.colorbar(im1, ax=axes[1], label="turns")
    for ax in axes:
        ax.set_xlabel("log10 forgetting alpha")
        ax.set_ylabel("log10 inner learning rate theta")
    fig.suptitle(f"memory width {args.width}, context {args.context}, momentum {args.eta}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def floquet(args):
    mm = MemoryMap(args.context, args.width, args.d, args.teacher_seed, args.init_seed, args.init_scale)
    n, p = mm.n, args.period
    theta, alpha, eta = 10.0 ** args.log_theta, 10.0 ** args.log_alpha, args.eta
    w = mm.w0[None, :].copy()
    s = np.zeros_like(w)
    for _ in range(args.transient):
        w, s = mm.period(w, s, np.array([theta]), np.array([alpha]), eta)
    z = np.concatenate([w[0], s[0]])

    def residual(z):
        y, jac = z, np.eye(2 * n)
        for _ in range(p):
            y, j1 = mm.period_jacobian(y, theta, alpha, eta)
            jac = j1 @ jac
        return y - z, jac

    history = []
    for _ in range(args.max_newton):
        res, jac = residual(z)
        history.append(float(np.linalg.norm(res)))
        if history[-1] < 1e-12 * max(1.0, np.linalg.norm(z)):
            break
        z = z - np.linalg.solve(jac - np.eye(2 * n), res)
    res, jac = residual(z)
    history.append(float(np.linalg.norm(res)))
    mult = np.linalg.eigvals(jac)
    order = np.argsort(-np.abs(mult))
    sep = None
    if p > 1:
        y, _ = mm.period_jacobian(z, theta, alpha, eta)
        sep = float(np.linalg.norm(y - z))
    out = {"log10_theta": args.log_theta, "log10_alpha": args.log_alpha, "eta": eta, "period": p,
           "newton": history, "separation_after_one_period": sep,
           "multipliers_top6": [[float(mult[i].real), float(mult[i].imag)] for i in order[:6]],
           "spectral_radius": float(np.abs(mult[order[0]])),
           "recall_error": float(mm.recall_error(z[None, :n])[0])}
    print(json.dumps(out))
    if args.out:
        dest = DATA / "explore" / args.out
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "floquet.json").write_text(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "floquet"):
        sp = sub.add_parser(name)
        sp.add_argument("--context", type=int, default=8)
        sp.add_argument("--width", type=int, default=3)
        sp.add_argument("--d", type=int, default=2)
        sp.add_argument("--teacher-seed", type=int, default=28011)
        sp.add_argument("--init-seed", type=int, default=128011)
        sp.add_argument("--init-scale", type=float, default=0.2)
        sp.add_argument("--eta", type=float, default=0.0, help="momentum of the memory update")
        sp.add_argument("--transient", type=int, default=2000, help="periods of the context before measuring")
    sc = sub.choices["scan"]
    sc.add_argument("--theta", type=float, nargs=2, default=[-2.0, 1.0], help="log10 inner learning rate range")
    sc.add_argument("--alpha", type=float, nargs=2, default=[-5.0, -0.5], help="log10 forgetting range")
    sc.add_argument("--resolution", type=int, nargs=2, default=[120, 120])
    sc.add_argument("--samples", type=int, default=1024)
    sc.add_argument("--max-period", type=int, default=64)
    sc.add_argument("--tol", type=float, default=1e-8)
    sc.add_argument("--workers", type=int, default=6)
    sc.add_argument("--out", required=True)
    fq = sub.choices["floquet"]
    fq.add_argument("--log-theta", type=float, required=True)
    fq.add_argument("--log-alpha", type=float, required=True)
    fq.add_argument("--period", type=int, default=1)
    fq.add_argument("--max-newton", type=int, default=30)
    fq.add_argument("--out", default=None)
    args = parser.parse_args()
    {"scan": scan, "floquet": floquet}[args.command](args)


if __name__ == "__main__":
    main()
