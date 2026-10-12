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

Variants (beyond the plain Titans rule): per-layer inner rates (theta for the
hidden layer, theta_o for the output layer), coupled weight decay (a ridge
term in the token loss) and chunked updates (each memory update uses the mean
gradient over C consecutive tokens, as in chunked test-time training). With
C = T, coupled decay lambda, eta = 0 and alpha = 0, the period map is exactly
one full-batch gradient step of the two-rate training map with ridge lambda.
With --scale-by-chunk, each update's rates are multiplied by C/T so that one
pass over the context always applies the same total step.

Subcommands
  scan     Grid at fixed eta in either (log10 theta, log10 alpha) or the rate
           plane (log10 theta_o, log10 theta): iterate the period map from a
           fixed initial memory and classify the attractor (diverged,
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

    def __init__(self, context=8, width=3, d=2, teacher_seed=28011, init_seed=128011, init_scale=0.2,
                 chunk=1, coupled_decay=0.0, scale_by_chunk=False, shuffle_seed=None):
        self.net = TwoLayerNet(d=int(d), width=int(width), activation="tanh", bias=True, loss_scale=0.5,
                               ridge=float(coupled_decay))
        data = make_teacher_regression(seed=int(teacher_seed), d=int(d), teacher_width=3, n_train=int(context),
                                       n_validation=1, n_test=1)
        self.keys, self.values = data.train_x, data.train_y
        self.T, self.C = int(context), int(chunk)
        if self.C < 1 or self.T % self.C:
            raise ValueError("the chunk size must divide the context length")
        self.updates = self.T // self.C
        self.rate_scale = self.C / self.T if scale_by_chunk else 1.0
        self.n = self.net.n_parameters
        self.mask_h = np.zeros(self.n)
        self.mask_h[:self.net.hidden_parameter_count] = 1.0
        self.w0 = TanhRegressor(int(d), int(width)).initialize(seed=int(init_seed), scale=float(init_scale))
        # optional noise: every pass over the context applies the chunks in a fresh random order
        self.shuffle = None if shuffle_seed is None else np.random.default_rng(int(shuffle_seed))

    def rates(self, theta, theta_o):
        """Per-parameter rates of one update, shape (cells, n), from hidden and output rates of shape (cells,)."""
        return self.rate_scale * (np.asarray(theta)[:, None] * self.mask_h
                                  + np.asarray(theta_o)[:, None] * (1.0 - self.mask_h))

    def update(self, w, s, u, rates, alpha, eta):
        """Memory update number u of a period (tokens u*C .. u*C+C-1), batched over cells."""
        part = slice(u * self.C, (u + 1) * self.C)
        g = self.net.gradient(w, self.keys[part], self.values[part])
        s = eta * s - rates * g
        return (1.0 - alpha[:, None]) * w + s, s

    def period(self, w, s, rates, alpha, eta):
        order = range(self.updates) if self.shuffle is None else self.shuffle.permutation(self.updates)
        for u in order:
            w, s = self.update(w, s, u, rates, alpha, eta)
        return w, s

    def recall_error(self, w):
        """Mean squared recall error over the context keys (an inference-visible observable)."""
        pred = self.net.predict(w, self.keys)                 # shape (keys, cells)
        return 0.5 * np.mean((pred - self.values[:, None]) ** 2, axis=0)

    def period_jacobian(self, z, rates, alpha, eta):
        """Period map value and Jacobian for one state z = (W, S); rates of shape (n,), scalar alpha and eta."""
        n = self.n
        w, s = z[:n].copy(), z[n:].copy()
        jac = np.eye(2 * n)
        eye = np.eye(n)
        for u in range(self.updates):
            part = slice(u * self.C, (u + 1) * self.C)
            rh = rates[:, None] * self.net.hessian(w, self.keys[part], self.values[part])
            step = np.zeros((2 * n, 2 * n))
            step[:n, :n] = (1.0 - alpha) * eye - rh
            step[:n, n:] = eta * eye
            step[n:, :n] = -rh
            step[n:, n:] = eta * eye
            jac = step @ jac
            w1, s1 = self.update(w[None, :], s[None, :], u, rates[None, :], np.array([alpha]), eta)
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
    mm = MemoryMap(**task["memory"])
    thetas, theta_os, alphas, eta = task["theta"], task["theta_o"], task["alpha"], task["eta"]
    transient, samples, max_period, tol = task["transient"], task["samples"], task["max_period"], task["tol"]
    cells = len(thetas)
    rates = mm.rates(thetas, theta_os)
    w = np.repeat(mm.w0[None, :], cells, axis=0)
    s = np.zeros_like(w)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(transient):
            w, s = mm.period(w, s, rates, alphas, eta)
        orbit = np.empty((samples, cells, 2 * mm.n))
        for j in range(samples):
            orbit[j] = np.concatenate([w, s], axis=1)
            w, s = mm.period(w, s, rates, alphas, eta)
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
    log_x = np.linspace(*(args.theta_o if args.plane == "rates" else args.alpha), cols)
    r, c = np.divmod(np.arange(rows * cols), cols)
    blocks = np.array_split(np.arange(rows * cols), args.workers * 4)
    memory = memory_kwargs(args)
    tasks = []
    for i in blocks:
        theta = 10.0 ** log_theta[r[i]]
        if args.plane == "rates":
            theta_o, alpha = 10.0 ** log_x[c[i]], np.full(len(i), args.fixed_alpha)
        else:
            theta_o, alpha = theta * args.ratio, 10.0 ** log_x[c[i]]
        tasks.append({"memory": memory, "theta": theta, "theta_o": theta_o, "alpha": alpha, "eta": args.eta,
                      "transient": args.transient, "samples": args.samples, "max_period": args.max_period,
                      "tol": args.tol})
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(classify_block, tasks))
    fields = {k: np.concatenate([p[k] for p in parts]).reshape(rows, cols) for k in parts[0]}
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scan.npz", log10_theta=log_theta, log10_x=log_x, plane=args.plane, **fields)
    finite, period = fields["finite"], fields["period"]
    summary = {"cells": int(finite.size), "diverged": int((~finite).sum()),
               "T_periodic_memory": int((period == 1).sum()),
               "period_counts": {int(k): int((period == k).sum()) for k in range(2, args.max_period + 1)
                                 if (period == k).sum()},
               "moving": int((finite & (period == 0)).sum()), "seconds": round(time.perf_counter() - start, 1)}
    (out / "metadata.json").write_text(json.dumps({"args": vars(args), "summary": summary}, indent=1))
    print(json.dumps(summary))
    figure(fields, log_theta, log_x, out / "scan.png", args)


def memory_kwargs(args):
    return {"context": args.context, "width": args.width, "d": args.d, "teacher_seed": args.teacher_seed,
            "init_seed": args.init_seed, "init_scale": args.init_scale, "chunk": args.chunk,
            "coupled_decay": args.coupled_decay, "scale_by_chunk": args.scale_by_chunk,
            "shuffle_seed": getattr(args, "shuffle_seed", None)}


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
        ax.set_xlabel("log10 output-layer rate theta_o" if args.plane == "rates" else "log10 forgetting alpha")
        ax.set_ylabel("log10 inner (hidden-layer) learning rate theta")
    fig.suptitle(f"memory width {args.width}, context {args.context}, chunk {args.chunk}, momentum {args.eta}, "
                 f"coupled decay {args.coupled_decay}" + (", rates scaled by chunk" if args.scale_by_chunk else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def floquet(args):
    mm = MemoryMap(**memory_kwargs(args))
    n, p = mm.n, args.period
    theta = 10.0 ** args.log_theta
    theta_o = 10.0 ** args.log_theta_o if args.log_theta_o is not None else theta * args.ratio
    alpha = 0.0 if args.log_alpha is None else 10.0 ** args.log_alpha
    eta = args.eta
    rates = mm.rates(np.array([theta]), np.array([theta_o]))
    w = mm.w0[None, :].copy()
    s = np.zeros_like(w)
    for _ in range(args.transient):
        w, s = mm.period(w, s, rates, np.array([alpha]), eta)
    z = np.concatenate([w[0], s[0]])

    def residual(z):
        y, jac = z, np.eye(2 * n)
        for _ in range(p):
            y, j1 = mm.period_jacobian(y, rates[0], alpha, eta)
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
        y, _ = mm.period_jacobian(z, rates[0], alpha, eta)
        sep = float(np.linalg.norm(y - z))
    out = {"log10_theta": args.log_theta, "theta_o": theta_o, "log10_alpha": args.log_alpha, "eta": eta,
           "period": p, "chunk": args.chunk, "coupled_decay": args.coupled_decay,
           "newton": history, "separation_after_one_period": sep,
           "multipliers_top6": [[float(mult[i].real), float(mult[i].imag)] for i in order[:6]],
           "spectral_radius": float(np.abs(mult[order[0]])),
           "recall_error": float(mm.recall_error(z[None, :n])[0])}
    print(json.dumps(out))
    if args.out:
        dest = DATA / "explore" / args.out
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "floquet.json").write_text(json.dumps(out, indent=1))


def continue_orbit(args):
    """Natural continuation of a period-p orbit of the period map in log10 theta (hidden rate) at fixed theta_o.

    Starts from an orbit converged by iteration at the first rate; records the leading multipliers at each
    step and classifies the first unit-circle crossing (complex pair, real +1 or real -1)."""
    mm = MemoryMap(**memory_kwargs(args))
    n, p, eta = mm.n, args.period, args.eta
    alpha = 0.0 if args.log_alpha is None else 10.0 ** args.log_alpha
    theta_o = 10.0 ** args.log_theta_o

    def residual(z, rates):
        y, jac = z, np.eye(2 * n)
        for _ in range(p):
            y, j1 = mm.period_jacobian(y, rates, alpha, eta)
            jac = j1 @ jac
        return y - z, jac

    def solve(z, rates):
        res, jac = residual(z, rates)
        for _ in range(args.max_newton):
            norm0 = float(np.linalg.norm(res))
            if norm0 < 1e-12 * max(1.0, np.linalg.norm(z)):
                return z, jac, norm0
            step = np.linalg.solve(jac - np.eye(2 * n), -res)
            frac = 1.0
            for _ in range(12):
                trial = z + frac * step
                res_t, jac_t = residual(trial, rates)
                if np.all(np.isfinite(res_t)) and np.linalg.norm(res_t) < norm0:
                    break
                frac *= 0.5
            z, res, jac = trial, res_t, jac_t
        return None, None, float(np.linalg.norm(res))

    values = np.arange(args.theta_start, args.theta_stop + 0.5 * args.step, args.step)
    rates0 = mm.rates(np.array([10.0 ** values[0]]), np.array([theta_o]))
    w = mm.w0[None, :].copy()
    s_ = np.zeros_like(w)
    for _ in range(args.transient):
        w, s_ = mm.period(w, s_, rates0, np.array([alpha]), eta)
    z = np.concatenate([w[0], s_[0]])
    rows, crossing = [], None
    for lt in values:
        rates = mm.rates(np.array([10.0 ** lt]), np.array([theta_o]))[0]
        z_new, jac, res = solve(z, rates)
        if z_new is None:
            rows.append({"log10_theta": float(lt), "outcome": "lost", "residual": res})
            break
        z = z_new
        mult = np.linalg.eigvals(jac)
        order = np.argsort(-np.abs(mult))
        lead = mult[order[0]]
        row = {"log10_theta": float(lt), "spectral_radius": float(abs(lead)),
               "top": [[float(mult[i].real), float(mult[i].imag)] for i in order[:4]]}
        rows.append(row)
        if crossing is None and abs(lead) > 1.0:
            crossing = {"log10_theta": float(lt), "type": "complex (Neimark-Sacker)" if abs(lead.imag) > 1e-8 else
                        ("real +1" if lead.real > 0 else "real -1 (flip)"),
                        "turns": float(abs(np.angle(lead)) / (2 * np.pi)), "modulus": float(abs(lead)),
                        "previous_spectral_radius": rows[-2]["spectral_radius"] if len(rows) > 1 else None}
            if not args.beyond:
                break
    out = {"args": vars(args), "first_crossing": crossing, "rows": rows}
    print(json.dumps({"first_crossing": crossing, "steps": len(rows), "last": rows[-1]}))
    dest = DATA / "explore" / args.out
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "continuation.json").write_text(json.dumps(out, indent=1))


def criticality(args):
    """First Lyapunov coefficient of the synchronized memory's Neimark-Sacker crossing (hidden-rate direction).

    Continues the period-1 orbit from theta_start up to the bracket [lo, hi], bisects the unit-circle
    crossing of its leading complex pair, and evaluates the normal-form coefficient with finite-difference
    second and third derivatives of the period map (repeated with doubled steps)."""
    from neural_shrimp.criticality import finite_difference_forms, ns_coefficient

    mm = MemoryMap(**memory_kwargs(args))
    n, eta = mm.n, args.eta
    alpha = 0.0 if args.log_alpha is None else 10.0 ** args.log_alpha
    theta_o = 10.0 ** args.log_theta_o

    def rates_at(lt):
        return mm.rates(np.array([10.0 ** lt]), np.array([theta_o]))[0]

    def solve(z, rates):
        for _ in range(60):
            y, jac = mm.period_jacobian(z, rates, alpha, eta)
            res = y - z
            if np.linalg.norm(res) < 1e-13 * max(1.0, np.linalg.norm(z)):
                return z, jac
            z = z + np.linalg.solve(jac - np.eye(2 * n), -res)
        raise RuntimeError("Newton did not converge on the synchronized orbit")

    r0 = mm.rates(np.array([10.0 ** args.theta_start]), np.array([theta_o]))
    w = mm.w0[None, :].copy()
    s_ = np.zeros_like(w)
    for _ in range(args.transient):
        w, s_ = mm.period(w, s_, r0, np.array([alpha]), eta)
    z = np.concatenate([w[0], s_[0]])
    for lt in np.arange(args.theta_start, args.lo + 0.5 * args.step, args.step):
        z, _ = solve(z, rates_at(lt))
    lo, hi, z_lo = args.lo, args.hi, z
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        z_mid, jac = solve(z_lo, rates_at(mid))
        radius = float(np.max(np.abs(np.linalg.eigvals(jac))))
        if radius < 1.0:
            lo, z_lo = mid, z_mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    z_c, jac_c = solve(z_lo, rates_at(0.5 * (lo + hi)))
    mult = np.linalg.eigvals(jac_c)
    lead = mult[np.argmax(np.abs(mult))]
    if abs(lead.imag) < 1e-8:
        out = {"outcome": "crossing multiplier is real, not Neimark-Sacker", "lead": [lead.real, lead.imag]}
        print(json.dumps(out))
        return
    rates_c = rates_at(0.5 * (lo + hi))

    def jvp(X, V):
        return np.array([mm.period_jacobian(x, rates_c, alpha, eta)[1] @ v for x, v in zip(X, V)])

    results = []
    for factor in (1.0, 2.0):
        B, C = finite_difference_forms(jvp, z_c, step_b=args.step_b * factor, step_c=args.step_c * factor)
        res = ns_coefficient(jac_c, B, C, mu=lead)
        results.append({"step_factor": factor, **{k: (float(v) if np.isscalar(v) and np.isreal(v) else str(v))
                                                    for k, v in res.items()}})
    out = {"args": vars(args), "crossing_log10_theta": 0.5 * (lo + hi), "bracket_width": hi - lo,
           "lead": [float(lead.real), float(lead.imag)], "modulus": float(abs(lead)),
           "turns": float(abs(np.angle(lead)) / (2 * np.pi)), "first_lyapunov": results,
           "classification": "supercritical (soft)" if float(results[0]["d"]) < 0 else "subcritical (hard)"}
    print(json.dumps({k: v for k, v in out.items() if k != "args"}))
    dest = DATA / "explore" / args.out
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "criticality.json").write_text(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "floquet", "continue", "criticality"):
        sp = sub.add_parser(name)
        sp.add_argument("--context", type=int, default=8)
        sp.add_argument("--width", type=int, default=3)
        sp.add_argument("--d", type=int, default=2)
        sp.add_argument("--teacher-seed", type=int, default=28011)
        sp.add_argument("--init-seed", type=int, default=128011)
        sp.add_argument("--init-scale", type=float, default=0.2)
        sp.add_argument("--eta", type=float, default=0.0, help="momentum of the memory update")
        sp.add_argument("--transient", type=int, default=2000, help="periods of the context before measuring")
        sp.add_argument("--chunk", type=int, default=1, help="tokens per memory update (mean gradient)")
        sp.add_argument("--coupled-decay", type=float, default=0.0, help="ridge term in the token loss")
        sp.add_argument("--scale-by-chunk", action="store_true", help="multiply each update's rates by chunk/context")
        sp.add_argument("--ratio", type=float, default=1.0, help="theta_o / theta when theta_o is not given")
        sp.add_argument("--shuffle-seed", type=int, default=None,
                        help="apply the chunks in a fresh random order on every pass (random dynamical system)")
    sc = sub.choices["scan"]
    sc.add_argument("--theta", type=float, nargs=2, default=[-2.0, 1.0], help="log10 inner learning rate range")
    sc.add_argument("--alpha", type=float, nargs=2, default=[-5.0, -0.5], help="log10 forgetting range")
    sc.add_argument("--plane", choices=("theta-alpha", "rates"), default="theta-alpha")
    sc.add_argument("--theta-o", type=float, nargs=2, default=[-0.2, 0.0], help="log10 output-rate range (rates plane)")
    sc.add_argument("--fixed-alpha", type=float, default=0.0, help="forgetting in the rates plane")
    sc.add_argument("--resolution", type=int, nargs=2, default=[120, 120])
    sc.add_argument("--samples", type=int, default=1024)
    sc.add_argument("--max-period", type=int, default=64)
    sc.add_argument("--tol", type=float, default=1e-8)
    sc.add_argument("--workers", type=int, default=6)
    sc.add_argument("--out", required=True)
    fq = sub.choices["floquet"]
    fq.add_argument("--log-theta", type=float, required=True)
    fq.add_argument("--log-alpha", type=float, default=None, help="omit for no forgetting")
    fq.add_argument("--log-theta-o", type=float, default=None)
    fq.add_argument("--period", type=int, default=1)
    fq.add_argument("--max-newton", type=int, default=30)
    fq.add_argument("--out", default=None)
    co = sub.choices["continue"]
    co.add_argument("--log-theta-o", type=float, required=True)
    co.add_argument("--log-alpha", type=float, default=None)
    co.add_argument("--theta-start", type=float, required=True, help="log10 hidden rate where the orbit is stable")
    co.add_argument("--theta-stop", type=float, required=True)
    co.add_argument("--step", type=float, default=0.002)
    co.add_argument("--period", type=int, default=1)
    co.add_argument("--max-newton", type=int, default=30)
    co.add_argument("--beyond", action="store_true", help="keep continuing after the first crossing")
    co.add_argument("--out", required=True)
    cr = sub.choices["criticality"]
    cr.add_argument("--log-theta-o", type=float, required=True)
    cr.add_argument("--log-alpha", type=float, default=None)
    cr.add_argument("--theta-start", type=float, required=True)
    cr.add_argument("--lo", type=float, required=True, help="log10 hidden rate just below the crossing")
    cr.add_argument("--hi", type=float, required=True, help="log10 hidden rate just above the crossing")
    cr.add_argument("--step", type=float, default=0.002)
    cr.add_argument("--step-b", type=float, default=1e-4)
    cr.add_argument("--step-c", type=float, default=1e-3)
    cr.add_argument("--out", required=True)
    args = parser.parse_args()
    {"scan": scan, "floquet": floquet, "continue": continue_orbit, "criticality": criticality}[args.command](args)


if __name__ == "__main__":
    main()
