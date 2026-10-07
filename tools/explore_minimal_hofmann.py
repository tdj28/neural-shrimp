"""Exploratory: Hofmann's rotational edge in the 2x2 depth-two linear network.

Model (Hofmann 2026, arXiv:2607.04993, Section 6.2, Proposition 16):
l(W1, W2) = 1/2 ||W2 W1 - diag(s1, s2)||_F^2, single rate a, no weight decay
unless --lam is given. On the aligned (balanced diagonal) manifold the top mode
follows the cubic map w -> w - a w (w^2 - s1), whose 2-cycle exists for
1 < a s1 < sqrt(5) - 1; the aligned 2-cycle is predicted to lose stability to
frame misalignment at a (s1 + s2) = 2 with complex two-step multipliers.

Subcommands (outputs under DATA/explore/minimal-model/hofmann):
  edge       exact two-step multipliers of the aligned 2-cycle across the
             predicted edge; locate the crossing of the complex pair by
             bisection; transversality; the other multipliers
  circle     iterate just past the edge from the aligned cycle plus a small
             misalignment; attractor amplitude, rotation of the two-step map in
             rotation-invariant coordinates, top Lyapunov exponents (one is the
             exact zero of the hidden-rotation symmetry), loss
  staircase  attractor-following sweep of a beyond the edge: rotation number,
             period (recurrence of invariants) and exponents per rate

Exploratory under docs/adaptive-research-protocol.md.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.minimal import DeepLinear2x2  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))
OUT = DATA / "explore" / "minimal-model" / "hofmann"
RATIOS = (0.7, 0.8, 0.9)


def two_step(model, a):
    p, q = model.aligned_cycle(a)
    return model.jacobian(q, a) @ model.jacobian(p, a), p


def misalignment_pair(model, a):
    """Complex pair of the aligned cycle's two-step derivative with the largest modulus."""
    m, p = two_step(model, a)
    values = np.linalg.eigvals(m)
    complex_values = [v for v in values if abs(v.imag) > 1e-12]
    if not complex_values:
        return None, values
    return max(complex_values, key=abs), values


def window(model):
    s_eff = model.s1 - model.lam
    return 1.0 / s_eff, (np.sqrt(5) - 1.0) / s_eff


def find_edge_bracket(model, points=400):
    """First rate in the 2-cycle window where the complex pair leaves the unit circle."""
    lo_w, hi_w = window(model)
    grid = np.linspace(lo_w + 1e-6, hi_w - 1e-6, points)
    previous = None
    for a in grid:
        mu, _ = misalignment_pair(model, a)
        outside = mu is not None and abs(mu) > 1
        if outside and previous is not None:
            return previous, a
        previous = None if outside else a
    raise ValueError("no complex-pair crossing inside the 2-cycle window")


def edge_rate(model):
    lo, hi = find_edge_bracket(model)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        mu, _ = misalignment_pair(model, mid)
        if mu is not None and abs(mu) < 1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cmd_edge(args):
    rows = []
    for ratio in RATIOS:
        model = DeepLinear2x2(1.0, ratio, args.lam)
        predicted = 2.0 / (1.0 + ratio)
        lo, hi = find_edge_bracket(model)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            mu, _ = misalignment_pair(model, mid)
            if mu is not None and abs(mu) < 1:
                lo = mid
            else:
                hi = mid
        a_star = 0.5 * (lo + hi)
        mu, values = misalignment_pair(model, a_star)
        h = 1e-6
        slope = (abs(misalignment_pair(model, a_star + h)[0]) - abs(misalignment_pair(model, a_star - h)[0])) / (2 * h)
        others = sorted((complex(v) for v in values if abs(v - mu) > 1e-9 and abs(v - np.conj(mu)) > 1e-9),
                        key=abs, reverse=True)
        sweep = []
        for a in np.linspace(max(a_star - 0.05, window(model)[0] + 1e-4), min(a_star + 0.05, window(model)[1] - 1e-4), 21):
            pair, _ = misalignment_pair(model, a)
            sweep.append({"a": float(a), "modulus": None if pair is None else float(abs(pair)),
                          "theta_over_2pi": None if pair is None else float(abs(np.angle(pair)) / (2 * np.pi))})
        rows.append({"s2_over_s1": ratio, "predicted_edge": predicted, "crossing": a_star,
                     "crossing_minus_predicted": a_star - predicted, "mu": [mu.real, mu.imag],
                     "theta_over_2pi": float(abs(np.angle(mu)) / (2 * np.pi)), "dmodulus_da": slope,
                     "other_multipliers": [[v.real, v.imag] for v in others], "sweep": sweep})
        print(f"s2/s1 = {ratio}: crossing a = {a_star:.12f} (predicted {predicted:.12f}, diff {a_star - predicted:.2e}); "
              f"theta/2pi = {rows[-1]['theta_over_2pi']:.6f}; d|mu|/da = {slope:.4f}; "
              f"other |mu|: {[round(abs(v), 6) for v in others]}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"edge-lam{args.lam}.json").write_text(json.dumps(rows, indent=1))


def batched_step(model, theta, a):
    return theta - a[:, None] * model.gradient(theta)


def run_attractor(model, a, theta, transient, samples, exponents=3, seed=0):
    """Iterate many rates at once; return final states, two-step invariant samples,
    top Lyapunov exponents and mean loss over the sample window."""
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(transient):
            theta = batched_step(model, theta, a)
        rng = np.random.default_rng(seed)
        q = np.linalg.qr(rng.normal(size=(n, 8, exponents)))[0]
        logs = np.zeros((n, exponents))
        feats = np.empty((samples, n, 12))
        loss = np.zeros(n)
        for k in range(samples):
            for _ in range(2):
                v = np.stack([q[:, :, j] - a[:, None] * model.hvp(theta, q[:, :, j]) for j in range(exponents)], -1)
                theta = batched_step(model, theta, a)
                q, r = np.linalg.qr(v)
                logs += np.log(np.abs(np.diagonal(r, axis1=1, axis2=2)))
                loss += model.loss(theta)
            feats[k] = model.invariants(theta)
    return theta, feats, logs / (2 * samples), loss / (2 * samples)


def rotation_and_period(feats, tol=1e-7):
    """Rotation (turns per two steps) in the principal plane of the invariants, and the
    smallest two-step recurrence lag (0 if none up to 64)."""
    n = feats.shape[1]
    rho = np.full(n, np.nan)
    share = np.full(n, np.nan)
    period = np.zeros(n, dtype=int)
    amplitude = np.full(n, np.nan)
    for i in range(n):
        x = feats[:, i, :]
        if not np.all(np.isfinite(x)):
            continue
        c = x - x.mean(0)
        _, s, vt = np.linalg.svd(c, full_matrices=False)
        z = c @ vt[:2].T
        ang = np.unwrap(np.arctan2(z[:, 1], z[:, 0]))
        rho[i] = abs(ang[-1] - ang[0]) / (2 * np.pi * (len(ang) - 1))
        share[i] = (s[:2] ** 2).sum() / max((s ** 2).sum(), 1e-300)
        amplitude[i] = float(np.sqrt(np.mean(np.sum(c * c, axis=1))))
        scale = max(1.0, float(np.max(np.abs(x))))
        tail = x[-256:]
        for lag in range(1, 65):
            if np.max(np.abs(tail[lag:] - tail[:-lag])) <= tol * scale:
                period[i] = 2 * lag
                break
    return rho, share, period, amplitude


def perturbed_aligned(model, a, size, seed=1):
    p, _ = model.aligned_cycle(a)
    rng = np.random.default_rng(seed)
    return p + size * rng.normal(size=8)


def cmd_circle(args):
    rows = []
    deltas = np.array(args.deltas)
    for ratio in RATIOS:
        model = DeepLinear2x2(1.0, ratio, args.lam)
        edge = edge_rate(model)
        a = edge * (1.0 + deltas)
        if args.sequential:
            # attractor following in the listed order, starting from the aligned cycle at the first rate
            theta_k = perturbed_aligned(model, a[0], 1e-3)
            outs = []
            for ak in a:
                res = run_attractor(model, np.array([ak]), theta_k[None, :], args.transient, args.samples)
                theta_k = res[0][0]
                outs.append(res)
            theta = np.concatenate([o[0] for o in outs])
            feats = np.concatenate([o[1] for o in outs], axis=1)
            lyap = np.concatenate([o[2] for o in outs])
            loss = np.concatenate([o[3] for o in outs])
        else:
            theta = np.stack([perturbed_aligned(model, ai, 1e-3) for ai in a])
            theta, feats, lyap, loss = run_attractor(model, a, theta, args.transient, args.samples)
        rho, share, period, amplitude = rotation_and_period(feats)
        mu = misalignment_pair(model, edge)[0]
        birth = float(abs(np.angle(mu)) / (2 * np.pi))
        for k in range(len(a)):
            rows.append({"s2_over_s1": ratio, "lam": args.lam, "edge": edge, "delta": float(deltas[k]), "a": float(a[k]),
                         "birth_theta_over_2pi": birth,
                         "rotation": float(rho[k]), "planar_share": float(share[k]), "period": int(period[k]),
                         "amplitude": float(amplitude[k]), "lyapunov": lyap[k].tolist(), "mean_loss": float(loss[k])})
            print(f"s2/s1 {ratio} delta {deltas[k]:.0e}: rotation {rho[k]:.5f} (birth {birth:.5f}) period {period[k]} "
                  f"amplitude {amplitude[k]:.3e} planar {share[k]:.3f} lyapunov {np.round(lyap[k], 5).tolist()} "
                  f"loss {loss[k]:.4e}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"circle-lam{args.lam}-{args.tag}.json").write_text(json.dumps(rows, indent=1))


def cmd_staircase(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ratios = RATIOS if args.ratios is None else tuple(args.ratios)
    fig, axes = plt.subplots(len(ratios), 1, figsize=(10, 3.2 * len(ratios)), constrained_layout=True, squeeze=False)
    axes = axes[:, 0]
    report = []
    for ax, ratio in zip(np.atleast_1d(axes), ratios):
        model = DeepLinear2x2(1.0, ratio, args.lam)
        edge = edge_rate(model)
        top = window(model)[1] - 1e-4
        a = np.linspace(edge * 1.0005, top, args.cells) if args.a_range is None else np.linspace(*args.a_range, args.cells)
        theta = np.stack([perturbed_aligned(model, min(ai, top), 1e-3) for ai in a])
        theta, feats, lyap, loss = run_attractor(model, a, theta, args.transient, args.samples)
        rho, share, period, amplitude = rotation_and_period(feats)
        rows = [{"a": float(a[i]), "rotation": float(rho[i]), "period": int(period[i]), "planar_share": float(share[i]),
                 "amplitude": float(amplitude[i]), "lyapunov": lyap[i].tolist(), "mean_loss": float(loss[i])}
                for i in range(len(a))]
        report.append({"s2_over_s1": ratio, "edge": edge, "rows": rows})
        lam2 = np.sort(lyap, axis=1)[:, ::-1]
        periodic = period > 0
        chaotic = (~periodic) & (lam2[:, 0] > 5e-4) & (lam2[:, 1] > 5e-4)
        quasi = (~periodic) & ~chaotic
        ax.plot(a[quasi], rho[quasi], ".", ms=2.5, color="#1baf7a", label="non-periodic, no second positive exponent")
        ax.plot(a[periodic], rho[periodic], ".", ms=3.5, color="#2a78d6", label="periodic (two-step recurrence)")
        ax.plot(a[chaotic], rho[chaotic], ".", ms=2.5, color="#a3a29b", label="two positive exponents (chaotic)")
        ax.set_title(f"s2/s1 = {ratio}: rotation of the two-step map beyond the rotational edge "
                     f"(a = {edge:.4f}); exploratory", loc="left", fontsize=9)
        ax.set_ylabel("rotation (turns / two steps)", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        values, counts = np.unique(period[periodic], return_counts=True)
        print(f"s2/s1 {ratio}: {periodic.sum()} periodic cells {dict(zip(values.tolist(), counts.tolist()))}, "
              f"{quasi.sum()} non-periodic without a second positive exponent, {chaotic.sum()} chaotic")
    np.atleast_1d(axes)[-1].set_xlabel("learning rate a (s1 = 1)", fontsize=9)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"staircase-lam{args.lam}-{args.tag}.png", dpi=160)
    (OUT / f"staircase-lam{args.lam}-{args.tag}.json").write_text(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["edge", "circle", "staircase"])
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--transient", type=int, default=200000)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--deltas", type=float, nargs="+", default=[1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2])
    parser.add_argument("--sequential", action="store_true", help="follow the attractor through the deltas in order")
    parser.add_argument("--tag", default="default")
    parser.add_argument("--ratios", type=float, nargs="+", default=None)
    parser.add_argument("--a-range", type=float, nargs=2, default=None)
    args = parser.parse_args()
    {"edge": cmd_edge, "circle": cmd_circle, "staircase": cmd_staircase}[args.command](args)


if __name__ == "__main__":
    main()
