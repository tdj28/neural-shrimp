"""Exploratory CIFAR pilot: critical two-step multiplier below the transition, two ways.

For each learning rate below the onset of sustained loss modulation, continue
gradient descent from the sweep's saved state (the period-2 orbit attracts
there), then
  (a) compute the dominant complex two-step multiplier by matrix-free Arnoldi
      (largest-imaginary-part search) at the final state, and
  (b) estimate the same multiplier from the even-step loss alone: the decaying
      spiral toward the 2-cycle has frequency arg(mu)/2pi (turns per two-step)
      and envelope decay |mu| per two-step (AR(2) fit of the detrended
      even-step loss; complex AR roots give mu).
Writes DATA/explore/cifar/spiral/spiral.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from neural_shrimp import cifar  # noqa: E402
import explore_cifar as ec  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))


def ar2_root(series):
    """Complex root of an AR(2) fit x_t = a1 x_{t-1} + a2 x_{t-2} (least squares)."""
    x = series - series.mean()
    design = np.stack([x[1:-1], x[:-2]], 1)
    a1, a2 = np.linalg.lstsq(design, x[2:], rcond=None)[0]
    roots = np.roots([1.0, -a1, -a2])
    root = max(roots, key=lambda r: r.imag)
    return complex(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--etas", type=float, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--threads", type=int, default=14)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    problem, model = ec.subset_a_problem(args.lam)
    rows = []
    for eta in args.etas:
        state = DATA / "explore" / "cifar" / "sweep" / f"state-lam{args.lam:g}-eta{eta:.6g}.npy"
        theta = torch.as_tensor(np.load(state))
        losses = np.empty(args.steps)
        started = time.perf_counter()
        for t in range(args.steps):
            value, gradient = problem.loss_and_grad(theta)
            losses[t] = float(value)
            theta = theta - eta * gradient
        seconds = (time.perf_counter() - started) / args.steps
        t1 = problem.update(theta, eta)
        t2 = problem.update(t1, eta)
        residual = float((t2 - theta).norm() / theta.norm())
        arnoldi = cifar.two_step_multipliers(problem, theta.numpy(), eta, k=2)
        even = losses[0::2]
        trend = np.polyval(np.polyfit(np.arange(len(even)), even, 2), np.arange(len(even)))
        detrended = even - trend
        root = ar2_root(detrended[len(detrended) // 2:])
        spectrum = np.abs(np.fft.rfft(detrended)) ** 2
        freqs = np.fft.rfftfreq(len(detrended))
        row = {"eta": eta, "log10_eta": float(np.log10(eta)), "steps": args.steps, "seconds_per_step": seconds,
               "two_step_relative_residual": residual,
               "arnoldi_mu": arnoldi["dominant_complex"], "arnoldi_modulus": arnoldi["dominant_complex_modulus"],
               "arnoldi_theta_over_2pi": arnoldi["dominant_complex_theta_over_2pi"],
               "arnoldi_partial": arnoldi["partial"],
               "loss_ar2_mu": [root.real, root.imag], "loss_ar2_modulus": abs(root),
               "loss_ar2_theta_over_2pi": abs(np.angle(root)) / (2 * np.pi),
               "loss_spectrum_peak": float(freqs[1 + int(np.argmax(spectrum[1:]))]),
               "envelope_first_last_fifth": [float(np.array_split(detrended, 5)[0].std()),
                                             float(np.array_split(detrended, 5)[-1].std())]}
        rows.append(row)
        print(json.dumps(row), flush=True)
        np.save(DATA / "explore" / "cifar" / "sweep" / f"spiral-state-lam{args.lam:g}-eta{eta:.6g}.npy", theta.numpy())
    out = DATA / "explore" / "cifar" / "spiral"
    out.mkdir(parents=True, exist_ok=True)
    (out / "spiral.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
