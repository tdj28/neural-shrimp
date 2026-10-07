"""Exploratory A3b: noisy precursor of the Neimark-Sacker bifurcation in the training loss.

Before the period-2 orbit loses stability, its two-step map has a weakly
damped complex multiplier pair mu. Small additive noise on each update (a
stand-in for minibatch noise) keeps exciting that mode, so the even-step
training-loss series should show a spectral peak at arg(mu) / 2 pi turns per
two steps, growing as |mu| approaches 1 (a noisy precursor). Beyond the
bifurcation the peak should sit at the rotation number of the invariant
circle. The largest FTLE predicts the damping (peak height and width) but not
the frequency, which comes from the multiplier angle.

For every cell of a slice: settle 40,000 noiseless steps; for period-2 cells,
compute the two-step multipliers from dense Jacobians; then record the loss
for ``--record`` steps with Gaussian noise of standard deviation ``--noise``
added to every parameter after each update (seed 28730), and report the
strongest peak of the even-step loss spectrum and its height relative to the
median spectral power.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.gdmaps import GridMap  # noqa: E402
from neural_shrimp.regimes import peak_frequencies  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))
NOISE_SEED = 28730


def peak_height(series, frequency, width=2):
    x = np.asarray(series, dtype=np.float64)
    x = (x - x.mean()) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x))
    i = int(np.argmin(np.abs(freq - frequency)))
    return float(power[max(i - width, 1):i + width + 1].max() / np.median(power[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=float, required=True)
    parser.add_argument("--hidden", type=float, nargs=2, required=True)
    parser.add_argument("--cells", type=int, default=40)
    parser.add_argument("--noise", type=float, nargs="+", default=[1e-6])
    parser.add_argument("--settle", type=int, default=40000)
    parser.add_argument("--record", type=int, default=16384)
    parser.add_argument("--nice", type=int, default=10)
    args = parser.parse_args()
    if 29000 <= args.seed < 30000:
        raise SystemExit("seeds 29000-29999 are reserved for confirmatory tests")
    os.nice(args.nice)
    out = DATA / "explore" / "consequence" / args.name
    out.mkdir(parents=True, exist_ok=False)
    net, x, y, theta0 = SYSTEMS[args.system](args.seed)
    lh = np.linspace(args.hidden[0], args.hidden[1], args.cells)
    grid = GridMap(net, x, y, 10.0 ** lh, np.full(args.cells, 10.0 ** args.output))
    theta = np.repeat(theta0[None], args.cells, axis=0)
    for _ in range(args.settle):
        theta = grid.step(theta)
    rows = []
    for i in range(args.cells):
        map_fn, jac = grid.cell_map(i)
        x0 = theta[i]
        x1 = map_fn(x0)
        is_p2 = bool(np.max(np.abs(map_fn(x1) - x0)) < 1e-8 and np.max(np.abs(x1 - x0)) > 1e-6)
        row = {"log10_hidden": float(lh[i]), "period2": is_p2}
        if is_p2:
            mu = np.linalg.eigvals(jac(x1) @ jac(x0))
            complex_mu = mu[np.abs(mu.imag) > 1e-12]
            top = complex_mu[np.argmax(np.abs(complex_mu))] if len(complex_mu) else None
            row["max_modulus"] = float(np.max(np.abs(mu)))
            if top is not None:
                row["complex_modulus"] = float(abs(top))
                row["predicted_frequency"] = float(abs(np.angle(top)) / (2 * np.pi))
        rows.append(row)
    settled = theta.copy()
    for k, sigma in enumerate(args.noise):
        rng = np.random.default_rng(NOISE_SEED + k)
        theta = settled.copy()
        losses = np.empty((args.record, args.cells))
        for t in range(args.record):
            theta, loss = grid.step(theta, return_loss=True)
            theta = theta + sigma * rng.normal(size=theta.shape)
            losses[t] = loss
        for i, row in enumerate(rows):
            series = losses[::2, i]
            if not np.all(np.isfinite(series)):
                continue
            peaks = peak_frequencies(series, count=3)
            entry = {"peaks": peaks}
            if "predicted_frequency" in row:
                entry["height_at_prediction"] = peak_height(series, row["predicted_frequency"])
                entry["gap_to_prediction"] = (min(abs(p - row["predicted_frequency"]) for p in peaks)
                                              if peaks else None)
            row[f"noise_{sigma:g}"] = entry
    (out / "precursor.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
