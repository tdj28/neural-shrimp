"""Validate the matrix-free multiplier and period-2 machinery on a known Neimark-Sacker.

Replays S2-28011 at log10 output rate -0.0856, just below and at the traced
Neimark-Sacker point of its period-2 orbit (log10 hidden 1.738439), through the
torch code used for CIFAR-10 (neural_shrimp.cifar: Problem, two_step_multipliers,
newton_period2), and compares with the dense NumPy reference (exact Jacobian,
Newton certification). Writes DATA/explore/cifar/validation/s2-replay.json.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp import cifar  # noqa: E402
from neural_shrimp.gdmaps import GridMap  # noqa: E402
from neural_shrimp.orbits import certify_orbit  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))


def main():
    torch.set_num_threads(4)
    net, x, y, theta0 = SYSTEMS["S2"](28011)
    log_output = -0.0856
    rows = []
    for log_hidden in (1.7370, 1.7380, 1.7384):
        grid = GridMap(net, x, y, [10.0 ** log_hidden], [10.0 ** log_output])
        theta = theta0[None].copy()
        for _ in range(60000):
            theta = grid.step(theta)
        map_fn, jacobian = grid.cell_map(0)
        cert = certify_orbit(map_fn, jacobian, theta[0], 2)
        orbit = np.asarray(cert["orbit"][0])
        dense = jacobian(map_fn(orbit)) @ jacobian(orbit)
        dense_values = np.linalg.eigvals(dense)
        dense_complex = max((v for v in dense_values if v.imag > 1e-12), key=abs)

        model = cifar.MLP(d=x.shape[1], width=net.width, k=1)
        problem = cifar.Problem(model, x, np.asarray(y)[:, None], lam=net.ridge)
        eta = torch.as_tensor(grid.rates[0].copy())
        loss_numpy = float(net.loss(orbit[None], x, y)[0])
        loss_torch = float(problem.loss(torch.as_tensor(orbit)))
        arnoldi = cifar.two_step_multipliers(problem, orbit, eta, k=4, tol=1e-13)
        mu = complex(*arnoldi["dominant_complex"])
        solved, history = cifar.newton_period2(problem, orbit + 1e-6 * np.random.default_rng(0).normal(size=orbit.size),
                                               eta, tol=1e-13)
        rows.append({
            "log10_hidden": log_hidden, "log10_output": log_output, "certified": bool(cert.get("verified")),
            "loss_numpy": loss_numpy, "loss_torch": loss_torch,
            "dense_dominant_complex": [dense_complex.real, dense_complex.imag],
            "dense_modulus": abs(dense_complex), "arnoldi_dominant_complex": [mu.real, mu.imag],
            "arnoldi_modulus": abs(mu), "abs_difference": abs(mu - dense_complex),
            "theta_over_2pi": abs(np.angle(mu)) / (2 * np.pi),
            "newton_final_relative_residual": history[-1]["relative_residual"],
            "newton_distance_to_dense_orbit": float(np.min([np.linalg.norm(solved.numpy() - np.asarray(p))
                                                            for p in cert["orbit"]])),
        })
        print(json.dumps(rows[-1]), flush=True)
    out = DATA / "explore" / "cifar" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "s2-replay.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
