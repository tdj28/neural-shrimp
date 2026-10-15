import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bifurcation_curves as bc  # noqa: E402
import flip_normal_form as fnf  # noqa: E402

torch = pytest.importorskip("torch")


def toy(q, g, nu, kappa):
    """(u, w) -> (-(1 + t0) u + q u^2 + g u^3 + nu w, 0.3 w + kappa u^2): a flip of the fixed point 0 at t0 = 0."""
    def phi(z, theta):
        u, w = z[0], z[1]
        return torch.stack([-(1 + theta[0]) * u + q * u ** 2 + g * u ** 3 + nu * w, 0.3 * w + kappa * u ** 2])
    return phi


def two_cycle(q, g, nu, kappa, t0, guess):
    """Newton on f(f(z)) = z for the toy map in numpy."""
    def f(z):
        u, w = z
        return np.array([-(1 + t0) * u + q * u ** 2 + g * u ** 3 + nu * w, 0.3 * w + kappa * u ** 2])
    z = np.array(guess, dtype=float)
    for _ in range(50):
        F = f(f(z)) - z
        J = np.zeros((2, 2))
        for k in range(2):
            e = np.zeros(2); e[k] = 1e-7
            J[:, k] = (f(f(z + e)) - f(f(z - e))) / 2e-7
        z = z - np.linalg.solve(J - np.eye(2), F)
    return z, f(z)


def test_decoupled_coefficient_is_g_plus_q_squared():
    fam = bc.Family(toy(-1.0, 0.5, 0.0, 0.0), (2,), torch.float64)
    nf = fnf.flip_cubic(fam, np.zeros(2), np.array([0.0]))
    assert abs(nf["c"] - 1.5) < 1e-6 and nf["crossing"]["next_modulus"] == pytest.approx(0.3)


@pytest.mark.parametrize("q,g,nu,kappa", [(-1.0, 0.5, 0.4, 0.7), (0.8, -0.2, -0.3, 0.5)])
def test_amplitude_and_multiplier_predictions(q, g, nu, kappa):
    fam = bc.Family(toy(q, g, nu, kappa), (2,), torch.float64)
    nf = fnf.flip_cubic(fam, np.zeros(2), np.array([0.0]))
    c, l = nf["c"], nf["l"]
    beta = 1e-4 if c > 0 else -1e-4                     # the side where the doubled orbit exists
    eta = np.sqrt(beta / c)
    za, zb = two_cycle(q, g, nu, kappa, beta, [eta, 0.0])
    assert abs(abs(l @ (za - zb)) / 2 / eta - 1) < 0.02


def test_refuses_a_fold():
    def phi(z, theta):
        return torch.stack([z[0] + theta[0] + z[0] ** 2, 0.3 * z[1]])
    fam = bc.Family(phi, (2,), torch.float64)
    with pytest.raises(ValueError):
        fnf.flip_cubic(fam, np.zeros(2), np.array([0.0]))
