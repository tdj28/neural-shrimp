import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_latent_loop as ell  # noqa: E402

torch = pytest.importorskip("torch")


def logistic3(x, r):
    for _ in range(3):
        x = r * x * (1 - x)
    return x


def turn_index(records):
    turns = [r["index"] for r in records if "turn_lambda_estimate" in r]
    assert turns, "no turn found"
    return turns[0]


def test_arclength_turns_at_the_period3_saddle_node():
    """The logistic period-3 window is born in a saddle-node at r = 1 + sqrt(8): continuing the stable 3-cycle
    toward it, lambda must turn there and the multiplier must pass through +1 onto the unstable branch."""
    r0 = 3.835
    x = torch.tensor([0.5], dtype=torch.float64)
    for _ in range(3000):
        x = r0 * x * (1 - x)
    records, reason = ell.pseudo_arclength(logistic3, x, r0, direction=-1, steps=200, ds=0.002, ds_max=0.01,
                                           tol=1e-12, k=1, beyond=0.003)
    i = turn_index(records)
    assert abs(records[i]["turn_lambda_estimate"] - (1 + np.sqrt(8))) < 1e-4, reason
    before = [r["real_multiplier_nearest_one"] for r in records[: i - 1]]
    after = [r["real_multiplier_nearest_one"] for r in records[i:]]
    assert before and after
    assert all(abs(m) < 1 for m in before) and all(m > 1 for m in after)


def test_arclength_fold_in_many_dimensions_uses_arnoldi():
    """x0 -> x0 + lam - x0^2 with 99 contracting coordinates driven by x0: fold at lam = 0 (multiplier
    1 - 2 x0 -> +1); the state is large enough for the Arnoldi path."""
    c = torch.linspace(0.1, 0.5, 99, dtype=torch.float64)

    def phi(x, lam):
        return torch.cat([x[:1] + lam - x[:1] ** 2, c * x[1:] + 0.1 * x[:1]])

    x0 = torch.cat([torch.tensor([0.2], dtype=torch.float64), 0.02 / (1 - c)])
    records, reason = ell.pseudo_arclength(phi, x0, 0.04, direction=-1, steps=120, ds=0.004, ds_max=0.02,
                                           tol=1e-11, k=4, beyond=0.01)
    i = turn_index(records)
    assert abs(records[i]["turn_lambda_estimate"]) < 1e-4, reason
    assert records[0]["spectral_radius"] == pytest.approx(0.6, abs=1e-6)
    assert all(r["real_multiplier_nearest_one"] < 1 for r in records[: i - 1])
    assert all(r["real_multiplier_nearest_one"] > 1 for r in records[i:])
    assert records[-1]["lambda"] >= 0.01 - 1e-12                 # continued beyond the turn on the other branch


def test_rotation_summary_tells_quasi_periodic_from_locked_motion():
    """An ellipse traversed with an irrational rotation never returns; a 2/13 resonance returns after 13 steps;
    the rotation number comes out in turns per step either way, also for an off-centre, skewed projection."""
    t = np.arange(4000)
    golden = (np.sqrt(5) - 1) / 2 * 0.25
    for rho, q in ((golden, None), (2 / 13, 13)):
        phase = 2 * np.pi * rho * t
        z = 0.3 + 0.1j + 2.0 * np.cos(phase) + 1j * (0.5 * np.sin(phase) + 0.4 * np.cos(phase))
        r, locked, d = ell.rotation_summary(z, max_q=60, lock_tol=1e-6)
        assert abs(r - rho) < 1e-3
        assert locked == q
