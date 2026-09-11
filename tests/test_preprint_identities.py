"""Synthetic checks through a two-state Markov chain, not neural target runs."""
from fractions import Fraction
import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "preprint_identity_audit",
    Path(__file__).resolve().parents[1] / "tools/audit_preprint_identities.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def refresh_transition(delta, beta, field=0.0):
    """Enumerate hold/refresh alternatives; rows old spin, columns new spin."""
    spins = (-1, 1)
    return np.array([
        [
            (1 - delta) * (new == old)
            + delta * (1 + new * np.tanh(beta * field)) / 2
            for new in spins
        ]
        for old in spins
    ])


@pytest.mark.parametrize("delta", [Fraction(1, 10), Fraction(2, 5), Fraction(1)])
def test_lag_table_against_independently_enumerated_chain(delta):
    beta = Fraction(7, 3)
    rows = audit.lag_table(delta, beta, 8)
    transition = refresh_transition(float(delta), float(beta))
    spins = np.array([-1.0, 1.0])
    stationary = np.full(2, 0.5)
    np.testing.assert_allclose(transition.sum(axis=1), 1)
    np.testing.assert_allclose(stationary @ transition, stationary)
    epsilon = 1e-6
    plus = refresh_transition(float(delta), float(beta), epsilon)
    minus = refresh_transition(float(delta), float(beta), -epsilon)
    for row in rows:
        lag = row["lag"]
        joint = stationary[:, None] * np.linalg.matrix_power(transition, lag)
        correlation = np.sum(joint * np.outer(spins, spins))
        after_impulse = np.linalg.matrix_power(transition, lag - 1)
        response = stationary @ ((plus - minus) / (2 * epsilon)) @ after_impulse @ spins
        assert row["numeric"]["q_update_law"] == pytest.approx(correlation, abs=2e-15)
        assert row["numeric"]["chi_update_law"] == pytest.approx(response, abs=1e-10)


def test_printed_identities_fail_partial_refresh_but_match_full_refresh():
    partial = audit.lag_table(max_lag=1)[0]["exact"]
    assert partial == {
        "q_update_law": "9/10",
        "q_printed_S40": "0",
        "chi_update_law": "1/10",
        "chi_printed_S41": "1",
    }
    for row in audit.lag_table(Fraction(1), Fraction(2), 5):
        assert row["exact"]["q_update_law"] == row["exact"]["q_printed_S40"]
        assert row["exact"]["chi_update_law"] == row["exact"]["chi_printed_S41"]


def test_finite_causal_matrix_can_amplify_despite_zero_spectral_radius():
    # A strict lower shift is a counterexample to equating finite-window
    # amplification with a nonzero eigenvalue radius. No paper matrix is used.
    causal = 2 * np.eye(5, k=-1)
    assert np.max(np.abs(np.linalg.eigvals(causal))) == 0
    assert np.linalg.norm(causal, 2) == pytest.approx(2)
    np.testing.assert_array_equal(np.linalg.matrix_power(causal, 5), np.zeros((5, 5)))


@pytest.mark.parametrize("delta,beta,lag", [
    (0, 1, 2), (-1, 1, 2), (2, 1, 2),
    (Fraction(1, 10), 0, 2), (Fraction(1, 10), 1, 0),
    (Fraction(1, 10), 1, 1.5), (Fraction(1, 10), 1, True),
])
def test_invalid_audit_parameters_fail(delta, beta, lag):
    with pytest.raises(ValueError):
        audit.lag_table(delta, beta, lag)
