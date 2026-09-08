from types import SimpleNamespace
import numpy as np
from neural_shrimp.utility import run_policy


class QuadraticFixture:
    """Analytic GD on x²/2, period-two at eta=2; no target network outcomes."""
    def training_map(self, x, y, lr_hidden, lr_output):
        return lambda theta: (1 - lr_hidden) * theta
    def loss(self, theta, x, y):
        return 0.5 * float(theta[0] ** 2)
    def predict(self, theta, x):
        return np.full(len(x), theta[0])


def fixture_config():
    return dict(steps=100, tail=60, max_period=4, atol=1e-8, rtol=1e-6,
                divergence_threshold=1e8, check_every=10, check_after=40,
                decay_factor=0.5, max_decays=2, stagnation_relative_improvement=.001)


def fixture_data():
    return SimpleNamespace(**{f"{split}_{kind}": np.zeros((3,1)) if kind == "x" else np.zeros(3)
                              for split in ("train", "validation", "test") for kind in ("x","y")})


def test_recurrence_rescue_preserves_paired_preintervention_path():
    fixed, a = run_policy(QuadraticFixture(), fixture_data(), [1.0], [2., 2.], "fixed", fixture_config())
    rescue, b = run_policy(QuadraticFixture(), fixture_data(), [1.0], [2., 2.], "recurrence_decay", fixture_config())
    assert fixed["terminal_test_mse"] == 1
    assert rescue["terminal_test_mse"] == 0
    assert rescue["events"][0]["step"] == 40
    np.testing.assert_array_equal(a["states"][:41], b["states"][:41])
    assert fixed["gradient_updates"] == rescue["gradient_updates"] == 100


def test_fixed_point_is_not_a_nontrivial_cycle_trigger():
    result, raw = run_policy(QuadraticFixture(), fixture_data(), [0.0], [2., 2.], "recurrence_decay", fixture_config())
    assert result["events"] == []
    assert len(raw["states"]) == 101


def test_divergence_is_retained_with_missing_terminal_quality():
    result, _ = run_policy(QuadraticFixture(), fixture_data(), [1.0], [100.,100.], "fixed", fixture_config())
    assert result["status"] == "divergent"
    assert result["terminal_test_mse"] is None
    assert result["gradient_updates"] < 100
