"""Known maps distinguish recurrence, prime period, attraction, and expansion."""

import unittest

import numpy as np

from neural_shrimp.controls import HenonMap, LogisticMap
from neural_shrimp.diagnostics import (
    certify_cycle, classify_tail, iterate_map, largest_lyapunov,
)


class DiagnosticTests(unittest.TestCase):
    def test_logistic_prime_two_and_exact_multiplier(self):
        dynamics = LogisticMap(3.2)
        trajectory = iterate_map(dynamics, [0.2], steps=300, tail=120)
        candidate = classify_tail(trajectory.states)
        self.assertEqual(candidate["classification"], "periodic_candidate")
        self.assertEqual(candidate["period"], 2)
        verified = certify_cycle(dynamics, trajectory.states[-1], 2, dynamics.jacobian)
        self.assertTrue(verified["verified"])
        self.assertTrue(verified["attracting"])
        self.assertAlmostEqual(verified["spectral_radius"], abs(-3.2**2 + 2 * 3.2 + 4), places=10)
        exact = (3.2 + 1 + np.array([-1, 1]) * np.sqrt((3.2 - 3) * (3.2 + 1))) / (2 * 3.2)
        np.testing.assert_allclose(np.sort(np.array(verified["orbit"])[:, 0]), exact, atol=1e-12)

    def test_logistic_prime_four_and_nonprimitive_rejection(self):
        dynamics = LogisticMap(3.5)
        trajectory = iterate_map(dynamics, [0.2], 300, tail=120)
        self.assertEqual(classify_tail(trajectory.states)["period"], 4)
        self.assertTrue(certify_cycle(dynamics, trajectory.states[-1], 4, dynamics.jacobian)["attracting"])
        two = LogisticMap(3.2)
        point = iterate_map(two, [0.2], 300).states[-1]
        invalid = certify_cycle(two, point, 4, two.jacobian)
        self.assertFalse(invalid["verified"])
        self.assertEqual(invalid["reason"], "nonprimitive_period")
        self.assertEqual(invalid["primitive_period"], 2)

    def test_henon_stable_fixed_point_and_known_eigenvalues(self):
        dynamics = HenonMap(a=0.0, b=0.3)
        trajectory = iterate_map(dynamics, [0.0, 0.0], 150, tail=60)
        self.assertEqual(classify_tail(trajectory.states)["classification"], "fixed_point")
        verified = certify_cycle(dynamics, trajectory.states[-1], 1, dynamics.jacobian)
        self.assertTrue(verified["attracting"])
        self.assertAlmostEqual(verified["spectral_radius"], np.sqrt(0.3), places=12)
        np.testing.assert_allclose(verified["orbit"][0], [1 / 0.7, 0.3 / 0.7], atol=1e-12)

    def test_unstable_and_neutral_roots_are_not_called_attracting(self):
        dynamics = LogisticMap(3.2)
        unstable = certify_cycle(dynamics, [1 - 1 / 3.2], 1, dynamics.jacobian)
        self.assertTrue(unstable["verified"])
        self.assertFalse(unstable["attracting"])
        self.assertEqual(unstable["stability"], "unstable")
        identity = lambda x: x.copy()
        neutral = certify_cycle(identity, [0.7, -0.2], 1, lambda x: np.eye(2))
        self.assertTrue(neutral["verified"])
        self.assertEqual(neutral["stability"], "neutral_or_unresolved")

    def test_bounded_irregularity_without_ftle_is_unresolved(self):
        trajectory = np.random.default_rng(42).uniform(-1, 1, size=(100, 2))
        self.assertEqual(classify_tail(trajectory)["classification"], "unresolved")
        self.assertEqual(classify_tail(trajectory, lyapunov=0.2)["classification"], "chaotic_candidate")
        self.assertEqual(classify_tail(trajectory, lyapunov={"exponent": 0.2, "status": "failed"})["classification"], "unresolved")

    def test_last_pair_recurrence_does_not_hide_transient_drift(self):
        states = np.arange(30, dtype=float)[:, None]
        states[-1] = states[-2]
        self.assertEqual(classify_tail(states)["classification"], "unresolved")
        # All consecutive differences pass atol, but accumulated drift does not.
        slow = np.arange(30, dtype=float)[:, None] * 0.4e-8
        self.assertEqual(classify_tail(slow, max_period=1, atol=1e-8, rtol=0)["classification"], "unresolved")

    def test_remote_root_is_not_attributed_to_observed_tail(self):
        contracting = lambda x: 0.5 * x
        result = certify_cycle(contracting, [10.0], 1, lambda x: np.array([[0.5]]))
        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "root_remote_from_candidate")

    def test_lyapunov_against_logistic_chaotic_and_periodic_controls(self):
        chaotic = LogisticMap(4.0)
        expansion = largest_lyapunov(chaotic, chaotic.jacobian, [0.213], steps=3000, burn_in=100)
        self.assertEqual(expansion["status"], "ok")
        self.assertAlmostEqual(expansion["exponent"], np.log(2), delta=0.01)
        periodic = LogisticMap(3.2)
        contraction = largest_lyapunov(periodic, periodic.jacobian, [0.2], steps=300, burn_in=100)
        self.assertAlmostEqual(contraction["exponent"], np.log(0.16) / 2, places=10)

    def test_singular_tangents_and_linear_spectrum(self):
        matrix = np.diag([0.5, 0.0])
        result = largest_lyapunov(lambda x: matrix @ x, lambda x: matrix,
                                  [0.2, 0.4], steps=50, burn_in=20)
        self.assertAlmostEqual(result["exponent"], np.log(0.5), places=12)
        self.assertEqual(result["negative_infinity_indices"], [1])
        zero = np.zeros((2, 2))
        collapsed = largest_lyapunov(lambda x: zero @ x, lambda x: zero,
                                     [0.2, 0.4], steps=10)
        self.assertIsNone(collapsed["exponent"])
        self.assertTrue(collapsed["largest_is_negative_infinity"])

    def test_divergence_and_tail_retention(self):
        divergent = iterate_map(lambda x: 2 * x, [1.0], 100, tail=5, divergence_threshold=100)
        self.assertEqual(divergent.status, "divergent")
        self.assertEqual(divergent.steps_completed, 7)
        self.assertEqual(divergent.states.shape, (5, 1))
        self.assertEqual(classify_tail(divergent.states, divergence_threshold=100)["classification"], "divergent")
        overflow = iterate_map(lambda x: x * x, [1e100], 10, divergence_threshold=1e300)
        self.assertEqual(overflow.status, "divergent")


if __name__ == "__main__":
    unittest.main()
