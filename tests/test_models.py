"""Derivative and data-isolation audits against independent calculations."""

import unittest

import numpy as np

from neural_shrimp.diagnostics import finite_difference_jacobian
from neural_shrimp.models import TanhRegressor, make_teacher_regression


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = TanhRegressor(d=2, width=3, ridge=0.017)
        self.data = make_teacher_regression(seed=7, d=2, n_train=11,
                                            n_validation=7, n_test=9)
        self.theta = self.model.initialize(seed=42, scale=0.7)
        self.theta += np.random.default_rng(91).normal(scale=0.1, size=len(self.theta))

    def test_analytic_gradient_against_scalar_loss_differences(self):
        numerical = np.empty_like(self.theta)
        for i in range(len(self.theta)):
            direction = np.zeros_like(self.theta)
            direction[i] = 1e-6
            numerical[i] = (self.model.loss(self.theta + direction, self.data.train_x, self.data.train_y)
                            - self.model.loss(self.theta - direction, self.data.train_x, self.data.train_y)) / 2e-6
        np.testing.assert_allclose(
            self.model.gradient(self.theta, self.data.train_x, self.data.train_y), numerical,
            rtol=2e-6, atol=2e-9)

    def test_hessian_contains_residual_curvature(self):
        gradient_fn = lambda t: self.model.gradient(t, self.data.train_x, self.data.train_y)
        numerical = finite_difference_jacobian(gradient_fn, self.theta)
        analytical = self.model.hessian(self.theta, self.data.train_x, self.data.train_y)
        np.testing.assert_allclose(analytical, numerical, rtol=3e-6, atol=3e-9)
        np.testing.assert_allclose(analytical, analytical.T, atol=1e-14)

    def test_training_map_jacobian_with_distinct_rates(self):
        training = self.model.training_map(self.data.train_x, self.data.train_y,
                                           lr_hidden=0.37, lr_output=1.13)
        numerical = finite_difference_jacobian(training, self.theta)
        np.testing.assert_allclose(training.jacobian(self.theta), numerical, rtol=3e-6, atol=3e-9)

    def test_small_step_descends_and_zero_rate_freezes_its_layer(self):
        training = self.model.training_map(self.data.train_x, self.data.train_y, 0.0, 1e-3)
        after = training(self.theta)
        count = self.model.hidden_parameter_count
        np.testing.assert_array_equal(after[:count], self.theta[:count])
        self.assertLess(self.model.loss(after, self.data.train_x, self.data.train_y),
                        self.model.loss(self.theta, self.data.train_x, self.data.train_y))

    def test_teacher_labels_and_independent_split_streams(self):
        other = make_teacher_regression(seed=7, d=2, n_train=50, n_validation=7, n_test=9)
        np.testing.assert_array_equal(other.validation_x, self.data.validation_x)
        np.testing.assert_array_equal(other.validation_y, self.data.validation_y)
        np.testing.assert_array_equal(other.test_x, self.data.test_x)
        meta = self.data.metadata
        expected = np.tanh(self.data.test_x @ np.array(meta["teacher_w"]) + meta["teacher_b"]) @ np.array(meta["teacher_v"]) + meta["teacher_c"]
        np.testing.assert_allclose(self.data.test_y, expected, atol=0.0)

    def test_training_batch_is_frozen_against_external_mutation(self):
        x, y = self.data.train_x.copy(), self.data.train_y.copy()
        training = self.model.training_map(x, y, 0.2, 0.3)
        before = training(self.theta)
        x[:] = 100
        y[:] = -100
        np.testing.assert_array_equal(training(self.theta), before)
        self.assertEqual(before.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
