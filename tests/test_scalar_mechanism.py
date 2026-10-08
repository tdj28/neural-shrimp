import numpy as np
from numpy.testing import assert_allclose

from neural_shrimp.criticality import finite_difference_forms
from neural_shrimp.minimal import TwoModeFactorization
from neural_shrimp.scalar_mechanism import ScalarFactorization, planar_stability


def test_scalar_matches_decoupled_four_weight_map():
    scalar = ScalarFactorization(target=1.2, ridge=0.2)
    model = TwoModeFactorization(c=0, lam=0.2)
    x = np.array([1.11, 0.83])
    theta = np.array([x[0], np.sqrt(0.5), x[1], np.sqrt(0.5)])
    assert_allclose(scalar.step(x, 0.4, 2.8), model.step(theta, 0.4, 1.12)[[0, 2]], atol=1e-15)
    assert_allclose(scalar.hessian(x), model.hessian(theta)[np.ix_([0, 2], [0, 2])])


def test_cycle_mean_difference_equations_are_exact():
    model = ScalarFactorization()
    mean, half = np.array([0.8, 1.1]), np.array([0.1, -0.2])
    gp, gm = model.gradient(mean+half), model.gradient(mean-half)
    expected = np.concatenate([(gp+gm)/2, (gp-gm)/2 - 2*half/np.array([0.3, 0.9])])
    assert_allclose(model.cycle_equations(mean, half, 0.3, 3), expected, atol=1e-15)


def test_composed_analytic_forms_against_finite_differences():
    model = ScalarFactorization()
    x, eta, ratio = np.array([1.1, 0.8]), 0.4, 2.8
    A, B, C = model.two_step_forms(x, eta, ratio)
    def jvp(points, vectors):
        return np.array([model.two_step_forms(p, eta, ratio)[0]@v for p, v in zip(points, vectors)])
    Bfd, Cfd = finite_difference_forms(jvp, x)
    a, b, c = np.array([0.3, -0.7]), np.array([0.2, 0.4]), np.array([-0.6, 0.1])
    assert_allclose(B(a, b), Bfd(a, b), rtol=1e-7, atol=1e-8)
    assert_allclose(C(a, b, c), Cfd(a, b, c), rtol=1e-6, atol=1e-7)
    h = 1e-6
    G = lambda p: model.step(model.step(p, eta, ratio), eta, ratio)
    assert_allclose(A@a, (G(x+h*a)-G(x-h*a))/(2*h), rtol=1e-8, atol=1e-8)


def test_balanced_equal_rate_steps_commute_and_have_real_monodromy():
    model = ScalarFactorization()
    A, _, _ = model.two_step_forms(np.array([1.1, 1.1]), 0.7, 1)
    assert np.max(np.abs(np.linalg.eigvals(A).imag)) == 0
    assert planar_stability(A)["discriminant"] >= -1e-14


def test_jury_margins_distinguish_planar_boundaries():
    assert planar_stability(0.9*np.array([[0, -1], [1, 0]]))["attracting"]
    assert not planar_stability(1.1*np.array([[0, -1], [1, 0]]))["attracting"]
    assert not planar_stability(np.diag([-1.1, 0.5]))["attracting"]
    assert not planar_stability(np.diag([1.1, 0.5]))["attracting"]


def test_explicit_balanced_cycle_and_both_multipliers():
    from neural_shrimp.scalar_mechanism import balanced_cycle
    for ridge in (0.05, 0.1, 0.2, 0.3):
        model = ScalarFactorization(ridge=ridge)
        for factor in (1.02, 1.15, np.sqrt(5)-1):
            eta = factor/(model.target-ridge)
            orbit, multipliers = balanced_cycle(model.target, ridge, eta)
            assert_allclose(model.step(orbit[0], eta, 1), orbit[1], atol=2e-15)
            assert_allclose(model.step(orbit[1], eta, 1), orbit[0], atol=2e-15)
            numeric = np.linalg.eigvals(model.two_step_forms(orbit[0], eta, 1)[0])
            assert_allclose(np.sort(numeric), np.sort(multipliers), atol=4e-14)


def test_closed_form_minimum_flip_with_unequal_rates():
    for ridge in (0.05, 0.1, 0.2, 0.3):
        model = ScalarFactorization(ridge=ridge)
        for ratio in (1, 2, np.sqrt(8), 4):
            d = model.target
            expected = 4/(d*(1+ratio) + np.sqrt(d*d*(1-ratio)**2+4*ratio*(d-2*ridge)**2))
            assert_allclose(model.flip_rate(ratio), expected, atol=1e-15)
