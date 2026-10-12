"""Small analytic controls for independently authored generic numerics."""
import numpy as np
import pytest

from codex.crosscheck.numerics import (
    cycle_points, fold_normal_form, fold_refine, monodromy, newton_cycle,
    orbit_diagnostics, quadratic_passage, refine_fold, tangent_product,
)


def test_newton_logistic_primitive_period_two():
    rate = 3.2
    step = lambda x: rate*x*(1-x)
    jvp = lambda x, v: rate*(1-2*x)*v
    result = newton_cycle(step, jvp, np.array([0.5]), 2)
    assert result["converged"], result
    expected = (rate+1-np.sqrt((rate-3)*(rate+1)))/(2*rate)
    assert abs(result["state"][0]-expected) < 1e-12
    diagnostics = orbit_diagnostics(step, jvp, result["state"], 2)
    assert diagnostics["closure_passed"] and diagnostics["primitive_separation_passed"]
    assert diagnostics["numerically_attracting"]
    assert diagnostics["proper_divisor_returns"][1] > .1


def test_period_eight_affine_charts_and_tangent_batch_orientation():
    # An affine map on each of eight disjoint local charts. A globally affine
    # isolated attracting orbit of primitive period>1 would be impossible.
    angles = np.arange(8)*np.pi/4
    centers = np.column_stack([np.cos(angles), np.sin(angles)])
    A = np.array([[.21, .03], [-.02, .16]])
    def step(x):
        k = np.argmin(np.linalg.norm(centers-x, axis=1))
        return centers[(k+1)%8] + A@(x-centers[k])
    jvp = lambda x, v: A@v
    answer = newton_cycle(step, jvp, centers[0]+np.array([.01, -.01]), 8)
    assert answer["converged"]
    assert np.linalg.norm(answer["state"]-centers[0]) < 1e-12
    matrix = monodromy(step, jvp, answer["state"], 8,
                       jvp_rows=lambda x, V: V@A.T, chunk=1)
    assert np.allclose(matrix, np.linalg.matrix_power(A, 8), rtol=1e-13, atol=1e-16)
    diagnostics = orbit_diagnostics(step, jvp, answer["state"], 8, matrix=matrix)
    assert diagnostics["closure_passed"] and diagnostics["primitive_separation_passed"]
    assert set(diagnostics["proper_divisor_returns"]) == {1, 2, 4}
    assert diagnostics["numerically_attracting"]


def test_nonsymmetric_temporal_order_is_not_reversed():
    A, B = np.array([[1., 2.], [0., 1.]]), np.array([[1., 0.], [3., 1.]])
    points = [np.array([0., 0.]), np.array([1., 0.]), np.array([0., 0.])]
    jvp = lambda x, v: (A if x[0] == 0 else B)@v
    v = np.array([.3, .7])
    assert np.array_equal(tangent_product(jvp, points, v), B@A@v)
    assert not np.allclose(B@A@v, A@B@v)


def polynomial_step(x, a):
    return np.array([x[0]+2*a+3*x[0]**2, .25*x[1]])


def polynomial_jvp(x, v, a):
    return np.array([(1+6*x[0])*v[0], .25*v[1]])


def test_spectral_fold_corrector_and_normal_form():
    mono = lambda x, a: np.diag([1+6*x[0], .25])
    da = lambda x, a: np.array([2., 0.])
    gradient = lambda x, a, l, r: (np.array([6*l[0]*r[0], 0.]), 0.)
    result = fold_refine(polynomial_step, mono, da, gradient,
                         np.array([.02, .03]), -.003, residual_scale=1.)
    assert result["converged"], result
    assert np.linalg.norm(result["state"]) < 1e-12 and abs(result["alpha"]) < 1e-12
    nf = fold_normal_form(polynomial_step, polynomial_jvp, np.zeros(2), 0., 1,
                          matrix=mono(np.zeros(2), 0.), reference_right=np.array([1., 0.]),
                          residual_scale=1., parameter_derivative=da,
                          directional_second=lambda x,r,a: np.array([6*r[0]**2, 0.]))
    assert abs(nf["a"]-2) < 1e-12 and abs(nf["b"]-3) < 1e-9
    assert abs(nf["ab"]-6) < 1e-9
    assert nf["exact_callback_a"] == 2 and nf["exact_callback_b"] == 3
    negative = fold_normal_form(polynomial_step, polynomial_jvp, np.zeros(2), 0., 1,
                                matrix=mono(np.zeros(2), 0.), reference_right=np.array([-1., 0.]),
                                residual_scale=1.)
    assert negative["a"] < 0 and negative["b"] < 0
    assert abs(negative["ab"]-nf["ab"]) < 1e-10


def test_matrix_free_extended_fold_positive_control():
    result = refine_fold(polynomial_step, polynomial_jvp, np.array([.02, .03]), -.003,
                         1, np.array([1., .01]), residual_scale=1., max_newton=8)
    assert result["converged"], result
    assert np.linalg.norm(result["state"]) < 1e-11 and abs(result["alpha"]) < 1e-11


def test_normal_form_refuses_flip_and_unconverged_root():
    with pytest.raises(ValueError, match="real \\+1"):
        fold_normal_form(lambda x,a: -x, lambda x,v,a: -v, np.zeros(1), 0., 1,
                         matrix=np.array([[-1.]]), residual_scale=1.)
    with pytest.raises(ValueError, match="converged periodic"):
        fold_normal_form(polynomial_step, polynomial_jvp, np.array([.1,.2]), 0., 1)
    with pytest.raises(ValueError, match="isolated"):
        fold_normal_form(lambda x,a: x, lambda x,v,a: v, np.zeros(2), 0., 1,
                         matrix=np.eye(2), residual_scale=1.)


def test_unconverged_linear_solve_is_not_promoted():
    # Translation has no fixed point and singular return-minus-identity.
    result = newton_cycle(lambda x: x+1, lambda x,v: v, np.array([1.]), 1,
                          max_linear_iterations=1)
    assert not result["converged"] and result["reason"] == "linear_solve_failed"


def test_passage_exact_clock_strict_exit_and_censoring():
    # First iterate is exactly threshold and must not count as an exit.
    answer = quadratic_passage(0., 1., 1., .1, .1)
    assert answer["discrete"] == 2
    assert quadratic_passage(0., 1., 1., .1, .1, max_iter=1)["discrete"] is None
    a, b = 4.834227003692353, 25.42851228205931
    U = .1/b
    assert [quadratic_passage(-U,a,b,e,U)["discrete"] for e in [3e-6,1e-5,3e-5,1e-4]] == [144,71,34,14]
    with pytest.raises(ValueError):
        quadratic_passage(0., -1., 1., .1, .1)


def test_spectral_contraction_does_not_promote_nonorbit():
    result = orbit_diagnostics(lambda x: .5*x, lambda x,v: .5*v, np.ones(2), 1)
    assert result["spectral_contraction_passed"]
    assert not result["closure_passed"] and not result["numerically_attracting"]
