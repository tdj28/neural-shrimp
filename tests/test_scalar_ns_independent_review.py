"""Synthetic controls for the separate exact-polynomial reviewer."""
import importlib.util
from fractions import Fraction
from pathlib import Path
from flint import arb, ctx

spec = importlib.util.spec_from_file_location("independent_ns_review", Path(__file__).parents[1] / "tools/review_scalar_ns.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_full_polynomial_derivatives_keep_factorials_and_high_degrees():
    p = review.Polynomial({(7, 3, 2): Fraction(3, 7)})
    assert p.diff(0, 0, 1).terms == {(5, 2, 2): Fraction(54)}
    assert p.diff(2, 2, 2).terms == {}
    with ctx.workprec(128):
        assert p.diff(0, 0, 1).evaluate([arb(2), arb(3), arb(5)]) == 388800


def test_expanded_composition_and_determinant_against_exact_rationals():
    inputs = dict(target="5/4", ridge="1/4", ratio="4/1")
    f, g, h, determinant = review.build(inputs)
    u, v, eta = Fraction(1, 2), Fraction(3, 4), Fraction(1, 8)
    def step(x):
        a, b = x
        return (a-eta*((a*b-Fraction(5, 4))*b+a/4),
                b-4*eta*((a*b-Fraction(5, 4))*a+b/4))
    expected_f = step((u, v))
    expected_g = step(expected_f)
    with ctx.workprec(512):
        z = [review.exact(x) for x in (u, v, eta)]
        assert all(f[i].evaluate(z) == review.exact(expected_f[i]) for i in range(2))
        assert all(g[i].evaluate(z) == review.exact(expected_g[i]) for i in range(2))
        a = [[g[i].diff(j).evaluate(z) for j in range(2)] for i in range(2)]
        assert determinant.evaluate(z) == a[0][0]*a[1][1]-a[0][1]*a[1][0]
        assert h[2].evaluate(z) == determinant.evaluate(z)-1
