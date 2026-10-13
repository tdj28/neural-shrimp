"""Synthetic exact-rational enclosure tests; never load a model or checkpoint."""
import importlib.util
from itertools import product
from pathlib import Path
from fractions import Fraction
import math

import numpy as np
import pytest
from flint import arb, arb_mat, ctx, fmpq

PATH = Path(__file__).resolve().parents[1] / "codex/certify-llm/ball_linear.py"
SPEC = importlib.util.spec_from_file_location("codex_llm_ball_linear", PATH)
linear = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(linear)


@pytest.fixture(autouse=True)
def precision():
    old_prec, old_threads = ctx.prec, ctx.threads
    ctx.prec, ctx.threads = 128, 1
    yield
    ctx.prec, ctx.threads = old_prec, old_threads


def rational(value):
    q = Fraction(float(value))
    return fmpq(q.numerator, q.denominator)


@pytest.mark.parametrize("prec", [16, 53, 128])
@pytest.mark.parametrize("value", [0.0, -0.0, .1, math.nextafter(.1, 1.),
    np.nextafter(0., 1.), np.finfo(np.float64).max, -np.finfo(np.float64).max,
    np.float16(.3), np.float32(.1)])
def test_scalar_exact_binary_value_even_at_low_precision(prec, value):
    ctx.prec = prec
    answer = linear.exact_dyadic(value)
    assert answer.is_exact()
    assert answer.fmpq() == rational(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_constant_rejected(value):
    with pytest.raises(ValueError):
        linear.exact_dyadic(value)


@pytest.mark.parametrize("value", [1, "0.1", Fraction(1, 10), np.longdouble(.1)])
def test_no_silent_nonbinary64_conversion(value):
    with pytest.raises(TypeError):
        linear.exact_dyadic(value)


@pytest.mark.parametrize("chunk_rows,chunk_cols", [(1, 1), (2, 3), (32, 256)])
def test_random_point_dot_products_enclose_exact_rational_oracle(chunk_rows, chunk_cols):
    rng = np.random.default_rng(831)
    weights = rng.normal(size=(7, 11))
    values = rng.normal(size=11)
    bias = rng.normal(size=7)
    result = linear.affine(weights, [linear.exact_dyadic(v) for v in values], bias,
        chunk_rows=chunk_rows, chunk_cols=chunk_cols)
    for i, actual in enumerate(result):
        expected = sum((rational(weights[i, j])*rational(values[j]) for j in range(11)), rational(bias[i]))
        assert actual.contains(expected)
        assert actual.rad() < arb((1, -110))


def test_interval_box_all_corners_with_mixed_weight_signs():
    weights = np.array([[1.5, -.125, 3.], [-2., 1., -.5]])
    center = [1., -.5, .25]
    radii = [.125, .0625, .25]
    balls = [arb(c, r) for c, r in zip(center, radii)]
    actual = linear.affine(weights, balls, chunk_rows=1, chunk_cols=2)
    for signs in product((-1, 1), repeat=3):
        corner = [rational(c) + s*rational(r) for c, s, r in zip(center, signs, radii)]
        for i in range(2):
            exact = sum((rational(weights[i, j])*corner[j] for j in range(3)), fmpq(0))
            assert actual[i].contains(exact)


def test_extreme_exponents_and_cancellation():
    huge, tiny = np.finfo(float).max, np.nextafter(0., 1.)
    weights = np.array([[huge, huge, tiny], [tiny, -tiny, huge]])
    values = [arb(1), arb(-1), linear.exact_dyadic(tiny)]
    answer = linear.affine(weights, values, chunk_cols=1)
    for i in range(2):
        exact = sum((rational(weights[i, j])*values[j].fmpq() for j in range(3)), fmpq(0))
        assert answer[i].contains(exact)
        assert answer[i].is_finite()


def test_primal_tangent_matrix_bias_only_primal_and_shared_weight_conversion():
    weights = np.array([[1., -2.], [.5, 3.]])
    x = arb_mat([[arb(3), arb(.125)], [arb(-1), arb(.25)]])
    bias = np.array([10., -20.])
    cache = linear.DyadicWeightCache(max_elements=4)
    answer = linear.affine(weights, x, bias, bias_columns=(0,), cache=cache)
    assert isinstance(answer, arb_mat)
    assert answer[0, 0] == 15 and answer[1, 0] == -21.5
    assert answer[0, 1] == -.375 and answer[1, 1] == .8125
    assert cache.misses == 1 and cache.elements == 4
    assert x[0, 0] == 3 and weights[0, 0] == 1


def test_bias_default_all_columns_and_explicit_none():
    weights = np.array([[2.]])
    x = arb_mat([[3, 4]])
    default = linear.affine(weights, x, np.array([5.]))
    absent = linear.affine(weights, x, np.array([5.]), bias_columns=())
    assert default[0, 0] == 11 and default[0, 1] == 13
    assert absent[0, 0] == 6 and absent[0, 1] == 8


def test_noncontiguous_transpose_float32_exactness():
    base = np.arange(24, dtype=np.float32).reshape(4, 6)/np.float32(10)
    weights = base[:, ::2].T
    x = [arb(i) for i in range(1, 5)]
    answer = linear.affine(weights, x, chunk_rows=2, chunk_cols=2)
    for i in range(3):
        expected = sum((rational(weights[i, j])*(j+1) for j in range(4)), fmpq(0))
        assert answer[i].contains(expected)


def test_cache_reuse_mutation_precision_and_size_bound():
    weights = np.array([[1.1, -2.5], [.5, 3.]])
    x = [arb(1), arb(2)]
    cache = linear.DyadicWeightCache(max_elements=4)
    old = linear.affine(weights, x, cache=cache)
    assert cache.misses == 1
    linear.affine(weights, x, cache=cache)
    assert cache.hits == 1
    weights[0, 0] = 8.
    changed = linear.affine(weights, x, cache=cache)
    assert not changed[0].overlaps(old[0])
    assert changed[0] == 3
    assert cache.misses == 2 and cache.elements == 4
    ctx.prec = 64
    linear.affine(weights, x, cache=cache)
    assert cache.misses == 3 and cache.peak_elements == 4
    cache.clear()
    assert cache.elements == 0


def test_cache_lru_never_retains_more_than_budget_or_oversize_block():
    weights = np.arange(48, dtype=float).reshape(6, 8)
    cache = linear.DyadicWeightCache(max_elements=9)
    linear.affine(weights, [arb(1)]*8, chunk_rows=2, chunk_cols=2, cache=cache)
    assert cache.elements <= 9 and cache.peak_elements <= 9
    assert cache.misses == 12
    small = linear.DyadicWeightCache(max_elements=3)
    linear.affine(weights, [arb(1)]*8, cache=small)
    assert small.elements == 0 and small.misses == 1
    disabled = linear.DyadicWeightCache(max_elements=0)
    linear.affine(weights, [arb(1)]*8, cache=disabled)
    assert disabled.elements == 0


def test_context_is_not_changed():
    ctx.prec = 73
    before = (ctx.prec, ctx.threads)
    linear.affine(np.array([[.1]]), [arb(1)])
    assert (ctx.prec, ctx.threads) == before


@pytest.mark.parametrize("kwargs", [dict(chunk_rows=0), dict(chunk_cols=-1),
    dict(chunk_rows=True), dict(bias_columns=(1,)), dict(bias_columns=(0, 0)),
    dict(cache={})])
def test_invalid_options_fail(kwargs):
    with pytest.raises((TypeError, ValueError)):
        linear.affine(np.ones((1, 1)), [arb(1)], **kwargs)


@pytest.mark.parametrize("weights,x,bias", [
    (np.ones((1, 2)), [arb(1)], None),
    (np.ones((1, 1), dtype=int), [arb(1)], None),
    (np.ones((0, 1)), [arb(1)], None),
    (np.array([[float('inf')]]), [arb(1)], None),
    (np.ones((1, 1)), [arb('nan')], None),
    (np.ones((1, 1)), [1.], None),
    (np.ones((1, 1)), [arb(1)], np.zeros(2)),
    (np.ones((1, 1)), [arb(1)], np.array([float('nan')])),
])
def test_invalid_data_fail_closed(weights, x, bias):
    with pytest.raises((TypeError, ValueError)):
        linear.affine(weights, x, bias)
