"""Synthetic CPU-only regressions for signed directional batch arithmetic."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from flint import arb, arb_mat, ctx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'codex/certify-llm'))
from ball_jet import BallQwenJet, Jet, linear_sequence, rmsnorm, softmax
from ball_linear import DyadicWeightCache
from ball_qwen import Dual, exact

spec = importlib.util.spec_from_file_location('tiny_jet_fixture', ROOT/'tests/test_codex_llm_ball_qwen.py')
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


@pytest.fixture(autouse=True)
def precision():
    old_threads = ctx.threads
    ctx.threads = 1
    with ctx.workprec(192):
        yield
    ctx.threads = old_threads


def models(count=8):
    scalar, reference = fixture_module.fixture()
    jet = BallQwenJet(scalar.config, scalar.weights, scalar.prefix_ids, scalar.vector,
        steering_layer=scalar.steering_layer, alpha=.071674, log10_temperature=-.115,
        direction_count=count)
    return jet, scalar, reference


def test_signed_arithmetic_matches_each_single_direction():
    x = Jet.make(2., [1., -2., .5, 0.])
    y = Jet.make(.75, [-.25, 1., 2., 0.])
    operations = [lambda a,b:a+b, lambda a,b:a-b, lambda a,b:a*b,
        lambda a,b:a/b, lambda a,b:(a+b).exp(), lambda a,b:(a+b).rsqrt(),
        lambda a,b:2/a - b + 1]
    for operation in operations:
        actual = operation(x,y)
        for k in range(4):
            # Dual has no reverse division; express the same last operation.
            a,b = Dual(x.value,x.tangent[k]), Dual(y.value,y.tangent[k])
            if operation is operations[-1]:
                expected = a.reciprocal()*2-b+1
            else:
                expected = operation(a,b)
            assert actual.value.overlaps(expected.value)
            assert actual.tangent[k].overlaps(expected.tangent)


def test_affine_shares_weights_and_bias_only_primal_columns():
    weights = np.array([[1., -2.], [.5, 3.]])
    a = [Jet.make(3., [.125, -.5]), Jet.make(-1., [.25, .75])]
    b = [Jet.make(-2., [0., .5]), Jet.make(.5, [1., -1.])]
    cache = DyadicWeightCache(max_elements=4)
    result = linear_sequence(weights, [a,b], np.array([10., -20.]), cache=cache)
    assert cache.misses == 1 and cache.elements == 4
    for token, actual in zip([a,b], result):
        for i in range(2):
            primal = exact([10., -20.][i]) + sum((exact(weights[i,j])*token[j].value for j in range(2)),arb(0))
            assert actual[i].value == primal
            for k in range(2):
                tangent = sum((exact(weights[i,j])*token[j].tangent[k] for j in range(2)),arb(0))
                assert actual[i].tangent[k] == tangent


def test_signed_cancellation_is_not_replaced_by_independent_interval_cube():
    values = [Jet.make(1., [1., -1.]), Jet.make(2., [-1., 1.])]
    result = linear_sequence(np.array([[1., 1.]]), [values])[0][0]
    assert result.value == 3
    assert all(v.is_exact() and v == 0 for v in result.tangent)


def test_softmax_each_direction_sums_to_zero_and_common_shift_cancels():
    q = [Jet.make(10000., [1., 2., -1.]), Jet.make(9999., [-1., 2., .5])]
    p = softmax(q)
    shifted = softmax([v-10000 for v in q])
    assert sum((v.value for v in p),arb(0)).contains(1)
    for k in range(3):
        assert sum((v.tangent[k] for v in p),arb(0)).contains(0)
        assert all(a.tangent[k].overlaps(b.tangent[k]) for a,b in zip(p,shifted))
    assert all(v.tangent[1].contains(0) for v in p)


def test_full_tiny_map_eight_signed_directions_match_scalar_map():
    jet, scalar, _ = models()
    x = np.array([.08,-.11,.16,.03])
    directions = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
        [.1,.2,-.15,.07],[-.1,-.2,.15,-.07],[0,0,0,0],[1,-1,1,-1]],dtype=float)
    value, derivative = jet.forward(x,directions)
    assert (derivative.nrows(),derivative.ncols()) == (4,8)
    for k,v in enumerate(directions):
        expected, expected_j = scalar.forward(x,v)
        for i in range(4):
            assert value[i].overlaps(expected[i])
            assert derivative[i,k].overlaps(expected_j[i])
    assert all(derivative[i,6].is_exact() and derivative[i,6] == 0 for i in range(4))
    assert all(derivative[i,4].overlaps(-derivative[i,5]) for i in range(4))
    assert max(float(v.rad()) for v in value) < 1e-45
    assert max(float(derivative[i,k].rad()) for i in range(4) for k in range(8)) < 1e-45


def test_matrix_direction_orientation_and_default_zero():
    jet, _, _ = models(2)
    x = [.08,-.11,.16,.03]
    directions = np.array([[.1,.2,-.15,.07],[-.1,.4,.03,-.05]])
    f,a = jet.forward(x,directions)
    matrix = arb_mat([[exact(directions[k,i]) for k in range(2)] for i in range(4)])
    g,b = jet.forward(x,matrix)
    assert all(u.overlaps(v) for u,v in zip(f,g))
    assert all(a[i,k].overlaps(b[i,k]) for i in range(4) for k in range(2))
    _,zero = jet.forward(x)
    assert all(zero[i,k] == 0 for i in range(4) for k in range(2))


def test_interval_state_and_interval_direction_columns_enclose_interior_jvps():
    jet, _, _ = models(2)
    x = np.array([.08,-.11,.16,.03])
    directions = np.array([[.1,.2,-.15,.07],[-.1,.4,.03,-.05]])
    state_box = [arb(exact(v),(1,-28)) for v in x]
    direction_box = arb_mat([[arb(exact(directions[k,i]),(1,-29)) for k in range(2)] for i in range(4)])
    outer_f,outer_j = jet.forward(state_box,direction_box)
    rng = np.random.default_rng(1013)
    for _ in range(3):
        sample_x = x + rng.uniform(-1,1,4)*2**-28
        sample_d = directions + rng.uniform(-1,1,(2,4))*2**-29
        inner_f,inner_j = jet.forward(sample_x,sample_d)
        assert all(a.contains(b) for a,b in zip(outer_f,inner_f))
        assert all(outer_j[i,k].contains(inner_j[i,k]) for i in range(4) for k in range(2))


def test_inherited_cache_precision_and_rebinding_guards():
    jet, _, _ = models(2)
    jet.prefix_cache()
    with ctx.workprec(256):
        with pytest.raises(ValueError,match='precision'):
            jet.forward([.1]*4)
    jet.alpha = exact(.07)
    with pytest.raises(ValueError,match='dependencies'):
        jet.forward([.1]*4)


@pytest.mark.parametrize('directions', [np.ones((4,2)),np.ones((2,4),dtype=int),
    np.ones((2,4),dtype=np.longdouble),np.full((2,4),np.nan),arb_mat(2,4),[[1.]*4]*2])
def test_bad_direction_admission(directions):
    jet, _, _ = models(2)
    with pytest.raises((TypeError,ValueError)):
        jet.forward([.1]*4,directions)


def test_domain_and_mixed_dimension_failures():
    with pytest.raises(ValueError,match='zero'):
        Jet.make(arb(0,1),[1.]).reciprocal()
    with pytest.raises(ValueError,match='positive'):
        Jet.make(-1.,[1.]).rsqrt()
    with pytest.raises(ValueError,match='counts'):
        Jet.make(1.,[1.])+Jet.make(2.,[1.,2.])
    with pytest.raises(ValueError,match='dimension'):
        rmsnorm([Jet.make(1.,[1.])],[],1e-6)
    with pytest.raises(ValueError,match='nonfinite'):
        softmax([Jet(arb(1),(arb('nan'),))])
    jet, _, _ = models(2)
    with pytest.raises(ValueError,match='explicit'):
        jet.scalar(1.,1.)


@pytest.mark.parametrize('count',[0,-1,True,1.5])
def test_invalid_direction_count(count):
    with pytest.raises(ValueError,match='direction_count'):
        models(count)
