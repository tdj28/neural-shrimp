"""Synthetic-only checks: no Hopfield imports, ODE calls, or saved outcomes."""
from fractions import Fraction
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from flint import arb, ctx
from scipy.integrate import quad
from scipy.special import ndtr, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import gaussian_mixture_readout as core


def vec(a):
    return np.array(a, dtype=np.float64)


def prob(A, B, mu, sign=1, sigma=1.):
    return core.probability(core.define_classifier(vec(A), vec(B), sigma), vec(mu), sign)


def contains(result, expected, tol=2e-15):
    assert result['complete'] and result['qualified'], result
    lo, hi = result['interval']
    assert lo-tol <= expected <= hi+tol
    assert hi-lo <= 1e-4
    assert abs(result['nominal']-expected) <= max(1e-8, hi-lo)


@pytest.mark.parametrize('A,B,mu,expected', [
    ([0, 0], [1, 0], [500, -300], .5),
    ([1, 0], [0, 0], [2, 500], ndtr(2)),
    ([1, 0], [0, 3], [-1, 90], ndtr(-1)),
    ([1, 0], [1, 0], [0, 7], .5),
    ([100, 0], [100, 0], [-1, 0], ndtr(-1)),
    ([.1, 0], [.2, 0], [.3, 0], ndtr(.3)),
])
def test_closed_branches(A, B, mu, expected):
    contains(prob(A, B, mu), expected)


def test_equal_overlap_reference_ba_is_three_quarters():
    A, B = vec([20, 0]), vec([20, 0])
    d = core.define_classifier(A, B, 1.)
    values = []
    for first in (-1, 1):
        for second in (-1, 1):
            r = core.probability(d, [{'coefficient': first, 'vector': A},
                                     {'coefficient': second, 'vector': B}], first)
            assert r['qualified']
            values.append(r['nominal'])
    assert np.mean(values) == .75


def test_scalar_disjoint_intervals_not_single_threshold():
    result = prob([1, 0], [3, 0], [-2, 0])
    assert result['qualified']
    root = result['diagnostics']['positive_decision_root']
    assert 0 < root[0] <= root[1] < 3
    assert result['nominal'] > .8
    assert ndtr(-2) < .03


@pytest.mark.parametrize('a,b,c,mt,mz', [
    (1., 2., .5, 3., .5), (.1, 1., .3, .9, .7),
    (.7, .4, .8, -.5, .2), (2., 1., 1.5, -1., -.3),
])
def test_rank_two_against_independent_density_boundary(a, b, c, mt, mz):
    # Direct log-sum of the four isotropic Gaussian densities. L is affine
    # in z, so two evaluations give the conditional boundary independently.
    def ll(t, z, first):
        means = [(first*a+s*b, first*c) for s in (-1, 1)]
        return logsumexp([-.5*((t-x)**2+(z-y)**2) for x, y in means])
    def integrand(u):
        t = u+mt
        l0 = ll(t, 0., 1)-ll(t, 0., -1)
        l1 = ll(t, 1., 1)-ll(t, 1., -1)
        threshold = -l0/(l1-l0)
        return ndtr(mz-threshold)*np.exp(-u*u/2)/np.sqrt(2*np.pi)
    expected, error = quad(integrand, -10, 10, epsabs=2e-12, epsrel=0)
    assert error < 1e-9
    contains(prob([a, c], [b, 0], [mt, mz]), expected)


def test_sign_and_nuisance_orientation_symmetries():
    A, B, mu = vec([1., .5]), vec([2., 0.]), vec([.3, -.2])
    results = [core.probability(core.define_classifier(A, b, 1.), m, s)
               for b, m, s in [(B, mu, 1), (-B, mu, 1), (B, -mu, -1)]]
    assert all(x['qualified'] for x in results)
    assert max(x['nominal'] for x in results)-min(x['nominal'] for x in results) < 1e-12


def test_exact_rank_never_snaps_tiny_perpendicular_component():
    d = core.define_classifier(vec([1., 1e-200]), vec([2., 0.]), 1.)
    assert d['branch'] == 'rank-two'
    assert core._decode(d['exact']['D']) > 0
    result = core.probability(d, vec([3., 1e-200]), 1)
    assert (result['diagnostics'].get('exact_rank') == 2 or
            result['diagnostics'].get('branch') == 'rank-two')
    # Numerical unavailability is allowed; changing the model's rank is not.


@pytest.mark.parametrize('c', [1e-4, 1e-10, 1e-30])
def test_near_collinear_rank_two_retains_certificate(c):
    r = prob([1., c], [2., 0.], [3., c])
    scalar = prob([1., 0.], [2., 0.], [3., 0.])
    assert r['qualified'], r
    assert r['diagnostics']['exact_rank'] == 2
    assert abs(r['nominal']-scalar['nominal']) < 1e-7


def test_high_separation_underflow_does_not_create_ties():
    r = prob([100, 0], [100, 0], [3, 0])
    contains(r, ndtr(3))
    assert r['nominal'] > .99


def test_mathematical_one_mode_is_exact_not_rounded_product():
    B, rho = vec([.1, .7, 1.3]), .3
    d = core.define_classifier(rho*B, B, 1., one_mode_rho=rho)
    assert d['branch'] == 'collinear'
    assert core._decode(d['exact']['D']) == 0
    assert core._decode(d['exact']['AB']) == Fraction.from_float(rho)*core._dot(B, B)
    assert core.probability(d, [{'coefficient': rho, 'vector': B}], 1)['qualified']


@pytest.mark.parametrize('B,rho', [([20., 1.], .3), ([100., 1.], .1), ([1000., 1.], .7)])
def test_scalar_irrational_boundary_high_separation(B, rho):
    B = vec(B)
    d = core.define_classifier(rho*B, B, 1., one_mode_rho=rho)
    r = core.probability(d, [{'coefficient': 1+rho, 'vector': B}], 1)
    assert r['qualified'], r
    assert r['diagnostics']['positive_decision_root'][0] <= np.linalg.norm(B)
    assert r['nominal'] > .999


def test_exact_mean_sum_retains_small_terms_after_large_cancellation():
    terms = [{'coefficient': 1., 'vector': vec([1e20, 0])},
             {'coefficient': 1., 'vector': vec([1, 0])},
             {'coefficient': -1., 'vector': vec([1e20, 0])}]
    d = core.define_classifier(vec([1, 0]), vec([0, 0]), 1.)
    contains(core.probability(d, terms, 1), ndtr(1))
    contains(core.linear_probability(vec([1, 0]), terms, 1, 1.), ndtr(1))


def test_baseline_translation_terms_invariant():
    A, B, mu, baseline, offset = map(vec, ([1, .5], [2, 0], [.3, .4], [1000, -500], [.125, -.25]))
    d = core.define_classifier(A, B, 1.)
    first = core.probability(d, [{'coefficient': 1., 'vector': mu}], 1)
    translated = core.probability(d, [
        {'coefficient': 1., 'vector': mu},
        {'coefficient': 1., 'vector': baseline},
        {'coefficient': 1., 'vector': offset},
        {'coefficient': -1., 'vector': baseline},
        {'coefficient': -1., 'vector': offset}], 1)
    assert first == translated


def test_outward_serialization_encloses_exact_rational_endpoints():
    with ctx.workprec(128):
        for q in [Fraction(1, 3), Fraction(-1, 7), Fraction(1, 10**400), Fraction(-1, 10**400)]:
            ball = core._ball(q)
            lo, hi = core._bounds(ball)
            assert Fraction.from_float(lo) <= q <= Fraction.from_float(hi)
            assert core._ball(lo) <= ball.lower()
            assert core._ball(hi) >= ball.upper()


def test_definition_and_mean_codec_roundtrip(tmp_path):
    d = core.define_classifier(vec([1., .5]), vec([2., 0.]), 1.)
    # Real JSON and NPZ round trip, not merely deepcopy.
    meta = {k: v for k, v in d.items() if k not in ('A', 'B')}
    (tmp_path/'definition.json').write_text(json.dumps(meta, allow_nan=False))
    np.savez_compressed(tmp_path/'arrays.npz', A=d['A'], B=d['B'], mean=vec([3., .5]))
    loaded = json.loads((tmp_path/'definition.json').read_text())
    with np.load(tmp_path/'arrays.npz', allow_pickle=False) as arrays:
        loaded.update(A=arrays['A'], B=arrays['B'])
        result = core.probability(loaded, arrays['mean'], 1)
    assert result == core.probability(d, vec([3., .5]), 1)


@pytest.mark.parametrize('bad', [True, 0, 2, 1., '1'])
def test_invalid_sign_retained(bad):
    r = prob([1, 0], [2, 0], [3, 0], bad)
    assert not r['complete'] and not r['qualified'] and r['error']


@pytest.mark.parametrize('mutation', ['float32', 'nan', 'shape', 'sigma', 'binding'])
def test_invalid_inputs_retained(mutation):
    A, B = vec([1, 0]), vec([2, 0])
    sigma = 1.
    if mutation == 'float32': A = A.astype('float32')
    if mutation == 'nan': A[0] = np.nan
    if mutation == 'shape': B = B[:, None]
    if mutation == 'sigma': sigma = 0.
    d = core.define_classifier(A, B, sigma)
    if mutation == 'binding': d['exact']['D'] = ['1', '1']
    r = core.probability(d, vec([3, 0]), 1)
    assert not r['complete'] and not r['qualified']


def test_subdivision_failure_retains_partial_certificate(monkeypatch):
    monkeypatch.setattr(core, 'MAX_LEAVES', 1)
    r = prob([1, .5], [2, 0], [3, .5])
    assert not r['qualified'] and not r['complete']
    assert r['interval'] is not None
    assert len(r['diagnostics']['attempts']) == 3
    assert all('subdivision budget' in x['error'] for x in r['diagnostics']['attempts'])


def test_root_budget_failure_retained(monkeypatch):
    monkeypatch.setattr(core, 'MAX_ROOT_ITERATIONS', 0)
    r = prob([1, 0], [2, 0], [3, 0])
    assert not r['complete']
    assert all('root iteration budget' in x['error'] for x in r['diagnostics']['attempts'])


def test_nominal_failure_retains_certificate(monkeypatch):
    def fail(*args):
        raise core.NumericalUnavailable('synthetic nominal failure')
    monkeypatch.setattr(core, '_nominal', fail)
    r = prob([1, .5], [2, 0], [3, .5])
    assert not r['complete'] and r['interval'] is not None
    assert len(r['diagnostics']['attempts']) == 3


def test_nominal_outside_certificate_is_gate_failure(monkeypatch):
    monkeypatch.setattr(core, '_nominal', lambda *args: (.1, [{'estimated_error': 0.}]))
    r = prob([1, .5], [2, 0], [3, .5])
    assert r['complete'] and not r['qualified']
    assert r['interval'] is not None and r['nominal'] == .1


def test_tiny_nominal_probability_range_roundoff_is_retained():
    r = prob([20., 1e-10], [40., 0.], [60., 0.])
    assert r['qualified'] and 0 <= r['nominal'] <= 1
    assert 'raw_nominal' in r['diagnostics']
    assert abs(r['diagnostics']['nominal_range_correction']) <= 1e-14


def test_derivative_extrema_and_stable_formula_match_direct_density():
    with ctx.workprec(128):
        a, b = arb(1), arb(3)
        for t in [-8., -3., -1., -.2, 0., .2, 1., 3., 8.]:
            stable = core._g(core._ball(t), a, b)
            direct = 2*a*core._ball(t)+(b*(core._ball(t)-a)).cosh().log()-(b*(core._ball(t)+a)).cosh().log()
            assert stable.overlaps(direct)
        values = [float(core._gp(arb(x), a, b)) for x in [0, .2, 1, 2, 3]]
        assert values == sorted(values)


def test_precision_context_restored():
    initial = ctx.prec
    prob([1, 0], [2, 0], [3, 0])
    assert ctx.prec == initial


def test_no_io_or_neural_imports_in_core():
    import ast
    tree = ast.parse(Path(core.__file__).read_text())
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any('hopfield' in (m or '') or 'two_write' in (m or '') for m in modules)
    calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert 'open' not in calls
