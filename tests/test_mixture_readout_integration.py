"""Independent synthetic checks of EXP020's scientific interfaces.

No historical neural arrays are read. The oscillator fixture supplies analytic
flow and tangent data through the real older packet APIs. Direct four-density
integration below does not use the producer's likelihood reduction or roots.
"""
from pathlib import Path
import copy
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import gaussian_mixture_readout as probability
import run_phase_memory_recovery as codec
from test_refined_two_write_integration import pipeline


def direct_density_accuracy(A, B, mean, first_sign):
    """Independent moderate-scale 2D oracle, B parallel to the first axis.

    Integrate the physical first coordinate. For each point, directly compare
    four Gaussian component densities and root the second-coordinate boundary.
    No logcosh, projected Gram determinant or producer split locations enter.
    """
    assert B[1] == 0 and A[1] > 0
    def log_ratio(t, z):
        y = np.array([t, z])
        plus = [-.5 * np.sum((y - A - s * B) ** 2) for s in (-1, 1)]
        minus = [-.5 * np.sum((y + A - s * B) ** 2) for s in (-1, 1)]
        return float(logsumexp(plus) - logsumexp(minus))
    def integrand(t):
        boundary = brentq(lambda z: log_ratio(t, z), -500., 500., xtol=1e-12)
        conditional = ndtr(first_sign * (mean[1] - boundary))
        return conditional * np.exp(-.5 * (t - mean[0]) ** 2) / np.sqrt(2 * np.pi)
    value, error = quad(integrand, mean[0] - 12., mean[0] + 12.,
                        epsabs=2e-10, epsrel=0., limit=300)
    assert error < 2e-9
    return value


@pytest.mark.parametrize('first,second', [(-1,-1),(-1,1),(1,-1),(1,1)])
def test_certificate_contains_direct_four_density_integral_under_mean_mismatch(first, second):
    A, B = np.array([1.3, .7]), np.array([2.1, 0.])
    mean = first * A + second * B + np.array([.17, -.23])
    oracle = direct_density_accuracy(A, B, mean, first)
    definition = probability.define_classifier(A, B, sigma=1.)
    result = probability.probability(definition, mean, first)
    assert result['qualified'], result
    lo, hi = result['interval']
    assert hi - lo <= 1e-4
    assert lo - 2e-9 <= oracle <= hi + 2e-9
    assert abs(result['nominal'] - oracle) < 2e-9


def test_identical_write_templates_have_unavoidable_reference_overlap():
    A = B = np.array([7., 0.])
    definition = probability.define_classifier(A, B, sigma=1.)
    results = [probability.probability(definition,
        [dict(coefficient=float(s1), vector=A), dict(coefficient=float(s2), vector=B)], s1)
        for s1 in (-1,1) for s2 in (-1,1)]
    assert all(row['qualified'] for row in results)
    # The mixed-sign classes have exactly the same observation distribution.
    # Perfect separation of the other two classes still leaves accuracy <=3/4.
    lower = sum(row['interval'][0] for row in results) / 4
    upper = sum(row['interval'][1] for row in results) / 4
    assert lower <= .75 <= upper
    assert upper <= .75 + 2e-15
    assert results[1]['nominal'] == results[2]['nominal'] == .5


def test_exact_physical_mean_translation_does_not_round_away_small_signal():
    definition = probability.define_classifier(np.array([1.]), np.array([0.]), sigma=1.)
    terms = [dict(coefficient=1., vector=np.array([1e16])),
             dict(coefficient=1., vector=np.array([1.])),
             dict(coefficient=-1., vector=np.array([1e16]))]
    result = probability.probability(definition, terms, 1)
    assert result['qualified']
    assert result['interval'][0] <= ndtr(1.) <= result['interval'][1]
    assert result['nominal'] > .84
    assert float(1e16 + 1. - 1e16) == 0.


def test_analytic_oscillator_saved_packet_to_real_mixture_evaluator(pipeline, tmp_path):
    """A complete real-code cell, including both references and three methods."""
    import mixture_readout_evaluation as evaluation
    p = pipeline
    cell = next(row for row in evaluation.cell_census(p['phases'])
                if row['context_id'] == p['context']['id'] and row['window'] == 'k2')
    outer = dict(context_id=cell['context_id'], status='complete', qualified=True,
                 acquisition=p['physical'], evaluation=p['seal'])
    sealed = evaluation.seal_cell(cell, outer, p['references']['rk45'])
    assert sealed['complete'], sealed
    assert all(row['available'] for row in sealed['definitions'].values())
    record = codec.save_packet(tmp_path, 'classifiers', sealed)
    sealed = codec.load_packet(record['path'])
    definition_digest = codec.packet_digest(sealed)
    scores = {method: [copy.deepcopy(row) for row in p['scores'].values()
        if row['task']['method'] == method and row['task']['arm'] == 'identity'
        and row['task']['first_sign'] != 0 and row['task']['second_sign'] != 0]
        for method in evaluation.methods_for_cell(cell)}
    assert all(len(rows) == 4 for rows in scores.values())
    result = evaluation.evaluate_cell(cell, sealed, scores)
    assert codec.packet_digest(sealed) == definition_digest
    assert result['complete'], result
    assert result['qualified'], result
    # Independent panel cells retain 40 declared probability slots.
    assert len(result['calculations']) == len(evaluation.calculation_census(cell)) == 40
    assert all(row['result']['qualified'] for row in result['calculations'])
    record = codec.save_packet(tmp_path, 'calculated-cell', result)
    assert codec.packet_digest(codec.load_packet(record['path'])) == codec.packet_digest(result)
