"""Pure synthetic/algebraic tests; no Hopfield reference, pulse, fit or timing."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

# Works both in this exclusive external draft folder and its eventual tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import phase_memory as pm


def circle_samples(count=128, period=2*np.pi):
    t = np.arange(count, dtype=np.float64)*period/count
    w = 2*np.pi/period
    values = np.column_stack((np.cos(w*t), np.sin(w*t), .3*np.sin(2*w*t)))
    derivative = np.column_stack((-w*np.sin(w*t), w*np.cos(w*t), .6*w*np.cos(2*w*t)))
    return t, values, derivative


def circle(count=128, period=2*np.pi):
    t, values, derivative = circle_samples(count, period)
    return pm.PeriodicHermiteTemplate(t, values, derivative, period, raw_endpoint=values[0].copy())


def test_periodic_hermite_preserves_knots_derivatives_and_physical_time_units():
    template = circle(period=5.)
    np.testing.assert_array_equal(template.evaluate(template.times), template.values)
    np.testing.assert_array_equal(template.evaluate(template.times, derivative=True), template.derivatives)
    query = np.array([-.7, 0., .15, 1.2, 5., 11.1])
    w = 2*np.pi/5.
    expected = np.column_stack((np.cos(w*query), np.sin(w*query), .3*np.sin(2*w*query)))
    expected_d = np.column_stack((-w*np.sin(w*query), w*np.cos(w*query), .6*w*np.cos(2*w*query)))
    np.testing.assert_allclose(template.evaluate(query), expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(template.evaluate(query, derivative=True), expected_d, rtol=0, atol=5e-6)
    np.testing.assert_allclose(template.evaluate(query+5.), template.evaluate(query), rtol=0, atol=3e-15)


def test_nonperiodic_cubic_hermite_is_exact_on_irregular_knots():
    t = np.array([0., .3, 1.1, 2.])
    f = lambda x: np.column_stack((x**3-2*x+1, 2*x*x+x, 3*x+2))
    d = lambda x: np.column_stack((3*x*x-2, 4*x+1, np.full(len(x), 3.)))
    segment = pm.HermiteSegment(t, f(t), d(t))
    q = np.array([0., .07, .3, .85, 1.6, 2.])
    np.testing.assert_allclose(segment.evaluate(q), f(q), rtol=0, atol=2e-15)
    np.testing.assert_allclose(segment.evaluate(q, derivative=True), d(q), rtol=0, atol=3e-15)
    np.testing.assert_allclose(segment.evaluate(np.array(2.)), f(np.array([2.]))[0], rtol=0, atol=0)
    with pytest.raises(ValueError, match='inside'):
        segment.evaluate(np.array([2.01]))


def test_raw_endpoint_mismatch_retained_separately_and_input_arrays_owned():
    t, x, d = circle_samples()
    endpoint = x[0]+np.array([.1, .2, -.3])
    template = pm.PeriodicHermiteTemplate(t, x, d, 2*np.pi, raw_endpoint=endpoint)
    original = x[0].copy()
    x[:] = 999.; d[:] = 99.; endpoint[:] = 888.
    np.testing.assert_array_equal(template.evaluate(np.array(2*np.pi)), original)
    np.testing.assert_allclose(template.closure()['endpoint_difference'], [.1, .2, -.3], atol=1e-15)
    assert template.closure()['closure_qualification_owned_by_caller']
    assert template.closure()['maximum_absolute_difference'] == .3
    assert not template.values.flags.writeable and not template.raw_endpoint.flags.writeable


@pytest.mark.parametrize('mutation', ['duplicate-time', 'endpoint-included', 'nan', 'infinite-derivative', 'float32', 'wrong-shape', 'zero-period'])
def test_template_rejects_incompatible_inputs(mutation):
    t, x, d = circle_samples(); period = 2*np.pi
    if mutation == 'duplicate-time': t[2] = t[1]
    elif mutation == 'endpoint-included': t[-1] = period
    elif mutation == 'nan': x[2, 1] = np.nan
    elif mutation == 'infinite-derivative': d[4, 0] = np.inf
    elif mutation == 'float32': x = x.astype(np.float32)
    elif mutation == 'wrong-shape': d = d[:-1]
    else: period = 0.
    with pytest.raises(ValueError):
        pm.PeriodicHermiteTemplate(t, x, d, period, raw_endpoint=np.array([1., 0., 0.]))


def test_neutral_left_svd_on_nonnormal_monodromy_and_time_normalization():
    basis = np.array([[1., 2., .3], [.2, 1., .4], [.1, .2, 1.]])
    inverse = np.linalg.inv(basis)  # Synthetic ground truth only; production helper never inverts.
    matrix = basis @ np.diag([1., .25, 1e-20]) @ inverse
    velocity = 3.*basis[:, 0]
    result = pm.neutral_left_phase(matrix, velocity)
    expected = inverse[0]/3.
    np.testing.assert_allclose(result['phase_gradient'], expected, rtol=1e-13, atol=1e-14)
    assert result['passed'] and result['unique_singular_direction']
    assert abs(result['phase_gradient']@velocity-1) <= 1e-15
    assert result['neutral_left_relative_residual'] < 1e-13


@pytest.mark.parametrize('matrix,flow', [
    (np.eye(3), np.array([1., 0., 0.])),
    (np.diag([1., 1., .2]), np.array([1., 0., 0.])),
    (np.diag([1.01, .3, .2]), np.array([1., 0., 0.])),
    (np.diag([1., .3, .2]), np.array([0., 1., 0.]))])
def test_neutral_ambiguity_or_non_neutral_flow_retains_failure_diagnostics(matrix, flow):
    with pytest.raises(pm.PhaseQualificationError) as caught:
        pm.neutral_left_phase(matrix, flow)
    assert 'singular_values' in caught.value.diagnostics
    assert 'neutral_flow_relative_residual' in caught.value.diagnostics


def test_phase_null_plane_is_not_euclidean_flow_normal_and_ties_are_fixed():
    z, flow = np.array([1., 2., 3.]), np.array([1., 0., 0.])
    result = pm.pulse_directions(z, flow)
    null = result['directions']['phase_null']
    assert abs(z@null) < 1e-14
    assert abs(flow@null) > .9
    assert result['null_coordinate_index'] == 0
    for direction in result['directions'].values():
        assert abs(np.linalg.norm(direction)-1.) < 1e-15
    tied = pm.pulse_directions(np.array([1., 0., 0.]), flow)
    assert tied['null_coordinate_index'] == 1
    np.testing.assert_array_equal(tied['directions']['phase_null'], [0., 1., 0.])
    assert tied['classifier_orientations']['phase_null'] == 1
    reverse = pm.pulse_directions(np.array([-1., 0., 0.]), -flow)
    assert reverse['classifier_orientations']['neuron_one'] == -1


def test_pulse_does_not_hide_lost_binary64_displacement_or_rescale():
    initial, direction = np.array([1e20, 0., 0.]), np.array([1., 0., 0.])
    pulse = pm.realize_pulse(initial, direction, 1e-4, 1)
    assert pulse['realized_norm'] == 0.
    np.testing.assert_array_equal(pulse['requested_delta'], [1e-4, 0., 0.])
    np.testing.assert_array_equal(pulse['addition_rounding_residual'], [-1e-4, 0., 0.])
    np.testing.assert_array_equal(initial, [1e20, 0., 0.])


def test_arithmetic_overflow_is_not_an_implicit_normalization_pass():
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(ValueError, match='normalization'):
            pm.pulse_directions(np.array([1e150, 1e150, 0.]), np.array([1e200, -1e200, 0.]))
        with pytest.raises(ValueError, match='overflowed'):
            pm.realize_pulse(np.zeros(3), np.array([1., 0., 0.]), 1e308, 1)


def test_clock_grid_and_tanh_chain_rule_from_unperturbed_template():
    template = circle(period=7.)
    readout = pm.clock_readout(template, .8, 4)
    expected_times = 7.*(3+(np.arange(32)+.5)/32)
    np.testing.assert_array_equal(readout.times, expected_times)
    assert readout.times[0] > 3*7. and readout.times[-1] < 4*7.
    potential = template.evaluate(readout.times+.8)
    activation = np.tanh(potential)
    derivative = (1.-activation**2)*template.evaluate(readout.times+.8, derivative=True)
    np.testing.assert_array_equal(readout.reference_activations, activation)
    np.testing.assert_allclose(readout.weights, derivative/np.sum(derivative**2), rtol=0, atol=0)
    assert readout.score(activation) == 0.
    assert not readout.weights.flags.writeable


def test_small_true_time_shift_is_read_without_phase_fitting():
    template = circle(512, period=5.)
    readout = pm.clock_readout(template, .3, 4)
    shift = 1e-5
    shifted = np.tanh(template.evaluate(readout.times+.3+shift))
    assert abs(readout.score(shifted)/shift-1.) < 2e-6
    # A phase-aligned RMS would erase this information. This score retains it.
    assert readout.score(shifted) > 0.


def test_balanced_accuracy_gaussian_formula_zero_null_and_below_chance_retained():
    readout = pm.clock_readout(circle(), 0., 4)
    derivative = readout.weights*readout.derivative_energy
    plus = readout.reference_activations+1e-5*derivative
    minus = readout.reference_activations-1e-5*derivative
    result = pm.balanced_sign_readout(readout, plus, minus, sigma=1e-4)
    expected_sd = 1e-4*np.sqrt(np.sum(readout.weights**2))
    expected = .5*(1.+math.erf((1e-5/expected_sd)/np.sqrt(2.)))
    assert result['score_noise_sd'] == expected_sd
    assert abs(result['balanced_accuracy']-expected) < 1e-12
    null = pm.balanced_sign_readout(readout, readout.reference_activations, readout.reference_activations)
    assert null['balanced_accuracy'] == .5
    reverse = pm.balanced_sign_readout(readout, minus, plus, sigma=1e-4)
    assert reverse['balanced_accuracy'] < .5
    assert abs(reverse['balanced_accuracy']+result['balanced_accuracy']-1.) < 1e-15
    oriented = pm.balanced_sign_readout(readout, minus, plus, sigma=1e-4, orientation=-1)
    assert oriented['balanced_accuracy'] == result['balanced_accuracy']


def test_constant_or_fully_saturated_template_cannot_supply_clock_readout():
    t = np.array([0., .5])
    for states in (np.zeros((2, 3)), np.full((2, 3), 100.)):
        template = pm.PeriodicHermiteTemplate(t, states, np.zeros((2, 3)), 1., raw_endpoint=states[0])
        with pytest.raises(pm.PhaseQualificationError, match='Unobservable'):
            pm.clock_readout(template, 0., 1)


@pytest.mark.parametrize('sigma', [0., -1., np.nan, np.inf, True, '0.1'])
def test_noise_specification_is_not_silently_coerced(sigma):
    readout = pm.clock_readout(circle(), 0., 1)
    with pytest.raises(ValueError):
        pm.balanced_sign_readout(readout, readout.reference_activations, readout.reference_activations, sigma=sigma)


@pytest.mark.parametrize('delay,epoch', [(True, 0.), (3, 0.), (4, -1.), (4, 2*np.pi)])
def test_readout_requires_declared_delay_and_known_phase(delay, epoch):
    with pytest.raises(ValueError):
        pm.clock_readout(circle(), epoch, delay)


def test_phase_only_emulator_predicts_signed_displacement_and_zero_response():
    template = circle()
    query = np.array([0., .2, 7., 13.])
    z, delta = np.array([2., 0., 0.]), np.array([.01, -.1, .3])
    result = pm.phase_only_prediction(template, delta, z, query, phase_epoch=.3)
    assert result['time_displacement'] == .02
    np.testing.assert_array_equal(result['states'], template.evaluate(query+.3+.02))
    np.testing.assert_array_equal(result['activations'], np.tanh(result['states']))
    null = pm.phase_only_prediction(template, np.zeros(3), z, query, phase_epoch=.3)
    np.testing.assert_array_equal(null['states'], template.evaluate(query+.3))


def noncommuting_segment():
    a = np.array([[0., .4, 0.], [0., 0., 0.], [0., 0., 0.]])
    b = np.array([[0., 0., 0.], [.2, 0., 0.], [0., 0., -.1]])
    t = np.array([0., .4, 1.])
    values = np.asarray([np.eye(3)+s*a+s*s*b for s in t])
    derivatives = np.asarray([a+2*s*b for s in t])
    return pm.HermiteSegment(t, values, derivatives), values[-1], a, b


def test_full_linear_response_uses_nonperiodic_propagator_and_correct_product_order():
    segment, matrix, a, b = noncommuting_segment()
    delta = np.array([.2, -.3, .1]); times = np.array([0., .5, 1., 1.5, 2., 8.])
    result = pm.full_linear_response(segment, matrix, delta, times)
    expected = np.asarray([(np.eye(3)+(t%1)*a+(t%1)**2*b)@np.linalg.matrix_power(matrix, int(t))@delta for t in times])
    np.testing.assert_allclose(result['state_delta'], expected, rtol=0, atol=3e-16)
    wrong = matrix@(np.eye(3)+.5*a+.25*b)@delta
    assert not np.allclose(wrong, result['state_delta'][3])
    np.testing.assert_array_equal(result['state_delta'][0], delta)
    np.testing.assert_allclose(result['state_delta'][4], matrix@matrix@delta, rtol=0, atol=1e-16)
    scalar = pm.full_linear_response(segment, matrix, delta, np.array(1.5))
    assert scalar['state_delta'].shape == (3,)


def test_full_linear_response_preserves_synthetic_contraction_and_neutral_phase():
    t = np.array([0., 1.])
    rates = np.array([0., -2., -4.])
    values = np.asarray([np.diag(np.exp(s*rates)) for s in t])
    derivative = np.asarray([np.diag(rates*np.exp(s*rates)) for s in t])
    segment = pm.HermiteSegment(t, values, derivative)
    result = pm.full_linear_response(segment, values[-1], np.ones(3), np.array([0., 1., 4., 8.]))
    expected = np.exp(np.outer([0., 1., 4., 8.], rates))
    np.testing.assert_allclose(result['state_delta'], expected, rtol=2e-15, atol=1e-20)


@pytest.mark.parametrize('mutation', ['wrong-monodromy', 'nonidentity-start', 'too-far', 'negative-time', 'periodic-template', 'boolean-cap'])
def test_full_linear_response_rejects_wrong_binding_or_scope(mutation):
    segment, matrix, _, _ = noncommuting_segment()
    query, bound = np.array([1.5]), 8
    if mutation == 'wrong-monodromy': matrix = matrix+np.eye(3)*.01
    elif mutation == 'nonidentity-start':
        values = segment.values.copy(); values[0] *= 2
        segment = pm.HermiteSegment(segment.times, values, segment.derivatives)
    elif mutation == 'too-far': query = np.array([9.])
    elif mutation == 'negative-time': query = np.array([-1.])
    elif mutation == 'periodic-template': segment = circle()
    else: bound = True
    with pytest.raises(ValueError):
        pm.full_linear_response(segment, matrix, np.ones(3), query, max_cycles=bound)


def test_tanh_linearization_matches_symmetric_finite_difference_and_exposes_saturation():
    x, direction = np.array([[0., .5, 10.]]), np.array([[.2, -.1, 2.]])
    predicted = pm.linear_activation_delta(x, direction)
    step = 1e-6
    numerical = (np.tanh(x+step*direction)-np.tanh(x-step*direction))/(2*step)
    np.testing.assert_allclose(predicted, numerical, rtol=0, atol=1e-10)
    assert abs(predicted[0, 2]) < 2e-8
    assert direction[0, 2] == 2.


def test_helpers_need_no_file_or_integration_capability(monkeypatch):
    import builtins
    import scipy.integrate
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure mathematical helper attempted file access or integration')
    monkeypatch.setattr(builtins, 'open', forbidden)
    monkeypatch.setattr(scipy.integrate, 'solve_ivp', forbidden)
    template = circle()
    readout = pm.clock_readout(template, 0., 4)
    packet = pm.pulse_directions(np.array([1., 0., 0.]), np.array([1., 0., 0.]))
    delta = pm.realize_pulse(np.zeros(3), packet['directions']['phase_write'], 1e-4, 1)['realized_delta']
    plus = pm.phase_only_prediction(template, delta, np.array([1., 0., 0.]), readout.times)['activations']
    minus = pm.phase_only_prediction(template, -delta, np.array([1., 0., 0.]), readout.times)['activations']
    assert 0 <= pm.balanced_sign_readout(readout, plus, minus)['balanced_accuracy'] <= 1
