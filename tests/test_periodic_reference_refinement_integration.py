"""Nine-maximum synthetic oscillator; never loads historical neural outcomes.

The analytic flow is a stable radial oscillator with a third coordinate
tracking Re((x+i*y)**9). Its primitive orbit has nine genuine x3 maxima.
This supplies an independent exact-flow fixture for the unchanged peak gate.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import periodic_reference_refinement as acq
import periodic_reference_refinement_evaluation as evaluate
import run_phase_memory_recovery as codec

K = .2
LAMBDA = .3
ROTATION = np.array([[0., -1.], [1., 0.]])
PERIOD = 2*np.pi
SEED = np.array([1., 0., 1.])


def wave(x):
    u = complex(*x)
    return float((u**9).real)


def wave_gradient(x):
    value = 9*complex(*x)**8
    return np.array([value.real, -value.imag])


def wave_hessian(x):
    value = 72*complex(*x)**7
    return np.array([[value.real, -value.imag], [-value.imag, -value.real]])


def rhs(t, state, alpha, gamma):
    del t, alpha, gamma
    xy = state[:2]
    flow = ROTATION@xy + K*(1.-xy@xy)*xy
    return np.r_[flow, wave_gradient(xy)@flow-LAMBDA*(state[2]-wave(xy))]


def jacobian(t, state, alpha, gamma):
    del t, alpha, gamma
    xy = state[:2]
    flow = ROTATION@xy + K*(1.-xy@xy)*xy
    planar = ROTATION + K*(1.-xy@xy)*np.eye(2)-2*K*np.outer(xy, xy)
    matrix = np.zeros((3, 3))
    matrix[:2, :2] = planar
    matrix[2, :2] = wave_hessian(xy)@flow + wave_gradient(xy)@planar + LAMBDA*wave_gradient(xy)
    matrix[2, 2] = -LAMBDA
    return matrix


def exact_flow(initial, elapsed):
    xy = initial[:2]
    c, s = np.cos(elapsed), np.sin(elapsed)
    rotation = np.array([[c, -s], [s, c]])
    decay = np.exp(-2*K*elapsed)
    scale = 1/np.sqrt(xy@xy + (1-xy@xy)*decay)
    end_xy = scale*(rotation@xy)
    end = np.r_[end_xy, wave(end_xy)+np.exp(-LAMBDA*elapsed)*(initial[2]-wave(xy))]
    planar = scale*rotation - (1-decay)*scale**3*np.outer(rotation@xy, xy)
    tangent = np.zeros((3, 3))
    tangent[:2, :2] = planar
    tangent[2, :2] = wave_gradient(end_xy)@planar-np.exp(-LAMBDA*elapsed)*wave_gradient(xy)
    tangent[2, 2] = np.exp(-LAMBDA*elapsed)
    return end, tangent


def exact_solver(fun, span, initial, *, t_eval, **kwargs):
    """Injected analytic flow/adjoint, with honest synthetic event metadata."""
    origin = float(span[0])
    def dense(query):
        query = np.asarray(query)
        values = []
        for t in query.reshape(-1):
            if len(initial) == 12:
                x, tangent = exact_flow(initial[:3], float(t)-origin)
                value = np.r_[x, (tangent@initial[3:].reshape(3, 3)).ravel()]
            else:
                # The fixture's canonical orbit is known analytically.
                origin_state, _ = exact_flow(SEED, origin)
                _, tangent = exact_flow(origin_state, float(t)-origin)
                value = np.linalg.solve(tangent.T, initial)
            values.append(initial.copy() if t == origin else value)
        result = np.asarray(values).T
        return result[:, 0] if query.ndim == 0 else result.reshape((len(initial),)+query.shape)
    fun(span[0], initial)
    values = dense(t_eval)
    fun(span[1], values[:, -1])
    result = SimpleNamespace(t=t_eval.copy(), y=values, sol=dense, success=True,
        message='analytic nine-maximum stable oscillator', nfev=2, njev=0, nlu=0)
    if 'events' in kwargs:
        # Used only for verified synthetic candidates on this exact orbit.
        assert np.allclose(initial[:3], SEED, rtol=0., atol=1e-12)
        events = np.arange(10, dtype=np.float64)*(PERIOD/9)
        events = events[events <= span[1]]
        result.t_events = [events]
        result.y_events = [dense(events).T]
        assert max(abs(kwargs['events'](float(t), y)) for t, y in zip(events, result.y_events[0], strict=True)) < 2e-12
    return result


def reference(setting='core'):
    return dict(status='complete', qualified=True, setting=setting, alpha=0., gamma=0., period=PERIOD,
        certificate=dict(section_state=SEED.copy(), period=PERIOD), orbit=dict(states=SEED[None].copy()))


def phase_rows():
    return [dict(phase_index=i, phase_fraction=(i+.37)/24., split='development' if i < 8 else 'heldout') for i in range(24)]


def test_fixture_derivatives_and_fundamental_matrix_are_independent():
    x = np.array([.97, .13, .8])
    h = 1e-6
    fd = np.column_stack([(rhs(0., x+h*e, 0., 0.)-rhs(0., x-h*e, 0., 0.))/(2*h) for e in np.eye(3)])
    np.testing.assert_allclose(jacobian(0., x, 0., 0.), fd, rtol=2e-9, atol=2e-8)
    elapsed = .7
    _, tangent = exact_flow(x, elapsed)
    fd_flow = np.column_stack([(exact_flow(x+h*e, elapsed)[0]-exact_flow(x-h*e, elapsed)[0])/(2*h) for e in np.eye(3)])
    np.testing.assert_allclose(tangent, fd_flow, rtol=2e-9, atol=2e-8)
    numerical = solve_ivp(lambda t, y: rhs(t, y, 0., 0.), (0., elapsed), x,
                         method='DOP853', rtol=1e-12, atol=1e-14, max_step=.01)
    np.testing.assert_allclose(numerical.y[:, -1], exact_flow(x, elapsed)[0], rtol=1e-10, atol=1e-11)


def test_actual_scipy_finds_nine_primitive_maxima_on_the_synthetic_orbit():
    forward, _ = acq._forward(SEED, PERIOD, 0., 0., acq.PRIMARY,
        rhs=rhs, jacobian=jacobian, solver=solve_ivp, events=True)
    certificate = acq._certificate(forward, 0., 0., rhs=rhs, jacobian=jacobian)
    assert certificate['qualified']
    assert certificate['peak_count'] == 9
    np.testing.assert_allclose(forward['trajectory']['states'][-1], SEED, rtol=0., atol=1e-12)
    expected = np.arange(1, 9)*PERIOD/9
    actual = forward['maxima']['times'][certificate['interior_mask']]
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-13)


@pytest.fixture(scope='module')
def synthetic_packets():
    old = reference()
    shots = {m: acq.refine_cycle(old, m, rhs=rhs, jacobian=jacobian, solver=exact_solver) for m in acq.METHODS}
    pair = acq.crossverify_reference(old, shots, phase_rows(), rhs=rhs, jacobian=jacobian, solver=exact_solver)
    return old, shots, pair


def test_unmodified_nine_peak_crossverification_and_shared_global_adjoint(synthetic_packets):
    old, shots, pair = synthetic_packets
    assert all(s['converged'] for s in shots.values())
    assert pair['complete'] and pair['qualified']
    assert len(pair['verifications']) == 4
    for path in pair['verifications'].values():
        assert path['certificate']['peak_count'] == 9
        assert path['certificate']['qualified']
        assert np.min(path['certificate']['interior_return_distances']) > .4
    assert pair['canonical']['normalization_count'] == 1
    terminal = pair['canonical']['terminal_adjoint']
    for base in pair['bases'].values():
        assert np.array_equal(base['raw_endpoints']['final_z'], terminal)
        assert base['new_normalizations'] == 0
        assert len(base['phases']) == 24
        assert base['diagnostics']['adjoint_relative_periodicity'] < 1e-12


def test_real_acquisition_saved_codec_and_evaluator(synthetic_packets, tmp_path):
    old, shots, pair = synthetic_packets
    for method, shot in shots.items():
        seal = codec.save_packet(tmp_path, method, shot)
        loaded = codec.load_packet(seal['path'])
        evaluate.validate_shoot_packet(loaded, old, rhs=rhs, jacobian=jacobian)
    seal = codec.save_packet(tmp_path, 'reference-pair', pair)
    loaded = codec.load_packet(seal['path'])
    evaluate.validate_reference_packet(loaded, shots, old, phase_rows(), rhs=rhs, jacobian=jacobian)
    events = {}
    for method, base in pair['bases'].items():
        event = acq.acquire_refined_event(base, 7, rhs=rhs, jacobian=jacobian, solver=exact_solver)
        seal = codec.save_packet(tmp_path, method+'-event', event)
        loaded_event = codec.load_packet(seal['path'])
        evaluate.validate_event_packet(loaded_event, base, rhs=rhs, jacobian=jacobian)
        events[loaded_event['id']] = loaded_event
    result = evaluate.summarize({p['id']: p for p in shots.values()}, {pair['id']: loaded},
        events, {'core': old, 'failed-near-edge': reference('failed-near-edge')}, phase_rows(),
        rhs=rhs, jacobian=jacobian)
    assert result['counts']['shoots_complete'] == 2
    assert result['counts']['references_qualified'] == 1
    assert result['counts']['events_complete'] == 2
    assert result['counts']['contexts_qualified'] == 1
    assert len(result['contexts']) == 48 and len(result['nested_slots']) == 208
    assert not result['complete'] and not result['qualified']
    assert not result['eight_period_memory_qualified']
