"""Draft phase-memory mathematics, without files, fits or ODE integration.

Inputs are supplied finite float64 arrays. These helpers neither qualify an
orbit nor infer an isochron away from it. The future caller owns source/input
seals, reference/adjoint integration, census and all experiment-level gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr

DELAYS = (1, 2, 4, 8)
SAMPLES = 32


class PhaseQualificationError(ValueError):
    """Mathematical qualification failure with diagnostics for caller retention."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _array(value, name, shape=None):
    result = np.asarray(value)
    if (result.dtype != np.float64 or (shape is not None and result.shape != shape)
            or not np.isfinite(result).all()):
        raise ValueError(f'{name} requires finite float64 values with shape {shape}')
    return result


def _scalar(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f'{name} requires a finite {"positive " if positive else ""}real scalar')
    return float(value)


def _frozen(value):
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _norm(value, name):
    value = float(np.linalg.norm(value))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} requires positive finite norm')
    return value


def _samples(times, values, derivatives):
    t = _array(times, 'sample times')
    v, d = _array(values, 'sample values'), _array(derivatives, 'sample derivatives')
    if (t.ndim != 1 or len(t) < 2 or t[0] != 0 or not np.all(np.diff(t) > 0)
            or v.ndim < 2 or v.shape[0] != len(t) or d.shape != v.shape
            or any(size == 0 for size in v.shape[1:])):
        raise ValueError('Increasing zero-origin times and matching nonempty value/derivative arrays required')
    return _frozen(t), _frozen(v), _frozen(d)


def _hermite(times, values, derivatives, queries, *, derivative=False):
    q = _array(queries, 'query times')
    if np.any(q < times[0]) or np.any(q > times[-1]):
        raise ValueError('Nonperiodic queries must be inside the saved interval')
    indices = np.clip(np.searchsorted(times, q, side='right')-1, 0, len(times)-2)
    h = times[indices+1]-times[indices]
    a = (q-times[indices])/h
    shape = q.shape+(1,)*(values.ndim-1)
    a, h = a.reshape(shape), h.reshape(shape)
    left, right = values[indices], values[indices+1]
    dl, dr = derivatives[indices], derivatives[indices+1]
    if derivative:
        answer = ((6*a*a-6*a)*left/h + (3*a*a-4*a+1)*dl
                  + (-6*a*a+6*a)*right/h + (3*a*a-2*a)*dr)
    else:
        answer = ((2*a**3-3*a*a+1)*left + (a**3-2*a*a+a)*h*dl
                  + (-2*a**3+3*a*a)*right + (a**3-a*a)*h*dr)
    if not np.isfinite(answer).all():
        raise ValueError('Hermite result overflowed; no fallback')
    return answer


@dataclass(frozen=True, init=False)
class HermiteSegment:
    """Closed nonperiodic sampled interval; may contain a matrix propagator."""
    times: np.ndarray
    values: np.ndarray
    derivatives: np.ndarray

    def __init__(self, times, values, derivatives):
        for name, value in zip(('times', 'values', 'derivatives'), _samples(times, values, derivatives), strict=True):
            object.__setattr__(self, name, value)

    def evaluate(self, times, *, derivative=False):
        if type(derivative) is not bool:
            raise ValueError('derivative selector must be boolean')
        return _hermite(self.times, self.values, self.derivatives, times, derivative=derivative)


@dataclass(frozen=True, init=False)
class PeriodicHermiteTemplate:
    """Endpoint-excluded orbit samples, explicitly wrapped to the first sample.

    raw_endpoint is retained separately and is never silently substituted into
    the periodic template. Uniform sample count and acceptable closure remain
    future caller gates; this mathematical helper supports irregular knots.
    Derivatives are with respect to physical time, not normalized phase.
    """
    times: np.ndarray
    values: np.ndarray
    derivatives: np.ndarray
    raw_endpoint: np.ndarray
    period: float

    def __init__(self, times, values, derivatives, period, *, raw_endpoint):
        t, v, d = _samples(times, values, derivatives)
        period = _scalar(period, 'period', positive=True)
        endpoint = _array(raw_endpoint, 'raw endpoint', v.shape[1:])
        if t[-1] >= period:
            raise ValueError('Periodic samples must exclude the endpoint')
        for name, value in [('times', t), ('values', v), ('derivatives', d),
                            ('period', period), ('raw_endpoint', _frozen(endpoint))]:
            object.__setattr__(self, name, value)

    def evaluate(self, times, *, derivative=False):
        if type(derivative) is not bool:
            raise ValueError('derivative selector must be boolean')
        query = np.remainder(_array(times, 'periodic query times'), self.period)
        return _hermite(np.r_[self.times, self.period],
            np.concatenate((self.values, self.values[:1])),
            np.concatenate((self.derivatives, self.derivatives[:1])), query, derivative=derivative)

    def closure(self):
        delta = self.raw_endpoint-self.values[0]
        return dict(endpoint_difference=delta.copy(), maximum_absolute_difference=float(np.abs(delta).max()),
            relative_norm=float(np.linalg.norm(delta)/max(1., np.linalg.norm(self.values[0]))),
            periodic_wrap_uses_initial_sample=True, closure_qualification_owned_by_caller=True)


def neutral_left_phase(monodromy, flow, *, residual_atol=1e-7,
                       unique_singular_rtol=1e-12, normalization_cosine_min=1e-12):
    """SVD right-null vector of M.T-I, normalized once so Z.T @ flow = 1.

    The unique-neutral and residual checks do not certify primitive period or
    transverse stability. No matrix inversion or tiny-multiplier inverse is
    used. Failed diagnostics are retained on PhaseQualificationError.
    """
    matrix, velocity = _array(monodromy, 'monodromy', (3, 3)), _array(flow, 'flow', (3,))
    tolerance = _scalar(residual_atol, 'residual tolerance', positive=True)
    rank_tolerance = _scalar(unique_singular_rtol, 'uniqueness tolerance', positive=True)
    cosine_min = _scalar(normalization_cosine_min, 'normalization cosine minimum', positive=True)
    if rank_tolerance >= 1 or cosine_min >= 1:
        raise ValueError('Relative uniqueness and cosine tolerances must be below one')
    flow_norm = _norm(velocity, 'flow')
    _, singular, vh = np.linalg.svd(matrix.T-np.eye(3), full_matrices=False)
    unit = vh[-1]
    product = float(unit @ velocity)
    cosine = abs(product)/flow_norm
    diagnostics = dict(singular_values=singular.copy(), normalization_product=product,
        normalization_cosine=cosine, neutral_flow_relative_residual=float(np.linalg.norm(matrix@velocity-velocity)/flow_norm),
        unique_singular_direction=bool(singular[-2] > rank_tolerance*max(1., singular[0])))
    if not diagnostics['unique_singular_direction'] or cosine < cosine_min:
        raise PhaseQualificationError('Ambiguous neutral direction or ill-conditioned phase normalization', diagnostics)
    z = unit/product
    diagnostics.update(phase_gradient=z.copy(), normalization_residual=float(abs(z@velocity-1.)),
        neutral_left_relative_residual=float(np.linalg.norm(matrix.T@z-z)/np.linalg.norm(z)))
    if (not np.isfinite(z).all() or not np.isfinite(diagnostics['neutral_left_relative_residual'])
            or not np.isfinite(diagnostics['neutral_flow_relative_residual'])
            or not np.isfinite(diagnostics['normalization_residual'])
            or diagnostics['neutral_left_relative_residual'] > tolerance
            or diagnostics['neutral_flow_relative_residual'] > tolerance
            or diagnostics['normalization_residual'] > 1e-12):
        raise PhaseQualificationError('Neutral phase residual gate failed', diagnostics)
    diagnostics['passed'] = True
    return diagnostics


def pulse_directions(phase_gradient, flow):
    """Same state-norm controls; the null plane is orthogonal to Z, not flow."""
    z, velocity = _array(phase_gradient, 'phase gradient', (3,)), _array(flow, 'flow', (3,))
    z_norm = _norm(z, 'phase gradient')
    normalization = float(z@velocity)
    if not np.isfinite(normalization) or abs(normalization-1.) > 1e-7:
        raise ValueError('Time-valued phase normalization Z.T @ flow = 1 required')
    write = z/z_norm
    projection = np.eye(3)-np.outer(write, write)
    index = int(np.argmax(np.linalg.norm(projection, axis=0)))
    null = projection[:, index]/_norm(projection[:, index], 'phase-null projection')
    one = np.array([1., 0., 0.])
    directions = dict(phase_write=write, phase_null=null, neuron_one=one)
    if (any(abs(np.linalg.norm(d)-1.) > 1e-12 for d in directions.values())
            or abs(float(z@null))/z_norm > 1e-12):
        raise ValueError('Unit or phase-null direction check failed')
    return dict(directions=directions, null_coordinate_index=index,
        phase_response_per_unit_norm={name: float(z@d) for name, d in directions.items()},
        classifier_orientations=dict(phase_write=1, phase_null=1, neuron_one=-1 if z[0] < 0 else 1))


def realize_pulse(initial, direction, epsilon, sign):
    """Record requested versus rounded state displacement; never rescale it."""
    initial, direction = _array(initial, 'initial state', (3,)), _array(direction, 'direction', (3,))
    epsilon = _scalar(epsilon, 'epsilon', positive=True)
    if (type(sign) is not int or sign not in (-1, 1)
            or abs(np.linalg.norm(direction)-1.) > 1e-12):
        raise ValueError('Unit direction and integer sign -1 or +1 required')
    requested = sign*epsilon*direction
    realized_initial = initial+requested
    if not np.isfinite(realized_initial).all():
        raise ValueError('Pulse addition overflowed')
    actual = realized_initial-initial
    actual_norm = float(np.linalg.norm(actual))
    if not np.isfinite(actual).all() or not np.isfinite(actual_norm):
        raise ValueError('Realized displacement diagnostic overflowed')
    return dict(initial=realized_initial, requested_delta=requested, realized_delta=actual,
        realized_norm=actual_norm, addition_rounding_residual=actual-requested)


@dataclass(frozen=True)
class ClockReadout:
    times: np.ndarray
    reference_activations: np.ndarray
    weights: np.ndarray
    derivative_energy: float
    phase_epoch: float
    period: float
    delay_cycles: int

    def __post_init__(self):
        period = _scalar(self.period, 'readout period', positive=True)
        epoch = _scalar(self.phase_epoch, 'readout phase epoch')
        energy = _scalar(self.derivative_energy, 'derivative energy', positive=True)
        times = _array(self.times, 'readout times', (SAMPLES,))
        reference = _array(self.reference_activations, 'reference activations', (SAMPLES, 3))
        weights = _array(self.weights, 'readout weights', (SAMPLES, 3))
        if (type(self.delay_cycles) is not int or self.delay_cycles not in DELAYS
                or not 0 <= epoch < period or np.max(np.abs(reference)) > 1.
                or not np.array_equal(times, period*(self.delay_cycles-1+(np.arange(SAMPLES)+.5)/SAMPLES))
                or not np.isclose(np.sum(weights*weights)*energy, 1., rtol=1e-12, atol=1e-14)):
            raise ValueError('Clock readout schema, sample grid or normalization differs')
        for name, value in [('times', times), ('reference_activations', reference), ('weights', weights)]:
            object.__setattr__(self, name, _frozen(value))

    def score(self, activations):
        values = _array(activations, 'readout activations', (SAMPLES, 3))
        if np.max(np.abs(values)) > 1.:
            raise ValueError('Deterministic mean activations must lie in [-1,1]')
        answer = float(np.sum(self.weights*(values-self.reference_activations)))
        if not np.isfinite(answer):
            raise ValueError('Readout score overflowed')
        return answer


def clock_readout(template, phase_epoch, delay_cycles):
    """32 midpoint samples across the cycle ending at delay_cycles*T.

    Template values are neuron potentials. The derivative of tanh is applied
    before constructing the time-shift matched filter; no phase optimization.
    """
    if not isinstance(template, PeriodicHermiteTemplate) or template.values.shape[1:] != (3,):
        raise ValueError('Three-potential periodic template required')
    epoch = _scalar(phase_epoch, 'phase epoch')
    if not 0 <= epoch < template.period or type(delay_cycles) is not int or delay_cycles not in DELAYS:
        raise ValueError('Known epoch in [0,T) and declared delay 1,2,4,8 required')
    times = template.period*(delay_cycles-1+(np.arange(SAMPLES)+.5)/SAMPLES)
    state = template.evaluate(times+epoch)
    reference = np.tanh(state)
    derivative = (1.-reference**2)*template.evaluate(times+epoch, derivative=True)
    energy = float(np.sum(derivative*derivative))
    if not np.isfinite(energy) or energy <= 0:
        raise PhaseQualificationError('Unobservable time-shift readout', dict(derivative_energy=energy))
    weights = derivative/energy
    if not np.isfinite(weights).all() or not np.isfinite(np.linalg.norm(weights)):
        raise PhaseQualificationError('Readout normalization overflowed', dict(derivative_energy=energy))
    return ClockReadout(_frozen(times), _frozen(reference), _frozen(weights), energy, epoch, template.period, delay_cycles)


def balanced_sign_readout(readout, plus, minus, *, sigma=1e-4, orientation=1):
    """Exact BA under independent Gaussian observation noise, no noisy ODE.

    The reference template is calibrated/noiseless. Noise is per activation
    sample, so score standard deviation is sigma*||w||, with no sqrt(n) again.
    """
    if not isinstance(readout, ClockReadout) or type(orientation) is not int or orientation not in (-1, 1):
        raise ValueError('Frozen clock readout and integer orientation -1 or +1 required')
    sigma = _scalar(sigma, 'observation noise', positive=True)
    q_plus, q_minus = readout.score(plus), readout.score(minus)
    score_sd = sigma*_norm(readout.weights, 'readout weights')
    standardized = np.array([orientation*q_plus, -orientation*q_minus])/score_sd
    if not np.isfinite(standardized).all() or score_sd <= 0 or not np.isfinite(score_sd):
        raise ValueError('Noise-normalized score overflowed or underflowed')
    return dict(q_plus=q_plus, q_minus=q_minus, sigma=sigma, orientation=orientation,
        score_noise_sd=score_sd, standardized_correct_means=standardized.tolist(),
        balanced_accuracy=float(np.mean(ndtr(standardized))),
        noise_scope='Independent activation-sample measurement noise; noiseless calibrated reference.')


def phase_only_prediction(template, delta_x, phase_gradient, elapsed_times, *, phase_epoch=0.):
    """First-order phase reset followed by the unchanged periodic waveform.

    The caller explicitly labels requested versus realized displacement. The
    phase-memory primary null uses requested sign*epsilon*d; a realized-pulse
    prediction is only a rounding diagnostic. This is not a finite-amplitude
    isochron calculation or an additional neural integration.
    """
    if not isinstance(template, PeriodicHermiteTemplate) or template.values.shape[1:] != (3,):
        raise ValueError('Three-potential periodic template required')
    delta, z = _array(delta_x, 'pulse displacement', (3,)), _array(phase_gradient, 'phase gradient', (3,))
    elapsed = _array(elapsed_times, 'elapsed times')
    epoch = _scalar(phase_epoch, 'phase epoch')
    if np.any(elapsed < 0) or not 0 <= epoch < template.period:
        raise ValueError('Nonnegative elapsed times and epoch in [0,T) required')
    displacement = float(z@delta)
    if not np.isfinite(displacement):
        raise ValueError('Phase displacement overflowed')
    state = template.evaluate(elapsed+epoch+displacement)
    return dict(time_displacement=displacement, states=state, activations=np.tanh(state))


def full_linear_response(segment, monodromy, delta_x, elapsed_times, *, max_cycles=8):
    """Phi(s) @ M**k @ delta_x; the fundamental matrix is NOT periodic.

    segment contains a full [0,T] matrix propagator and its time derivative,
    including identity at zero and the supplied monodromy at T. Polynomial
    interpolation is a numerical representation, not an integration-error
    certificate. Nonnegative cycles only; there is no inverse multiplier.
    """
    matrix, delta = _array(monodromy, 'monodromy', (3, 3)), _array(delta_x, 'pulse displacement', (3,))
    elapsed = _array(elapsed_times, 'elapsed times')
    if (not isinstance(segment, HermiteSegment) or segment.values.shape[1:] != (3, 3)
            or not np.array_equal(segment.values[0], np.eye(3))
            or not np.array_equal(segment.values[-1], matrix)
            or np.any(elapsed < 0) or type(max_cycles) is not int or max_cycles < 0):
        raise ValueError('Bound identity-to-monodromy segment and nonnegative horizon required')
    period = float(segment.times[-1])
    cycles_float = np.floor(elapsed/period)
    if np.any(cycles_float > max_cycles):
        raise ValueError('Elapsed horizon exceeds the declared cycle bound')
    cycles = cycles_float.astype(np.int64)
    remainder = elapsed-cycles*period
    pieces = segment.evaluate(remainder).reshape(-1, 3, 3)
    result = np.empty((elapsed.size, 3))
    for count in np.unique(cycles):
        mask = cycles.ravel() == count
        result[mask] = np.einsum('nij,j->ni', pieces[mask], np.linalg.matrix_power(matrix, int(count))@delta)
    if not np.isfinite(result).all():
        raise ValueError('Linear response overflowed; no fallback')
    return dict(cycles=cycles.copy(), remainder=remainder.copy(), state_delta=result.reshape(elapsed.shape+(3,)))


def linear_activation_delta(reference_states, state_delta):
    """First-order tanh response only, distinct from tanh(x+delta)-tanh(x)."""
    reference, delta = _array(reference_states, 'reference states'), _array(state_delta, 'state response')
    if reference.shape != delta.shape or reference.ndim < 1 or reference.shape[-1] != 3:
        raise ValueError('Matching three-potential reference and response required')
    answer = (1.-np.tanh(reference)**2)*delta
    if not np.isfinite(answer).all():
        raise ValueError('Activation linearization overflowed')
    return answer
