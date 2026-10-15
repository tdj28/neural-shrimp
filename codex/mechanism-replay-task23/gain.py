"""Own finite-twin accounting around the unchanged independent CPU Qwen map.

No producer implementation is imported. All target calls belong to a separately
reviewed profile/run card; importing this module does not load a model.
"""
import numpy as np
import torch
import torch.nn.functional as functional

SOURCE_INDICES = (25, 29, 19, 35)
LOG_T = (.125, .129, .11900000000000001, .135)
ALPHA = .0655
EPS = (1e-6, 1e-9)
IDENTITY_ABS_TOLERANCE = 1e-10  # prospective floating accounting check, not an interval bound


class UndefinedGain(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('undefined finite gain or renormalization')


def original_directions(dimension=896):
    """Select original positions from both complete201-row RNG blocks."""
    rng = np.random.default_rng(0)
    return np.stack([rng.normal(size=(201, dimension))[list(SOURCE_INDICES)]
                     for _ in EPS])


def finite_tensor(x):
    return (isinstance(x, torch.Tensor) and x.dtype == torch.float64
            and x.device.type == 'cpu' and bool(torch.isfinite(x).all()))


def split_map(model, states, alpha, temperatures, cache):
    """Expose the final normalized hidden state through the original blocks.

    temperatures are positive physicalT per batch row, constructed with the
    producer's NumPy float64 power convention. No extra temperature or bias.
    """
    if states.ndim != 2 or not finite_tensor(states):
        raise ValueError('finite CPU float64 batch required')
    if not finite_tensor(temperatures) or temperatures.shape != (len(states),) or not bool((temperatures > 0).all()):
        raise ValueError('one positive physical temperature per row required')
    alpha = model._scalar(alpha)
    cache = model._external_cache(cache, alpha)
    hidden = model._state(states)[:, None, :]
    positions = torch.tensor([len(model.prefix_ids)], dtype=torch.long)
    for index in range(model.layers):
        hidden, _ = model._block(hidden, index, alpha, positions, cache.layers[index])
    h = model._norm(hidden[:, -1, :], model.weights['model.norm.weight'])
    z = functional.linear(h, model.embedding)
    p = torch.softmax(z / temperatures[:, None], dim=-1, dtype=torch.float64)
    feedback = p @ model.embedding
    for name, value in [('hidden', h), ('logits', z), ('probability', p), ('feedback', feedback)]:
        if not finite_tensor(value):
            raise UndefinedGain({'field': name, 'reason': 'nonfinite or wrong effective dtype/device'})
    return h, z, p, feedback


def initial_twins(base, directions):
    if not finite_tensor(base) or base.ndim != 2:
        raise ValueError('finite base batch required')
    g = torch.as_tensor(directions, dtype=torch.float64, device='cpu')
    if g.shape != (len(EPS), *base.shape) or not finite_tensor(g):
        raise ValueError('two full finite direction batches required')
    base_norm = torch.linalg.vector_norm(base, dim=1, keepdim=True)
    dir_norm = torch.linalg.vector_norm(g, dim=2, keepdim=True)
    if not bool((base_norm > 0).all()) or not bool((dir_norm > 0).all()):
        raise UndefinedGain({'field': 'initialization', 'reason': 'zero base/direction norm'})
    twins = [base + g[k] / dir_norm[k] * eps * base_norm for k, eps in enumerate(EPS)]
    return torch.cat([base] + twins, dim=0)


def norms(x):
    return torch.linalg.vector_norm(x, dim=1).detach().numpy().copy()


def readout_observables(z, p, temperatures):
    entropy = -(p * torch.log(torch.clamp(p, min=1e-300))).sum(-1)
    top = torch.topk(z, 2, dim=-1, largest=True, sorted=True)
    gap = (top.values[:, 0] - top.values[:, 1]) / temperatures
    return {'entropy': entropy.detach().numpy().copy(),
            'gap': gap.detach().numpy().copy(),
            'p_top': p.max(dim=-1).values.detach().numpy().copy(),
            'top2_logits': top.values.detach().numpy().copy(),
            'top2_ids': top.indices.detach().numpy().copy()}


def finite_step(states, hidden, feedback, z, p, temperatures):
    """All actual norm operands and both finite-twin splits, then renormalize.

    An undefined row is retained in exception diagnostics and stops the attempt;
    there is no floor, replacement perturbation, row skipping, or silent retry.
    """
    if len(states) % 3 or states.shape != hidden.shape or states.shape != feedback.shape:
        raise ValueError('three equal base/first-twin/second-twin batches required')
    n = len(states) // 3
    xb, hb, fb = states[:n], hidden[:n], feedback[:n]
    operand = {'x_norm': norms(xb), 'h_norm': norms(hb), 'F_norm': norms(fb),
               'dx': np.stack([norms(states[(k+1)*n:(k+2)*n] - xb) for k in range(2)]),
               'dh': np.stack([norms(hidden[(k+1)*n:(k+2)*n] - hb) for k in range(2)]),
               'dF': np.stack([norms(feedback[(k+1)*n:(k+2)*n] - fb) for k in range(2)])}
    invalid = {key: np.argwhere(~np.isfinite(value) | (value <= 0)).tolist()
               for key, value in operand.items() if np.any(~np.isfinite(value) | (value <= 0))}
    if invalid:
        raise UndefinedGain({'invalid_indices': invalid, 'norm_operands': {k: v.tolist() for k, v in operand.items()}})
    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        dx, dh, df = operand['dx'], operand['dh'], operand['dF']
        rx, rh, rf = dx / operand['x_norm'], dh / operand['h_norm'], df / operand['F_norm']
        growth = np.log(df / dx)
        net, read = np.log(rh / rx), np.log(rf / rh)
        boundary_step = np.log(operand['F_norm'] / operand['x_norm'])
        residual = growth - net - read - boundary_step[None, :]
    gains = {'log_growth': growth, 'log_g_net': net, 'log_g_read': read,
             'boundary_step': boundary_step, 'identity_residual': residual}
    if any(not np.isfinite(v).all() for v in gains.values()):
        raise UndefinedGain({'field': 'gain arithmetic', 'norm_operands': {k: v.tolist() for k, v in operand.items()},
                             'gains': {k: v.tolist() for k, v in gains.items()}})
    if np.any(np.abs(residual) > IDENTITY_ABS_TOLERANCE):
        raise UndefinedGain({'field': 'floating accounting identity', 'tolerance': IDENTITY_ABS_TOLERANCE,
                             'norm_operands': {k: v.tolist() for k, v in operand.items()},
                             'gains': {k: v.tolist() for k, v in gains.items()}})
    new = [fb]
    for k, eps in enumerate(EPS):
        delta = feedback[(k+1)*n:(k+2)*n] - fb
        scale = torch.as_tensor(eps * operand['F_norm'] / df[k], dtype=torch.float64)[:, None]
        new.append(fb + delta * scale)
    next_states = torch.cat(new, dim=0)
    if not finite_tensor(next_states):
        raise UndefinedGain({'field': 'renormalization', 'reason': 'nonfinite output'})
    actual_next_dx = np.stack([norms(new[k+1] - fb) for k in range(2)])
    if not np.isfinite(actual_next_dx).all() or np.any(actual_next_dx <= 0):
        raise UndefinedGain({'field': 'renormalization', 'reason': 'zero/nonfinite represented next offset',
                             'actual_next_dx': actual_next_dx.tolist()})
    obs = readout_observables(z[:n], p[:n], temperatures[:n])
    if any(not np.isfinite(v).all() for v in obs.values()):
        raise UndefinedGain({'field': 'readout observables', 'reason': 'nonfinite'})
    return {**operand, **gains, **obs, 'next_dx': actual_next_dx}, next_states


def period_diagnostic(tail):
    """Literal final-state recurrence, not all-phase primitive/stability proof."""
    tail = np.asarray(tail, dtype=np.float64)
    if tail.ndim != 3 or not len(tail) or not np.isfinite(tail).all():
        raise ValueError('finite nonempty time/cell/state tail required')
    rows = []
    for c in range(tail.shape[1]):
        s = tail[:, c]
        threshold = 1e-8 * max(1., float(np.linalg.norm(s[-1])))
        distances = [float(np.linalg.norm(s[-1-q] - s[-1])) for q in range(1, min(65, len(s)))]
        period = next((q for q, distance in enumerate(distances, 1) if distance < threshold), 0)
        rows.append({'period': period, 'threshold': threshold, 'lag_distances': distances,
                     'tail_states': len(s), 'scope': 'last-state recurrence only'})
    return rows


class Accumulator:
    def __init__(self, initial_base, expected_steps):
        if type(expected_steps) is not int or expected_steps < 1:
            raise ValueError('positive fixed record length required')
        self.initial_norm = norms(initial_base)
        if not np.isfinite(self.initial_norm).all() or np.any(self.initial_norm <= 0):
            raise UndefinedGain({'field': 'initial norm'})
        self.expected_steps, self.count = expected_steps, 0
        self.sums = {key: np.zeros((2, len(initial_base))) for key in ('log_growth', 'log_g_net', 'log_g_read')}
        self.obs_sums = {key: np.zeros(len(initial_base)) for key in ('entropy', 'gap', 'p_top')}
        self.gap_counts = np.zeros((2, len(initial_base)), dtype=np.int64)
        self.max_step_identity_abs = np.zeros((2, len(initial_base)))
        self.nonzero_step_identity_count = np.zeros((2, len(initial_base)), dtype=np.int64)

    def add(self, record):
        if self.count >= self.expected_steps:
            raise ValueError('record overrun')
        for key in self.sums:
            value = record[key]
            if value.shape != self.sums[key].shape or not np.isfinite(value).all():
                raise ValueError('wrong finite gain shape')
            self.sums[key] += value
        for key in self.obs_sums:
            self.obs_sums[key] += record[key]
        self.gap_counts += np.stack([record['gap'] < 1., record['gap'] < 2.])
        self.max_step_identity_abs = np.maximum(self.max_step_identity_abs, np.abs(record['identity_residual']))
        self.nonzero_step_identity_count += record['identity_residual'] != 0
        self.count += 1

    def finish(self, final_base, tail):
        if self.count != self.expected_steps:
            raise ValueError('incomplete measurement cannot become a final summary')
        boundary = np.log(norms(final_base) / self.initial_norm) / self.count
        means = {key: value / self.count for key, value in self.sums.items()}
        residual = means['log_growth'] - means['log_g_net'] - means['log_g_read'] - boundary[None, :]
        if not np.isfinite(residual).all() or np.any(np.abs(residual) > IDENTITY_ABS_TOLERANCE):
            raise UndefinedGain({'field': 'summed floating accounting identity',
                                 'residual': residual.tolist(), 'tolerance': IDENTITY_ABS_TOLERANCE})
        periods = period_diagnostic(tail)
        result = []
        for c in range(len(self.initial_norm)):
            lyap = means['log_growth'][:, c].tolist()
            result.append({'measurement_steps': self.count, 'lyapunov': lyap, 'eps': list(EPS),
                           'mean_log_g_net': float(means['log_g_net'][0, c]),
                           'mean_log_g_read': float(means['log_g_read'][0, c]),
                           'both_twin_mean_log_g_net': means['log_g_net'][:, c].tolist(),
                           'both_twin_mean_log_g_read': means['log_g_read'][:, c].tolist(),
                           'boundary_term': float(boundary[c]), 'identity_residual': residual[:, c].tolist(),
                           'identity_abs_tolerance': IDENTITY_ABS_TOLERANCE,
                           'max_step_identity_abs': self.max_step_identity_abs[:, c].tolist(),
                           'nonzero_step_identity_count': self.nonzero_step_identity_count[:, c].tolist(),
                           'mean_entropy': float(self.obs_sums['entropy'][c] / self.count),
                           'mean_gap': float(self.obs_sums['gap'][c] / self.count),
                           'mean_p_top': float(self.obs_sums['p_top'][c] / self.count),
                           'frac_gap_below_1': float(self.gap_counts[0, c] / self.count),
                           'frac_gap_below_2': float(self.gap_counts[1, c] / self.count),
                           **periods[c],
                           'literal_chaotic': periods[c]['period'] == 0 and all(x > 1e-3 for x in lyap),
                           'numerically_defined': True})
        return result
