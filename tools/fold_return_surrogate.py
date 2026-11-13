"""Autonomous, branch-aware departure map. NumPy only; no network at rollout.

Coordinates are v/U. A table is a numerical model, not an exact invariant map.
Mixed-label/elapsed-time intervals are explicitly marked as unresolved branches.
"""
import numpy as np


class DepartureDetector:
    """Streaming two-phase channel exit; independent offline audit is required."""
    def __init__(self, width, transverse_limit=.05):
        self.width = width
        self.limit = transverse_limit
        self.streak = [0, 0]
        self.entry = [None, None]
        self.pending = [None, None]

    def update(self, step, v, transverse):
        phase = step % 2
        inside = abs(v) < self.width and transverse < self.limit
        if inside:
            if not self.streak[phase]:
                self.entry[phase] = step
            self.streak[phase] += 1
            self.pending[phase] = None
            return None
        if self.streak[phase] >= 10:
            self.pending[phase] = {'entry': self.entry[phase], 'last_inside': step-2}
        self.streak[phase] = 0
        candidate = self.pending[phase]
        if candidate is None:
            return None
        if step-candidate['last_inside'] > 32:
            self.pending[phase] = None
            return None
        if v >= 3*self.width and transverse < self.limit:
            self.pending[phase] = None
            return {**candidate, 'exit': step, 'v_over_U': float(v/self.width),
                    'phase': phase, 'transverse': float(transverse)}
        return None


def make_model(records):
    rows = sorted(records, key=lambda r: r['q'])
    q = np.array([r['q'] for r in rows])
    if len(q) < 2 or np.any(np.diff(q) <= 0) or not np.isfinite(q).all():
        raise ValueError('Strictly ordered distinct finite training coordinates required')
    return {'version': 1, 'coordinate': 'v/U', 'domain': [float(q[0]), float(q[-1])],
            'rule': 'Linear interpolation within equal-time return branches; nearest endpoint otherwise. All mixed branches flagged. No extrapolation.',
            'rows': rows}


def predict(model, q, method='piecewise'):
    rows = model['rows']; grid = np.array([r['q'] for r in rows])
    if not np.isfinite(q) or q < grid[0] or q > grid[-1]:
        return {'kind': 'outside_domain', 'q': float(q)}
    j = int(np.clip(np.searchsorted(grid, q, side='right')-1, 0, len(grid)-2))
    a, b = rows[j:j+2]
    nearest = a if q-a['q'] <= b['q']-q else b
    coherent = a['kind'] == b['kind'] and (a['kind'] != 'return' or a['updates'] == b['updates'])
    result = {'kind': nearest['kind'], 'updates': nearest['updates'],
              'branch_resolved': coherent, 'bracket': [a['q'], b['q']]}
    if nearest['kind'] == 'return':
        if method == 'piecewise' and coherent:
            f = (q-a['q'])/(b['q']-a['q'])
            result['next_q'] = float((1-f)*a['next_q']+f*b['next_q'])
        else:
            result['next_q'] = nearest['next_q']
    return result


def rollout(model, q, cycles=128, method='piecewise'):
    """Runs unassisted: only the initial scalar is supplied by the network."""
    events = []; elapsed = 0
    for k in range(cycles):
        prediction = predict(model, q, method)
        elapsed += prediction.get('updates', 0)
        events.append({'cycle': k+1, 'q': float(q), 'elapsed': elapsed, **prediction})
        if prediction['kind'] != 'return':
            break
        q = prediction['next_q']
    return events


def refinement_points(records, count):
    """Development-only refinement: labels, time jumps, slope and coverage."""
    rows = sorted(records, key=lambda r: r['q']); ranked = []
    for a, b in zip(rows, rows[1:]):
        width = b['q']-a['q']
        change = 0.
        if a['kind'] != b['kind']:
            change += 8
        elif a['kind'] == 'return':
            change += min(8., abs(a['next_q']-b['next_q'])*4)
            change += min(8., abs(a['updates']-b['updates'])/8)
        else:
            change += 1
        # Width prevents repeated microscopic refinement of one discontinuity.
        ranked.append((width*(1+change), (a['q']+b['q'])/2))
    return sorted(q for _, q in sorted(ranked, reverse=True)[:count])
