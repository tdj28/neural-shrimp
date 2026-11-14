"""Unwrapped return coordinate for the FGR3 natural-departure table (post-test diagnostic).

A qualified departure is the first same-phase reading with v >= 3U. Its position q+ therefore wraps: a start that
arrives one channel step earlier leaves one step earlier, with q+ jumping from the top of the window back to 3.
This tool removes the wrap. With q- the same-phase reading two updates before the departure, the fraction of a
channel step already taken past the line is f = (q+ - 3)/(q+ - q-), and the departure time in channel steps,
tau = t/2 - f, is continuous across a wrap (one side has f -> 1 at time t, the other f -> 0 at time t - 2).

Inputs are the saved full states of the FGR3 training and validation runs (read-only). The tool reads no model and
runs no network. Its interval rule and tolerances are fixed in this file; the fresh trajectories were already seen
when this diagnostic was written, so every fresh-trajectory score it prints is post-test.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import backup_file, sha, write_json
from fold_return_surrogate import predict as table_predict

DATA = Path('/data2/neural-shrimp-data')
RAW = DATA/'explore/llm-soft-loop/global-return'
TRAIN = RAW/'20261113-fgr3-training-01/run'
VALID = RAW/'20261113-fgr3-validation-01/run'
LINE = 3.0            # departure line, v/U
TAU_TOL = 0.5         # largest |delta tau| (channel steps) between neighbors called continuous


class States:
    """Lazy reader of one batch directory's update shards."""
    def __init__(self, batch_dir):
        self.shards = []
        for p in sorted(Path(batch_dir).glob('updates-*.npz')):
            lo, hi = map(int, p.stem.split('-')[1:3])
            self.shards.append((lo, hi, p))
        self.cache = {}

    def at(self, step):
        for lo, hi, p in self.shards:
            if lo <= step <= hi:
                if p not in self.cache:
                    self.cache = {p: np.load(p)['states']}
                return self.cache[p][step-lo]
        raise KeyError(step)


def projector(inputs):
    with np.load(inputs) as z:
        return z['centre'].copy(), z['left'].copy(), z['right'].copy()


def q_of(x, centre, left, U):
    return float((x-centre)@left/U)


def unwrap(t, q_plus, q_minus):
    if not (q_minus < LINE <= q_plus):
        raise ValueError(f'Departure is not the first reading past the line: q-={q_minus}, q+={q_plus}')
    f = (q_plus-LINE)/(q_plus-q_minus)
    return t/2-f, f


def table_rows(U, centre, left):
    model = json.loads((VALID/'frozen-model.json').read_text())
    rows = []
    cache = {}
    for r in sorted(model['rows'], key=lambda r: r['q']):
        row = dict(r)
        if r['kind'] == 'return':
            S = cache.setdefault(r['batch'], States(TRAIN/r['batch']))
            x_plus = S.at(r['updates'])[r['row']]
            x_minus = S.at(r['updates']-2)[r['row']]
            qp = q_of(x_plus, centre, left, U)
            if abs(qp-r['next_q']) > 1e-9:
                raise ValueError(f'Saved state does not reproduce the recorded departure at q={r["q"]}')
            row['q_minus'] = q_of(x_minus, centre, left, U)
            row['tau'], row['f'] = unwrap(r['updates'], r['next_q'], row['q_minus'])
        rows.append(row)
    return model, rows


def fresh_pairs(U, centre, left, run=VALID):
    result = json.loads((Path(run)/'result.json').read_text())
    S = States(Path(run)/'trajectories')
    pairs, departures = [], []
    for i, (events, outcome) in enumerate(zip(result['events'], result['outcomes'])):
        for k, e in enumerate(events):
            x_plus = S.at(e['exit'])[i]; x_minus = S.at(e['exit']-2)[i]
            qp = q_of(x_plus, centre, left, U)
            if abs(qp-e['v_over_U']) > 1e-9:
                raise ValueError('Saved fresh state does not reproduce the recorded departure')
            departures.append({'trajectory': i, 'index': k, 'exit': e['exit'], 'q': e['v_over_U'],
                               'q_minus': q_of(x_minus, centre, left, U)})
        for k in range(len(events)):
            a = events[k]
            if k+1 < len(events):
                b = departures[-len(events)+k+1]
                t = b['exit']-a['exit']
                tau, f = unwrap(t, b['q'], b['q_minus'])
                pairs.append({'trajectory': i, 'index': k, 'q': a['v_over_U'], 'kind': 'return', 'updates': t,
                              'next_q': b['q'], 'q_minus': b['q_minus'], 'tau': tau, 'f': f})
            elif outcome['kind'] in ('reference_p3', 'other_recurrence_candidate'):
                pairs.append({'trajectory': i, 'index': k, 'q': a['v_over_U'], 'kind': outcome['kind'],
                              'updates': None})
            else:
                pairs.append({'trajectory': i, 'index': k, 'q': a['v_over_U'], 'kind': 'censored',
                              'updates': None})
    return result, pairs, departures


def intervals(rows):
    """Adjacent table samples are joined when both return, share parity and differ by < TAU_TOL in tau."""
    out = []
    for a, b in zip(rows, rows[1:]):
        joined = (a['kind'] == b['kind'] == 'return' and a['updates'] % 2 == b['updates'] % 2
                  and abs(a['tau']-b['tau']) < TAU_TOL)
        out.append({'lo': a['q'], 'hi': b['q'], 'resolved': bool(joined)})
    return out


def runs_of_resolved(iv):
    runs, start = [], None
    for k, s in enumerate(iv):
        if s['resolved'] and start is None:
            start = k
        if (not s['resolved'] or k == len(iv)-1) and start is not None:
            end = k if s['resolved'] else k-1
            runs.append((iv[start]['lo'], iv[end]['hi'], end-start+1))
            start = None
    return runs


class ExitStep:
    """q- as a function of q+, fitted on all table departures (the exit states lie on one curve)."""
    def __init__(self, rows, degree=6):
        r = [x for x in rows if x['kind'] == 'return']
        qp = np.array([x['next_q'] for x in r]); qm = np.array([x['q_minus'] for x in r])
        self.lo, self.hi = qp.min(), qp.max()
        self.coef = np.polynomial.chebyshev.chebfit(self.scale(qp), qm, degree)
        self.residual = float(np.max(abs(self(qp)-qm)))

    def scale(self, q):
        return (np.asarray(q)-(self.lo+self.hi)/2)/((self.hi-self.lo)/2)

    def __call__(self, q):
        return np.polynomial.chebyshev.chebval(self.scale(q), self.coef)

    def wrap(self, tau, parity):
        """Inverse of unwrap(): the departure time and position for a given tau and time parity."""
        half = 0.5 if parity else 0.
        k = np.ceil(tau-half-1e-12)
        t2 = k+half                      # t/2
        f = t2-tau                       # in [0, 1)
        lo, hi = LINE, self.hi+0.05
        for _ in range(200):
            mid = (lo+hi)/2
            g = (mid-LINE)/(mid-self(mid))
            lo, hi = (mid, hi) if g < f else (lo, mid)
        return int(round(2*t2)), (lo+hi)/2


def unwrapped_predict(rows, iv, q, step):
    grid = np.array([r['q'] for r in rows])
    if q < grid[0] or q > grid[-1]:
        return {'kind': 'outside_domain'}
    j = int(np.clip(np.searchsorted(grid, q, side='right')-1, 0, len(grid)-2))
    if not iv[j]['resolved']:
        return {'kind': 'unresolved', 'bracket': [grid[j], grid[j+1]]}
    a, b = rows[j], rows[j+1]
    w = (q-a['q'])/(b['q']-a['q'])
    tau = (1-w)*a['tau']+w*b['tau']
    t, qn = step.wrap(tau, a['updates'] % 2)
    return {'kind': 'return', 'tau': tau, 'updates': t, 'next_q': qn}


def region_of(q, runs):
    for lo, hi, _ in runs:
        if lo <= q <= hi:
            return lo, hi
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True); p.add_argument('--backup', required=True)
    args = p.parse_args()
    out, backup = Path(args.out), Path(args.backup)
    out.mkdir(parents=True, exist_ok=False); backup.mkdir(parents=True, exist_ok=False)
    U = json.loads((TRAIN/'plan.json').read_text())  # bound below; width read from the parent curve plan
    parent = Path(U['parent_curve']); U = json.loads((parent/'plan.json').read_text())['width']
    centre, left, right = projector(TRAIN/'inputs.npz')
    if sha(TRAIN/'inputs.npz') != sha(VALID/'inputs.npz'):
        raise ValueError('Training and validation inputs differ')
    model, rows = table_rows(U, centre, left)
    result, pairs, departures = fresh_pairs(U, centre, left)
    step = ExitStep(rows)
    iv = intervals(rows)
    runs = runs_of_resolved(iv)
    # Round-trip check of the inverse on every table return.
    trip = [abs(step.wrap(r['tau'], r['updates'] % 2)[1]-r['next_q']) for r in rows if r['kind'] == 'return']
    trip_t = all(step.wrap(r['tau'], r['updates'] % 2)[0] == r['updates'] for r in rows if r['kind'] == 'return')
    # Fresh one-cycle comparison, reset at each observed departure.
    long_runs = [r for r in runs if r[1]-r[0] >= 0.02]
    scored = []
    for pr in pairs:
        old = table_predict(model, pr['q'])
        new = unwrapped_predict(rows, iv, pr['q'], step)
        scored.append({**pr, 'table': old, 'unwrapped': new, 'region': region_of(pr['q'], runs)})
    summary = {}
    for name, sel in (('all', lambda s: True),
                      ('resolved', lambda s: s['unwrapped']['kind'] != 'unresolved'),
                      ('unresolved', lambda s: s['unwrapped']['kind'] == 'unresolved'),
                      ('resolved_long_runs', lambda s: s['unwrapped']['kind'] != 'unresolved'
                       and any(a <= s['q'] <= b for a, b, _ in long_runs)),
                      ('resolved_short_windows', lambda s: s['unwrapped']['kind'] != 'unresolved'
                       and not any(a <= s['q'] <= b for a, b, _ in long_runs))):
        S = [s for s in scored if sel(s)]
        ret = [s for s in S if s['kind'] == 'return']
        both = [s for s in ret if s['unwrapped']['kind'] == 'return']
        tab = [s for s in ret if s['table']['kind'] == 'return']
        def rmse(xs, key):
            e = [s[key]['next_q']-s['next_q'] for s in xs]
            return float(np.sqrt(np.mean(np.square(e)))) if e else None
        summary[name] = {
            'departures': len(S), 'returns': len(ret),
            'captures': sum(s['kind'] == 'reference_p3' for s in S),
            'unwrapped_time_correct': sum(s['unwrapped']['updates'] == s['updates'] for s in both),
            'unwrapped_predicted_returns': len(both),
            'unwrapped_coordinate_RMSE_U': rmse(both, 'unwrapped'),
            'unwrapped_max_abs_error_U': float(max((abs(s['unwrapped']['next_q']-s['next_q']) for s in both), default=0)),
            'unwrapped_tau_RMSE_steps': (float(np.sqrt(np.mean([(s['unwrapped']['tau']-s['tau'])**2 for s in both])))
                                         if both else None),
            'unwrapped_tau_max_abs_error_steps': float(max((abs(s['unwrapped']['tau']-s['tau']) for s in both), default=0)),
            'unwrapped_tau_error_quantiles_50_90_99': (np.quantile([abs(s['unwrapped']['tau']-s['tau']) for s in both],
                                                       [.5, .9, .99]).tolist() if both else None),
            'table_routed_returns': len(tab),
            'table_time_correct': sum(s['table']['updates'] == s['updates'] for s in tab),
            'table_coordinate_RMSE_U': rmse(tab, 'table'),
        }
    # Where the 30 table stops and 14 fresh captures lie.
    caps_table = [r['q'] for r in rows if r['kind'] != 'return']
    caps_fresh = [s['q'] for s in scored if s['kind'] == 'reference_p3']
    resolved_q = lambda q: unwrapped_predict(rows, iv, q, step)['kind'] == 'return'
    # Runs of consecutive fresh departures inside resolved intervals (autonomous horizon of the unwrapped model).
    streaks = []
    for i in range(len(result['events'])):
        c = 0
        for s in [s for s in scored if s['trajectory'] == i]:
            if s['unwrapped']['kind'] == 'return' and s['kind'] == 'return' and s['unwrapped']['updates'] == s['updates']:
                c += 1
            else:
                streaks.append(c); c = 0
        streaks.append(c)
    audit = {
        'status': 'post-test descriptive diagnostic; fresh trajectories were seen before this rule was written',
        'bindings': {'frozen_model_sha256': sha(VALID/'frozen-model.json'),
                     'training_manifest_sha256': sha(TRAIN/'manifest.json'),
                     'validation_manifest_sha256': sha(VALID/'manifest.json'),
                     'inputs_sha256': sha(TRAIN/'inputs.npz')},
        'width_U': U, 'line': LINE, 'tau_tolerance_channel_steps': TAU_TOL,
        'table_returns': sum(r['kind'] == 'return' for r in rows),
        'odd_time_table_returns': sum(r['kind'] == 'return' and r['updates'] % 2 for r in rows),
        'q_minus_range': [min(r['q_minus'] for r in rows if r['kind'] == 'return'),
                          max(r['q_minus'] for r in rows if r['kind'] == 'return')],
        'exit_step_fit_max_residual_U': step.residual,
        'inverse_round_trip_max_error_U': float(max(trip)), 'inverse_round_trip_times_exact': trip_t,
        'resolved_intervals': sum(s['resolved'] for s in iv), 'intervals': len(iv),
        'resolved_runs_at_least_0.02_wide': [{'lo': a, 'hi': b, 'intervals': n} for a, b, n in long_runs],
        'resolved_measure': float(sum(s['hi']-s['lo'] for s in iv if s['resolved'])),
        'domain_measure': float(rows[-1]['q']-rows[0]['q']),
        'table_stops_in_resolved_intervals': sum(resolved_q(q) for q in caps_table), 'table_stops': len(caps_table),
        'fresh_captures_from_resolved_intervals': sum(resolved_q(q) for q in caps_fresh), 'fresh_captures': len(caps_fresh),
        'one_cycle': summary,
        'fresh_correct_streaks': {'count': len(streaks), 'max': max(streaks),
                                  'histogram': np.bincount(streaks).tolist()},
        'limits': ('A reset-at-observed-departure score; tau tolerance and interval rule were chosen after the fresh '
                   'trajectories were seen. Confirmation requires new fresh trajectories against a frozen rule.'),
    }
    write_json(out/'table-unwrapped.json', rows)
    write_json(out/'fresh-scored.json', scored)
    write_json(out/'fresh-departures.json', departures)
    write_json(out/'intervals.json', iv)
    write_json(out/'exit-step.json', {'lo': step.lo, 'hi': step.hi, 'coef': step.coef.tolist(),
                                      'max_residual_U': step.residual})
    write_json(out/'audit.json', audit)
    files = []
    for f in sorted(out.glob('*.json')):
        files.append({'path': f.name, 'bytes': f.stat().st_size, 'sha256': sha(f)})
        backup_file(f, out, backup)
    write_json(out/'manifest.json', files); backup_file(out/'manifest.json', out, backup)
    print(json.dumps(audit, indent=1))


if __name__ == '__main__':
    main()
