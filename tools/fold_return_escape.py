"""FGR4 zoom analysis (exploratory) and the frozen FGR5 escape-rate test.

zoom     : uncertainty fraction p(delta) of the departure-to-departure outcome for bases in the unresolved interval M,
           the exponent alpha in p ~ delta^alpha, the exact-replicate check, and the measure of terminal slivers.
predict  : FROZEN. A transfer-operator (Ulam) estimate of the per-departure termination probability from the FGR4
           uniform bases alone; writes prediction.json. No fresh-trajectory data are read.
score    : FROZEN. Scores the FGR5 fresh trajectories against a prediction whose hash is committed in
           docs/fold-return-fgr5-prediction.json, plus the pointwise one-cycle test on the two long resolved runs.

Frozen rules (docs/fold-return-fgr5-plan.json): 28 equal bins on [3, 4.12]; termination = verified period-3
approach or other recurrence candidate; censored bases excluded and counted; 2000 bootstrap resamples within bins.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import backup_file, sha, write_json
import fold_return_unwrap as uw

DOMAIN = (3.0, 4.12)
BINS = 28
BOOT = 2000
BOOT_SEED = 20261114
TERMINAL = ('reference_p3', 'other_recurrence_candidate')
LONG_RUNS = ((3.0, 3.0503125000000004), (3.56, 4.12))   # from the FGR3 table (fold_return_unwrap audit)
C3 = {'time_correct_fraction': 0.97, 'median_tau_error': 0.002, 'p99_tau_error': 0.05}


def load_zoom(parts):
    records = []
    for d in parts:
        records += [{**r, 'run': str(d)} for r in json.loads((Path(d)/'records.json').read_text())]
    return records


def bin_of(q, bins=BINS, domain=DOMAIN):
    lo, hi = domain
    return int(np.clip(np.floor((q-lo)/(hi-lo)*bins), 0, bins-1))


def ulam(points, bins=BINS, domain=DOMAIN):
    """points: (q, next_q or None for termination). Returns the substochastic matrix and termination vector."""
    P = np.zeros((bins, bins)); term = np.zeros(bins); n = np.zeros(bins)
    for q, nq in points:
        i = bin_of(q, bins, domain); n[i] += 1
        if nq is None:
            term[i] += 1
        else:
            P[i, bin_of(nq, bins, domain)] += 1
    if np.any(n == 0):
        raise ValueError('Empty Ulam bin; the uniform design must cover every bin')
    P /= n[:, None]; term /= n
    w, V = np.linalg.eig(P.T)
    k = int(np.argmax(w.real)); lam = float(w[k].real)
    nu = np.abs(V[:, k].real); nu /= nu.sum()
    return {'lambda': lam, 'kappa': 1-lam, 'qsd': nu.tolist(), 'termination_by_bin': term.tolist(),
            'counts': n.astype(int).tolist(), 'qsd_weighted_termination': float(nu@term)}


def base_points(records):
    pts, censored = [], 0
    for r in records:
        if r['role'] != 'base':
            continue
        if r['kind'] == 'return':
            pts.append((r['q'], r['next_q']))
        elif r['kind'] in TERMINAL:
            pts.append((r['q'], None))
        else:
            censored += 1
    return pts, censored


def bootstrap(pts, bins=BINS, reps=BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    by_bin = {}
    for p in pts:
        by_bin.setdefault(bin_of(p[0], bins), []).append(p)
    out = []
    for _ in range(reps):
        sample = []
        for b, ps in by_bin.items():
            idx = rng.integers(0, len(ps), len(ps))
            sample += [ps[i] for i in idx]
        out.append(ulam(sample, bins)['kappa'])
    return np.quantile(out, [.025, .5, .975]).tolist()


def garwood(c, conf=.95):
    a = (1-conf)/2
    lo = 0. if c == 0 else chi2.ppf(a, 2*c)/2
    return float(lo), float(chi2.ppf(1-a, 2*c+2)/2)


def outcome_key(r, criterion):
    if r['kind'] != 'return':
        return ('end', r['kind'])
    return ('return', r['updates']) if criterion == 'time' else ('return', r['updates'] % 2, r['tau'])


def different(a, b, criterion):
    ka, kb = outcome_key(a, criterion), outcome_key(b, criterion)
    if criterion == 'time' or ka[0] != 'return' or kb[0] != 'return':
        return ka != kb
    return ka[1] != kb[1] or abs(ka[2]-kb[2]) >= uw.TAU_TOL


def add_tau(records, U, centre, left):
    cache = {}
    for r in records:
        if r['kind'] != 'return':
            continue
        S = cache.setdefault((r['run'], r['batch']), uw.States(Path(r['run'])/r['batch']))
        row = r['batch_row']
        qp = uw.q_of(S.at(r['updates'])[row], centre, left, U)
        if abs(qp-r['next_q']) > 1e-9:
            raise ValueError('Saved zoom state does not reproduce the departure')
        r['q_minus'] = uw.q_of(S.at(r['updates']-2)[row], centre, left, U)
        r['tau'], r['f'] = uw.unwrap(r['updates'], r['next_q'], r['q_minus'])


def uncertainty(records, deltas, criterion, reps=BOOT, seed=BOOT_SEED):
    base = {r['base']: r for r in records if r['role'] == 'base'}
    rows = []
    for d in deltas:
        pairs = [(base[r['base']], r) for r in records if r['role'] == 'partner' and r['delta'] == d]
        if not pairs:
            continue
        pairs = [(a, b) for a, b in pairs if a['kind'] in TERMINAL+('return',) and b['kind'] in TERMINAL+('return',)]
        flags = np.array([different(a, b, criterion) for a, b in pairs])
        rows.append({'delta': d, 'pairs': len(pairs), 'uncertain': int(flags.sum()),
                     'fraction': float(flags.mean()) if len(flags) else None, 'flags': flags})
    def fit(rows):
        use = [r for r in rows if 0 < r['delta'] <= 1e-4 and r['uncertain'] >= 10]
        if len(use) < 3:
            return None
        x = np.log10([r['delta'] for r in use]); y = np.log10([r['fraction'] for r in use])
        return float(np.polyfit(x, y, 1)[0])
    alpha = fit(rows)
    rng = np.random.default_rng(seed); boot = []
    for _ in range(reps):
        res = []
        for r in rows:
            f = r['flags'][rng.integers(0, len(r['flags']), len(r['flags']))]
            res.append({**r, 'uncertain': int(f.sum()), 'fraction': float(f.mean()) if f.sum() else 1e-300})
        a = fit(res)
        if a is not None:
            boot.append(a)
    for r in rows:
        r.pop('flags')
    return {'criterion': criterion, 'levels': rows, 'alpha': alpha,
            'alpha_bootstrap_95': np.quantile(boot, [.025, .975]).tolist() if len(boot) > reps//2 else None,
            'bootstrap_fits': len(boot),
            'fit_rule': 'least squares of log10 fraction on log10 delta over levels with delta <= 1e-4 and >= 10 uncertain pairs'}


def score_runs(result, pairs):
    """Per-departure termination counts: every departure whose next outcome was observed."""
    D = sum(1 for p in pairs if p['kind'] in ('return',)+TERMINAL)
    C = sum(1 for p in pairs if p['kind'] in TERMINAL)
    no_departure = sum(1 for ev, o in zip(result['events'], result['outcomes']) if not ev and o['kind'] in TERMINAL)
    return D, C, no_departure


def committed_prediction():
    receipt = ROOT/'docs/fold-return-fgr5-prediction.json'
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    committed = subprocess.check_output(['git', 'show', f'{head}:docs/fold-return-fgr5-prediction.json'], cwd=ROOT)
    if hashlib.sha256(committed).hexdigest() != sha(receipt):
        raise ValueError('Prediction receipt is not committed unchanged')
    return json.loads(receipt.read_text())


def save_all(out, backup, files):
    out, backup = Path(out), Path(backup)
    out.mkdir(parents=True, exist_ok=False); backup.mkdir(parents=True, exist_ok=False)
    manifest = []
    for name, obj in files.items():
        write_json(out/name, obj); backup_file(out/name, out, backup)
        manifest.append({'path': name, 'bytes': (out/name).stat().st_size, 'sha256': sha(out/name)})
    write_json(out/'manifest.json', manifest); backup_file(out/'manifest.json', out, backup)


def geometry():
    parent = Path(json.loads((uw.TRAIN/'plan.json').read_text())['parent_curve'])
    U = json.loads((parent/'plan.json').read_text())['width']
    centre, left, _ = uw.projector(uw.TRAIN/'inputs.npz')
    return U, centre, left


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['zoom', 'predict', 'score'])
    p.add_argument('--zoom', nargs='+', required=True, help='FGR4 part run directories')
    p.add_argument('--fresh', help='FGR5 run directory (score only)')
    p.add_argument('--extra', nargs='*', default=[], help='zoom extension runs (zoom analysis only)')
    p.add_argument('--out', required=True); p.add_argument('--backup', required=True)
    args = p.parse_args()
    records = load_zoom(args.zoom)
    plans = [json.loads((Path(d)/'plan.json').read_text()) for d in args.zoom]
    if any(pl != plans[0] for pl in plans):
        raise ValueError('Zoom parts disagree on the plan')
    bindings = {str(Path(d)/'manifest.json'): sha(Path(d)/'manifest.json') for d in args.zoom}
    if args.mode == 'predict':
        pts, censored = base_points(records)
        est = ulam(pts)
        pred = {'status': 'frozen FGR5 prediction from FGR4 uniform bases only; no fresh data read',
                'bindings': bindings, 'bins': BINS, 'domain': DOMAIN, 'bases_used': len(pts),
                'bases_censored': censored, 'kappa': est['kappa'], 'kappa_bootstrap_95': bootstrap(pts),
                'sensitivity_bins': {b: ulam(pts, b)['kappa'] for b in (14, 56)},
                'termination_by_bin': est['termination_by_bin'], 'qsd': est['qsd'], 'counts': est['counts'],
                'flag_censored_over_1_percent': censored > 0.01*(len(pts)+censored)}
        save_all(args.out, args.backup, {'prediction.json': pred})
        print(json.dumps({k: v for k, v in pred.items() if k not in ('qsd', 'termination_by_bin', 'counts')}, indent=1))
    elif args.mode == 'zoom':
        U, centre, left = geometry()
        spec = plans[0]['zoom']
        deltas = list(spec['deltas'])
        for d in args.extra:
            ext = json.loads((Path(d)/'plan.json').read_text())['zoom']
            if any(ext[k] != spec[k] for k in ('domain', 'bases', 'seed', 'unresolved_interval', 'max_updates')):
                raise ValueError('Extension run does not share the parent bases')
            deltas += [x for x in ext['deltas'] if x not in deltas]
            bindings[str(Path(d)/'manifest.json')] = sha(Path(d)/'manifest.json')
        records += load_zoom(args.extra)
        deltas = sorted(deltas, reverse=True)
        add_tau(records, U, centre, left)
        base = {r['base']: r for r in records if r['role'] == 'base'}
        reps = [r for r in records if r['role'] == 'replicate']
        same = [all(r.get(k) == base[r['base']].get(k) for k in ('kind', 'updates', 'next_q')) for r in reps]
        m_lo, m_hi = spec['unresolved_interval']
        inside = [r for r in base.values() if m_lo <= r['q'] < m_hi]
        everywhere = list(base.values())
        pts, censored = base_points(records)
        audit = {'status': 'exploratory FGR4 zoom analysis', 'bindings': bindings,
                 'evaluations': len(records), 'replicates': len(reps), 'replicates_identical': sum(same),
                 'terminal_fraction_inside_M': sum(r['kind'] in TERMINAL for r in inside)/len(inside),
                 'terminal_fraction_whole_domain': sum(r['kind'] in TERMINAL for r in everywhere)/len(everywhere),
                 'bases_inside_M': len(inside), 'bases': len(everywhere), 'censored_bases': censored,
                 'outcome_kinds': {k: sum(r['kind'] == k for r in records) for k in {r['kind'] for r in records}},
                 'uncertainty_time': uncertainty(records, deltas, 'time'),
                 'uncertainty_tau': uncertainty(records, deltas, 'tau')}
        save_all(args.out, args.backup, {'audit.json': audit, 'records-with-tau.json': records})
        print(json.dumps(audit, indent=1))
    else:
        pred = committed_prediction()
        if pred['bindings'] != bindings:
            raise ValueError('Prediction was made from different zoom data')
        U, centre, left = geometry()
        result, pairs, departures = uw.fresh_pairs(U, centre, left, Path(args.fresh))
        D, C, no_dep = score_runs(result, pairs)
        lo, hi = garwood(C)
        kappa_ci = [lo/D, hi/D]
        # Calibration: expected terminations from the per-bin termination fractions at the observed departures.
        E = float(sum(pred['termination_by_bin'][bin_of(pq['q'])] for pq in pairs
                      if pq['kind'] in ('return',)+TERMINAL))
        # Pointwise one-cycle test on the two long resolved runs, unwrapped FGR3 table, reset at each departure.
        model, rows = uw.table_rows(U, centre, left)
        iv = uw.intervals(rows); step = uw.ExitStep(rows)
        long = [pq for pq in pairs if pq['kind'] == 'return' and any(a <= pq['q'] <= b for a, b in LONG_RUNS)]
        errs, correct, abstained = [], 0, 0
        for pq in long:
            pr = uw.unwrapped_predict(rows, iv, pq['q'], step)
            if pr['kind'] != 'return':
                abstained += 1; continue
            correct += pr['updates'] == pq['updates']; errs.append(abs(pr['tau']-pq['tau']))
        errs = np.array(errs)
        c3 = {'pairs': len(long), 'abstained': abstained, 'time_correct': correct,
              'time_correct_fraction': correct/len(long) if long else None,
              'median_tau_error': float(np.median(errs)) if len(errs) else None,
              'p99_tau_error': float(np.quantile(errs, .99)) if len(errs) else None}
        c3['passed'] = bool(long) and c3['time_correct_fraction'] >= C3['time_correct_fraction'] and \
            c3['median_tau_error'] <= C3['median_tau_error'] and c3['p99_tau_error'] <= C3['p99_tau_error']
        score = {'status': 'frozen FGR5 score', 'prediction_sha256': sha(ROOT/'docs/fold-return-fgr5-prediction.json'),
                 'fresh_manifest_sha256': sha(Path(args.fresh)/'manifest.json'),
                 'trajectories': len(result['outcomes']),
                 'outcomes': {k: sum(o['kind'] == k for o in result['outcomes'])
                              for k in {o['kind'] for o in result['outcomes']}},
                 'departures_with_observed_next_outcome': D, 'terminations_after_a_departure': C,
                 'terminations_before_any_departure': no_dep,
                 'kappa_observed': C/D, 'kappa_observed_95': kappa_ci, 'kappa_predicted': pred['kappa'],
                 'C1_escape_rate_passed': kappa_ci[0] <= pred['kappa'] <= kappa_ci[1],
                 'expected_terminations_calibration': E, 'observed_terminations_95': [lo, hi],
                 'C2_calibration_passed': lo <= E <= hi,
                 'C3_pointwise_long_runs': c3}
        save_all(args.out, args.backup, {'score.json': score, 'fresh-pairs.json': pairs})
        print(json.dumps(score, indent=1))


if __name__ == '__main__':
    main()
