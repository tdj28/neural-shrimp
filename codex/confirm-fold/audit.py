"""Independent offline Task 5 audit; no producer or model modules imported.

Reconstructs scalar predictions, first exits and full-state projections from
hash-bound JSON/NPZ files. Integrity and scientific success are separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

EPS = (3e-6, 1e-5, 3e-5, 1e-4)
ENTRANCES = ('full', 'midpoint', 'full+transverse')
REPO = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def scalar_prediction(u0, a, b, epsilon, upper, cap=100000):
    vals = (u0, a, b, epsilon, upper)
    if not all(math.isfinite(float(x)) for x in vals) or min(a, b, epsilon, upper) <= 0 or u0 > upper:
        raise ValueError('invalid scalar inputs')
    u = float(u0)
    count = None
    for n in range(1, cap + 1):
        u += a * epsilon + b * u * u
        if u > upper:
            count = n
            break
    c = math.sqrt(b / (a * epsilon))
    integral = (math.atan(upper*c) - math.atan(u0*c)) / math.sqrt(a*b*epsilon)
    return count, integral


def first_exit(trace, upper):
    """Post-G trace: strict first exit, no initial sample in this array."""
    u = np.asarray(trace, dtype=np.float64)
    if u.ndim != 1:
        raise ValueError('one-dimensional coordinate trace required')
    hits = np.flatnonzero(u > upper)
    count = int(hits[0]) + 1 if len(hits) else None
    n = len(u) if count is None else count
    finite_prefix = bool(np.isfinite(u[:n]).all())
    empty_tail = bool(count is None or np.isnan(u[n:]).all())
    return dict(count=count, recorded=n, finite_prefix=finite_prefix,
                nan_after_first_exit=empty_tail, passed=finite_prefix and empty_tail)


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, value, **details):
        self.rows.append(dict(check=name, passed=bool(value), **details))
        return bool(value)

    def error(self, name, error):
        self.add(name, False, error=str(error))

    @property
    def passed(self):
        return bool(self.rows) and all(x['passed'] for x in self.rows)


def checked_artifact(record, checks, label):
    path = Path(record['path'])
    try:
        ok = checks.add(label, path.is_file() and digest(path) == record['sha256']
                        and path.stat().st_size == record['bytes'], path=str(path))
    except (OSError, KeyError, TypeError) as e:
        checks.error(label, e)
        return None
    return path if ok else None


def close(x, y, *, atol=1e-12, rtol=1e-10):
    return bool(np.isclose(x, y, atol=atol, rtol=rtol))


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def audit_target(target, result_dir=None):
    c = Checks()
    label = target['label']
    rows = target.get('rows', [])
    expected_ids = [f'{label}-{i}-{e}' for i in range(4) for e in ENTRANCES]
    c.add('exact_planned_trial_ids', [r.get('id') for r in rows] == expected_ids)
    c.add('target_identity', label in ('A', 'B') and target['period'] == (2 if label == 'A' else 8)
          and target['coordinate'] == ('alpha' if label == 'A' else 'log10T'))
    setup_path = checked_artifact(target['setup'], c, 'setup_metadata_hash')
    report = dict(label=label, checks=c.rows, rows=[], primary_all_pass=False, arctan_all_pass=False)
    def fill_missing(status):
        available = {row['id']: row for row in report['rows']}
        report['rows'] = [available.get(row['id'], dict(id=row['id'], status=status, primary_pass=False, arctan_pass=False)) for row in rows]

    if target['status'] != 'ready':
        c.add('blocked_target_explicit', target['status'] == 'setup_gate_failed')
        report.update(integrity_pass=c.passed, status='setup_gate_failed',
                      rows=[dict(id=r['id'], status='not_run_setup_gate', primary_pass=False,
                                 arctan_pass=False) for r in rows])
        return report
    setup_arrays = checked_artifact(target['setup_arrays'], c, 'setup_arrays_hash')
    input_path = checked_artifact(target['inputs'], c, 'frozen_inputs_hash')
    if not all((setup_path, setup_arrays, input_path)):
        report.update(integrity_pass=False, status='input_hash_failure')
        fill_missing('not_audited_input_hash_failure')
        return report
    try:
        setup = json.loads(setup_path.read_text())
        arrays = load_npz(input_path)
        source = load_npz(setup_arrays)
        x, r, l, w, starts, M = (arrays[k] for k in ('x_c', 'r', 'l', 'w', 'starts', 'matrix'))
        dim = len(x)
        c.add('finite_float64_inputs', all(v.dtype == np.float64 and np.isfinite(v).all() for v in arrays.values()))
        c.add('input_dimensions', x.shape == r.shape == l.shape == w.shape == (dim,)
              and starts.shape == (3, dim) and M.shape == (dim, dim))
        c.add('setup_success', setup.get('passed') is True and bool(setup.get('gates')) and all(setup['gates'].values()))
        c.add('setup_arrays_match', all(np.array_equal(arrays[k], source[k]) for k in ('x_c', 'r', 'l', 'matrix')))
        raw_r, raw_l = source['raw_r'], source['raw_l']
        ad_a = float(raw_l @ source['parameter_derivative'])
        ad_b = float(.5 * raw_l @ source['directional_second'])
        c.add('saved_ad_vector_contractions', close(ad_a, setup['raw_a'], atol=1e-12, rtol=1e-12)
              and close(ad_b, setup['raw_b'], atol=1e-12, rtol=1e-12))
        c.add('orientation_from_saved_raw_modes', np.array_equal(r, np.sign(setup['raw_b'])*raw_r)
              and np.array_equal(l, np.sign(setup['raw_b'])*raw_l))
        c.add('critical_mode_residuals', np.linalg.norm(M @ r-r) < 1e-8
              and np.linalg.norm(M.T @ l-l)/max(np.linalg.norm(l),np.finfo(float).tiny) < 1e-8)
        a, b, U = target['a_positive'], target['b_positive'], target['U']
        c.add('ad_coefficients_match', a == abs(setup['raw_a']) == abs(target['raw_a'])
              and b == abs(setup['raw_b']) == abs(target['raw_b']) and a > 0 and b > 0)
        c.add('width_from_b', U == .1/b)
        side = int(np.sign(target['raw_a'] * target['raw_b']))
        c.add('ghost_side', side in (-1, 1) and side == target['ghost_direction'] == setup['ghost_parameter_direction'])
        c.add('parameter_center', target['parameter_c'] == setup['parameter_c'])
        c.add('fixed_coordinate', target['log10T'] == -.140 if label == 'A' else target['alpha'] == .0717)
        c.add('mode_normalization', abs(np.linalg.norm(r)-1) <= 1e-8 and abs(l @ r-1) <= 1e-8)
        c.add('seed_identity', target['seed'] == (10501 if label == 'A' else 10502))
        z = np.random.default_rng(target['seed']).normal(size=dim)
        z -= r * (l @ z)
        z /= np.linalg.norm(z)
        c.add('transverse_rng_reconstruction', np.array_equal(w, z))
        c.add('transverse_normalization', close(np.linalg.norm(w), 1.) and abs(l @ w) < 1e-8)
        expected_starts = np.stack((x-U*r, x, x-U*r+U*w))
        c.add('entrance_states', np.array_equal(starts, expected_starts))
        projection_tol = 128*np.finfo(float).eps*max(1., np.linalg.norm(l))*(1+np.linalg.norm(x)+np.linalg.norm(starts, axis=1).max())
        c.add('initial_scalar_coordinates', np.allclose((starts-x) @ l, [-U, 0., -U], rtol=0, atol=projection_tol))
        spectrum = np.linalg.eigvals(M)
        k = int(np.argmin(abs(spectrum-1)))
        complement = np.delete(spectrum, k)
        c.add('reconstructed_clean_fold', abs(spectrum[k].real-1) <= 1e-8 and abs(spectrum[k].imag) <= 1e-10
              and (len(complement) == 0 or np.max(abs(complement)) < .95))
        predictions = {}
        for i, row in enumerate(rows):
            group, j = divmod(i, 3)
            v0 = 0. if j == 1 else -U
            c.add(row['id']+'/identity', row['group'] == group and row['entrance_index'] == j
                  and row['eps'] == EPS[group] and row['entrance'] == ENTRANCES[j]
                  and row['u_in'] == v0 and row['u_out'] == U and row['target'] == label)
            c.add(row['id']+'/parameter', row['parameter'] == target['parameter_c'] + side*EPS[group])
            discrete, arc = scalar_prediction(v0, a, b, row['eps'], U)
            predictions[row['id']] = (discrete, arc)
            c.add(row['id']+'/frozen_prediction', discrete == row['quadratic_G']
                  and close(arc, row['arctan_G'], atol=1e-10, rtol=1e-12))
        c.add('frozen_iteration_cap', target['max_g'] == max(512, 4*max(v[0] for v in predictions.values()))
              and target['max_g'] <= 4096)
    except (KeyError, ValueError, TypeError, OSError, IndexError, ZeroDivisionError, np.linalg.LinAlgError) as e:
        c.error('input_structure', e)
        report.update(integrity_pass=False, status='input_structure_failure')
        fill_missing('not_audited_input_structure_failure')
        return report
    if result_dir is None:
        c.add('runtime_present', False)
        report.update(integrity_pass=False, status='not_run',
                      rows=[dict(id=row['id'], status='not_run', primary_pass=False, arctan_pass=False) for row in rows])
        return report
    folder = Path(result_dir)
    try:
        result = json.loads((folder/'results.json').read_text())
        report['runtime'] = {k: result.get(k) for k in ('status', 'freeze_commit', 'plan_sha256', 'started_utc',
                              'freeze_pushed_at_utc', 'first_target_outcome_at_utc')}
        c.add('runtime_target', result['target'] == label)
        c.add('runtime_complete', result['status'] == 'completed')
        c.add('runtime_trial_ids', [row.get('id') for row in result['rows']] == expected_ids)
        runtime_rows = {row['id']: row for row in result['rows']}
        c.add('no_duplicate_runtime_trials', len(runtime_rows) == len(result['rows']) == 12)
        groups = result.get('groups', [])
        c.add('complete_group_ids', [g['index'] for g in groups] == list(range(4)))
        group_map = {g['index']: g for g in groups}
        c.add('exact_trace_inventory', {f.name for f in folder.glob('group-*.npz')} == {f'group-{g}.npz' for g in range(4)})
    except (OSError, KeyError, ValueError, TypeError) as e:
        c.error('runtime_structure', e)
        report.update(integrity_pass=False, status='runtime_structure_failure')
        fill_missing('not_audited_runtime_structure_failure')
        return report
    for group_index in range(4):
        try:
            group = group_map[group_index]
            trace_path = folder/f'group-{group_index}.npz'
            c.add(f'group-{group_index}/hash', digest(trace_path) == group['trace_sha256'])
            trace = load_npz(trace_path)
            u, tr = trace['u'], trace['transverse']
            state_trace = trace['state_trace']
            c.add(f'group-{group_index}/shape', u.shape == tr.shape == (group['G_evaluations'], 3)
                  and state_trace.shape == (len(u), 3, dim) and 0 < len(u) <= target['max_g'])
            c.add(f'group-{group_index}/starts', np.array_equal(trace['initial_states'], starts))
            parameter = target['parameter_c'] + side*EPS[group_index]
            c.add(f'group-{group_index}/physical_parameters',
                  group['alpha'] == (parameter if label == 'A' else target['alpha'])
                  and group['log10T'] == (parameter if label == 'B' else target['log10T'])
                  and group['batch_model_steps'] == target['period']*len(u))
            counts = []
            for j in range(3):
                planned = rows[group_index*3+j]
                rid = planned['id']
                actual = runtime_rows[rid]
                c.add(rid+'/immutable_row_fields', all(actual.get(k) == v for k, v in planned.items()))
                observed = first_exit(u[:, j], U)
                c.add(rid+'/finite_trace_and_tail', observed['passed'])
                n = observed['count']
                counts.append(n)
                valid_tr = np.isfinite(tr[:observed['recorded'], j]).all() and (tr[:observed['recorded'], j] >= 0).all()
                valid_tr = valid_tr and (n is None or np.isnan(tr[n:, j]).all())
                c.add(rid+'/transverse_trace', valid_tr)
                saved_states = state_trace[:observed['recorded'], j]
                finite_states = np.isfinite(saved_states).all()
                empty_state_tail = n is None or np.isnan(state_trace[n:, j]).all()
                c.add(rid+'/full_trace_finiteness', finite_states and empty_state_tail)
                projected = (saved_states-x) @ l
                distances = np.linalg.norm(saved_states-x-projected[:, None]*r, axis=1)
                c.add(rid+'/every_state_projection', np.allclose(projected, u[:observed['recorded'], j], atol=projection_tol, rtol=0))
                c.add(rid+'/every_transverse_norm', np.allclose(distances, tr[:observed['recorded'], j], atol=projection_tol, rtol=1e-10))
                c.add(rid+'/count_and_clock', actual.get('measured_G') == n
                      and actual.get('measured_F') == (None if n is None else n*target['period'])
                      and actual.get('recorded_G') == observed['recorded'])
                c.add(rid+'/status', actual.get('status') == ('censored' if n is None else 'observed'))
                if n is None:
                    c.add(rid+'/censoring_at_cap', len(u) == target['max_g'])
                    c.add(rid+'/censored_state_fields', np.isnan(trace['previous_states'][j]).all()
                          and np.isnan(trace['crossing_states'][j]).all()
                          and np.array_equal(trace['states_at_stop'][j], saved_states[-1]))
                    margins = dict(pre_exit_margin=None, crossing_margin=None)
                else:
                    previous, crossing = trace['previous_states'][j], trace['crossing_states'][j]
                    c.add(rid+'/finite_crossing_states', np.isfinite(previous).all() and np.isfinite(crossing).all())
                    up, uc = float((previous-x) @ l), float((crossing-x) @ l)
                    stored_previous = float((starts[j]-x) @ l) if n == 1 else u[n-2, j]
                    c.add(rid+'/crossing_state_projections', abs(up-stored_previous) <= projection_tol
                          and abs(uc-u[n-1, j]) <= projection_tol)
                    c.add(rid+'/strict_full_state_exit', up <= U and uc > U)
                    c.add(rid+'/bracketing_states_in_trace', np.array_equal(previous, starts[j] if n == 1 else state_trace[n-2,j])
                          and np.array_equal(crossing, state_trace[n-1,j]))
                    transverse_at_exit = np.linalg.norm(crossing-x-uc*r)
                    c.add(rid+'/crossing_transverse_norm', close(transverse_at_exit, tr[n-1, j], atol=projection_tol))
                    margins = dict(pre_exit_margin=float(U-stored_previous), crossing_margin=float(u[n-1,j]-U),
                                   projection_comparison_tolerance=float(projection_tol),
                                   margins_above_projection_tolerance=bool(min(U-stored_previous,u[n-1,j]-U)>projection_tol))
                nq, na = predictions[rid]
                de = None if n is None else n-nq
                ae = None if n is None else n-na
                passed = n is not None and abs(de) <= 1
                apassed = n is not None and abs(ae) <= 1
                c.add(rid+'/errors_and_verdicts', actual.get('quadratic_error') == de
                      and (actual.get('arctan_error') is None if ae is None else close(actual.get('arctan_error'), ae))
                      and actual.get('primary_pass') is passed and actual.get('arctan_pass') is apassed)
                report['rows'].append(dict(id=rid, measured_G=n, quadratic_G=nq, arctan_G=na,
                                           quadratic_error=de, arctan_error=ae, primary_pass=passed,
                                           arctan_pass=apassed, max_transverse=float(np.max(tr[:observed['recorded'], j])), **margins))
            c.add(f'group-{group_index}/stop_rule', len(u) == (target['max_g'] if any(n is None for n in counts) else max(counts)))
        except (OSError, KeyError, ValueError, TypeError, IndexError) as e:
            c.error(f'group-{group_index}/reconstruction', e)
    fill_missing('missing_or_invalid_trace')
    report['primary_all_pass'] = len(report['rows']) == 12 and all(row['primary_pass'] for row in report['rows'])
    report['arctan_all_pass'] = len(report['rows']) == 12 and all(row['arctan_pass'] for row in report['rows'])
    c.add('aggregate_verdicts', result.get('primary_all_pass') is report['primary_all_pass']
          and result.get('arctan_all_pass') is report['arctan_all_pass'])
    report.update(integrity_pass=c.passed, status=result.get('status'), checks=c.rows)
    return report


def audit_study(plan_path, result_dirs, *, repo=REPO, expected_freeze=None):
    plan_path, repo = Path(plan_path), Path(repo)
    plan = json.loads(plan_path.read_text())
    c = Checks()
    c.add('schema', plan['schema'] == 1)
    c.add('target_inventory', [t['label'] for t in plan['targets']] == ['A', 'B'])
    c.add('execution_contract', plan['execution']['epsilon_order'] == list(EPS)
          and plan['execution']['entrances'] == list(ENTRANCES)
          and plan['execution']['device'] == 'cpu' and plan['execution']['dtype'] == 'float64')
    for source in plan['source']:
        path = repo/source['path']
        try:
            c.add('source/'+source['path'], path.resolve().is_relative_to(repo.resolve())
                  and digest(path) == source['sha256'])
        except OSError as e:
            c.error('source/'+source['path'], e)
    targets = [audit_target(t, result_dirs.get(t['label'])) for t in plan['targets']]
    freezes = set()
    for t in targets:
        runtime = t.get('runtime')
        if runtime is None:
            continue
        prefix = t['label']+'/'
        c.add(prefix+'plan_hash', runtime.get('plan_sha256') == digest(plan_path))
        freeze = runtime.get('freeze_commit')
        freezes.add(freeze)
        c.add(prefix+'freeze_hash', isinstance(freeze, str) and len(freeze) == 40
              and all(ch in '0123456789abcdef' for ch in freeze)
              and (expected_freeze is None or freeze == expected_freeze))
        try:
            pushed = datetime.fromisoformat(runtime['freeze_pushed_at_utc'])
            started = datetime.fromisoformat(runtime['started_utc'])
            first = datetime.fromisoformat(runtime['first_target_outcome_at_utc'])
            created = datetime.fromisoformat(plan['created_utc'])
            c.add(prefix+'prospective_timestamps', all(t.tzinfo is not None for t in (pushed,started,first,created))
                  and created <= pushed < started <= first)
            committed = subprocess.check_output(['git','show',f'{freeze}:codex/confirm-fold/design.json'],cwd=repo,stderr=subprocess.PIPE)
            c.add(prefix+'plan_in_freeze_tree', hashlib.sha256(committed).hexdigest() == digest(plan_path))
            for source in plan['source']:
                content = subprocess.check_output(['git','show',f"{freeze}:{source['path']}"],cwd=repo,stderr=subprocess.PIPE)
                c.add(prefix+'frozen_source/'+source['path'], hashlib.sha256(content).hexdigest() == source['sha256'])
        except (TypeError, ValueError, KeyError, subprocess.CalledProcessError, OSError) as e:
            c.error(prefix+'freeze_binding', e)
    c.add('same_freeze_all_runtime', len(freezes) <= 1)
    integrity = c.passed and all(t['integrity_pass'] for t in targets)
    return dict(schema=1, scope='Independent saved-data and freeze-tree audit; no model replay or mathematical certificate. Push timing uses retained runtime receipts, not an independent public timestamp.',
                plan_sha256=digest(plan_path), checks=c.rows, targets=targets, integrity_pass=integrity,
                primary_all_pass=integrity and all(t['primary_all_pass'] for t in targets),
                arctan_all_pass=integrity and all(t['arctan_all_pass'] for t in targets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--result', action='append', default=[], help='A=/path/to/run or B=/path/to/run')
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--freeze')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    locations = {}
    for item in args.result:
        label, path = item.split('=', 1)
        if label not in ('A','B') or label in locations:
            raise ValueError('unique A/B result locations required')
        locations[label] = path
    report = audit_study(args.design, locations, repo=args.repo, expected_freeze=args.freeze)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('integrity_pass','primary_all_pass','arctan_all_pass')}))
    if not report['integrity_pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
