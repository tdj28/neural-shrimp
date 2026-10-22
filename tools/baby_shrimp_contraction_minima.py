"""BSC5: actual leading-pair-radius section minima near the moving trace saddle.

The smooth trace/determinant jets predict coordinates only. Every searched
objective value comes from a freshly corrected P18 orbit. Section minima are
not declared to be intrinsic ridges or exact intersections.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '8'
import numpy as np
from baby_shrimp_memory_connection_probe import (
    ROOT, DATA, SOURCES as PRIOR_SOURCES, require, digest, save, checked,
    saddle_basis, seed_records, replay_seed)

SOURCES = list(dict.fromkeys(PRIOR_SOURCES + [
    'tools/baby_shrimp_contraction_minima.py', 'tools/contraction_section_minima.py']))


def validate_plan(plan):
    c = plan['contraction_minima']
    require(plan['id'] == 'BSC5-contraction-minima' and plan['period'] == 18,
            'wrong experiment or orbit family')
    require(type(plan['maximum_evaluations']) is int and 1 <= plan['maximum_evaluations'] <= 300,
            'invalid point budget')
    require(c['jet_steps'] == [.001, .0003] and c['refinement_step'] == .0003,
            'undeclared predictor stencil scales')
    require(c['longitudinal_steps'] == [.002, .0006, .0002], 'undeclared inward steps')
    require(c['section_mesh'] == [-.0005, -.00025, -.000125, 0., .000125, .00025, .0005],
            'undeclared section support or mesh')
    require(c['validation_offsets'] == [.0001, .00003] and c['coordinate_tolerance'] == 1e-5,
            'undeclared minimum precision')
    require(c['maximum_refinements'] == 2 and c['maximum_beta_correction'] == 1e-7
            and c['height_tolerance'] == 1e-12 and c['center_correction_tolerance'] == 1e-5,
            'undeclared center/root gate')
    require(c['section_maximum_evaluations'] == 40 and c['radius_floor_minimum'] == 1e-14
            and c['signal_multiple'] == 10 and c['expanded_columns'] == 12,
            'undeclared section/spectral gates')
    return c


def field_jet(values, step):
    a = np.asarray(values, float)
    require(a.shape == (3, 3) and np.isfinite(a).all() and np.isfinite(step) and step > 0,
            'invalid stencil')
    h = float(step)
    g = np.array([(a[1, 2]-a[1, 0])/(2*h), (a[2, 1]-a[0, 1])/(2*h)])
    xy = (a[2, 2]-a[2, 0]-a[0, 2]+a[0, 0])/(4*h*h)
    H = np.array([[(a[1, 2]-2*a[1, 1]+a[1, 0])/h**2, xy],
                  [xy, (a[2, 1]-2*a[1, 1]+a[0, 1])/h**2]])
    return {'value': float(a[1, 1]), 'gradient': g.tolist(), 'hessian': H.tolist()}


def local_jet(ev, center, beta, step):
    center = np.asarray(center, float)
    points = [[checked(ev, center+step*np.array([dx, dy]), beta)
               for dx in (-1, 0, 1)] for dy in (-1, 0, 1)]
    fields = {name: field_jet([[p[name] for p in line] for line in points], step)
              for name in ('tau', 'delta')}
    H, g = np.asarray(fields['tau']['hessian']), np.asarray(fields['tau']['gradient'])
    eigenvalues, _ = saddle_basis(H)
    correction = -np.linalg.solve(H, g)
    return {'center': center.tolist(), 'beta': float(beta), 'step': step,
            'point_indices': [[p['index'] for p in line] for line in points],
            'fields': fields, 'hessian_eigenvalues': eigenvalues.tolist(),
            'correction': correction.tolist(), 'correction_norm': float(np.linalg.norm(correction))}


def predict_field(jet, name, displacement):
    f, q = jet['fields'][name], np.asarray(displacement, float)
    return float(f['value'] + np.asarray(f['gradient']) @ q + .5*q @ np.asarray(f['hessian']) @ q)


def null_arms(hessian):
    values, Q = saddle_basis(hessian)
    # Opposite signed eigenvalues give two null lines, four directed rays.
    a, b = np.sqrt(values[1]), np.sqrt(-values[0])
    arms = []
    for family, sign in enumerate((1., -1.)):
        v = Q @ np.array([a, sign*b]); v /= np.linalg.norm(v)
        for ray in (1., -1.):
            tangent = ray*v
            arms.append({'arm_id': f'line-{family}-ray-{int(ray):+d}',
                         'tangent': tangent.tolist(),
                         'normal': np.array([-tangent[1], tangent[0]]).tolist()})
    return arms


def predictor_derivatives(bundle, seed):
    current = seed['center']
    lower = [r for r in bundle['receipt']['accepted_centers']
             if r.get('kind') == 'continued saddle' and r['parameter'] < current['parameter']]
    require(lower, 'missing lower continuation predictor')
    previous = max(lower, key=lambda r: r['parameter'])
    db = current['parameter']-previous['parameter']
    hp = (current['tau']-previous['tau'])/db
    cp = (np.asarray(current['center'])-previous['center'])/db
    require(np.isfinite(hp) and hp < 0 and np.isfinite(cp).all(), 'invalid measured predictor')
    return {'lower': {k: previous[k] for k in ('parameter', 'center', 'tau')},
            'upper': {k: current[k] for k in ('parameter', 'center', 'tau')},
            'height_secant': float(hp), 'center_secant': cp.tolist(),
            'scope': 'Previously measured neighboring corrected saddles; predictor only, not an exact derivative.'}


def refine_step(jet, derivatives, maximum_beta_correction):
    H, g = np.asarray(jet['fields']['tau']['hessian']), np.asarray(jet['fields']['tau']['gradient'])
    dp = -np.linalg.solve(H, g)
    # Height at the predicted stationary point, before changing beta.
    stationary_height = jet['fields']['tau']['value'] + .5*g @ dp
    db = -stationary_height/derivatives['height_secant']
    require(abs(db) <= maximum_beta_correction and np.linalg.norm(dp) < .001,
            'root predictor left its declared local trust region')
    next_center = np.asarray(jet['center']) + dp + np.asarray(derivatives['center_secant'])*db
    return {'old_center': jet['center'], 'old_beta': jet['beta'],
            'stationary_height_predictor': float(stationary_height),
            'step_center': dp.tolist(), 'step_beta': float(db),
            'new_center': next_center.tolist(), 'new_beta': float(jet['beta']+db)}


def expanded_check(ev, point, columns):
    from baby_shrimp_memory import qualified_ritz
    from baby_shrimp_memory_finish import pair_gap, complex_values
    directory = ev.out/f"point-{point['index']:04d}"
    orbit = np.load(directory/'points.npy', allow_pickle=False)
    orb = ev.cell(ev.origin+ev.scales*np.asarray(point['coordinate']), point['beta'])
    pts = [ev.torch.as_tensor(p.reshape((1,)+orb.shape), dtype=ev.torch.float64,
                             device=ev.loop.device) for p in orbit]
    rows = []
    dest = ev.out/'expanded-spectrum'; dest.mkdir()
    for phase in (0, 6, 12):
        require(time.monotonic()-ev.started < ev.args.seconds, 'budget expired before expanded check')
        rotated = pts[phase:]+pts[:phase]
        q = np.random.default_rng(5801+phase).normal(size=(896, columns))
        t = time.monotonic()
        r = qualified_ritz(lambda v: ev.apply(orb, rotated, v), q)
        vals = r['values']
        row = {'phase': phase, 'columns': columns, 'random_seed': 5801+phase,
               'multipliers': [[float(v.real), float(v.imag)] for v in vals],
               'relative_residuals': r['relative_residuals'].tolist(), 'history': r['history'],
               'tau': float(sum(vals[:2]).real), 'delta': float(np.prod(vals[:2]).real),
               'radius': float(max(abs(vals))), 'seconds': time.monotonic()-t}
        original = next(p for p in point['phases'] if p['phase'] == phase)
        row['pair_relative_difference'] = float(pair_gap(vals[:2], complex_values(original['multipliers'])[:2])/original['radius'])
        np.savez_compressed(dest/f'phase-{phase}.npz', **{k: r[k] for k in
                           ('values', 'vectors', 'direct', 'relative_residuals', 'q', 'w')})
        row['passed'] = bool(row['delta'] < 0 and row['radius'] < 1 and row['pair_relative_difference'] < 1e-5)
        rows.append(row)
        save(dest/'check.json', {'point_index': point['index'], 'rows': rows, 'complete': False})
        require(row['passed'], 'expanded independent subspace disagrees or loses tracked regime')
    variations = {k: float(np.ptp([p[k] for p in point['phases']]+[p[k] for p in rows]))
                  for k in ('tau', 'delta', 'radius')}
    result = {'point_index': point['index'], 'rows': rows, 'variations': variations, 'complete': True,
              'scope': 'Independent random 12-column Ritz replay and cyclic comparisons; not a full-spectrum error bound.'}
    save(dest/'check.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='docs/baby-shrimp-zoom/contraction-minima-plan.json')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--seconds', type=int, default=10800)
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    require(90 < args.seconds <= 10800, 'local runtime must be bounded within three hours')
    started = time.monotonic(); plan_path = ROOT/args.plan
    plan = json.loads(plan_path.read_text()); cfg = validate_plan(plan)
    commit = args.source_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit),
            'exact immutable 40-character source commit required')
    hashes = {p: digest(ROOT/p) for p in SOURCES}
    for path, sha in {**hashes, args.plan: digest(plan_path)}.items():
        require(hashlib.sha256(subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT)).hexdigest() == sha,
                'execution source must be committed: '+path)
    out = args.out.resolve()
    require(out.is_relative_to((DATA/'explore').resolve()) and not out.exists(), 'exclusive external output required')
    out.mkdir(parents=True)
    def emit(stage, **kw):
        row = {'time': datetime.now(timezone.utc).isoformat(), 'stage': stage, **kw}
        with (out/'progress.jsonl').open('a') as f: f.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)
    protocol = {'commit': commit, 'source_sha256': hashes, 'plan': plan,
                'plan_sha256': digest(plan_path), 'args': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                'started_utc': datetime.now(timezone.utc).isoformat(), 'prior_audit': plan['prior_audit']}
    save(out/'protocol.json', protocol)
    result = {'complete': False, 'baseline': None, 'seed_replay': None, 'expanded_spectrum': None,
              'local_jets': [], 'refinement': [], 'sections': [], 'center_qualified': False}
    ev = None
    def checkpoint(): save(out/'probe-progress.json', result)
    try:
        from baby_shrimp_memory_orbit_alignment import load_receipt, METADATA
        from baby_shrimp_saddle_memory import Evaluator
        from baby_shrimp_memory_resume import seed_evaluator
        from contraction_section_minima import search_section_minima
        bound = plan['prior_audit']; path = DATA/bound['path']
        require(path.stat().st_size == bound['bytes'], 'prior audit size mismatch')
        bundle = load_receipt(path, bound['sha256'])
        require(all(plan[k] == bundle['scientific_metadata'][k] for k in METADATA if k != 'id'),
                'map or coordinate metadata changed')
        seed = seed_records(bundle, {'before_beta': .006, 'after_beta': .007})[-1]
        derivatives = predictor_derivatives(bundle, seed)
        protocol['predictor_derivatives'] = derivatives
        save(out/'protocol.json', protocol)
        evaluation_args = argparse.Namespace(**vars(args))
        evaluation_args.seconds = int(args.seconds-(time.monotonic()-started)-90)
        require(evaluation_args.seconds > 0, 'preflight consumed budget')
        ev = Evaluator(evaluation_args, plan, out, emit)
        result['baseline'] = ev.baseline(); checkpoint()
        seed_evaluator(ev, {'warm': [(np.asarray(seed['point']['coordinate']), seed['point']['beta'],
                                     seed['orbit'][0], seed['bases'])]})
        result['seed_replay'] = replay_seed(ev, seed, plan); checkpoint()
        center, beta = np.asarray(seed['point']['coordinate']), seed['point']['beta']
        for step in cfg['jet_steps']:
            result['local_jets'].append(local_jet(ev, center, beta, step)); checkpoint()
            emit('local_jet_complete', step=step, beta=beta, evaluations=ev.attempts)
        for iteration in range(cfg['maximum_refinements']):
            jet = result['local_jets'][-1]
            if abs(jet['fields']['tau']['value']) <= cfg['height_tolerance'] and jet['correction_norm'] <= cfg['center_correction_tolerance']:
                break
            row = refine_step(jet, derivatives, cfg['maximum_beta_correction'])
            center, beta = np.asarray(row['new_center']), row['new_beta']
            p = checked(ev, center, beta)
            row['point_index'] = p['index']; result['refinement'].append(row); checkpoint()
            result['local_jets'].append(local_jet(ev, center, beta, cfg['refinement_step'])); checkpoint()
            emit('refined_center', iteration=iteration, beta=beta, trace=p['tau'], evaluations=ev.attempts)
        jet = result['local_jets'][-1]
        require(abs(jet['fields']['tau']['value']) <= cfg['height_tolerance']
                and jet['correction_norm'] <= cfg['center_correction_tolerance'],
                'local saddle/root remains unresolved; no forced center for minima')
        accepted = checked(ev, center, beta, cyclic=True)
        require(max(abs(p['tau']) for p in accepted['phases']) <= cfg['height_tolerance'],
                'center height tolerance fails in another cyclic phase')
        result['center_point'] = accepted
        result['expanded_spectrum'] = expanded_check(ev, accepted, cfg['expanded_columns'])
        checkpoint()
        require(max(abs(p['tau']) for p in result['expanded_spectrum']['rows']) <= cfg['height_tolerance'],
                'center height tolerance fails in expanded independent subspace')
        radius_floor = max(cfg['radius_floor_minimum'], result['expanded_spectrum']['variations']['radius'])
        require(radius_floor <= 1e-11, 'objective variation too large for declared local minima search')
        result['radius_floor'] = radius_floor
        result['center_qualified'] = True; checkpoint()
        # Refresh all three warm spectral bases at the qualified moving center.
        directory = out/f"point-{accepted['index']:04d}"
        center_orbit = np.load(directory/'points.npy', allow_pickle=False)
        bases = {}
        for phase in (0,6,12):
            with np.load(directory/f'phase-{phase}.npz', allow_pickle=False) as a: bases[phase] = a['q'].copy()
        seed_evaluator(ev, {'warm': [(center, beta, center_orbit[0], bases)]})
        arms = null_arms(jet['fields']['tau']['hessian'])
        result['arms'] = arms; checkpoint()
        # Every arm gets the outer section before any arm advances inward.
        for longitudinal in cfg['longitudinal_steps']:
            for arm in arms:
                tangent, normal = np.asarray(arm['tangent']), np.asarray(arm['normal'])
                origin = center + longitudinal*tangent
                section = {**arm, 'longitudinal': longitudinal, 'center': center.tolist(), 'beta': beta,
                           'origin': origin.tolist(), 'mesh': cfg['section_mesh'], 'search': None,
                           'complete': False, 'events': [], 'predictor_checks': []}
                result['sections'].append(section); checkpoint()
                def objective(s):
                    p = checked(ev, origin+float(s)*normal, beta)
                    orbit = np.load(out/f"point-{p['index']:04d}"/'points.npy', allow_pickle=False)
                    distances = [float(np.max(np.linalg.norm(np.roll(orbit, shift, axis=0)-center_orbit,
                                                               axis=1)/np.linalg.norm(center_orbit, axis=1)))
                                 for shift in range(18)]
                    alignment = {'reference_point_index': accepted['index'],
                                 'best_right_roll': int(np.argmin(distances)),
                                 'all_alignment_maximum_relative_changes': distances}
                    q = np.asarray(p['coordinate'])-center
                    prediction = {k: predict_field(jet, k, q) for k in ('tau','delta')}
                    if prediction['delta'] < 0:
                        prediction['radius'] = float((np.sqrt(prediction['tau']**2-4*prediction['delta'])+abs(prediction['tau']))/2)
                    section['predictor_checks'].append({'s': float(s), 'point_index': p['index'],
                         'predicted': prediction, 'measured': {k: p[k] for k in ('tau','delta','radius')},
                         'orbit_alignment': alignment})
                    checkpoint()
                    require(alignment['best_right_roll'] == 0 and distances[0] < 1e-4,
                            'section sample lost declared full-state neighborhood or cyclic phase')
                    return {'radius': p['radius'], 'index': p['index']}
                def event(row):
                    section['events'].append(row); checkpoint()
                section['search'] = search_section_minima(objective, cfg['section_mesh'],
                    coordinate_tolerance=cfg['coordinate_tolerance'], max_evaluations=cfg['section_maximum_evaluations'],
                    radius_floor=radius_floor, signal_multiple=cfg['signal_multiple'],
                    validation_offsets=cfg['validation_offsets'], record=event)
                section['complete'] = section['search']['complete']
                checkpoint(); emit('section_complete', arm_id=arm['arm_id'], longitudinal=longitudinal,
                                   status=section['search'].get('status'), evaluations=ev.attempts)
                require(section['search']['status'] != 'evaluation_failed',
                        'section objective evaluation failed; preserve partial evidence and stop')
        result.update(complete=all(s['complete'] for s in result['sections']), execution_finished=True,
                      evaluations=ev.attempts, completed_utc=datetime.now(timezone.utc).isoformat(),
                      scope='Exploratory local section minima of the computed leading-pair radius; no full-spectrum, intrinsic-ridge, crossing or exact-superstability certificate.')
        save(out/'result.json', result)
    except BaseException as error:
        save(out/'failure.json', {'type': type(error).__name__, 'error': str(error),
             'evaluations': None if ev is None else ev.attempts, 'time': datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        checkpoint()
        save(out/'manifest.json', {'complete': result['complete'], 'files': [
            {'path': str(p.relative_to(out)), 'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json']})


if __name__ == '__main__':
    main()
