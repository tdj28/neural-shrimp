"""Exact tangent growth on FGR3's saved fresh trajectories; CPU only.

No frozen post-capture padding contributes. Windows are fixed in the plan,
not chosen for their observed growth. Capture conditioning is explicit.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import sha, write_json, backup_file
from fold_return_followup import Engine, Evidence
from audit_fold_return_followup import verify_manifest, load_batch


def summarize_window(logs, eligible, alignment, planned_length=None):
    if planned_length is None: planned_length = len(logs)
    records = []
    for row in range(logs.shape[1]):
        n = min(int(eligible[row]), len(logs))
        measured = logs[alignment:n, row]
        half = len(measured)//2
        records.append({'row': row, 'eligible_updates': n, 'alignment_updates': min(n, alignment),
                        'measured_updates': len(measured),
                        'mean': float(np.mean(measured)) if len(measured) else None,
                        'first_half': float(np.mean(measured[:half])) if half else None,
                        'second_half': float(np.mean(measured[half:])) if half else None,
                        'full_planned_window': n == planned_length})
    return records


def execute(args):
    plan = json.loads(Path(args.plan).read_text())
    run, audit = Path(args.validation), Path(args.audit)
    binding = verify_manifest(run)
    audited = json.loads((audit/'audit.json').read_text())
    if audited['bindings']['manifest_sha256'] != binding['manifest_sha256']:
        raise ValueError('Validation audit does not bind this raw run')
    rows = json.loads((audit/'rows.json').read_text())
    history, _, valid, n = load_batch(run/'trajectories')
    if n != plan['batch_size']: raise ValueError('Unexpected validation batch')
    evidence = Evidence(args.out, args.backup)
    evidence.save('plan.json', plan)
    import hashlib, subprocess
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    committed = subprocess.check_output(['git','show',f'{head}:tools/fold_return_growth.py'], cwd=ROOT)
    if hashlib.sha256(committed).hexdigest() != sha(__file__):
        raise ValueError('Uncommitted tangent runner')
    evidence.save('validation-binding.json', {'raw': str(run), 'audit': str(audit),
                    'manifest_sha256': binding['manifest_sha256'], 'audit_sha256': sha(audit/'audit.json'),
                    'rows_sha256': sha(audit/'rows.json'), 'growth_source_sha256': sha(__file__)})
    started = time.monotonic()
    try:
        engine = Engine(plan, evidence)
        rng = np.random.default_rng(2026111303)
        stops = np.array([r['capture']['first_128_match_start'] if r['capture']['first_128_match_start'] is not None
                          else r['valid_through'] for r in rows])
        settings = plan['growth']; L = settings['window_updates']; summaries = []
        windows = [('period3_control', None, 384, 129)] + [
            (f'window-{start:05d}', start, L, settings['alignment_updates']) for start in settings['window_starts']]
        for name, start, length, alignment in windows:
            if start is None:
                eligible = np.full(n, length)
            else:
                eligible = np.clip(stops-start, 0, length)
            if not eligible.max():
                evidence.save(name+'-skipped.json', {'reason': 'No trajectory survives to this predetermined start',
                                                     'start': start, 'eligible': eligible.tolist()})
                continue
            v = rng.normal(size=(n, 896)); v /= np.linalg.norm(v, axis=1)[:, None]
            evidence.save(name+'-initial.npz', {'directions': v.copy(), 'eligible_updates': eligible})
            logs = []; primal_errors = []; offsets = []; finite_errors = []
            for k in range(int(eligible.max())):
                if start is None:
                    x = np.tile(engine.orbit[k%3], (n,1))
                    next_state = np.tile(engine.orbit[(k+1)%3], (n,1))
                else:
                    x = history[start+k]; next_state = history[start+k+1]
                y, jv = engine.jvp(x, v)
                norm = np.linalg.norm(jv, axis=1)
                if not np.isfinite(norm).all() or (norm<=0).any():
                    raise ValueError('Nonfinite or zero tangent cannot yield a finite logarithm')
                error = np.max(abs(y-next_state), axis=1)
                active = k < eligible
                if np.max(error[active]) > 1e-11:
                    evidence.save(name+f'-primal-failure-{k:04d}.npz', {'x':x, 'saved_next':next_state, 'computed_next':y})
                    raise ValueError('AD primal disagrees with the saved full-network map')
                if k < settings['offset_check_updates']:
                    finite = []; disagreement = []
                    for relative in settings['offset_checks']:
                        h = relative*np.linalg.norm(x, axis=1)
                        perturbed, _, _ = engine.step(x+h[:,None]*v)
                        finite_jv = (perturbed-y)/h[:,None]
                        finite.append(np.log(np.linalg.norm(finite_jv, axis=1)))
                        disagreement.append(np.linalg.norm(finite_jv-jv, axis=1)/norm)
                    offsets.append(np.array(finite)); finite_errors.append(np.array(disagreement))
                logs.append(np.log(norm)); primal_errors.append(error)
                v = jv/norm[:,None]
                if (k+1)%128==0 or k+1==eligible.max():
                    lo = k+1-len(logs)
                    evidence.save(f'{name}/growth-{lo:04d}-{k:04d}.npz', {
                        'log_norm': np.array(logs), 'primal_max_abs': np.array(primal_errors),
                        'eligible_updates': eligible, 'last_directions': v.copy(),
                        'offset_log_norm': np.array(offsets), 'offset_relative_JVP_error': np.array(finite_errors)})
                    logs.clear(); primal_errors.clear(); offsets.clear(); finite_errors.clear()
                    status = {'window': name, 'update': k+1, 'seconds': time.monotonic()-started}
                    write_json(evidence.out/'status.json', status); print(json.dumps(status), flush=True)
            growth = np.concatenate([np.load(p)['log_norm'] for p in sorted((evidence.out/name).glob('growth-*.npz'))])
            summary = {'name': name, 'start': start, 'rows': summarize_window(growth, eligible, alignment, length)}
            if start is None:
                expected = float(np.log(max(abs(np.linalg.eigvals(engine.M))))/3)
                observed = np.array([r['mean'] for r in summary['rows']])
                summary['expected_per_update'] = expected
                summary['max_control_error'] = float(max(abs(observed-expected)))
                summary['control_passed'] = bool(np.all(observed<0) and summary['max_control_error']<.005)
                if not summary['control_passed']:
                    evidence.save('failed-control.json', summary)
                    raise ValueError('Attracting P3 tangent control failed')
            evidence.save(name+'-summary.json', summary); summaries.append(summary)
            if time.monotonic()-started > plan['local_wall_seconds_per_stage']:
                raise TimeoutError('Growth stage wall bound reached')
        evidence.save('result.json', {'windows': summaries, 'wall_seconds': time.monotonic()-started})
        evidence.finish()
    except BaseException as exc:
        if 'logs' in locals() and logs:
            try:
                evidence.save(f'{name}/interrupted-prefix-{k:04d}.npz', {
                    'log_norm':np.array(logs), 'primal_max_abs':np.array(primal_errors),
                    'eligible_updates':eligible, 'offset_log_norm':np.array(offsets),
                    'offset_relative_JVP_error':np.array(finite_errors)})
            except Exception:
                pass  # Preserve the original failure; existing partial files remain.
        write_json(evidence.out/'failure.json', {'type':type(exc).__name__, 'message':str(exc)})
        backup_file(evidence.out/'failure.json', evidence.out, evidence.backup)
        raise


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', default=str(ROOT/'docs/fold-return-autonomous-plan.json'))
    for name in ('validation','audit','out','backup'): p.add_argument('--'+name, required=True)
    execute(p.parse_args())
