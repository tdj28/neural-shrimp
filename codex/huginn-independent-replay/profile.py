"""Bounded full-size synthetic cost profile, never a target-data reader."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import time

OWNED = Path('/data2/neural-shrimp-data/codex/huginn-independent-replay')
GIB = 1024 ** 3
ENVELOPE = {'cpu_threads': 1, 'supervisor_slots': 1, 'nice': 10, 'address_space_gib': 4,
            'timeout_seconds': 120, 'startup_available_gib': 20, 'runtime_available_gib': 16}


def available_bytes():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def check_plan(plan, directory):
    if (plan.get('schema') != 'codex-huginn-synthetic-profile-v1'
            or plan.get('synthetic_only') is not True or plan.get('target_access_authorized') is not False
            or plan.get('counts') != {'A_questions': 64, 'A_arms': 4, 'A_bootstrap': 2000,
                                     'B_questions': 200, 'B_arms': 4, 'B_bootstrap': 10000}):
        raise ValueError('profile contract mismatch')
    env = plan.get('envelope')
    if not isinstance(env, dict) or set(env) != set(ENVELOPE) or any(type(env[k]) is not int or env[k] != v for k, v in ENVELOPE.items()):
        raise ValueError('profile envelope mismatch')
    if set(plan.get('sources', {})) != {'statistics.py', 'profile.py'}:
        raise ValueError('source inventory mismatch')
    for name, digest in plan['sources'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'source drift: {name}')


def fixtures():
    spectra, a_rows, b_rows = {}, [], []
    for q in range(64):
        digest = f'{q:064x}'
        spectra[q] = {'fixed_point_residual_F': 1e-6, 'token_ids_sha256': digest,
                      'arnoldi_converged': {'LM': True, 'SR': True, 'LR': True},
                      'spectra_of_F': {'LM': [[.60 + .004*q, .04]], 'SR': [[-.2, .05]], 'LR': [[.60 + .004*q, .04]]}}
        for arm in (1., 1.05, 1.08, 1.10):
            a_rows.append({'index': q, 'lam': arm, 'split': 'test', 'seed': q, 'gold': 0,
                'gold_ids_sha256': digest, 'native_residual_rows': [[(q+1)*arm, 2., 3., 4.] for _ in range(32)],
                'by_steps': {str(k): {'loglik': [-1., -2., -3., -4.]} for k in range(24, 33)}})
    for arm in ('1.0', '1.05', '1.15', '1.20'):
        for q in range(200):
            b_rows.append({'index': q, 'arm': arm, 'lam': float(arm), 'split': 'test', 'steps': 32,
                'prompt_style': 'cot8', 'seed': q, 'correct': bool(q % 3 == 0 or (arm == '1.05' and q % 5 == 0)),
                'correct_strict': bool(q % 3 == 0), 'new_tokens': 256 if q % 4 == 0 else 17+q,
                'strict_pred': '1' if q % 3 == 0 else None})
    return spectra, a_rows, b_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    plan_bytes = args.plan.read_bytes(); plan = json.loads(plan_bytes)
    directory = Path(__file__).resolve().parent
    check_plan(plan, directory)
    if args.output != OWNED / plan['attempt'] or plan['attempt'] != 'synthetic-profile-001':
        raise ValueError('only the single task-owned synthetic profile is admitted')
    source = os.environ.get('TASK8_SOURCE_COMMIT', '')
    if len(source) != 40 or any(c not in '0123456789abcdef' for c in source):
        raise ValueError('missing source commit binding')
    if os.environ.get('TASK8_PLAN_SHA256') != hashlib.sha256(plan_bytes).hexdigest():
        raise ValueError('missing plan hash binding')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '1': raise ValueError('thread environment must be fixed before imports')
    if os.getpriority(os.PRIO_PROCESS, 0) != 10: raise ValueError('nice10 required')
    if resource.getrlimit(resource.RLIMIT_AS) != (4*GIB, 4*GIB): raise ValueError('hard 4GiB AS required')
    if available_bytes() < 20*GIB: raise ValueError('startup headroom insufficient')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan.json').write_bytes(plan_bytes)
    for name in plan['sources']:
        (args.output/name).write_bytes((directory/name).read_bytes())
    low = [available_bytes()]
    def guard(signum, frame):
        low[0] = min(low[0], available_bytes())
        if low[0] < 16*GIB: raise RuntimeError('runtime memory headroom below floor')
        if time.monotonic() - start >= 120: raise TimeoutError('profile wall deadline')
    signal.signal(signal.SIGALRM, guard); signal.setitimer(signal.ITIMER_REAL, 1, 1)
    events = []
    def event(stage, **fields):
        record = {'stage': stage, 'elapsed_seconds': time.monotonic()-start, **fields}
        events.append(record)
        with (args.output/'events.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    status = {'synthetic_only': True, 'target_accessed': False, 'certificate_passed': False,
              'source_commit': source, 'plan_sha256': hashlib.sha256(plan_bytes).hexdigest()}
    try:
        event('source_captured')
        spec = importlib.util.spec_from_file_location('task8_profile_reducers', args.output/'statistics.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        spectra, a_rows, b_rows = fixtures()
        event('fixtures_ready', a_rows=len(a_rows), b_rows=len(b_rows))
        at = time.monotonic(); part_a = module.analyze_a(spectra, a_rows)
        event('part_a_completed', seconds=time.monotonic()-at)
        with (args.output/'synthetic-A.json').open('x') as f: json.dump(part_a, f)
        at = time.monotonic(); part_b = module.analyze_b(b_rows)
        event('part_b_completed', seconds=time.monotonic()-at)
        with (args.output/'synthetic-B.json').open('x') as f: json.dump(part_b, f)
        check_plan(plan, directory); guard(None, None)
        status['status'] = 'synthetic_profile_completed'
    except BaseException as exc:
        status.update(status='synthetic_profile_failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        status.update(elapsed_seconds=time.monotonic()-start, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      min_available_bytes=low[0], stages=events)
        with (args.output/'status.json').open('x') as f: json.dump(status, f, indent=2)


if __name__ == '__main__':
    main()
