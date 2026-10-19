"""Changed Task34 metadata gates only; synthetic files and mocked Git/Flint.

The positive case stops at the first point decode. No actual model, Arb,
original artifact, Git operation, or previously accepted generic test is used.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace


class ReachedDecode(Exception):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(',', ':')) + '\n')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def fixture(root, change):
    """Every mutation is re-pinned, so refusals exercise semantics, not hashes."""
    root.mkdir()
    raw = root / 'refinement'
    raw.mkdir()
    source = 'synthetic-original-source'
    prefix_source = 'synthetic-prefix-source'
    prefix = root / 'own-prefix'
    meta = dict(cell=change.get('prefix_cell', 'cascade-p3'), precision=2048,
                source_commit=prefix_source, versions={'synthetic': 'only'},
                pinned_assets={'synthetic': 'asset'}, prefix_ids=[1, 2, 3],
                steering_layer=12, alpha_ratio=list((.0650).as_integer_ratio()),
                log10_temperature_ratio=list((-.14798).as_integer_ratio()),
                temperature='synthetic-temperature', rotary='synthetic-rotary')
    prefix_pin = write(prefix / 'inputs.json', meta)
    map_record = {k: meta[k] for k in ('pinned_assets', 'prefix_ids',
                  'steering_layer', 'alpha_ratio', 'log10_temperature_ratio',
                  'temperature', 'rotary')}
    card = dict(cell=change.get('card_cell', 'cascade-p3'),
                period=change.get('card_period', 3), precision=2048,
                alpha=change.get('alpha', .0650), log10T=change.get('log10T', -.14798),
                prefix_output=str(prefix), prefix_source_commit=prefix_source,
                inputs=[prefix_pin])
    card_path = root / 'original-card.json'
    card_pin = write(card_path, card)
    card_relative = 'synthetic/original-card.json'
    source_bytes = {'synthetic/evaluator.py': b'# synthetic evaluator, never executed\n',
                    card_relative: card_path.read_bytes()}
    source_hashes = {k: hashlib.sha256(v).hexdigest() for k, v in source_bytes.items()}
    zero = {'mid': [0, 0], 'rad': [0, 0]}
    centers = [[zero for _ in range(896)] for _ in range(3)]
    definition_pin = write(root / 'definition.json',
                           dict(period=3, dimension=896, map=map_record, centers=centers))
    iteration = change.get('iteration', 16)
    outputs = [write(raw / 'report.json', dict(period=3, precision=2048,
                    corrections_applied=change.get('report_iteration', iteration))),
               write(raw / f'iteration-{iteration:02d}-centers.json', centers)]
    for phase in range(3):
        outputs.append(write(raw / f'iteration-{iteration:02d}-phase-{phase}.json',
                             dict(value=[zero] * 896, tangent=[zero] * 896)))
    argv = ['synthetic-python', 'synthetic-evaluator', '--card', str(card_path),
            '--out', str(raw)]
    receipt_pin = write(root / 'receipt.json', dict(source_commit=source,
                       returncode=0, failure=None, worker_reaped=True,
                       worker_group_session_absent=True, output_files=outputs, argv=argv))
    admission_pin = write(root / 'admission.json', dict(source_commit=source, argv=argv,
                         source_blobs=[dict(relative_path=k, sha256=v)
                                       for k, v in source_hashes.items()]))
    plan = dict(cell=change.get('plan_cell', 'cascade-p3'),
                period=change.get('plan_period', 3), source_commit=source,
                source_sha256=source_hashes, receipt=receipt_pin,
                admission=admission_pin, definition=definition_pin, card=card_pin,
                card_relative_path=card_relative, map=map_record,
                point_versions=meta['versions'], output=str(raw), iteration=iteration)
    write(root / 'synthetic-plan.json', plan)
    return plan, source_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if not out.is_relative_to(Path('/tmp')):
        raise ValueError('synthetic output must be under /tmp')
    out.mkdir(exist_ok=False)
    # These are only this process's limits/affinity, never another worker's.
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    start = time.monotonic()
    here = Path(__file__).resolve().parent
    source_path = here / 'point_bridge.py'
    source_hash = sha(source_path)
    spec = importlib.util.spec_from_file_location('task34_synthetic_point_bridge', source_path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    sys.modules['flint'] = SimpleNamespace(ctx=SimpleNamespace(prec=1024, threads=1))
    cases = [
        ('accepted_task34_metadata_reaches_decode', {}, 'ReachedDecode'),
        ('reject_task33_period2', {'plan_period': 2}, 'period'),
        ('reject_task33_named_plan', {'plan_cell': 'task33-p3'}, 'named point cell'),
        ('reject_wrong_card_period', {'card_period': 2}, 'original point map'),
        ('reject_task33_named_card', {'card_cell': 'task33-p3'}, 'original point cell'),
        ('reject_task33_alpha', {'alpha': .06468}, 'original point map'),
        ('reject_task33_log_temperature', {'log10T': -.148}, 'original point map'),
        ('reject_task33_prefix', {'prefix_cell': 'task33-p3'}, 'prefix point cell'),
        ('reject_consistent_but_nonfinal_iteration', {'iteration': 15}, 'final iteration'),
        ('reject_nonfinal_original_report', {'report_iteration': 15}, 'final iteration'),
    ]
    rows = []
    for name, change, expected in cases:
        plan, source_bytes = fixture(out / name, change)
        calls = []

        def fake_git(argv, *, cwd):
            if argv[:2] != ['git', 'show'] or cwd != out:
                raise AssertionError('unexpected synthetic Git interface')
            prefix = plan['source_commit'] + ':'
            if not argv[2].startswith(prefix):
                raise AssertionError('unexpected synthetic source')
            relative = argv[2][len(prefix):]
            calls.append(relative)
            return source_bytes[relative]

        def stop_before_decode(value):
            raise ReachedDecode('metadata accepted; no point decoding performed')

        def never_pack(value):
            raise AssertionError('pack should never run in metadata controls')

        bridge.subprocess = SimpleNamespace(check_output=fake_git)
        try:
            bridge.authenticate(plan, repository=out, unpack=stop_before_decode,
                                pack=never_pack, precision=1024)
            actual = 'unexpected-return'
        except ReachedDecode:
            actual = 'ReachedDecode'
        except ValueError as error:
            actual = str(error)
        except Exception:
            actual = traceback.format_exc()
        passed = actual == expected
        if expected == 'ReachedDecode':
            passed = passed and calls == list(source_bytes)
        rows.append(dict(name=name, change=change, expected=expected, actual=actual,
                         passed=passed, synthetic_source_calls=calls))
    stable = sha(source_path) == source_hash
    report = dict(schema='task34-point-bridge-changed-metadata-controls-v1',
                  source_sha256=source_hash, controls_sha256=sha(Path(__file__)),
                  source_unchanged=stable, passed=sum(x['passed'] for x in rows),
                  failed=sum(not x['passed'] for x in rows) + (not stable), rows=rows,
                  elapsed_seconds_before_report=time.monotonic() - start,
                  observed_ru_maxrss_kib_before_report=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Synthetic metadata only; mocked Git and Flint context; stopped before point decode. No target, model, real Git/Arb, or generic endpoint controls.',
                  limits=dict(cpu_seconds=10, as_bytes=1024**3, logical_cpus=1,
                              nice=os.getpriority(os.PRIO_PROCESS, 0)))
    write(out / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('passed', 'failed', 'elapsed_seconds_before_report')}))
    return 0 if report['failed'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
