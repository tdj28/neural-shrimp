#!/usr/bin/env python3
"""EXP015: frozen, bounded saved-reference analysis; no integration or fitting."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

import fresh_preview_runtime as runtime
import run_hopfield_refinement as legacy
import run_phase_memory_recovery as codec
import phase_transverse_gain as gain
import phase_transverse_gain_inputs as inputs
import numpy as np

ROOT = runtime.ROOT
OLD = Path('/data2/neural-shrimp-data/raw/EXP-014/attempt-002')
OLD_SOURCE = '406b89553b956bac1b97ef05413c799e383a9103'
OLD_RECEIPT = '2494966914cb68d8c5286a5dca6710212ade171ead42c9cfaed82cebd44c12a8'
PLAN_REL = 'experiments/hopfield/EXP-015-plan.json'
REVIEW_REL = 'docs/EXP-015-local-review.json'
NEW_SOURCES = [
    'tools/run_phase_transverse_gain.py', 'tools/phase_transverse_gain.py',
    'tools/phase_transverse_gain_inputs.py', 'tests/test_phase_transverse_gain.py',
    'tests/test_phase_transverse_gain_inputs.py', 'tests/test_phase_transverse_gain_run.py',
    'docs/EXP-015-protocol.md',
]
CONTRACT = dict(
    experiment_id='EXP-015', schema_version='EXP015-saved-reference-v1',
    exploratory=True, outcome_aware=True, cells=192, realized_pulse_rows=1152,
    settings=2, phase_contexts=16, directions=3, clock_windows=4,
    new_integrations=0, model_fits=0, threads=1, child_workers=0,
    producer_seconds=300, per_cell_seconds=10, sealing_seconds=60,
    owned_rss_bytes=4*1024**3, launch_available_ram_bytes=8*1024**3,
    reserve_ram_bytes=4*1024**3, raw_budget_bytes=256*1024**2,
    required_packages=dict(numpy='2.5.3', scipy='1.18.1'),
    paid_compute=False, paid_review=False,
)
SCOPE = ('Exploratory saved-reference local phase/transverse decomposition, selected after EXP014. '
         'All192 cells retained; no ODE solve, decoder fit, best-direction selection, setting means, '
         'original-study requalification, finite-pulse chaos or geometry-utility claim.')


class ProducerStop(BaseException):
    pass


class CellStop(BaseException):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def binding(path, root=None):
    path = Path(path)
    return dict(path=str(path.relative_to(root)) if root else str(path),
                bytes=path.stat().st_size, sha256=legacy.sha(path))


def normalized(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    return value


def inventory(directory):
    directory = Path(directory)
    result = []
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Evidence aliases are forbidden')
        if path.is_file() and path != directory/'receipt.json':
            result.append(binding(path, directory))
    return result


def build_plan():
    if legacy.sha(OLD/'receipt.json') != OLD_RECEIPT:
        raise ValueError('Original receipt differs')
    receipt = read(OLD/'receipt.json')
    if receipt['source_commit'] != OLD_SOURCE:
        raise ValueError('Original source differs')
    source_paths = list(dict.fromkeys(codec.SOURCE_PATHS + NEW_SOURCES))
    cells = gain.cell_census()
    if len(cells) != 192 or len({cell['id'] for cell in cells}) != 192:
        raise ValueError('Exact192 unique cells required')
    return dict(**CONTRACT, scope=SCOPE, mathematical_contract=normalized(gain.CONTRACT),
                gates=normalized(gain.GATES), cells_census=cells,
                original_directory=str(OLD), original_source=OLD_SOURCE,
                original_receipt_sha256=OLD_RECEIPT,
                input_files=inputs.input_file_bindings(OLD),
                source_files=[binding(ROOT/path, ROOT) for path in source_paths],
                original_setting_qualification={'core':False, 'failed-near-edge':False})


def validate(plan):
    if plan != build_plan():
        raise ValueError('Exact source/input/contract/192-cell plan reconstruction differs')
    return dict(cells=192, realized_pulse_rows=1152, sources=len(plan['source_files']),
                inputs=len(plan['input_files']), new_integrations=0)


def source_guard(plan, plan_path, expected_head):
    if not re.fullmatch('[0-9a-f]{40}', expected_head):
        raise ValueError('Exact expected source commit required')
    if legacy.git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Clean checkout required')
    head = legacy.git('rev-parse', 'HEAD')
    branch = legacy.git('branch', '--show-current')
    if head != expected_head or not branch.startswith('codex/'):
        raise ValueError('Source HEAD or branch differs')
    if str(Path(plan_path).resolve().relative_to(ROOT)) != PLAN_REL:
        raise ValueError('Canonical plan required')
    for item in plan['source_files'] + [binding(plan_path, ROOT), binding(ROOT/REVIEW_REL, ROOT)]:
        data = subprocess.check_output(['git', '-C', str(ROOT), 'show', head+':'+item['path']])
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Declared committed source differs: '+item['path'])
    review = read(ROOT/REVIEW_REL)
    if review.get('status') != 'ready_for_exploratory_execution' or review.get('plan_sha256') != legacy.sha(plan_path):
        raise ValueError('Exact-plan local review required')
    if legacy.git('ls-remote', '--exit-code', 'origin', 'refs/heads/'+branch).split()[0] != head:
        raise ValueError('Exact private branch freeze must be pushed')
    return dict(source_commit=head, pushed_branch=branch, remote_verified_at=legacy.now(),
                review=binding(ROOT/REVIEW_REL, ROOT))


def resource_check(directory, launch=False):
    mem = {row.split(':')[0]:int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines()
           if row.startswith(('MemAvailable:', 'MemTotal:'))}
    rss = int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    raw = sum(path.stat().st_size for path in Path(directory).rglob('*') if path.is_file())
    row = dict(at=legacy.now(), identity=runtime.identity(os.getpid()), owned_rss_bytes=rss,
               available_ram_bytes=mem['MemAvailable'], raw_bytes=raw,
               disk_free_bytes=shutil.disk_usage(directory).free, load_average=list(os.getloadavg()))
    runtime.append(Path(directory)/'resource-monitor.jsonl', row)
    reserve = CONTRACT['launch_available_ram_bytes'] if launch else CONTRACT['reserve_ram_bytes']
    if (rss > CONTRACT['owned_rss_bytes'] or raw > CONTRACT['raw_budget_bytes']
            or mem['MemAvailable'] < reserve or row['disk_free_bytes'] < CONTRACT['raw_budget_bytes']):
        raise ProducerStop('Owned resource ceiling or shared reserve reached')
    return row


def attempt_directory(data_root, attempt):
    if not re.fullmatch('attempt-[0-9]{3}', attempt):
        raise ValueError('Exclusive attempt-NNN required')
    path = (Path(data_root)/'EXP-015'/attempt).resolve()
    if any(path == base.resolve() or base.resolve() in path.parents for base in (ROOT, OLD)):
        raise ValueError('Output must be external to repository and original immutable evidence')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def compatibility_ids(cell):
    return [f'{cell["setting"]}--p{cell["phase_index"]}--d{dose}--{cell["direction"]}--s{sign:+d}'
            for dose in range(3) for sign in (-1,1)]


def missing_row(cell, status='not_started', error=None):
    return dict(cell=cell, status=status, qualified=False,
                qualified_algebraic=False, qualified_ideal_neutral=False, error=error,
                compatibility_ids=compatibility_ids(cell),
                compatibility_counts=dict(required=6, complete=0, failed=0, unavailable=6))


def check_result(cell, result):
    flags = ('qualified','qualified_algebraic','qualified_ideal_neutral')
    expected = compatibility_ids(cell)
    if (result['cell'] != cell or result['status'] not in ('complete','failed')
            or result['complete'] is not (result['status']=='complete')
            or any(type(result[key]) is not bool for key in flags)
            or [row['id'] for row in result['compatibility']] != expected
            or any(row['status'] not in ('complete','failed') for row in result['compatibility'])):
        raise ValueError('Result cell/status/qualification/six-pulse identity differs')
    complete = sum(row['status']=='complete' for row in result['compatibility'])
    if result['status']=='complete' and complete != 6:
        raise ValueError('Completed cell needs all six compatibility records')
    if result['qualified'] != result['qualified_ideal_neutral'] or (result['qualified_ideal_neutral'] and not result['qualified_algebraic']):
        raise ValueError('Qualification implication differs')
    return dict(required=6, complete=complete, failed=6-complete, unavailable=0)


def summary_from_rows(rows):
    return dict(required_cells=192, status_counts=dict(Counter(row['status'] for row in rows)),
                calculated_cells=sum(row['status']=='complete' for row in rows),
                all_gates_pass_cells=sum(row['qualified'] is True for row in rows),
                algebraically_qualified_cells=sum(row['qualified_algebraic'] is True for row in rows),
                ideal_neutral_qualified_cells=sum(row['qualified_ideal_neutral'] is True for row in rows),
                realized_pulse_rows={key:sum(row['compatibility_counts'][key] for row in rows)
                                     for key in ('required','complete','failed','unavailable')},
                setting_means=None, original_study_requalified=False)


def run(plan_path, data_root, attempt, expected_head):
    plan = read(plan_path)
    validate(plan)
    provenance = source_guard(plan, plan_path, expected_head)
    if any(importlib.metadata.version(name) != version for name, version in CONTRACT['required_packages'].items()):
        raise ValueError('Frozen numerical package versions differ')
    directory = attempt_directory(data_root, attempt)
    start = time.monotonic()
    deadline = start + CONTRACT['producer_seconds']
    cells = plan['cells_census']
    rows = {cell['id']:missing_row(cell) for cell in cells}
    manifest = dict(**provenance, plan=plan, plan_sha256=legacy.sha(plan_path), started_at=legacy.now(),
                    producer_pid=os.getpid(), producer_identity=runtime.identity(os.getpid()),
                    python=sys.version, executable=sys.executable, argv=sys.argv,
                    platform=platform.platform(), thread_environment={key:os.environ.get(key) for key in legacy.THREAD_ENV})
    runtime.write(directory/'manifest.json', manifest)
    runtime.write(directory/'planned-cells.json', cells)
    runtime.write(directory/'initial-terminal-records.json', list(rows.values()))
    error = postflight_error = None
    stage = 'input_seal'
    active_deadline = None
    handlers = {sig:signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}

    def stopped(signum, frame):
        if signum != signal.SIGALRM or time.monotonic() >= deadline:
            raise ProducerStop('Whole producer deadline or external interruption')
        resource_check(directory)
        if active_deadline is not None and time.monotonic() >= active_deadline:
            raise CellStop('Per-cell deadline')

    for sig in handlers:
        signal.signal(sig, stopped)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    try:
        resource_check(directory, launch=True)
        bundle = inputs.load_bound_inputs(OLD)
        seal = codec.save_packet(directory, 'inputs-seal', bundle)
        bundle = codec.load_packet(seal['path'])
        runtime.append(directory/'events.jsonl', dict(event='inputs_sealed', at=legacy.now(),
                       monotonic_ns=time.monotonic_ns(), sha256=seal['sha256'], semantic_sha256=seal['semantic_sha256']))
        (directory/'cells').mkdir()
        stage = 'cells'
        runtime.append(directory/'events.jsonl', dict(event='cells_started', at=legacy.now(), monotonic_ns=time.monotonic_ns()))
        for cell in cells:
            resource_check(directory)
            active_deadline = time.monotonic()+CONTRACT['per_cell_seconds']
            began = time.monotonic()
            row = missing_row(cell, 'failed')
            try:
                result = gain.evaluate_cell(cell, **inputs.cell_inputs(bundle, cell))
                compatibility_counts = check_result(cell, result)
                packet = codec.save_packet(directory/'cells', cell['id'], result)
                row.update(status=result['status'], qualified=bool(result['qualified']),
                           qualified_algebraic=result['qualified_algebraic'],
                           qualified_ideal_neutral=result['qualified_ideal_neutral'],
                           compatibility_counts=compatibility_counts,
                           packet_path=str(Path(packet['path']).relative_to(directory)),
                           packet_sha256=packet['sha256'], packet_semantic_sha256=packet['semantic_sha256'])
            except (Exception, CellStop) as exc:
                row.update(status='failed', error=repr(exc), qualified=False,
                           qualified_algebraic=False, qualified_ideal_neutral=False)
            except BaseException as exc:
                row.update(status='interrupted', error=repr(exc), qualified=False,
                           qualified_algebraic=False, qualified_ideal_neutral=False)
                raise
            finally:
                active_deadline = None
                row['seconds'] = time.monotonic()-began
                blocked = signal.pthread_sigmask(signal.SIG_BLOCK, set(handlers))
                try:
                    rows[cell['id']] = row
                    runtime.append(directory/'cell-journal.jsonl', row)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
        stage = 'postflight'
    except BaseException as exc:
        error = repr(exc)
        runtime.write(directory/'producer-error.json', dict(stage=stage, error=error, at=legacy.now()))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(ProducerStop('Sealing deadline')))
        signal.setitimer(signal.ITIMER_REAL, CONTRACT['sealing_seconds'])
        try:
            terminal = list(rows.values())
            runtime.write(directory/'terminal-records.json', terminal)
            try:
                validate(plan)
                if legacy.git('rev-parse','HEAD') != expected_head or legacy.git('status','--porcelain','--untracked-files=all'):
                    raise ValueError('Source checkout changed during analysis')
            except Exception as exc:
                postflight_error = repr(exc)
            runtime.write(directory/'postflight.json', dict(error=postflight_error, source_valid=postflight_error is None))
            summary = dict(**summary_from_rows(terminal), complete=error is None and all(row['status']=='complete' for row in terminal),
                           source_valid=postflight_error is None, error=error, postflight_error=postflight_error,
                           experiment_id='EXP-015', elapsed_seconds=time.monotonic()-start, scope=SCOPE)
            runtime.write(directory/'summary.json', summary)
            items = inventory(directory)
            runtime.write(directory/'receipt.json', dict(**provenance, experiment_id='EXP-015',
                          plan_sha256=legacy.sha(plan_path), artifacts=items, artifact_bytes=sum(item['bytes'] for item in items),
                          status='complete' if summary['complete'] and summary['source_valid'] else 'incomplete_or_invalid',
                          sealed_at=legacy.now()))
        except BaseException as exc:
            runtime.write(directory/'sealing-error.json', dict(error=repr(exc), at=legacy.now()))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return dict(directory=str(directory), **summary, receipt_sha256=legacy.sha(directory/'receipt.json'))


def verify(directory):
    directory = Path(directory).resolve()
    receipt = read(directory/'receipt.json')
    manifest = read(directory/'manifest.json')
    plan = manifest['plan']
    if receipt['artifacts'] != inventory(directory):
        raise ValueError('Immutable inventory differs')
    if receipt['source_commit'] != manifest['source_commit'] or receipt['plan_sha256'] != manifest['plan_sha256']:
        raise ValueError('Manifest/receipt source identity differs')
    data = subprocess.check_output(['git','-C',str(ROOT),'show',receipt['source_commit']+':'+PLAN_REL])
    if json.loads(data) != plan or hashlib.sha256(data).hexdigest() != receipt['plan_sha256']:
        raise ValueError('Committed plan differs')
    for item in plan['source_files']+[manifest['review']]:
        data = subprocess.check_output(['git','-C',str(ROOT),'show',receipt['source_commit']+':'+item['path']])
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Committed source/review differs')
    validate(plan)
    cells = plan['cells_census']
    rows = read(directory/'terminal-records.json')
    if read(directory/'planned-cells.json') != cells or [row['cell'] for row in rows] != cells:
        raise ValueError('Exact ordered192-cell denominator differs')
    if read(directory/'initial-terminal-records.json') != [missing_row(cell) for cell in cells]:
        raise ValueError('Initial complete denominator differs')
    if any(row['compatibility_ids'] != compatibility_ids(row['cell']) for row in rows):
        raise ValueError('Explicit1152 realized pulse identities differ')
    for row, cell in zip(rows, cells, strict=True):
        counts = row['compatibility_counts']
        if (any(type(row['cell'][key]) is not type(cell[key]) for key in cell)
                or set(counts) != {'required','complete','failed','unavailable'}
                or any(type(value) is not int or value < 0 for value in counts.values())
                or counts['required'] != 6 or sum(counts[key] for key in ('complete','failed','unavailable')) != 6):
            raise ValueError('Typed six-pulse denominator differs')
        if 'packet_path' not in row:
            if counts != dict(required=6,complete=0,failed=0,unavailable=6) or any(row[key] is not False for key in ('qualified','qualified_algebraic','qualified_ideal_neutral')):
                raise ValueError('Unavailable cell cannot claim pulse results or qualification')
        if row['status']=='not_started' and row != missing_row(cell):
            raise ValueError('Not-started cell differs from initial ledger')
    journal = [json.loads(line) for line in (directory/'cell-journal.jsonl').read_text().splitlines()] if (directory/'cell-journal.jsonl').exists() else []
    dispatched = [row for row in rows if row['status'] != 'not_started']
    if journal != dispatched:
        raise ValueError('Cell journal differs')
    sealed = (directory/'inputs-seal.json').exists()
    producer_error = read(directory/'producer-error.json') if (directory/'producer-error.json').exists() else None
    seal_status = 'absent'
    if dispatched and not sealed:
        raise ValueError('Analyzed cells require sealed inputs')
    if sealed:
        try:
            bundle = codec.load_packet(directory/'inputs-seal.json')
        except Exception:
            if dispatched or producer_error is None or producer_error['stage'] != 'input_seal':
                raise
            seal_status = 'invalid_partial_bytes_preserved'
        else:
            seal_status = 'validated'
            fresh = inputs.load_bound_inputs(OLD)
            if codec.packet_digest(bundle) != codec.packet_digest(fresh):
                raise ValueError('Exact original-input seal replay differs')
            events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()] if (directory/'events.jsonl').exists() else []
            seals = [row for row in events if row['event']=='inputs_sealed']
            starts = [row for row in events if row['event']=='cells_started']
            if dispatched and (len(seals)!=1 or len(starts)!=1 or seals[0]['monotonic_ns']>=starts[0]['monotonic_ns']):
                raise ValueError('Input seal chronology differs')
            if seals and (len(seals)!=1 or seals[0]['sha256']!=legacy.sha(directory/'inputs-seal.json') or seals[0]['semantic_sha256']!=codec.packet_digest(bundle)):
                raise ValueError('Input seal event differs')
    replayed = 0
    for row in dispatched:
        if 'packet_path' not in row:
            if row['status'] not in ('failed','interrupted') or any(row[key] is not False for key in ('qualified','qualified_algebraic','qualified_ideal_neutral')) or not row['error']:
                raise ValueError('Missing result needs retained explicit failure')
            continue
        if row['packet_path'] != 'cells/'+row['cell']['id']+'.json':
            raise ValueError('Cell packet path differs')
        path = directory/row['packet_path']
        if legacy.sha(path) != row['packet_sha256']:
            raise ValueError('Cell packet file differs')
        result = codec.load_packet(path)
        if check_result(row['cell'], result) != row['compatibility_counts']:
            raise ValueError('Six-pulse compatibility denominator differs')
        if codec.packet_digest(result) != row['packet_semantic_sha256']:
            raise ValueError('Cell semantic identity differs')
        operational_failure = row['status'] in ('interrupted','failed') and row['error'] is not None
        if operational_failure:
            if any(row[key] is not False for key in ('qualified','qualified_algebraic','qualified_ideal_neutral')):
                raise ValueError('Operationally failed cell cannot be qualified')
        elif row['status'] != result['status'] or any(row[key] != result[key] for key in ('qualified','qualified_algebraic','qualified_ideal_neutral')):
            raise ValueError('Cell completion or qualification differs')
        replay = gain.evaluate_cell(row['cell'], **inputs.cell_inputs(bundle,row['cell']))
        codec.compare(result, normalized(replay), row['cell']['id'])
        replayed += 1
    summary = read(directory/'summary.json')
    postflight = read(directory/'postflight.json')
    error = producer_error['error'] if producer_error else None
    for key, value in summary_from_rows(rows).items():
        if summary[key] != value:
            raise ValueError('Summary differs: '+key)
    complete = error is None and all(row['status']=='complete' for row in rows)
    if (summary['error'] != error or summary['complete'] != complete
            or summary['source_valid'] != (postflight['error'] is None)
            or summary['postflight_error'] != postflight['error']):
        raise ValueError('Summary error/source/completion conclusion differs')
    status = 'complete' if complete and postflight['error'] is None else 'incomplete_or_invalid'
    if receipt['status'] != status or receipt['artifact_bytes'] != sum(item['bytes'] for item in receipt['artifacts']):
        raise ValueError('Receipt conclusion differs')
    identity = runtime.identity(manifest['producer_pid'])
    return dict(complete=complete and postflight['error'] is None, source_commit=receipt['source_commit'],
                receipt_sha256=legacy.sha(directory/'receipt.json'), required_cells=192, replayed_cells=replayed,
                all_gates_pass_cells=summary['all_gates_pass_cells'],
                algebraically_qualified_cells=summary['algebraically_qualified_cells'],
                ideal_neutral_qualified_cells=summary['ideal_neutral_qualified_cells'],
                producer_original_identity_absent=identity is None or identity != manifest['producer_identity'],
                new_integrations=0, input_seal_status=seal_status,
                strict_saved_array_replay=bool(replayed))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--out', required=True)
    p = sub.add_parser('validate'); p.add_argument('--plan', required=True)
    p = sub.add_parser('run')
    for name in ('plan','data-root','attempt','expected-head'):
        p.add_argument('--'+name, required=True)
    p = sub.add_parser('verify'); p.add_argument('--directory', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        runtime.write(args.out, build_plan()); result = binding(args.out)
    elif args.command == 'validate':
        result = validate(read(args.plan))
    elif args.command == 'run':
        result = run(args.plan,args.data_root,args.attempt,args.expected_head)
    else:
        result = verify(args.directory)
    print(json.dumps(normalized(result), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
