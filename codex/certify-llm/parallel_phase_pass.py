"""Bounded Linux subprocess scheduling for eight authenticated point images.

Stdlib only: this controller never imports a model or performs mathematical
admission of output balls. The caller's committed plan must authenticate this
module, the complete immutable source snapshot and request. A sampled aggregate
RSS watchdog is conservative accounting, NOT a hard cgroup memory limit.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
OWNED = Path('/data2/neural-shrimp-data/codex/llm-p8-certificate')
PHASES = 8
GIB = 1024**3
WORKER_LIMIT = 18 * GIB
RESERVE_SECONDS = 600.0
POLL_SECONDS = .05
WORKER_KEY = 'codex/certify-llm/phase_point.py'


class PassFailure(RuntimeError):
    """A partial pass is retained, but no complete eight-phase tuple is issued."""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def valid_hash(value, length=64):
    return type(value) is str and len(value) == length and all(c in '0123456789abcdef' for c in value)


def write_json(path, value):
    # A controller artifact is written once. Live updates go to the event log.
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def proc_record(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]),
                    sid=int(fields[3]), start_ticks=int(fields[19]),
                    rss_bytes=max(0, int(fields[21])) * os.sysconf('SC_PAGE_SIZE'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def resource_snapshot(controller_pid, groups):
    records = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                row = proc_record(int(path.name))
            except (PermissionError, ValueError, IndexError):
                continue
            if row is not None:
                records[row['pid']] = row
    included = {controller_pid}
    while True:
        extra = {pid for pid, row in records.items()
                 if row['ppid'] in included or row['pgid'] in groups}
        new = included | extra
        if new == included:
            break
        included = new
    rows = [records[pid] for pid in included if pid in records]
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        memory[key] = int(value.strip().split()[0]) * 1024
    if 'MemAvailable' not in memory:
        raise PassFailure('Linux MemAvailable is required for admission')
    return dict(rss_bytes=sum(row['rss_bytes'] for row in rows),
                available_bytes=memory['MemAvailable'], processes=rows)


def inventory(directory):
    if not directory.exists():
        return {}
    files = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise PassFailure('worker artifacts must not be symlinks')
        if path.is_file():
            files[str(path.relative_to(directory))] = digest(path)
    return files


def validate_success(out, phase, request, request_sha256):
    """Receipt/source/byte checks only; scientific endpoint audit is external."""
    status = json.loads((out / 'status.json').read_bytes())
    inputs = json.loads((out / 'inputs.json').read_bytes())
    if (status.get('status') != 'phase_point_completed'
            or type(status.get('phase_index')) is not int or status['phase_index'] != phase
            or status.get('request_sha256') != request_sha256
            or status.get('certificate_passed') is not False):
        raise PassFailure('worker did not issue the expected completed phase status')
    expected = dict(request_sha256=request_sha256, source_commit=request['source_commit'],
                    phase_index=phase, map_sha256=request['map_sha256'],
                    backend_sha256=request['backend_sha256'], source_sha256=request['source_sha256'],
                    certificate_passed=False)
    selected = {key: inputs.get(key) for key in expected}
    if json.dumps(selected, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise PassFailure('worker input/source/map/backend bindings differ')
    if digest(out / 'request.json') != request_sha256:
        raise PassFailure('worker retained request bytes differ')
    for filename, key in [('input.json', 'input_sha256'), ('live-input.json', 'live_input_sha256'),
                          ('point.json', 'point_sha256')]:
        if status.get(key) != digest(out / filename):
            raise PassFailure('worker completed output hashes differ')
    return status


def _signal_group(item, sig):
    """Only signal a launched group with a still-observed original identity."""
    current = proc_record(item['process'].pid)
    leader = item['identity']
    if current is not None and leader is not None and current['start_ticks'] == leader['start_ticks'] and current['pgid'] == leader['pid']:
        allowed = True
    else:
        allowed = any((now := proc_record(pid)) is not None
                      and now['start_ticks'] == start and now['pgid'] == item['process'].pid
                      for pid, start in item['members'].items())
    if allowed:
        try:
            os.killpg(item['process'].pid, sig)
        except ProcessLookupError:
            pass


def cleanup(items):
    # All groups share the grace period, rather than each consuming it serially.
    for item in items:
        _signal_group(item, signal.SIGTERM)
    end = time.monotonic() + 2.0
    while any(item['process'].poll() is None for item in items) and time.monotonic() < end:
        time.sleep(.02)
    for item in items:
        _signal_group(item, signal.SIGKILL)
    end = time.monotonic() + 2.0
    while any(item['process'].poll() is None for item in items) and time.monotonic() < end:
        time.sleep(.02)
    for item in items:
        if item['process'].poll() is None:
            # Retained in receipt; never claim a verified termination.
            item['cleanup_unreaped'] = True


def _bootstrap(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-pid', required=True, type=int)
    parser.add_argument('--parent-start', required=True, type=int)
    parser.add_argument('--worker', required=True, type=Path)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--scheduler-sha256', required=True)
    parser.add_argument('--wall-seconds', required=True, type=float)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('worker_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if (sys.platform != 'linux' or not valid_hash(args.worker_sha256)
            or not valid_hash(args.scheduler_sha256) or digest(Path(__file__)) != args.scheduler_sha256):
        raise PassFailure('Linux and a pinned worker are required')
    def parent_alive():
        parent = proc_record(args.parent_pid)
        return os.getppid() == args.parent_pid and parent is not None and parent['start_ticks'] == args.parent_start
    if not parent_alive():
        raise PassFailure('controller identity disappeared before bootstrap')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    if not parent_alive():
        raise PassFailure('controller identity disappeared while arming parent-death guard')
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    if not (0 < args.wall_seconds <= 600) or not math.isfinite(args.wall_seconds):
        raise PassFailure('bounded worker wall allowance required')
    for kind, requested in [(resource.RLIMIT_AS, WORKER_LIMIT),
                            (resource.RLIMIT_CPU, math.ceil(args.wall_seconds) + 1)]:
        soft, hard = resource.getrlimit(kind)
        bound = min([requested] + [v for v in (soft, hard) if v >= 0])
        resource.setrlimit(kind, (bound, bound))
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    if digest(args.worker) != args.worker_sha256:
        raise PassFailure('worker source changed before isolated execution')
    write_json(args.record, dict(pid=os.getpid(), ppid=os.getppid(), identity=proc_record(os.getpid()),
        worker_sha256=args.worker_sha256, scheduler_sha256=args.scheduler_sha256, parent_death_signal='SIGKILL',
        address_space_limits=list(resource.getrlimit(resource.RLIMIT_AS)),
        cpu_seconds_limits=list(resource.getrlimit(resource.RLIMIT_CPU)), nice=os.getpriority(os.PRIO_PROCESS, 0)))
    remaining = args.worker_args[1:] if args.worker_args[:1] == ['--'] else args.worker_args
    os.execve(sys.executable, [sys.executable, '-I', '-B', str(args.worker), *remaining], env)


def run_parallel_pass(request_path, request_sha256, candidate_directory, *, deadline,
                      max_workers=4, per_worker_timeout=600.0,
                      aggregate_rss_bytes=32 * GIB, minimum_available_bytes=12 * GIB,
                      worker_path=None):
    """Retain all attempts; return a receipt only after all eight phases succeed.

    ``deadline`` is an absolute monotonic timestamp. Every new launch requires
    at least the frozen 600-second reserve. Passing this infrastructure gate is
    not authorization for a target run. Worker code is pinned by the request;
    this scheduler/source snapshot must additionally be pinned by the outer plan.
    No worker may spawn descendants: PDEATHSIG protects its process on abrupt
    controller death, while normal controller cleanup terminates owned groups.
    """
    if sys.platform != 'linux' or threading.current_thread() is not threading.main_thread():
        raise ValueError('Linux main-thread controller required for signal ownership')
    if (type(max_workers) is not int or max_workers != 4
            or type(per_worker_timeout) not in (float, int) or not math.isfinite(per_worker_timeout)
            or not 0 < per_worker_timeout <= 600
            or type(deadline) not in (float, int) or not math.isfinite(deadline)
            or type(aggregate_rss_bytes) is not int or not 0 < aggregate_rss_bytes <= 32 * GIB
            or type(minimum_available_bytes) is not int or minimum_available_bytes < 12 * GIB):
        raise ValueError('frozen CPU4/CPU1, wall and memory envelope required')
    candidate = Path(candidate_directory).resolve(); request_path = Path(request_path).resolve()
    worker = (Path(worker_path) if worker_path is not None else HERE / 'phase_point.py').resolve()
    if (len(candidate.parents) < 3 or candidate.parents[2] != OWNED.resolve()
            or candidate.parent.name != 'run' or not candidate.name.startswith('candidate-')
            or request_path != candidate / 'request.json' or not candidate.is_dir()
            or (candidate.parents[1] / 'receipt.json').exists()):
        raise ValueError('one live owned candidate request directory required')
    if any(candidate.glob('phase-*')) or any((candidate / name).exists() for name in
        ('parallel-pass.json', 'parallel-events.jsonl', 'parallel-workers')):
        raise ValueError('fresh phase outputs and controller artifact paths required')
    raw = request_path.read_bytes()
    if not valid_hash(request_sha256) or hashlib.sha256(raw).hexdigest() != request_sha256:
        raise ValueError('exact externally pinned request bytes required')
    request = json.loads(raw)
    if (type(request) is not dict or not valid_hash(request.get('source_commit'), 40)
            or request['source_commit'] != os.environ.get('CERT_SOURCE_COMMIT')
            or type(request.get('source_sha256')) is not dict
            or not valid_hash(request['source_sha256'].get(WORKER_KEY))
            or request['source_sha256'][WORKER_KEY] != digest(worker)):
        raise ValueError('pinned worker source and matching snapshot identity required')
    for key in ('map_sha256', 'backend_sha256'):
        if not valid_hash(request.get(key)):
            raise ValueError('request map/backend identity required')
    scheduler_sha256 = digest(Path(__file__))
    controller = proc_record(os.getpid())
    if controller is None:
        raise ValueError('observable controller process identity required')
    metadata = candidate / 'parallel-workers'; metadata.mkdir()
    started = time.monotonic(); all_items = []; active = []; receipts = {}; peak = 0; error = None
    previous_handlers = {}
    def event(name, **fields):
        with (candidate / 'parallel-events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=name, elapsed_seconds=time.monotonic() - started,
                utc=datetime.now(timezone.utc).isoformat(), **fields), allow_nan=False) + '\n')
    def interrupted(signum, frame):
        raise PassFailure(f'controller received signal {signum}')
    def resources():
        nonlocal peak
        snap = resource_snapshot(os.getpid(), {item['process'].pid for item in all_items})
        peak = max(peak, snap['rss_bytes'])
        for item in all_items:
            members = [row for row in snap['processes'] if row['pgid'] == item['process'].pid]
            original = item['identity']
            still_owned = any((original is not None and row['pid'] == original['pid']
                               and row['start_ticks'] == original['start_ticks'])
                              or item['members'].get(row['pid']) == row['start_ticks'] for row in members)
            if still_owned:
                item['members'].update({row['pid']: row['start_ticks'] for row in members})
                item['peak_group_rss_bytes'] = max(item['peak_group_rss_bytes'], sum(row['rss_bytes'] for row in members))
        if snap['rss_bytes'] >= aggregate_rss_bytes:
            raise PassFailure('sampled aggregate RSS limit reached')
        if snap['available_bytes'] < minimum_available_bytes:
            raise PassFailure('minimum available-memory headroom lost')
        return snap
    def finish(item, reason=None):
        phase = item['phase']; process = item['process']; out = candidate / f'phase-{phase:02d}'
        if phase in receipts:
            return receipts[phase]
        code = process.poll(); validation_error = None; status = None
        remaining_members = [pid for pid, start in item['members'].items()
            if pid != process.pid and (now := proc_record(pid)) is not None
            and now['start_ticks'] == start and now['state'] != 'Z']
        if remaining_members and reason is None:
            validation_error = 'unexpected live worker descendants remain'
        if code == 0 and reason is None and validation_error is None:
            try:
                status = validate_success(out, phase, request, request_sha256)
            except Exception as exc:
                validation_error = repr(exc)
        elif reason is None and validation_error is None:
            validation_error = f'worker exit code {code}'
        item['log'].close()
        try:
            artifacts = inventory(out)
        except Exception as exc:
            artifacts = {}; validation_error = repr(exc)
        receipt = dict(phase_index=phase, pid=process.pid, process_identity=item['identity'],
            worker_sha256=request['source_sha256'][WORKER_KEY], request_sha256=request_sha256,
            command=item['command'], exit_code=code, verified_exit=code is not None,
            status='completed' if code == 0 and reason is None and validation_error is None else 'partial_or_failed',
            termination_reason=reason, validation_error=validation_error,
            elapsed_seconds=time.monotonic() - item['started'], peak_group_rss_bytes=item['peak_group_rss_bytes'],
            artifacts=artifacts, remaining_owned_descendants=remaining_members,
            log_sha256=digest(item['log_path']), certificate_passed=False)
        bootstrap = metadata / f'phase-{phase:02d}.bootstrap.json'
        receipt['bootstrap_sha256'] = digest(bootstrap) if bootstrap.is_file() else None
        if status is not None:
            receipt['point_sha256'] = status['point_sha256']
        write_json(metadata / f'phase-{phase:02d}.receipt.json', receipt)
        receipts[phase] = receipt; event('worker_retained', phase_index=phase, status=receipt['status'], exit_code=code)
        return receipt
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        resources(); event('pass_started', controller=controller, request_sha256=request_sha256,
            max_workers=max_workers, per_worker_timeout=per_worker_timeout, deadline_monotonic=deadline,
            reserve_seconds=RESERVE_SECONDS, aggregate_rss_bytes=aggregate_rss_bytes,
            minimum_available_bytes=minimum_available_bytes, sampled_rss_is_hard_cap=False)
        next_phase = 0
        while next_phase < PHASES or active:
            if time.monotonic() >= deadline:
                raise PassFailure('shared pass deadline reached')
            resources()
            for item in list(active):
                if item['process'].poll() is not None:
                    row = finish(item); active.remove(item)
                    if row['status'] != 'completed':
                        raise PassFailure(f'phase {item["phase"]} did not complete with valid evidence')
                elif time.monotonic() - item['started'] >= per_worker_timeout:
                    raise PassFailure(f'phase {item["phase"]} worker timeout reached')
            while next_phase < PHASES and len(active) < max_workers:
                if deadline - time.monotonic() < RESERVE_SECONDS:
                    raise PassFailure('insufficient frozen worker reserve; remaining phases unstarted')
                resources()
                if (digest(request_path) != request_sha256
                        or digest(worker) != request['source_sha256'][WORKER_KEY]
                        or digest(Path(__file__)) != scheduler_sha256):
                    raise PassFailure('request or worker source changed before launch')
                phase = next_phase; log_path = metadata / f'phase-{phase:02d}.log'
                log = log_path.open('xb')
                command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--_worker-bootstrap',
                    '--parent-pid', str(os.getpid()), '--parent-start', str(controller['start_ticks']),
                    '--worker', str(worker), '--worker-sha256', request['source_sha256'][WORKER_KEY],
                    '--scheduler-sha256', scheduler_sha256,
                    '--wall-seconds', str(per_worker_timeout), '--record', str(metadata / f'phase-{phase:02d}.bootstrap.json'),
                    '--', '--request', str(request_path), '--request-sha256', request_sha256,
                    '--phase', str(phase), '--output', str(candidate / f'phase-{phase:02d}')]
                env = os.environ.copy()
                for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                    env[name] = '1'
                env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
                # Do not let SIGTERM land between Popen and ownership registration.
                blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
                try:
                    try:
                        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True, close_fds=True)
                    except BaseException:
                        log.close(); raise
                    item = dict(phase=phase, process=process, identity=proc_record(process.pid), members={},
                        started=time.monotonic(), command=command, log=log, log_path=log_path, peak_group_rss_bytes=0)
                    all_items.append(item); active.append(item); next_phase += 1
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
                event('worker_started', phase_index=phase, pid=process.pid, identity=item['identity'])
            if active:
                time.sleep(POLL_SECONDS)
        if digest(request_path) != request_sha256 or digest(Path(__file__)) != scheduler_sha256:
            raise PassFailure('request or scheduler source changed during the pass')
    except BaseException as exc:
        error = repr(exc)
        # A second ordinary termination signal must not interrupt bounded cleanup.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        cleanup(all_items)
        for item in all_items:
            finish(item, reason=error)
        event('pass_failed', error=error)
    finally:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
    terminations = []
    for item in all_items:
        remaining = [dict(pid=pid, start_ticks=start) for pid, start in item['members'].items()
            if (now := proc_record(pid)) is not None and now['start_ticks'] == start and now['state'] != 'Z']
        terminations.append(dict(phase_index=item['phase'], leader_exit_code=item['process'].poll(),
            remaining_observed_owned_processes=remaining,
            verified_termination=item['process'].poll() is not None and not remaining))
    if any(not row['verified_termination'] for row in terminations):
        error = error or 'an owned worker termination could not be verified'
    report = dict(schema='codex-parallel-point-pass-v1',
        status='completed' if error is None and len(receipts) == PHASES else 'partial_or_failed',
        request_sha256=request_sha256, source_commit=request['source_commit'],
        scheduler_sha256=scheduler_sha256, worker_sha256=request['source_sha256'][WORKER_KEY],
        controller_identity=controller, error=error, elapsed_seconds=time.monotonic() - started,
        peak_sampled_rss_bytes=peak, sampled_rss_is_hard_cap=False, certificate_passed=False,
        scientific_endpoint_audit_performed=False,
        phase_receipts=[dict(phase_index=i, receipt_sha256=digest(metadata / f'phase-{i:02d}.receipt.json'),
                             status=receipts[i]['status']) for i in sorted(receipts)],
        unstarted_phases=[i for i in range(PHASES) if i not in receipts],
        termination_verification=terminations,
        max_workers=max_workers, worker_timeout_seconds=per_worker_timeout, reserve_seconds=RESERVE_SECONDS,
        aggregate_rss_bytes=aggregate_rss_bytes, minimum_available_bytes=minimum_available_bytes)
    write_json(candidate / 'parallel-pass.json', report)
    if report['status'] != 'completed':
        raise PassFailure(error or 'all eight authenticated phase completions required')
    return report


if __name__ == '__main__':
    if sys.argv[1:2] != ['--_worker-bootstrap']:
        raise SystemExit('This is a controller library; target launches require a separately frozen plan.')
    _bootstrap(sys.argv[2:])
