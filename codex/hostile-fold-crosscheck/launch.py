"""Task7's single resource pilot: Task5 snapshot pattern + reviewed PID ownership.

No new numerical method. Work has a 300s absolute deadline; owned cleanup has
at most ten additional seconds. MemAvailable is sampled, not a hard RAM cap.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
OWNED = Path('/data2/neural-shrimp-data/codex/llm-hostile-fold-crosscheck')
GIB = 1024**3
OWNERSHIP_FILE = 'codex/certify-llm/derivative_campaign.py'
OWNERSHIP_SHA = '708eb618fa31afbab4e27ce0e834c16724a1d03985f1bfbef57bf8eb202bba03'
WORKER_FILES = ('codex/hostile-fold-crosscheck/assets.py',
                'codex/hostile-fold-crosscheck/resource_pilot.py',
                'codex/confirm-fold/fold_setup.py', 'codex/crosscheck/qwen_map.py',
                'codex/crosscheck/numerics.py')
CAPS = dict(cpu_threads=2, nice=10, memory_gib=12, timeout_seconds=300,
            minimum_start_available_gib=28, minimum_runtime_available_gib=16)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def establish_limits():
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    if nice > 10: raise RuntimeError('cannot raise inherited process priority')
    if nice < 10: os.nice(10-nice)
    if os.getpriority(os.PRIO_PROCESS, 0) != 10: raise RuntimeError('nice10 not established')
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    if hard != resource.RLIM_INFINITY and hard < 12*GIB:
        raise RuntimeError('inherited hard AS below12GiB')
    resource.setrlimit(resource.RLIMIT_AS, (12*GIB, 12*GIB))


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable absent')


def validate_arguments(args):
    if not args.attempt or Path(args.attempt).name != args.attempt or args.attempt.startswith('.'):
        raise ValueError('exclusive direct-child attempt name required')
    if args.plan != 'profile-001-plan.json': raise ValueError('only exact resource pilot plan is admitted')
    if re.fullmatch('[0-9a-f]{40}', args.source) is None or re.fullmatch('[0-9a-f]{64}', args.plan_sha256) is None:
        raise ValueError('literal external source and plan pins required')


def guard(deadline, stopped, *, startup=False):
    if stopped: raise InterruptedError(stopped[0])
    if time.monotonic() >= deadline: raise TimeoutError('absolute300s Task7 deadline')
    floor = (28 if startup else 16)*GIB
    value = available()
    if value < floor: raise RuntimeError(f'Task7 MemAvailable floor: {value} < {floor}')
    return value


def snapshot(repo, args, deadline, stopped):
    """Only Git and stdlib admission; never import Torch or inspect seed values."""
    def git(*words):
        guard(deadline, stopped)
        return subprocess.check_output(['git', *words], cwd=repo,
                    timeout=min(30, max(.001, deadline-time.monotonic())))
    if git('rev-parse', 'HEAD').decode().strip() != args.source:
        raise ValueError('current HEAD differs from exact source')
    planpath = repo/'codex/hostile-fold-crosscheck'/args.plan
    if sha(planpath) != args.plan_sha256: raise ValueError('external original plan hash differs')
    plan = json.loads(planpath.read_text())
    if plan.get('schema') != 'codex-hostile-fold-profile-v1' or type(plan.get('envelope')) is not dict or set(plan['envelope']) != set(CAPS) or any(type(plan['envelope'][k]) is not int or plan['envelope'][k] != v for k,v in CAPS.items()):
        raise ValueError('exact resource-only CPU2/12GiB/300s plan required')
    if set(plan.get('source_sha256', {})) != set(WORKER_FILES):
        raise ValueError('complete original worker import inventory required')
    paths = list(WORKER_FILES)+[OWNERSHIP_FILE, 'codex/hostile-fold-crosscheck/launch.py',
               'codex/hostile-fold-crosscheck/'+args.plan, 'codex/hostile-fold-crosscheck/profile-001-run-card.md']
    blobs = {}
    for name in paths:
        original = git('show', args.source+':'+name)
        if (repo/name).read_bytes() != original: raise ValueError('not original committed source: '+name)
        if name in plan['source_sha256'] and hashlib.sha256(original).hexdigest() != plan['source_sha256'][name]:
            raise ValueError('plan worker source hash differs: '+name)
        blobs[name] = original
    if hashlib.sha256(blobs[OWNERSHIP_FILE]).hexdigest() != OWNERSHIP_SHA:
        raise ValueError('reviewed process ownership helper differs')
    remote = git('ls-remote', '--exit-code', 'origin', 'refs/heads/main').decode().split()
    if len(remote) != 2 or remote[1] != 'refs/heads/main' or re.fullmatch('[0-9a-f]{40}', remote[0]) is None:
        raise ValueError('private main identity unavailable')
    # Coordinator fetches before admission. Missing remote objects fail closed.
    git('merge-base', '--is-ancestor', args.source, remote[0])
    guard(deadline, stopped, startup=True)
    return plan, blobs, remote[0]


def close_worker(child, identity, owner):
    """Use the previously reviewed birth-bound session/group and pidfd helpers."""
    if identity is None:
        # The direct Popen child is unreaped, so this exact PID cannot be reused.
        # Recover its birth handle for cleanup only, never scientific admission.
        identity = owner.Identity(owner.proc(child.pid))
    try:
        if owner.group(identity): owner.signal_group(identity, signal.SIGTERM)
        end = time.monotonic()+5
        while owner.group(identity) and time.monotonic() < end: time.sleep(.05)
        if owner.group(identity): owner.signal_group(identity, signal.SIGKILL)
        child.wait(timeout=5)
        if owner.group(identity): raise RuntimeError('owned worker group still present')
        return True
    finally:
        identity.close()


def monitor(child, identity, owner, deadline, stopped, observation):
    while True:
        remaining = guard(deadline, stopped)
        observation['samples'] += 1
        observation['minimum_available_bytes'] = min(observation['minimum_available_bytes'], remaining)
        code = child.poll()
        if code is not None:
            if owner.group(identity): raise RuntimeError('descendants outlived worker')
            return code
        time.sleep(min(.5, max(.001, deadline-time.monotonic())))


def execute(args):
    began = time.monotonic(); deadline = began+300
    establish_limits(); validate_arguments(args)
    attempt = OWNED/args.attempt; attempt.mkdir(parents=True, exist_ok=False)
    stopped = []; previous = {}
    def stop(signum, frame):
        if not stopped: stopped.append('launcher received signal'+str(signum))
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, stop)
    child = identity = owner = None; failure = None; code = None
    terminated = False; uncertain = []; manifest = {}
    observation = dict(samples=0, minimum_available_bytes=available())
    write(attempt/'admission-started.json', dict(source_commit=args.source, plan_sha256=args.plan_sha256,
        launcher_pid=os.getpid(), resource_envelope=CAPS, started_utc=datetime.now(timezone.utc).isoformat(),
        certificate_passed=False, resource_probe_only=True))
    try:
        guard(deadline, stopped, startup=True)
        plan, blobs, remote = snapshot(REPO, args, deadline, stopped)
        source = attempt/'source'
        for name, data in blobs.items():
            target = source/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        owner = module(source/OWNERSHIP_FILE, 'task7_reviewed_process_ownership')
        command = [sys.executable, '-I', '-B', '-u', str(source/'codex/hostile-fold-crosscheck/resource_pilot.py'),
                   '--plan', str(source/'codex/hostile-fold-crosscheck'/args.plan), '--output', str(attempt/'run')]
        manifest = dict(source_commit=args.source, plan_sha256=args.plan_sha256, remote_main=remote,
            remote_ancestry_verified_at=datetime.now(timezone.utc).isoformat(), command=command,
            launcher_identity=owner.serial_identity(owner.proc(os.getpid())), **CAPS,
            source=[dict(path=n, bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for n,b in blobs.items()],
            sampled_headroom_is_hard_cap=False, cleanup_seconds=10,
            available_before_worker_bytes=available())
        write(attempt/'manifest.json', manifest)
        env = os.environ.copy(); env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
            PYTHONDONTWRITEBYTECODE='1', TASK7_SOURCE_COMMIT=args.source, TASK7_PLAN_SHA256=args.plan_sha256)
        env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
        guard(deadline, stopped, startup=True)
        with (attempt/'stdout.log').open('xb') as out, (attempt/'stderr.log').open('xb') as err:
            child = subprocess.Popen(command, cwd=source, env=env, stdout=out, stderr=err,
                                     preexec_fn=establish_limits, start_new_session=True)
            # Popen ownership is retained before any fallible inspection or write.
            q = owner.proc(child.pid); identity = owner.Identity(q)
            if q['ppid'] != os.getpid() or q['pgid'] != child.pid or q['sid'] != child.pid or q['uid'] != os.getuid():
                raise RuntimeError('worker birth/session/parent ownership mismatch')
            command_deadline = min(deadline, time.monotonic()+5)
            while [os.fsdecode(v) for v in q['command']] != command:
                guard(deadline, stopped)
                if time.monotonic() >= command_deadline or q['state'] == 'Z':
                    raise RuntimeError('worker exact command not admitted before deadline')
                time.sleep(.01); q = owner.proc(child.pid)
                if q is None or q['start'] != identity.value['start']:
                    raise RuntimeError('worker birth changed before command admission')
            limits = next(line.split()[-3:-1] for line in (Path('/proc')/str(child.pid)/'limits').read_text().splitlines() if line.startswith('Max address space'))
            if q['nice'] != 10 or limits != [str(12*GIB)]*2:
                raise RuntimeError('worker live nice/AS differs')
            manifest['worker_identity'] = owner.serial_identity(q); manifest['owned_pid'] = child.pid
            write(attempt/'manifest.json', manifest)
            print(json.dumps(dict(attempt=str(attempt), source_commit=args.source, worker_identity=manifest['worker_identity'])), flush=True)
            code = monitor(child, identity, owner, deadline, stopped, observation)
            terminated = True
            if code != 0: raise RuntimeError('resource worker failed with exit'+str(code))
            status = json.loads((attempt/'run/status.json').read_text())
            if status.get('status') != 'resource_profile_completed' or status.get('resource_probe_only') is not True or status.get('fold_qualified') is not False or status.get('certificate_passed') is not False:
                raise RuntimeError('worker did not retain complete resource-only status')
    except BaseException as exc:
        failure = dict(error=repr(exc), traceback=traceback.format_exc())
        if child is not None:
            try:
                closing = identity; identity = None
                terminated = close_worker(child, closing, owner); code = child.returncode
            except BaseException as cleanup:
                uncertain.append(repr(cleanup))
        write(attempt/'failure.json', dict(failure, uncertain_ownership=uncertain, certificate_passed=False))
    finally:
        if identity is not None: identity.close()
        for signum, handler in previous.items(): signal.signal(signum, handler)
    result = dict(source_commit=args.source, plan_sha256=args.plan_sha256, exit_code=code,
        status='resource_pilot_completed' if failure is None else 'resource_pilot_failed',
        timed_out=bool(failure and 'TimeoutError' in failure['error']), elapsed_seconds=time.monotonic()-began,
        worker_spawned=child is not None, worker_termination_verified=terminated,
        uncertain_ownership=uncertain, resource_envelope=CAPS, certificate_passed=False,
        resource_probe_only=True, scientific_fold_result=False, runtime_headroom_observation=observation)
    write(attempt/'status.json', result)
    if uncertain:
        write(attempt/'finalization-pending.json', dict(reason='Owned writer/group unresolved; do not seal or retry.', certificate_passed=False))
    else:
        files=[dict(path=str(p.relative_to(attempt)), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(attempt.rglob('*')) if p.is_file()]
        write(attempt/'receipt.json', dict(result, artifacts=files))
        for p in attempt.rglob('*'):
            if p.is_file(): p.chmod(0o444)
    print(json.dumps(result), flush=True)
    return 0 if failure is None else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', required=True); p.add_argument('--source', required=True)
    p.add_argument('--plan', required=True); p.add_argument('--plan-sha256', required=True)
    return execute(p.parse_args())

if __name__=='__main__': sys.exit(main())
