"""Task19 finite return-map experiment: unchanged snapshot/owned-process lifecycle.

Lifecycle reused from Task7 without changing its existing source. The exact plan
sets the deadline; owned cleanup has at most ten additional seconds. Available
RAM and aggregate output bytes are sampled; address space is a hard limit.
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
OWNED = Path('/data2/neural-shrimp-data/codex/llm-return-map-crosscheck')
GIB = 1024**3
OWNERSHIP_FILE = 'codex/certify-llm/derivative_campaign.py'
OWNERSHIP_SHA = '9357d0aee5e5880b37c752a5c928c988480c63a1871db94f1fc3f8b3df70ee52'
WORKER_FILES = ('codex/return-map-crosscheck/reduce.py', 'codex/return-map-crosscheck/trajectory.py',
                'codex/return-map-crosscheck/runtime.py', 'codex/hostile-fold-crosscheck/assets.py',
                'codex/crosscheck/qwen_map.py')
CAPS = {mode: dict(cpu_threads=cpu, nice=10, memory_gib=memory, timeout_seconds=seconds,
            minimum_start_available_gib=start, minimum_runtime_available_gib=16,
            maximum_output_mib=512)
        for mode, cpu, memory, seconds, start in [('trajectory',2,12,4200,36), ('saved',1,4,300,20)]}


def specification(plan_name):
    names = {mode+'-001-plan.json': mode for mode in CAPS}
    if plan_name not in names:
        raise ValueError('only exact Task19 trajectory001/saved001 plans are admitted')
    mode = names[plan_name]
    return dict(schema='codex-return-map-v1', caps=dict(CAPS[mode]), mode=mode,
                attempt=mode+'-001', workers=WORKER_FILES, worker='runtime.py',
                card=mode+'-001-card.md', dispatch=mode+'-001-dispatch.sh',
                resource_probe_only=False)


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


def establish_limits(caps):
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    if nice > 10: raise RuntimeError('cannot raise inherited process priority')
    if nice < 10: os.nice(10-nice)
    if os.getpriority(os.PRIO_PROCESS, 0) != 10: raise RuntimeError('nice10 not established')
    bound = caps['memory_gib']*GIB
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    if hard != resource.RLIM_INFINITY and hard < bound:
        raise RuntimeError('inherited hard AS below mode bound')
    resource.setrlimit(resource.RLIMIT_AS, (bound, bound))


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable absent')


def validate_arguments(args):
    if not args.attempt or Path(args.attempt).name != args.attempt or args.attempt.startswith('.'):
        raise ValueError('exclusive direct-child attempt name required')
    if args.attempt != specification(args.plan)['attempt']:
        raise ValueError('exact corresponding mode attempt required')
    if re.fullmatch('[0-9a-f]{40}', args.source) is None or re.fullmatch('[0-9a-f]{64}', args.plan_sha256) is None:
        raise ValueError('literal external source and plan pins required')


def guard(deadline, stopped, caps, *, startup=False):
    if stopped: raise InterruptedError(stopped[0])
    if time.monotonic() >= deadline: raise TimeoutError('absolute Task19 plan deadline')
    floor = caps['minimum_start_available_gib' if startup else 'minimum_runtime_available_gib']*GIB
    value = available()
    if value < floor: raise RuntimeError(f'Task19 MemAvailable floor: {value} < {floor}')
    return value


def snapshot(repo, args, deadline, stopped):
    """Only Git and stdlib admission; never import Torch or inspect seed values."""
    spec = specification(args.plan); caps = spec['caps']
    def git(*words):
        guard(deadline, stopped, caps)
        return subprocess.check_output(['git', *words], cwd=repo,
                    timeout=min(30, max(.001, deadline-time.monotonic())))
    if git('rev-parse', 'HEAD').decode().strip() != args.source:
        raise ValueError('current HEAD differs from exact source')
    planpath = repo/'codex/return-map-crosscheck'/args.plan
    if sha(planpath) != args.plan_sha256: raise ValueError('external original plan hash differs')
    plan = json.loads(planpath.read_text())
    if plan.get('admission') != 'reviewed': raise ValueError('prepared plan is not launchable')
    if plan.get('schema') != spec['schema'] or type(plan.get('envelope')) is not dict or set(plan['envelope']) != set(caps) or any(type(plan['envelope'][k]) is not int or plan['envelope'][k] != v for k,v in caps.items()):
        raise ValueError('exact corresponding mode resource plan required')
    if plan.get('mode') != spec['mode']: raise ValueError('exact plan mode differs')
    if set(plan.get('source_sha256', {})) != set(spec['workers']):
        raise ValueError('complete original worker import inventory required')
    paths = list(spec['workers'])+[OWNERSHIP_FILE, 'codex/return-map-crosscheck/launch.py',
               'codex/return-map-crosscheck/'+args.plan, 'codex/return-map-crosscheck/'+spec['card'],
               'codex/return-map-crosscheck/'+spec['dispatch']]
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
    guard(deadline, stopped, caps, startup=True)
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


def monitor(child, identity, owner, deadline, stopped, observation, caps, output_root=None):
    while True:
        remaining = guard(deadline, stopped, caps)
        observation['samples'] += 1
        observation['minimum_available_bytes'] = min(observation['minimum_available_bytes'], remaining)
        if output_root is not None:
            size = sum(p.stat().st_size for p in output_root.rglob('*') if p.is_file())
            observation['maximum_output_bytes'] = max(observation.get('maximum_output_bytes', 0), size)
            if size > caps['maximum_output_mib']*1024**2:
                raise RuntimeError('Task19 sampled output byte cap exceeded')
        code = child.poll()
        if code is not None:
            if owner.group(identity): raise RuntimeError('descendants outlived worker')
            return code
        time.sleep(min(.5, max(.001, deadline-time.monotonic())))


def validate_completion(status, result, mode, source):
    """Complete finite computation may disagree with every producer claim."""
    if (status.get('status') != 'completed'
        or status.get('computational_completed') is not True
        or status.get('certificate_passed') is not False
        or status.get('experimental_claims') is not False
        or status.get('mode') != mode or status.get('source_commit') != source):
        raise RuntimeError('worker did not retain exact mode computational completion')
    if mode == 'trajectory':
        counts = dict(transient_steps=1000, recorded_steps=2000, native_step_calls=3000,
                      row_forwards=7000, seed=0)
        if any(type(result.get(k)) is not int or result[k] != v for k,v in counts.items()):
            raise RuntimeError('full fixed trajectory counts required')
        if result.get('offsets') != [1e-6,1e-9] or result.get('post_asset_hashes_equal') is not True:
            raise RuntimeError('both fixed twins and original asset postcheck required')
        windows=result.get('disjoint_windows',[]); running=result.get('running_means',[])
        if (len(windows)!=4 or [(r.get('start'),r.get('stop')) for r in windows]!=[(i,i+500) for i in range(0,2000,500)]
            or len(running)!=20 or [r.get('stop') for r in running]!=list(range(100,2001,100))):
            raise RuntimeError('every frozen twin window required')
    else:
        rows=result.get('rows')
        if (result.get('all65_admitted') is not True or type(rows) is not list or len(rows)!=65
            or any(type(r) is not dict or type(r.get('index')) is not int for r in rows)
            or [r['index'] for r in rows]!=list(range(65))):
            raise RuntimeError('all65 original ordered dispositions required')


def execute(args):
    began = time.monotonic(); validate_arguments(args)
    spec = specification(args.plan); caps = spec['caps']
    deadline = began+caps['timeout_seconds']; establish_limits(caps)
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
        launcher_pid=os.getpid(), resource_envelope=caps, started_utc=datetime.now(timezone.utc).isoformat(),
        certificate_passed=False, resource_probe_only=spec['resource_probe_only']))
    try:
        guard(deadline, stopped, caps, startup=True)
        plan, blobs, remote = snapshot(REPO, args, deadline, stopped)
        source = attempt/'source'
        for name, data in blobs.items():
            target = source/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        owner = module(source/OWNERSHIP_FILE, 'task19_reviewed_process_ownership')
        command = [sys.executable, '-I', '-B', '-u', str(source/'codex/return-map-crosscheck'/spec['worker']),
                   '--plan', str(source/'codex/return-map-crosscheck'/args.plan), '--output', str(attempt/'run')]
        manifest = dict(source_commit=args.source, plan_sha256=args.plan_sha256, remote_main=remote,
            remote_ancestry_verified_at=datetime.now(timezone.utc).isoformat(), command=command,
            launcher_identity=owner.serial_identity(owner.proc(os.getpid())), **caps,
            source=[dict(path=n, bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for n,b in blobs.items()],
            sampled_headroom_is_hard_cap=False, cleanup_seconds=10,
            available_before_worker_bytes=available())
        write(attempt/'manifest.json', manifest)
        env = os.environ.copy(); env.update(OMP_NUM_THREADS=str(caps['cpu_threads']), MKL_NUM_THREADS=str(caps['cpu_threads']), OPENBLAS_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
            PYTHONDONTWRITEBYTECODE='1', TASK19_SOURCE_COMMIT=args.source, TASK19_PLAN_SHA256=args.plan_sha256)
        env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
        guard(deadline, stopped, caps, startup=True)
        with (attempt/'stdout.log').open('xb') as out, (attempt/'stderr.log').open('xb') as err:
            child = subprocess.Popen(command, cwd=source, env=env, stdout=out, stderr=err,
                                     preexec_fn=lambda: establish_limits(caps), start_new_session=True)
            # Popen ownership is retained before any fallible inspection or write.
            q = owner.proc(child.pid); identity = owner.Identity(q)
            if q['ppid'] != os.getpid() or q['pgid'] != child.pid or q['sid'] != child.pid or q['uid'] != os.getuid():
                raise RuntimeError('worker birth/session/parent ownership mismatch')
            command_deadline = min(deadline, time.monotonic()+5)
            while [os.fsdecode(v) for v in q['command']] != command:
                guard(deadline, stopped, caps)
                if time.monotonic() >= command_deadline or q['state'] == 'Z':
                    raise RuntimeError('worker exact command not admitted before deadline')
                time.sleep(.01); q = owner.proc(child.pid)
                if q is None or q['start'] != identity.value['start']:
                    raise RuntimeError('worker birth changed before command admission')
            limits = next(line.split()[-3:-1] for line in (Path('/proc')/str(child.pid)/'limits').read_text().splitlines() if line.startswith('Max address space'))
            if q['nice'] != 10 or limits != [str(caps['memory_gib']*GIB)]*2:
                raise RuntimeError('worker live nice/AS differs')
            manifest['worker_identity'] = owner.serial_identity(q); manifest['owned_pid'] = child.pid
            write(attempt/'manifest.json', manifest)
            print(json.dumps(dict(attempt=str(attempt), source_commit=args.source, worker_identity=manifest['worker_identity'])), flush=True)
            code = monitor(child, identity, owner, deadline, stopped, observation, caps, attempt)
            terminated = True
            if code != 0: raise RuntimeError('Task19 worker failed with exit'+str(code))
            status = json.loads((attempt/'run/status.json').read_text())
            result = json.loads((attempt/'run/result.json').read_text())
            validate_completion(status, result, spec['mode'], args.source)
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
        status='return_map_completed' if failure is None else 'return_map_failed', mode=spec['mode'],
        timed_out=bool(failure and 'TimeoutError' in failure['error']), elapsed_seconds=time.monotonic()-began,
        worker_spawned=child is not None, worker_termination_verified=terminated,
        uncertain_ownership=uncertain, resource_envelope=caps, certificate_passed=False,
        resource_probe_only=False, finite_result_available=failure is None,
        runtime_headroom_observation=observation)
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
