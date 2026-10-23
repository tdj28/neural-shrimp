"""Incremental recovery with optional hash-only evidence acknowledgments of an already approved Huginn map.

No resource creation, execution retry, termination or runner-log edits. Gated plans may write only exact per-setting hash acknowledgments. This
collector copies immutable, completed chunks before large tails, preserving
failed transfers. It is opt-in and independently bounded by the existing
controller's lifetime. Scientific auditing is a separate subsequent step.

Example (use a NEW destination outside every repository)::

    python tools/huginn_incremental_retrieval.py --watch \
      --controller /path/to/existing/controller --pod EXACT_ID \
      --name EXACT_NAME --source FULL_COMMIT --approval EXISTING_APPROVAL \
      --out /path/to/new/incremental-copy

The original collector preserved the complete HCM3 dataset (see its receipt).
Subsequent lifecycle integration and fault tests are documented separately;
past retrieval success does not qualify a changed deployment. No model code
is imported and no local GPU is used.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shlex
import stat
import subprocess
import tarfile
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            digest.update(block)
    return digest.hexdigest()


def relative_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name or '\\' in name:
        raise ValueError('unsafe relative path')
    return name


def argument(command, flag):
    if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
        raise ValueError(f'exactly one {flag} required')
    return command[command.index(flag) + 1]


def verify_ownership(config, entries, pod_id, name, source, approval):
    """Bind exact expected identity to one new creation and its controller."""
    if not re.fullmatch(r'[0-9a-f]{40}', source) or not approval or approval == 'PENDING':
        raise ValueError('full source and existing approval required')
    if config['source_commit'] != source or config['approval'] != approval:
        raise ValueError('controller source or approval mismatch')
    if name not in config['names'] or pod_id in config['preexisting_ids']:
        raise ValueError('name not owned or pod was preexisting')
    cmd = config['commands'][config['names'].index(name)]
    for flag, expected in [('--name', name), ('--commit', source), ('--approval', approval), ('--jobset', 'toolseq')]:
        if argument(cmd, flag) != expected:
            raise ValueError(f'controller command mismatch: {flag}')
    job = shlex.split(argument(cmd, '--jobset-args'))
    if len(job) < 3 or job[2] != 'huginn_chaos_map.py' or argument(job, '--source-commit') != source:
        raise ValueError('not the bound Huginn job')
    if argument(job, '--mode') != 'scan':
        raise ValueError('not a scan')
    output = relative_name(argument(job, '--out'))
    if not output.startswith('llm-soft-loop/pod64/'):
        raise ValueError('output outside allowed scan tree')
    plan = relative_name(argument(job, '--plan'))
    if not re.fullmatch(r'docs/huginn-chaos-(map|zoom|tight-zoom|continuation|transition)-plan.json', plan):
        raise ValueError('unrecognized scan plan')
    creates = [e for e in entries if e.get('event') == 'created' and e.get('pod_id') == pod_id]
    if len(creates) != 1:
        raise ValueError('one unambiguous creation record required')
    e = creates[0]
    if (e.get('name'), e.get('commit'), e.get('approval')) != (name, source, approval):
        raise ValueError('creation identity mismatch')
    created = datetime.fromisoformat(e['time']).timestamp()
    if not math.isfinite(created) or created < config['started_epoch']:
        raise ValueError('pod predates controller')
    lifetime = float(config['hard_minutes']) * 60
    if not math.isfinite(lifetime) or lifetime <= 0:
        raise ValueError('invalid existing hard lifetime')
    return dict(pod_id=pod_id, name=name, source=source, approval=approval,
                created_epoch=created, hard_deadline_epoch=created + lifetime,
                plan=plan, remote_data='/root/neural-shrimp-data/explore/' + output)


def pinned_source(repo, source):
    """Read the upload allowlist from the *pinned* uploader, never current HEAD."""
    code = subprocess.run(['git', 'show', f'{source}:tools/runpod_soft_float64.py'],
                          cwd=repo, capture_output=True, check=True).stdout
    tree = ast.parse(code)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'archives')
    assignment = next(n for n in function.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'selected' for t in n.targets))
    if not isinstance(assignment.value, ast.IfExp):
        raise ValueError('unrecognized pinned upload selection; manual review required')
    paths = ast.literal_eval(assignment.value.body)
    if not isinstance(paths, tuple) or not paths or not all(isinstance(x, str) for x in paths):
        raise ValueError('invalid pinned archive paths')
    paths = tuple(relative_name(p) for p in paths)
    # The source archive is small, but stream to disk to keep memory bounded.
    with tempfile.TemporaryFile() as archive:
        subprocess.run(['git', 'archive', '--format=tar', source, *paths], cwd=repo,
                       stdout=archive, check=True)
        archive.seek(0)
        archive_hash = hashlib.sha256()
        for block in iter(lambda: archive.read(BLOCK), b''):
            archive_hash.update(block)
        archive.seek(0)
        files = {}
        with tarfile.open(fileobj=archive, mode='r|') as tar:
            for member in tar:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError('source archive contains nonregular entry')
                name = relative_name(member.name)
                digest = hashlib.sha256()
                stream = tar.extractfile(member)
                for block in iter(lambda: stream.read(BLOCK), b''):
                    digest.update(block)
                files[name] = {'sha256': digest.hexdigest(), 'bytes': member.size}
        if set(files) != set(paths):
            raise ValueError('pinned archive file list mismatch')
    return files, archive_hash.hexdigest()


# These remote programs only read known project paths. All data arguments are
# shell-quoted. The runner and watchdog logs are never modified.
REMOTE_COMMON = '''import os,sys,json,hashlib,stat
from pathlib import Path
p=Path(sys.argv[1])
if not (str(p).startswith('/root/neural-shrimp/') or str(p).startswith('/root/neural-shrimp-data/explore/llm-soft-loop/pod64/')):
 raise ValueError('outside project paths')
def regular(q):
 if not stat.S_ISREG(q.lstat().st_mode): raise ValueError('not a regular file')
def sha(q):
 regular(q);h=hashlib.sha256()
 with q.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return {'sha256':h.hexdigest(),'bytes':q.stat().st_size}
'''
REMOTE_HASH = REMOTE_COMMON + "print(json.dumps(sha(p)))\n"
REMOTE_READ = REMOTE_COMMON + '''regular(p)
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''): sys.stdout.buffer.write(b)
'''
REMOTE_SNAPSHOT = REMOTE_COMMON + '''
progress=p/'progress.jsonl'
if not progress.exists(): print(json.dumps({'ready':False}));sys.exit(0)
regular(progress)
if progress.stat().st_size>67108864: raise ValueError('progress exceeds bounded reader')
rows=[]
with progress.open() as f:
 for line in f:
  if not line.endswith('\\n'): continue
  rows.append(json.loads(line))
complete=[r['chunk_start'] for r in rows if r.get('phase')=='chunk_complete']
phases=[r.get('phase') for r in rows]
manifest=None
if 'complete' in phases and (p/'manifest.json').exists():
 try:
  regular(p/'manifest.json')
  if (p/'manifest.json').stat().st_size>1048576: raise ValueError('oversize manifest')
  manifest=json.loads((p/'manifest.json').read_text())
 except json.JSONDecodeError: pass
print(json.dumps({'ready':'native_gate' in phases,'chunks':complete,'manifest':manifest,
 'stopped':json.loads((p/'.stopped-after-setting').read_text()) if (p/'.stopped-after-setting').exists() else None,
 'barrier':next((r['barrier'] for r in reversed(rows) if r.get('phase')=='retrieval_wait'),None),
 'artifacts':[r['file'] for r in rows if r.get('phase')=='artifact_complete'],
 'native_gate':next((r for r in rows if r.get('phase')=='native_gate'),None),
 'throughput_gate':next((r for r in rows if r.get('phase')=='throughput_gate'),None)}))
'''


class SSHReader:
    """A transport containing no write or lifecycle API."""
    def __init__(self, identity, deadline, status_lookup=None):
        self.identity, self.deadline = identity, deadline
        if status_lookup is None:
            import runpod_pod
            status_lookup = runpod_pod.pod_status
        self.lookup = status_lookup
        self.base = None

    def refresh(self):
        try:
            status = self.lookup(self.identity['pod_id'])
        except SystemExit as error:
            raise RuntimeError('read-only pod status lookup failed') from error
        if status is None:
            return False
        if (status.get('id'), status.get('name')) != (self.identity['pod_id'], self.identity['name']):
            raise ValueError('live pod identity mismatch')
        ports = ((status.get('runtime') or {}).get('ports') or [])
        port = next((p for p in ports if p.get('privatePort') == 22 and p.get('isIpPublic')), None)
        if port is None:
            # Identity was freshly checked above. Do not discard a working,
            # host-key-checked connection solely because API metadata is absent.
            if self.base is not None:
                return True
            raise RuntimeError('owned pod has no SSH endpoint')
        from runpod_phase0 import ssh_base
        self.base = ssh_base(port['ip'], port['publicPort'])
        return True

    def acknowledge(self, root, payload):
        if root != self.identity['remote_data'] or payload.get('source') != self.identity['source']:
            raise ValueError('acknowledgment identity mismatch')
        index=payload.get('index')
        if type(index) is not int or not 0 <= index < 10000:
            raise ValueError('invalid acknowledgment index')
        code = """import json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);v=json.loads(sys.argv[2]);d=p/'.retrieval-acks';d.mkdir(exist_ok=True)
f=d/(str(v['index']).zfill(4)+'.json');t=f.with_suffix('.tmp')
if f.exists():
 assert json.loads(f.read_text())==v
else:
 with t.open('w') as s:json.dump(v,s,sort_keys=True);s.flush();os.fsync(s.fileno())
 os.replace(t,f)
"""
        subprocess.run(self.base+['python3 -c '+shlex.quote(code)+' '+shlex.quote(root)+' '+shlex.quote(json.dumps(payload))],
                       check=True,timeout=min(30,max(.1,self.deadline-time.time())),capture_output=True)

    def command(self, code, path):
        return self.base + ['python3 -c ' + shlex.quote(code) + ' ' + shlex.quote(path)]

    def json(self, code, path):
        timeout = min(60., self.deadline - time.time())
        if timeout <= 0:
            raise TimeoutError('existing hard deadline reached')
        # JSON outputs are bounded by the remote programs, not raw array bytes.
        result = subprocess.run(self.command(code, path), capture_output=True, timeout=timeout, check=True)
        if len(result.stdout) > 2 * BLOCK:
            raise ValueError('remote metadata exceeds bound')
        return json.loads(result.stdout)

    def snapshot(self, path):
        return self.json(REMOTE_SNAPSHOT, path)

    def metadata(self, path):
        return self.json(REMOTE_HASH, path)

    def stream(self, path, target, timeout, pulse):
        """Stream at most one MiB at a time; timeout never discards the partial."""
        digest, total = hashlib.sha256(), 0
        end = min(self.deadline, time.time() + timeout)
        with tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(self.command(REMOTE_READ, path), stdout=subprocess.PIPE, stderr=stderr)
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    last_pulse = 0.
                    while True:
                        remaining = end - time.time()
                        if remaining <= 0:
                            raise TimeoutError('bounded artifact transfer expired')
                        if time.time() - last_pulse >= 15:
                            pulse(total)
                            last_pulse = time.time()
                        if not selector.select(min(1., remaining)):
                            continue
                        block = os.read(process.stdout.fileno(), BLOCK)
                        if not block:
                            break
                        target.write(block)
                        digest.update(block)
                        total += len(block)
                if process.wait(timeout=max(.01, end-time.time())):
                    raise RuntimeError('read-only SSH transfer failed')
            finally:
                if process.poll() is None:
                    process.kill()  # only this local SSH subprocess, never a pod
                    process.wait()
                process.stdout.close()
        return {'sha256': digest.hexdigest(), 'bytes': total}


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def sync_directory(path):
    """Persist newly published directory entries, not just their file contents."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)



def continuation_artifacts(plan):
    names = {'benchmark.npz'}
    for i in range(plan['max_settings']):
        names.add(f'initial-{i:04d}.npz')
        for phase, key in [('t','transient'),('m','measure')]:
            for end in range(plan['checkpoint_steps'],plan[key]+1,plan['checkpoint_steps']):
                names.add(f'states-{i:04d}-{phase}-{end:05d}.npz')
    return names

class Collector:
    def __init__(self, out, identity, source_files, source_archive_sha, plan, reader,
                 *, max_attempts=3, transfer_timeout=180, deadline=None):
        self.out, self.identity, self.reader = Path(out), identity, reader
        self.sources, self.plan = source_files, plan
        self.deadline = min(deadline or identity['hard_deadline_epoch'], identity['hard_deadline_epoch'])
        self.max_attempts, self.transfer_timeout = max_attempts, transfer_timeout
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / '.collector.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.binding = {**identity, 'source_archive_sha256': source_archive_sha,
                        'source_files': source_files, 'deadline_epoch': self.deadline}
        path = self.out/'identity.json'
        if path.exists():
            if json.loads(path.read_text()) != self.binding:
                raise ValueError('destination belongs to a different retrieval')
        else:
            if any(p.name != '.collector.lock' for p in self.out.iterdir()):
                raise ValueError('new destination must be empty or already identity-bound')
            atomic_json(path, self.binding)
        self.attempts = {}
        log = self.out/'receipts.jsonl'
        if log.exists():
            for line in log.read_text().splitlines():
                event = json.loads(line)
                if event['event'] == 'transfer_started':
                    name = event['file']; self.attempts[name] = self.attempts.get(name, 0) + 1
        self.acknowledged = set()
        self.source_done = False
        self.verified = {}  # session-only; every restart revalidates local files
        # Real byte progress, not heartbeat timestamps or bytes retransmitted
        # after a failure. Watchdogs may retain the maximum across restarts.
        self.received_highwater = {}
        self.verification_bytes = 0
        self.status('ready', complete=False)

    def close(self):
        self.lock.close()

    def event(self, event, **fields):
        row = dict(time=time.time(), event=event, **fields)
        encoded = (json.dumps(row, sort_keys=True, allow_nan=False)+'\n').encode()
        fd = os.open(self.out/'receipts.jsonl', os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o600)
        try:
            if os.write(fd, encoded) != len(encoded):
                raise IOError('short receipt append')
            os.fsync(fd)
        finally:
            os.close(fd)

    def status(self, state, **fields):
        atomic_json(self.out/'status.json', dict(time=time.time(), pid=os.getpid(), state=state,
                    pod_id=self.identity['pod_id'],
                    progress_bytes=sum(self.received_highwater.values()),
                    verified_files=len(self.verified),
                    verified_bytes=sum(value[1]['bytes'] for value in self.verified.values()),
                    verification_bytes=self.verification_bytes, **fields))

    def pulse(self, local, received):
        self.received_highwater[local] = max(received, self.received_highwater.get(local, 0))
        self.status('transferring', file=local, received_bytes=received)

    def transfer(self, remote, local, expected=None):
        relative_name(local)
        destination = self.out/local
        destination.parent.mkdir(parents=True, exist_ok=True)
        cached = self.verified.get(local)
        if cached and destination.exists() and not destination.is_symlink():
            current = destination.stat()
            signature = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)
            if signature == cached[0]:
                if expected and cached[1]['sha256'] != expected:
                    raise ValueError(f'cached file differs from final manifest: {local}')
                return True
        metadata = self.reader.metadata(remote)
        if not SHA.fullmatch(metadata.get('sha256', '')) or metadata.get('bytes', -1) < 0:
            raise ValueError('invalid remote file metadata')
        if expected and metadata['sha256'] != expected:
            raise ValueError(f'remote hash differs from pinned or manifest value: {local}')
        if destination.exists() or destination.is_symlink():
            if not stat.S_ISREG(destination.lstat().st_mode):
                raise ValueError('existing destination is not regular')
            if destination.stat().st_size != metadata['bytes'] or digest_file(destination) != metadata['sha256']:
                raise ValueError(f'existing file mismatch; preserved: {local}')
            current = destination.stat()
            self.verified[local] = ((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns), metadata)
            self.event('skip_verified', file=local, **metadata)
            return True
        attempts = self.attempts.get(local, 0)
        if attempts >= self.max_attempts:
            self.event('attempt_limit', file=local, attempts=attempts)
            return False
        if time.time() >= self.deadline:
            raise TimeoutError('existing hard deadline reached')
        self.attempts[local] = attempts + 1
        partial = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.partial')
        self.event('transfer_started', file=local, attempt=attempts+1,
                   partial=str(partial.relative_to(self.out)), expected=metadata)
        try:
            with partial.open('xb') as stream:
                actual = self.reader.stream(remote, stream, min(self.transfer_timeout, self.deadline-time.time()),
                                            lambda n:self.pulse(local, n))
                stream.flush(); os.fsync(stream.fileno())
            if actual != metadata or partial.stat().st_size != metadata['bytes']:
                raise ValueError('truncated or hash-mismatched transfer')
            # Atomic no-replace publication: unlike rename(), link cannot replace
            # an existing artifact. The verified partial is then unlinked.
            os.link(partial, destination)
            partial.unlink()
            sync_directory(destination.parent)
            current = destination.stat()
            self.verified[local] = ((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns), actual)
            self.received_highwater[local] = max(actual['bytes'], self.received_highwater.get(local, 0))
            self.event('verified', file=local, **actual)
            return True
        except Exception as error:
            self.event('transfer_error', file=local, error=str(error),
                       partial=str(partial.relative_to(self.out)), preserved=partial.exists())
            return False

    def names(self, snapshot):
        count = self.plan['max_settings'] if self.plan.get('experiment') == 'HPC2' else self.plan['resolution'] ** 2
        batch = self.plan['batch']
        chunks = sorted(set(snapshot.get('chunks', [])))
        if any(type(i) is not int or i < 0 or i >= count or i % batch for i in chunks):
            raise ValueError('invalid completed chunk marker')
        compact = [f'chunk-{i:04d}.npz' for i in chunks]
        tails = ([f'tail-{i:04d}-{offset}.npz' for i in chunks for offset in range(len(self.plan['eps']))]
                 if self.plan.get('retain_full_tails') else [])
        if self.plan.get('experiment') == 'HPC2':
            allowed = continuation_artifacts(self.plan)
            artifacts = snapshot.get('artifacts', [])
            if len(artifacts) != len(set(artifacts)) or not set(artifacts) <= allowed | set(compact) | {'map.npz'}:
                raise ValueError('Invalid continuation artifact marker')
            tails = sorted(set(artifacts) & allowed)
        return compact, tails

    def step(self):
        if time.time() >= self.deadline:
            self.status('deadline_reached', complete=False)
            return 'deadline'
        if not self.reader.refresh():
            self.status('pod_absent', complete=False)
            self.event('pod_absent')
            return 'absent'
        root = self.identity['remote_data']
        snapshot = self.reader.snapshot(root)
        self.status('collecting', completed_chunks=len(snapshot.get('chunks', [])))
        if not snapshot.get('ready'):
            return 'waiting'
        manifest = snapshot.get('manifest')
        compact, tails = self.names(snapshot)
        initial = ['plan.json', 'runtime.json', 'steering-direction.npy']
        if manifest is not None:
            if not isinstance(manifest, dict) or any(not SHA.fullmatch(v) for v in manifest.values()):
                raise ValueError('invalid final manifest')
            required = set(initial + compact + tails + ['map.npz', 'summary.json', 'progress.jsonl'])
            count = self.plan['max_settings'] if self.plan.get('experiment') == 'HPC2' else self.plan['resolution']**2
            expected_chunks = len(range(0, count, self.plan['batch']))
            if self.plan.get('experiment') == 'HPC2':
                required = set(initial + compact + ['map.npz', 'summary.json', 'progress.jsonl']) | continuation_artifacts(self.plan)
                if set(tails) != continuation_artifacts(self.plan):
                    raise ValueError('Incomplete continuation artifact markers')
            if len(compact) != expected_chunks or set(manifest) != required:
                raise ValueError('final manifest or completed chunk set is incomplete')
        # Immutable provenance first, then ALL completed compact chunks.
        for name in initial:
            expected = self.sources[self.identity['plan']]['sha256'] if name == 'plan.json' else None
            if not self.transfer(root+'/'+name, 'data/'+name, expected or (manifest or {}).get(name)):
                return 'partial'
        runtime = json.loads((self.out/'data/runtime.json').read_text())
        if (runtime.get('source') != self.identity['source'] or
                runtime.get('plan_sha256') != self.sources[self.identity['plan']]['sha256']):
            raise ValueError('runtime source/plan binding mismatch')
        if not self.source_done:
            for name, metadata in self.sources.items():
                if not self.transfer('/root/neural-shrimp/'+name, 'source/'+name, metadata['sha256']):
                    return 'partial'
            self.source_done = True
        # Setup writes these once before the numerical run begins. Preserve
        # them independently of the map manifest and the runner's archive.
        environment_root = '/root/neural-shrimp-data/explore/llm-soft-loop/pod64/logs'
        for name in ('hardware.txt', 'libraries.txt'):
            if not self.transfer(environment_root+'/'+name, 'environment/'+name):
                return 'partial'
        for name in compact:
            self.transfer(root+'/'+name, 'data/'+name, (manifest or {}).get(name))
        if manifest is not None:
            for name in ['manifest.json', 'map.npz', 'summary.json', 'progress.jsonl']:
                self.transfer(root+'/'+name, 'data/'+name, manifest.get(name))
            saved = self.out/'data/manifest.json'
            if saved.exists() and json.loads(saved.read_text()) != manifest:
                raise ValueError('manifest changed during retrieval')
        # One missing large file per pass: then recheck for new compact chunks.
        for name in tails:
            path = self.out/'data'/name
            if not path.exists() and self.attempts.get('data/'+name, 0) < self.max_attempts:
                self.transfer(root+'/'+name, 'data/'+name, (manifest or {}).get(name))
                break
        barrier=snapshot.get('barrier')
        if self.plan.get('retrieval_barrier') and barrier is not None:
            index=barrier.get('index')
            if type(index) is not int or not 0 <= index < self.plan['max_settings'] or barrier.get('source') != self.identity['source']:
                raise ValueError('invalid evidence barrier identity')
            required={'plan.json','runtime.json','steering-direction.npy','benchmark.npz',f'initial-{index:04d}.npz',f'chunk-{index:04d}.npz'}
            required.update(n for n in continuation_artifacts(self.plan) if n.startswith(f'states-{index:04d}-'))
            hashes=barrier.get('files',{})
            if set(hashes)!=required or any(not isinstance(v,str) or not SHA.fullmatch(v) for v in hashes.values()):
                raise ValueError('incomplete evidence barrier')
            if index not in self.acknowledged and all((self.out/'data'/n).is_file() for n in required):
                for name,sha in hashes.items():
                    if digest_file(self.out/'data'/name)!=sha:
                        raise ValueError('evidence barrier hash mismatch')
                self.reader.acknowledge(root,barrier)
                self.acknowledged.add(index)
                self.event('setting_acknowledged',index=index,files=hashes)
        if manifest is not None:
            saved_manifest = self.out/'data/manifest.json'
            all_present = saved_manifest.is_file() and all((self.out/'data'/name).is_file() for name in manifest)
            # One full independent final pass, never quadratically rehash large
            # tails while the remaining files are still being collected.
            if all_present:
                self.status('verifying_manifest', complete=False)
                verified_bytes = 0
                for name, sha in manifest.items():
                    if time.time() >= self.deadline:
                        self.status('deadline_reached', complete=False)
                        return 'deadline'
                    path = self.out/'data'/name
                    if path.is_symlink() or digest_file(path) != sha:
                        raise ValueError(f'final local manifest mismatch; preserved: {name}')
                    verified_bytes += path.stat().st_size
                    self.verification_bytes = max(self.verification_bytes, verified_bytes)
                    self.status('verifying_manifest', complete=False, file='data/'+name)
            if all_present and json.loads(saved_manifest.read_text()) == manifest:
                sync_directory(self.out/'data')
                self.status('complete', complete=True, manifest_sha256=digest_file(saved_manifest),
                            source_archive_sha256=self.binding['source_archive_sha256'])
                self.event('all_manifest_files_verified', files=len(manifest),
                           manifest_sha256=digest_file(saved_manifest))
                return 'complete'
        self.status('waiting', complete=False, completed_chunks=len(compact))
        return 'partial'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--once', action='store_true')
    modes.add_argument('--watch', action='store_true')
    p.add_argument('--controller', type=Path, required=True)
    p.add_argument('--ledger', type=Path, default=ROOT.parent/'neural-shrimp-data/runpod/pods.jsonl')
    for flag in ('pod', 'name', 'source', 'approval'):
        p.add_argument('--'+flag, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hard-deadline-epoch', type=float, help='may shorten, never extend existing lifetime')
    p.add_argument('--poll-seconds', type=float, default=30)
    p.add_argument('--transfer-timeout', type=float, default=180)
    p.add_argument('--max-attempts', type=int, default=3)
    args = p.parse_args(argv)
    if not 10 <= args.poll_seconds <= 30 or not 1 <= args.max_attempts <= 3 or not 1 <= args.transfer_timeout <= 300:
        p.error('poll must be 10..30s, attempts 1..3, transfer timeout 1..300s')
    config = json.loads((args.controller/'config.json').read_text())
    entries = []
    lines = args.ledger.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1 or line.endswith('\n'):
                raise  # only a concurrently appended final partial line is ignored
    identity = verify_ownership(config, entries, args.pod, args.name, args.source, args.approval)
    deadline = args.hard_deadline_epoch or identity['hard_deadline_epoch']
    if not math.isfinite(deadline) or deadline > identity['hard_deadline_epoch']:
        p.error('deadline must not exceed the existing approved lifetime')
    # Refuse destinations in any Git checkout, including an unrelated repository.
    for parent in (args.out.resolve(), *args.out.resolve().parents):
        if (parent/'.git').exists():
            p.error('raw destination must be outside every repository')
    sources, archive_sha = pinned_source(ROOT, args.source)
    plan_bytes = subprocess.run(['git', 'show', f"{args.source}:{identity['plan']}"], cwd=ROOT,
                                capture_output=True, check=True).stdout
    if hashlib.sha256(plan_bytes).hexdigest() != sources[identity['plan']]['sha256']:
        raise ValueError('pinned plan binding mismatch')
    reader = SSHReader(identity, deadline)
    collector = Collector(args.out, identity, sources, archive_sha, json.loads(plan_bytes), reader,
                          deadline=deadline, max_attempts=args.max_attempts, transfer_timeout=args.transfer_timeout)
    consecutive_errors = 0
    try:
        while True:
            try:
                result = collector.step()
                consecutive_errors = 0
            except Exception as error:
                consecutive_errors += 1
                collector.event('collector_error', error=str(error), consecutive_errors=consecutive_errors)
                collector.status('error', complete=False, error=str(error))
                result = 'error'
            if result in ('complete', 'absent', 'deadline') or args.once or (consecutive_errors >= 3 and not collector.plan.get('retrieval_barrier')):
                return 0 if result == 'complete' else 2 if result in ('absent', 'deadline', 'error') else 1
            time.sleep(min(args.poll_seconds, max(0., deadline-time.time())))
    finally:
        collector.close()


if __name__ == '__main__':
    raise SystemExit(main())
