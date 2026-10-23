"""Mandatory, bounded per-file retrieval for future approved Huginn scans.

This module creates no resources. The worker supplies an exact already-created
identity, starts the read-only collector before numerics, and only reports
success after both numerical exit and the collector's verified final manifest.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from huginn_incremental_retrieval import (atomic_json, argument, digest_file,
                                          relative_name, verify_ownership)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'HUGINN_RETRIEVAL_PROTOCOL=2'


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else int(fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def is_huginn(jobset, arguments):
    job = shlex.split(arguments)
    return jobset == 'toolseq' and len(job) >= 3 and job[2] == 'huginn_chaos_map.py'


def validate_request(args, source, *, now=None, identity=process_identity):
    """Fail before creation unless a live matching controller protects this job."""
    if args.controller_root is None:
        raise ValueError('Huginn launch requires --controller-root and mandatory incremental retrieval')
    root = Path(args.controller_root).resolve()
    config = json.loads((root/'config.json').read_text())
    if Path(args.out).resolve() != root/'workers'/args.name:
        raise ValueError('worker output must be under the owning controller for watchdog visibility')
    if config['source_commit'] != source or config['approval'] != args.approval:
        raise ValueError('controller source or approval differs from worker')
    if config['names'].count(args.name) != 1:
        raise ValueError('worker must have one exact controller name')
    cmd = config['commands'][config['names'].index(args.name)]
    for flag, value in (('--name', args.name), ('--commit', source), ('--approval', args.approval),
                        ('--jobset', args.jobset), ('--jobset-args', args.jobset_args)):
        if argument(cmd, flag) != value:
            raise ValueError('controller command differs: '+flag)
    for flag, value in (('--run-minutes', args.run_minutes), ('--ready-minutes', args.ready_minutes),
                        ('--max-rate', args.max_rate), ('--max-cost', args.max_cost)):
        if float(argument(cmd, flag)) != value:
            raise ValueError('controller numerical bound differs: '+flag)
    if not is_huginn(args.jobset, args.jobset_args):
        raise ValueError('not the protected Huginn scan')
    hard, reserve = float(config['hard_minutes']), float(config['retrieval_minutes'])
    if not math.isfinite(hard) or not math.isfinite(reserve) or reserve <= 1 or hard <= reserve:
        raise ValueError('invalid nonextendable retrieval reserve')
    if args.max_rate > config['max_rate'] or args.max_cost > config['ceiling']/len(config['names']):
        raise ValueError('worker exceeds controller bounds')
    for parent in (root, *root.parents, Path(args.out).resolve(), *Path(args.out).resolve().parents):
        if (parent/'.git').exists():
            raise ValueError('controller and raw output must be outside repositories')
    state = json.loads((root/'state.json').read_text())
    ready = json.loads((root/'watchdog-ready.json').read_text())
    heartbeat = json.loads((root/'watchdog-heartbeat.json').read_text())
    now = time.time() if now is None else now
    if (state.get('finished') or identity(state['controller_pid']) != state['controller_ticks']
            or ready.get('ticks') is None or identity(ready['pid']) != ready['ticks']
            or heartbeat['pid'] != ready['pid']
            or not 0 <= now-heartbeat['time'] <= 120):
        raise ValueError('live controller and independent watchdog required before creation')
    runner = subprocess.run(['git', 'show', f'{source}:tools/runpod_huginn_chaos_runner.sh'],
                            cwd=ROOT, capture_output=True, check=True).stdout.decode()
    if PROTOCOL not in runner:
        raise ValueError('pinned Huginn runner lacks protected retrieval protocol; no pod created')
    return config


def remote_command(jobset, arguments, deadline):
    """Remote timeout stops the process tree before the original reserve starts.

The absolute deadline is evaluated remotely at execution, so an SSH connection
which starts late cannot extend the approved compute period. Ten seconds of
kill grace are included inside that period.
"""
    script = ('import os,sys,time; left=float(sys.argv[1])-time.time()-10; '
              'sys.exit(124) if left<=0 else None; '
              'os.execvp("timeout",["timeout","--signal=TERM","--kill-after=10s",'
              'f"{left:.3f}s",*sys.argv[2:]])')
    return shlex.join(['python3', '-c', script, str(deadline), 'bash',
                       '/root/neural-shrimp/tools/runpod_soft_runner.sh', jobset,
                       *shlex.split(arguments)])


def validate_scan_gates(out, binding):
    """Read the saved source/plan and native/throughput gates before success."""
    root = Path(out)/'data'
    plan_path = root/'plan.json'
    plan_sha = binding['source_files'][binding['plan']]['sha256']
    runtime = json.loads((root/'runtime.json').read_text())
    if (digest_file(plan_path) != plan_sha or runtime.get('source') != binding['source']
            or runtime.get('plan_sha256') != plan_sha or runtime.get('dtype') != 'float32'
            or runtime.get('tf32') is not False):
        raise ValueError('saved runtime/source/plan or arithmetic binding failed')
    plan = json.loads(plan_path.read_text())
    progress_path = root/'progress.jsonl'
    if progress_path.stat().st_size > 64*1024*1024:
        raise ValueError('progress exceeds bounded validation')
    rows = [json.loads(line) for line in progress_path.read_text().splitlines()]
    def one(phase):
        selected = [r for r in rows if r.get('phase') == phase]
        if len(selected) != 1:
            raise ValueError('one saved '+phase+' required')
        return selected[0]
    native, throughput = one('native_gate'), one('throughput_gate')
    error, repeat = native['error'], native['repeat']
    if not (math.isfinite(error) and 0 <= error <= 1e-6 and repeat == 0):
        raise ValueError('saved native gate failed')
    unit, projected, remaining = [throughput[k] for k in
                                 ('seconds_per_cell_step','projected_seconds_with_margin','remaining_seconds')]
    if (not all(math.isfinite(x) for x in (unit,projected,remaining))
            or unit <= 0 or projected <= 0 or projected > remaining-600):
        raise ValueError('saved throughput gate failed')
    total = plan['max_settings'] if plan.get('experiment') == 'HPC2' else plan['resolution']**2
    completed = [r for r in rows if r.get('phase') == 'chunk_complete']
    expected = list(range(0,total,plan['batch']))
    if (sorted(r['chunk_start'] for r in completed) != expected
            or any(r['total'] != total or r['completed'] != min(r['chunk_start']+plan['batch'],total)
                   for r in completed)):
        raise ValueError('saved completed chunk set failed')
    one('complete')
    if plan.get('experiment') == 'HPC2':
        replay = one('saved_step_gate')['relative_errors']
        if len(replay) != 6 or not all(math.isfinite(x) and 0 <= x <= 3e-5 for x in replay):
            raise ValueError('saved six-row replay gate failed')
        one('refinement_selected')
        if plan.get('retrieval_barrier'):
            acknowledged=[r['chunk_start'] for r in rows if r.get('phase')=='retrieval_verified']
            if acknowledged!=list(range(total)):
                raise ValueError('missing or out-of-order setting retrieval acknowledgment')
            gate=one('retrieval_throughput_gate')
            if not (0 < gate['projected_remaining'] <= gate['remaining']-60):
                raise ValueError('saved end-to-end throughput gate failed')
    return dict(native_gate_verified=True, throughput_gate_verified=True,
                source_plan_verified=True, completed_cells=total)


def verified_completion(out, identity, source_archive_sha, collector_returncode):
    """Validate the collector's completion attestation, never a progress marker."""
    out = Path(out)
    if collector_returncode != 0:
        raise ValueError('collector did not exit successfully')
    binding = json.loads((out/'identity.json').read_text())
    for key in ('pod_id', 'name', 'source', 'approval', 'hard_deadline_epoch'):
        if binding.get(key) != identity[key]:
            raise ValueError('collector identity differs: '+key)
    status = json.loads((out/'status.json').read_text())
    if (status.get('state') != 'complete' or status.get('complete') is not True
            or status.get('pod_id') != identity['pod_id']
            or status.get('source_archive_sha256') != source_archive_sha
            or binding.get('source_archive_sha256') != source_archive_sha):
        raise ValueError('verified complete collector status required')
    manifest_path = out/'data/manifest.json'
    if manifest_path.is_symlink() or digest_file(manifest_path) != status.get('manifest_sha256'):
        raise ValueError('verified final manifest missing or changed')
    manifest = json.loads(manifest_path.read_text())
    if not manifest:
        raise ValueError('empty final manifest')
    # The collector has streamed and independently hashed every file. Bind this
    # final success to its immutable per-file receipts as well as its exit code.
    verified = {}
    for line in (out/'receipts.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('event') in ('verified', 'skip_verified'):
            verified[row['file']] = row
    for name, sha in manifest.items():
        relative_name(name)
        path = out/'data'/name
        receipt = verified.get('data/'+name, {})
        if (path.is_symlink() or not path.is_file() or receipt.get('sha256') != sha
                or path.stat().st_size != receipt.get('bytes')):
            raise ValueError('manifest lacks a verified local file: '+name)
    gates = validate_scan_gates(out, binding)
    return dict(**gates, retrieval_mode='incremental_per_file', retrieval_complete=True,
                manifest_sha256=status['manifest_sha256'], manifest_files=len(manifest),
                data_root=str(out/'data'), archive_status='not_created_by_design',
                hash_verification='collector final full-file rehash; worker manifest/receipt cross-check')


class ProtectedRun:
    """Own only a local collector; resource lifecycle remains with the worker."""
    def __init__(self, controller, out, config, creation, args, source_archive_sha):
        self.out, self.controller = Path(out), Path(controller)
        self.identity = verify_ownership(config, [creation], creation['pod_id'], args.name,
                                         args.commit, args.approval)
        self.hard_deadline = self.identity['hard_deadline_epoch']
        self.compute_deadline = self.hard_deadline-60*float(config['retrieval_minutes'])
        self.retrieval_deadline = self.hard_deadline-60  # provider termination reserve
        self.source_archive_sha = source_archive_sha
        plan=json.loads(subprocess.check_output(['git','show',args.commit+':'+self.identity['plan']],cwd=ROOT))
        self.evidence_gate=plan.get('retrieval_barrier') is True
        if self.evidence_gate != (config.get('evidence_gate') is True):
            raise ValueError('controller evidence gate differs from pinned plan')
        self.numerical_started=False
        self.collector_restarts=0
        self.collector = None
        self.log = None
        local_sources = {name:digest_file(ROOT/'tools'/name) for name in
                         ('huginn_retrieval_guard.py', 'huginn_incremental_retrieval.py',
                          'runpod_soft_float64.py', 'runpod_scan_group.py')}
        self.fields = dict(**self.identity, local_control_sha256=local_sources,
                           worker_pid=os.getpid(), worker_ticks=process_identity(os.getpid()),
                           compute_deadline_epoch=self.compute_deadline, compute_stopped=False,
                           retrieval_complete=False, evidence_gate=self.evidence_gate)
        self.write('setup')

    def write(self, phase, **fields):
        self.fields.update(fields)
        atomic_json(self.out/'phase.json', dict(self.fields, phase=phase, updated_epoch=time.time()))

    def start(self):
        if time.time() >= self.compute_deadline-10:
            raise TimeoutError('compute budget exhausted before collector launch')
        self.log = (self.out/('incremental-stdout.txt' if self.collector_restarts==0 else f'incremental-restart-{self.collector_restarts}.txt')).open('x')
        command = [sys.executable, str(ROOT/'tools/huginn_incremental_retrieval.py'), '--watch',
                   '--poll-seconds', '10',
                   '--controller', str(self.controller), '--pod', self.identity['pod_id'],
                   '--name', self.identity['name'], '--source', self.identity['source'],
                   '--approval', self.identity['approval'], '--out', str(self.out/'incremental'),
                   '--hard-deadline-epoch', str(self.retrieval_deadline)]
        self.collector = subprocess.Popen(command, cwd=ROOT, stdout=self.log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
        self.write('setup', collector_pid=self.collector.pid,
                   collector_ticks=process_identity(self.collector.pid))
        until = min(time.time()+30, self.compute_deadline-10)
        while time.time() < until:
            if self.collector.poll() is not None:
                raise RuntimeError('mandatory collector exited before numerical launch')
            path = self.out/'incremental/status.json'
            if path.exists():
                status = json.loads(path.read_text())
                if status.get('pid') == self.collector.pid and status.get('pod_id') == self.identity['pod_id']:
                    if status.get('state') not in ('ready', 'collecting', 'waiting', 'transferring'):
                        raise RuntimeError('collector did not report healthy readiness')
                    return
            time.sleep(.2)
        raise TimeoutError('collector readiness not confirmed; numerical run refused')

    def supervise(self, process, *, clock=time.time, sleep=time.sleep):
        self.numerical_started=True
        self.write('running')
        failure_started = None
        failure_progress = 0
        while True:
            now = clock()
            numerical = process.poll()
            collected = self.collector.poll()
            if numerical is not None:
                # SSH rc255 is a transport failure, not evidence the remote
                # process stopped. Its absolute timeout still bounds its life.
                stopped = numerical != 255 or now >= self.compute_deadline
                if numerical != 0:
                    if failure_started is None:
                        failure_started = now
                    status_path = self.out/'incremental/status.json'
                    if status_path.exists():
                        status = json.loads(status_path.read_text())
                        if (status.get('pod_id') == self.identity['pod_id']
                                and status.get('pid') == self.collector.pid):
                            progress = sum(status.get(k, 0) for k in
                                           ('progress_bytes', 'verified_bytes', 'verification_bytes'))
                            if progress > failure_progress:
                                failure_progress, failure_started = progress, now
                self.write('retrieving', numerical_returncode=numerical, compute_stopped=stopped)
                if collected == 0:
                    receipt = verified_completion(self.out/'incremental', self.identity,
                                                   self.source_archive_sha, collected)
                    self.write('complete' if numerical == 0 else 'failed', **receipt)
                    return dict(receipt, runner_returncode=numerical,
                                success=numerical == 0, compute_stopped=stopped)
            if numerical is not None and numerical != 0 and getattr(self,'evidence_gate',False):
                raise RuntimeError('gated numerical exit; recover verified boundary before termination')
            if failure_started is not None and now-failure_started >= 120 and not getattr(self,'evidence_gate',False):
                raise RuntimeError('numerical process failed; partial retrieval made no new progress for 120s')
            if collected is not None and collected != 0:
                if not getattr(self,'evidence_gate',False):
                    raise RuntimeError('mandatory collector failed; preserving all retrieved/partial files')
                self.write('recovery',collector_returncode=collected,
                           recovery='numerics stop at setting barrier; retain resource within original deadline')
                if self.collector_restarts < 3 and now < self.compute_deadline-30:
                    self.collector_restarts += 1
                    if self.log is not None:self.log.close()
                    try:self.start()
                    except (RuntimeError,TimeoutError) as error:
                        self.write('recovery',recovery_error=str(error))

            if now >= self.retrieval_deadline:
                raise TimeoutError('termination reserve reached before verified retrieval')
            if numerical is None and now >= self.compute_deadline+15:
                raise TimeoutError('remote compute timeout did not return; refusing deadline extension')
            sleep(min(2, max(0, self.retrieval_deadline-now)))

    def salvage_before_termination(self, base, *, clock=time.time, sleep=time.sleep):
        """For gated runs, stop at the next verified boundary before deletion.

        A control-plane exception is not authority to destroy uncollected data.
        Only an acknowledged boundary or the original hard reserve ends recovery.
        """
        if not getattr(self,'evidence_gate',False):return {'enabled':False}
        if not getattr(self,'numerical_started',False):return {'enabled':True,'no_numerical_launch':True}
        root=self.identity['remote_data']
        command="import pathlib,sys; (pathlib.Path(sys.argv[1])/'.stop-after-setting').touch()"
        requested=False
        while clock()<self.retrieval_deadline:
            try:
                if not requested:
                    subprocess.run(base+['python3 -c '+shlex.quote(command)+' '+shlex.quote(root)],
                                   check=True,capture_output=True,timeout=min(30,max(.1,self.retrieval_deadline-clock())))
                    requested=True
                # Inspect the real remote boundary, not a stale local status.
                from huginn_incremental_retrieval import SSHReader
                reader=SSHReader(self.identity,self.retrieval_deadline);reader.base=base
                snapshot=reader.snapshot(root)
                barrier=snapshot.get('barrier')
                if barrier and requested:
                    receipts=self.out/'incremental/receipts.jsonl'
                    for line in receipts.read_text().splitlines():
                        row=json.loads(line)
                        if row.get('event')=='setting_acknowledged' and row.get('index')==barrier['index'] and row.get('files')==barrier['files']:
                            stopped=snapshot.get('stopped')
                            last=barrier['index']
                            if stopped==dict(index=last,source=self.identity['source']):
                                for name,sha in barrier['files'].items():
                                    if digest_file(self.out/'incremental/data'/relative_name(name))!=sha:
                                        raise ValueError('recovery boundary local hash mismatch')
                                self.write('recovered_partial',compute_stopped=True,recovery_setting=last)
                                return {'enabled':True,'verified_boundary':last}
                if self.collector is not None and self.collector.poll() is not None and self.collector_restarts<3 and clock()<self.compute_deadline-30:
                    self.collector_restarts+=1
                    if self.log is not None:self.log.close()
                    self.start()
                self.write('recovery',recovery='retaining resource for bounded evidence retrieval')
            except Exception as error:
                self.write('recovery',recovery_error=str(error))
            sleep(min(10,max(0,self.retrieval_deadline-clock())))
        return {'enabled':True,'deadline_reached':True,'complete_recovery':False}

    def close(self):
        if self.collector is not None and self.collector.poll() is None:
            self.collector.terminate()
            try:
                self.collector.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.collector.kill(); self.collector.wait(timeout=5)
        if self.log is not None:
            self.log.close()


REMOTE_DIAGNOSTICS = """import base64,hashlib,json,stat,sys
from pathlib import Path
root=Path('/root/neural-shrimp-data/explore/llm-soft-loop/pod64/logs')
name=sys.argv[1]
if not name or Path(name).name!=name or len(name)>120: raise ValueError('invalid job name')
rows=[]
for name in ['setup-uv.log','setup-torch.log','setup-packages.log','setup-model.log','hardware.txt','libraries.txt',name+'.log']:
 p=root/name
 if not p.exists(): rows.append({'name':name,'missing':True});continue
 if not stat.S_ISREG(p.lstat().st_mode): raise ValueError('nonregular diagnostic log')
 size=p.stat().st_size
 with p.open('rb') as f:
  f.seek(max(0,size-131072));data=f.read(131072)
 rows.append({'name':name,'remote_bytes':size,'captured_bytes':len(data),'tail_only':size>len(data),
  'captured_sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()})
print(json.dumps(rows))
"""


def retrieve_diagnostics(base, out, job_name, deadline):
    """Retain bounded log snapshots on success or failure, never repack tails."""
    seconds = min(20., deadline-time.time())
    if seconds <= 0:
        return dict(state='not_retrieved', reason='existing deadline exhausted')
    response = subprocess.run(base+['python3 -c '+shlex.quote(REMOTE_DIAGNOSTICS)+' '+shlex.quote(job_name)],
                              capture_output=True, check=True, timeout=seconds)
    if len(response.stdout) > 2*1024*1024:
        raise ValueError('diagnostic response exceeds bounded size')
    target = Path(out)/'diagnostics'
    target.mkdir(exist_ok=False)
    rows = json.loads(response.stdout)
    allowed = {'setup-uv.log','setup-torch.log','setup-packages.log','setup-model.log',
               'hardware.txt','libraries.txt',job_name+'.log'}
    if len(rows) != len(allowed) or {r['name'] for r in rows} != allowed:
        raise ValueError('diagnostic file list differs')
    for row in rows:
        if row.get('missing'):
            continue
        data = base64.b64decode(row.pop('base64'), validate=True)
        if (len(data)>131072 or len(data) != row['captured_bytes']
                or hashlib.sha256(data).hexdigest() != row['captured_sha256']):
            raise ValueError('diagnostic snapshot hash mismatch')
        with (target/relative_name(row['name'])).open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
    atomic_json(target/'receipt.json', dict(files=rows, captured_at_epoch=time.time(),
                                          note='bounded log snapshots; tail_only is explicit'))
    return dict(state='retrieved', receipt=str(target/'receipt.json'))


def terminate_and_record(pod_id, result, out, started, rp):
    """A racing watchdog must never prevent the worker's terminal receipt."""
    out = Path(out)
    result['wall_seconds'] = round(time.time()-started)
    result['termination_verified'] = False
    try:
        atomic_json(out/'result.json', result)
    except Exception as error:
        result['initial_receipt_error'] = str(error)
    try:
        direct = rp.pod_status(pod_id)
        if direct is not None:
            if direct.get('id') != pod_id or (result.get('name') and direct.get('name') != result['name']):
                raise ValueError('exact pod identity mismatch before termination')
            import argparse
            rp.cmd_terminate(argparse.Namespace(pod=pod_id))
    except BaseException as error:
        result['termination_error'] = str(error)
    finally:
        try:
            direct = rp.pod_status(pod_id)
            listed = pod_id in {p['id'] for p in rp.inventory()['pods']}
            result['termination_verified'] = direct is None and not listed
            result['termination_direct_absent'] = direct is None
            result['termination_inventory_absent'] = not listed
            if result['termination_verified']:
                rp.record({'event':'terminated_verified', 'pod_id':pod_id, 'verification':'worker_exact_recheck'})
        except BaseException as error:
            result['termination_verification_error'] = str(error)
        result['wall_seconds'] = round(time.time()-started)
        atomic_json(out/'result.json', result)
    return result['termination_verified']
