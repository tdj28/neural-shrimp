"""Real tiny subprocesses only: no checkpoint/model/target-artifact admission."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codex/certify-llm'))
import parallel_phase_pass as scheduler

SYNTHETIC = r'''
import argparse,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--request-sha256');p.add_argument('--phase',type=int);p.add_argument('--output');a=p.parse_args()
q=json.loads(Path(a.request).read_text());out=Path(a.output);out.mkdir()
def save(name,v): (out/name).write_text(json.dumps(v,sort_keys=True))
def sha(name):return hashlib.sha256((out/name).read_bytes()).hexdigest()
(out/'request.json').write_bytes(Path(a.request).read_bytes())
save('inputs.json',dict(request_sha256=a.request_sha256,source_commit=q['source_commit'],phase_index=a.phase,map_sha256=q['map_sha256'],backend_sha256=q['backend_sha256'],source_sha256=q['source_sha256'],certificate_passed=False))
save('runtime.json',dict(pid=os.getpid(),ppid=os.getppid(),env={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','CUDA_VISIBLE_DEVICES']},nice=os.getpriority(os.PRIO_PROCESS,0),limit=list(resource.getrlimit(resource.RLIMIT_AS)),flags=[sys.flags.isolated,sys.flags.dont_write_bytecode]))
mode=q.get('test_mode','success')
if mode=='hang':
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 time.sleep(30)
if mode=='crash' and a.phase==2:
 save('failure-evidence.json',{'retained':True});sys.exit(7)
if mode=='grandchild' and a.phase==0:
 child=subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(30)']);save('child.json',{'pid':child.pid});time.sleep(.2)
time.sleep(.01*(7-a.phase))
for name in ['input.json','live-input.json','point.json']:save(name,{'phase':a.phase,'synthetic':True})
status=dict(status='phase_point_completed',phase_index=a.phase,request_sha256=a.request_sha256,certificate_passed=False,input_sha256=sha('input.json'),live_input_sha256=sha('live-input.json'),point_sha256=sha('point.json'))
if mode=='bad_status' and a.phase==0:status['point_sha256']='0'*64
save('status.json',status)
'''


def setup_case(tmp_path, monkeypatch, mode='success'):
    owned = tmp_path / 'owned';candidate = owned / 'attempt' / 'run' / 'candidate-01';candidate.mkdir(parents=True)
    worker = tmp_path / 'synthetic_worker.py';worker.write_text(SYNTHETIC)
    request = dict(source_commit='a'*40,source_sha256={scheduler.WORKER_KEY:scheduler.digest(worker)},
                   map_sha256='b'*64,backend_sha256='c'*64,test_mode=mode)
    path = candidate / 'request.json';path.write_text(json.dumps(request,sort_keys=True))
    monkeypatch.setattr(scheduler,'OWNED',owned);monkeypatch.setenv('CERT_SOURCE_COMMIT','a'*40)
    return candidate,path,scheduler.digest(path),worker


def run(case, **kwargs):
    candidate,path,digest,worker=case
    return scheduler.run_parallel_pass(path,digest,candidate,deadline=time.monotonic()+610,worker_path=worker,**kwargs)


def test_eight_reordered_completions_canonical_and_real_bootstrap(tmp_path,monkeypatch):
    case=setup_case(tmp_path,monkeypatch);result=run(case);candidate=case[0]
    assert result['status']=='completed' and result['unstarted_phases']==[]
    assert [row['phase_index'] for row in result['phase_receipts']]==list(range(8))
    assert result['certificate_passed'] is False and result['scientific_endpoint_audit_performed'] is False
    assert result['sampled_rss_is_hard_cap'] is False and result['peak_sampled_rss_bytes']>0
    for i in range(8):
        runtime=json.loads((candidate/f'phase-{i:02d}'/'runtime.json').read_text())
        assert runtime['flags']==[1,1] and runtime['nice']>=10
        assert all(v=='1' for k,v in runtime['env'].items() if k!='CUDA_VISIBLE_DEVICES')
        assert runtime['env']['CUDA_VISIBLE_DEVICES']=='' and max(runtime['limit'])<=18*scheduler.GIB
        receipt=json.loads((candidate/'parallel-workers'/f'phase-{i:02d}.receipt.json').read_text())
        bootstrap=json.loads((candidate/'parallel-workers'/f'phase-{i:02d}.bootstrap.json').read_text())
        assert receipt['status']=='completed' and receipt['exit_code']==0 and receipt['verified_exit']
        assert bootstrap['pid']==runtime['pid'] and bootstrap['parent_death_signal']=='SIGKILL'
    events=[json.loads(line) for line in (candidate/'parallel-events.jsonl').read_text().splitlines()]
    assert sorted(e['phase_index'] for e in events if e['event']=='worker_started')==list(range(8))


@pytest.mark.parametrize('mode',['crash','bad_status'])
def test_failures_preserve_every_started_attempt(tmp_path,monkeypatch,mode):
    case=setup_case(tmp_path,monkeypatch,mode)
    with pytest.raises(scheduler.PassFailure):run(case)
    result=json.loads((case[0]/'parallel-pass.json').read_text())
    assert result['status']=='partial_or_failed'
    assert result['phase_receipts'] and len(result['phase_receipts'])+len(result['unstarted_phases'])==8
    for row in result['phase_receipts']:
        receipt=json.loads((case[0]/'parallel-workers'/f'phase-{row["phase_index"]:02d}.receipt.json').read_text())
        assert receipt['verified_exit']
    if mode=='crash':assert (case[0]/'phase-02'/'failure-evidence.json').is_file()


def test_per_worker_timeout_kills_ignoring_sigterm(tmp_path,monkeypatch):
    case=setup_case(tmp_path,monkeypatch,'hang');started=time.monotonic()
    with pytest.raises(scheduler.PassFailure,match='timeout'):run(case,per_worker_timeout=.25)
    assert time.monotonic()-started<8
    result=json.loads((case[0]/'parallel-pass.json').read_text())
    assert result['unstarted_phases']==[4,5,6,7]
    for i in range(4):
        receipt=json.loads((case[0]/'parallel-workers'/f'phase-{i:02d}.receipt.json').read_text())
        assert receipt['verified_exit'] and receipt['exit_code']<0
        assert scheduler.proc_record(receipt['pid']) is None


def test_frozen_reserve_refuses_all_launches(tmp_path,monkeypatch):
    candidate,path,digest,worker=setup_case(tmp_path,monkeypatch)
    with pytest.raises(scheduler.PassFailure,match='reserve'):
        scheduler.run_parallel_pass(path,digest,candidate,deadline=time.monotonic()+599,worker_path=worker)
    result=json.loads((candidate/'parallel-pass.json').read_text());assert result['unstarted_phases']==list(range(8))
    assert not list(candidate.glob('phase-*'))


@pytest.mark.parametrize('fault',['request_hash','snapshot','source','stale_phase','stale_controller','completed_attempt','layout'])
def test_admission_refuses_before_any_spawn(tmp_path,monkeypatch,fault):
    candidate,path,digest,worker=setup_case(tmp_path,monkeypatch)
    if fault=='request_hash':digest='d'*64
    if fault=='snapshot':monkeypatch.setenv('CERT_SOURCE_COMMIT','f'*40)
    if fault=='source':worker.write_text(SYNTHETIC+'\n# changed')
    if fault=='stale_phase':(candidate/'phase-07').mkdir()
    if fault=='stale_controller':(candidate/'parallel-pass.json').write_text('{}')
    if fault=='completed_attempt':(candidate.parents[1]/'receipt.json').write_text('{}')
    if fault=='layout':candidate=candidate.parent
    monkeypatch.setattr(scheduler.subprocess,'Popen',lambda *a,**k:pytest.fail('spawn before admission'))
    with pytest.raises(ValueError):scheduler.run_parallel_pass(path,digest,candidate,deadline=time.monotonic()+610,worker_path=worker)


@pytest.mark.parametrize('changes',[{'max_workers':True},{'max_workers':5},{'max_workers':2},
    {'per_worker_timeout':601},{'per_worker_timeout':float('nan')},{'per_worker_timeout':False},
    {'aggregate_rss_bytes':33*scheduler.GIB},{'minimum_available_bytes':11*scheduler.GIB}])
def test_envelope_refusals(tmp_path,monkeypatch,changes):
    with pytest.raises(ValueError):run(setup_case(tmp_path,monkeypatch),**changes)


def test_memory_guard_retains_started_jobs_and_kills_only_own(tmp_path,monkeypatch):
    case=setup_case(tmp_path,monkeypatch,'hang');actual=scheduler.resource_snapshot
    def measured(pid,groups):
        data=actual(pid,groups)
        if groups:data['rss_bytes']=32*scheduler.GIB
        return data
    monkeypatch.setattr(scheduler,'resource_snapshot',measured)
    with pytest.raises(scheduler.PassFailure,match='RSS'):run(case)
    result=json.loads((case[0]/'parallel-pass.json').read_text())
    assert result['peak_sampled_rss_bytes']==32*scheduler.GIB
    assert len(result['phase_receipts'])==1
    receipt=json.loads((case[0]/'parallel-workers'/'phase-00.receipt.json').read_text())
    assert receipt['verified_exit'] and scheduler.proc_record(receipt['pid']) is None


def test_insufficient_available_memory_no_spawn(tmp_path,monkeypatch):
    case=setup_case(tmp_path,monkeypatch)
    monkeypatch.setattr(scheduler,'resource_snapshot',lambda *a:dict(rss_bytes=1,available_bytes=1,processes=[]))
    with pytest.raises(scheduler.PassFailure,match='headroom'):run(case)
    result=json.loads((case[0]/'parallel-pass.json').read_text());assert result['unstarted_phases']==list(range(8))


def test_parent_identity_guard_refuses_unowned_bootstrap(tmp_path):
    worker=tmp_path/'tiny.py';worker.write_text('raise RuntimeError("must not execute")')
    record=tmp_path/'record.json'
    completed=subprocess.run([sys.executable,'-I','-B',str(Path(scheduler.__file__)),'--_worker-bootstrap',
        '--parent-pid',str(os.getpid()),'--parent-start','0','--worker',str(worker),
        '--worker-sha256',scheduler.digest(worker),'--scheduler-sha256',scheduler.digest(Path(scheduler.__file__)),'--wall-seconds','1','--record',str(record)],
        capture_output=True,text=True,timeout=5)
    assert completed.returncode!=0 and 'controller identity disappeared' in completed.stderr and not record.exists()


def test_unexpected_descendant_refuses_success_and_group_is_terminated(tmp_path,monkeypatch):
    case=setup_case(tmp_path,monkeypatch,'grandchild')
    with pytest.raises(scheduler.PassFailure):run(case)
    child=json.loads((case[0]/'phase-00'/'child.json').read_text())['pid']
    current=scheduler.proc_record(child)
    assert current is None or current['state']=='Z'
    receipt=json.loads((case[0]/'parallel-workers'/'phase-00.receipt.json').read_text())
    assert 'descendants' in receipt['validation_error'] and receipt['status']=='partial_or_failed'


@pytest.mark.parametrize('sig',[signal.SIGTERM,signal.SIGKILL])
def test_controller_termination_does_not_orphan_direct_workers(tmp_path,monkeypatch,sig):
    candidate,path,sha,worker=setup_case(tmp_path,monkeypatch,'hang')
    runner=tmp_path/'runner.py'
    runner.write_text('import sys,time\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(Path(scheduler.__file__).parent))+')\nimport parallel_phase_pass as s\ns.OWNED=Path('+repr(str(candidate.parents[2]))+')\ns.run_parallel_pass(Path('+repr(str(path))+'),'+repr(sha)+',Path('+repr(str(candidate))+'),deadline=time.monotonic()+610,worker_path=Path('+repr(str(worker))+'))\n')
    log=(tmp_path/'controller.log').open('wb')
    proc=subprocess.Popen([sys.executable,'-I','-B',str(runner)],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
    try:
        end=time.monotonic()+5
        # Wait for initialized bootstrap, not just Popen, to test actual death protection.
        while time.monotonic()<end and len(list((candidate/'parallel-workers').glob('*.bootstrap.json')))<4:
            assert proc.poll() is None;time.sleep(.02)
        bootstrap=list((candidate/'parallel-workers').glob('*.bootstrap.json'));assert len(bootstrap)==4
        pids=[json.loads(p.read_text())['pid'] for p in bootstrap]
        proc.send_signal(sig);proc.wait(timeout=8)
        end=time.monotonic()+3
        while time.monotonic()<end:
            remaining=[r for pid in pids if (r:=scheduler.proc_record(pid)) is not None and r['state']!='Z']
            if not remaining:break
            time.sleep(.02)
        assert not remaining
        if sig==signal.SIGTERM:
            report=json.loads((candidate/'parallel-pass.json').read_text())
            assert report['status']=='partial_or_failed' and all(row['status']=='partial_or_failed' for row in report['phase_receipts'])
    finally:
        if proc.poll() is None:proc.kill();proc.wait(timeout=5)
        log.close()


def test_changed_request_before_next_launch_retains_first_attempt(tmp_path,monkeypatch):
    case=setup_case(tmp_path,monkeypatch,'hang');actual=scheduler.resource_snapshot;changed=False
    def snapshot(pid,groups):
        nonlocal changed
        value=actual(pid,groups)
        if groups and not changed:
            case[1].write_bytes(case[1].read_bytes()+b' ');changed=True
        return value
    monkeypatch.setattr(scheduler,'resource_snapshot',snapshot)
    with pytest.raises(scheduler.PassFailure,match='changed'):run(case)
    report=json.loads((case[0]/'parallel-pass.json').read_text())
    assert len(report['phase_receipts'])==1 and report['unstarted_phases']==list(range(1,8))
    assert all(row['verified_termination'] for row in report['termination_verification'])


def test_pid_reuse_does_not_authorize_group_signal(monkeypatch):
    from types import SimpleNamespace
    item=dict(process=SimpleNamespace(pid=123),identity=dict(pid=123,start_ticks=10),members={123:10,456:20})
    monkeypatch.setattr(scheduler,'proc_record',lambda pid:dict(pid=pid,start_ticks=99,pgid=123))
    monkeypatch.setattr(scheduler.os,'killpg',lambda *a:pytest.fail('signalled reused PID group'))
    scheduler._signal_group(item,signal.SIGKILL)


def test_actual_parsed_request_bytes_are_hashed(tmp_path,monkeypatch):
    case=setup_case(tmp_path,monkeypatch);original=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:original(p)+b' ' if p==case[1] else original(p))
    monkeypatch.setattr(scheduler.subprocess,'Popen',lambda *a,**k:pytest.fail('spawned changed bytes'))
    with pytest.raises(ValueError,match='original|pinned'):run(case)
    assert not (case[0]/'parallel-workers').exists()
