"""Task7 launcher controls: tiny owned Python processes only; no model or Git."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import pytest

REPO = Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
L=load('task7_launcher_controls',REPO/'codex/hostile-fold-crosscheck/launch.py')
O=load('task7_existing_owner_controls',REPO/L.OWNERSHIP_FILE)

def args():return argparse.Namespace(attempt='tiny-pilot',source='a'*40,plan='profile-001-plan.json',plan_sha256='b'*64)

@pytest.mark.parametrize('change',[dict(attempt='../escape'),dict(attempt='.hidden'),dict(plan='other.json'),dict(source='x'*40),dict(plan_sha256='a'*63)])
def test_exact_arguments_refuse(change):
    a=args();vars(a).update(change)
    with pytest.raises(ValueError):L.validate_arguments(a)

def test_limits_idempotent_and_before_git(monkeypatch):
    priority=[0];limits=[L.GIB,18*L.GIB];calls=[]
    monkeypatch.setattr(L.os,'getpriority',lambda *x:priority[0]);monkeypatch.setattr(L.os,'nice',lambda x:priority.__setitem__(0,priority[0]+x))
    monkeypatch.setattr(L.resource,'getrlimit',lambda *x:tuple(limits));monkeypatch.setattr(L.resource,'setrlimit',lambda k,v:(calls.append(v),limits.__setitem__(slice(None),v)))
    L.establish_limits();L.establish_limits();assert priority==[10] and limits==[12*L.GIB]*2 and calls==[(12*L.GIB,)*2]*2

def test_tighter_hard_limit_refuses(monkeypatch):
    monkeypatch.setattr(L.os,'getpriority',lambda *x:10);monkeypatch.setattr(L.resource,'getrlimit',lambda *x:(L.GIB,11*L.GIB))
    monkeypatch.setattr(L.resource,'setrlimit',lambda *x:pytest.fail('cannot weaken hard limit'))
    with pytest.raises(RuntimeError,match='hard AS'):L.establish_limits()

@pytest.mark.parametrize('value,startup',[(27*L.GIB,True),(15*L.GIB,False)])
def test_headroom_guards(monkeypatch,value,startup):
    monkeypatch.setattr(L,'available',lambda:value)
    with pytest.raises(RuntimeError,match='MemAvailable'):L.guard(time.monotonic()+30,[],startup=startup)

def test_deadline_and_signal_guards():
    with pytest.raises(TimeoutError):L.guard(time.monotonic()-1,[])
    with pytest.raises(InterruptedError):L.guard(time.monotonic()+1,['SIGTERM'])


def snapshot_fixture(tmp_path,monkeypatch,*,fold=False):
    a=args();a.plan='fold-001-plan.json' if fold else a.plan;spec=L.specification(a.plan)
    repo=tmp_path/'repo';repo.mkdir();blobs={n:b'# tiny committed source\n' for n in spec['workers']}
    blobs[L.OWNERSHIP_FILE]=(REPO/L.OWNERSHIP_FILE).read_bytes()
    blobs['codex/hostile-fold-crosscheck/launch.py']=b'# tiny launcher\n'
    blobs['codex/hostile-fold-crosscheck/'+spec['card']]=b'fixed plan\n'
    plan=dict(schema=spec['schema'],envelope=dict(spec['caps']),source_sha256={n:L.hashlib.sha256(blobs[n]).hexdigest() for n in spec['workers']})
    rel='codex/hostile-fold-crosscheck/'+a.plan;blobs[rel]=json.dumps(plan).encode();a.plan_sha256=L.hashlib.sha256(blobs[rel]).hexdigest()
    for n,b in blobs.items():p=repo/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    def git(command,**kwargs):
        assert kwargs['timeout']<=30
        if command[1]=='rev-parse':return (a.source+'\n').encode()
        if command[1]=='show':return blobs[command[2].split(':',1)[1]]
        if command[1]=='ls-remote':return (a.source+'\trefs/heads/main\n').encode()
        if command[1]=='merge-base':return b''
        pytest.fail(command)
    monkeypatch.setattr(L.subprocess,'check_output',git);monkeypatch.setattr(L,'available',lambda:40*L.GIB)
    return a,repo,blobs,plan

def test_complete_snapshot_without_model_import(tmp_path,monkeypatch):
    a,repo,blobs,plan=snapshot_fixture(tmp_path,monkeypatch)
    result,captured,remote=L.snapshot(repo,a,time.monotonic()+10,[])
    assert result==plan and captured==blobs and remote==a.source

def test_changed_original_source_refused(tmp_path,monkeypatch):
    a,repo,blobs,plan=snapshot_fixture(tmp_path,monkeypatch);(repo/L.WORKER_FILES[0]).write_text('changed')
    with pytest.raises(ValueError,match='committed source'):L.snapshot(repo,a,time.monotonic()+10,[])

def test_external_plan_pin_refused(tmp_path,monkeypatch):
    a,repo,blobs,plan=snapshot_fixture(tmp_path,monkeypatch);a.plan_sha256='c'*64
    with pytest.raises(ValueError,match='plan hash'):L.snapshot(repo,a,time.monotonic()+10,[])

def test_float_envelope_refused(tmp_path,monkeypatch):
    a,repo,blobs,plan=snapshot_fixture(tmp_path,monkeypatch);plan['envelope']['cpu_threads']=2.0
    p=repo/'codex/hostile-fold-crosscheck'/a.plan;p.write_text(json.dumps(plan));a.plan_sha256=L.sha(p)
    with pytest.raises(ValueError,match='resource-only'):L.snapshot(repo,a,time.monotonic()+10,[])


def tiny_process():
    p=subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(30)'],start_new_session=True)
    return p,O.Identity(O.proc(p.pid))

@pytest.mark.parametrize('low_memory',[False,True])
def test_runtime_deadline_or_headroom_cleans_exact_owned_group(monkeypatch,low_memory):
    p,identity=tiny_process();monkeypatch.setattr(L,'available',lambda:(15 if low_memory else 40)*L.GIB)
    try:
        with pytest.raises((TimeoutError,RuntimeError)):
            L.monitor(p,identity,O,time.monotonic()+.02,[],dict(samples=0,minimum_available_bytes=40*L.GIB))
        assert L.close_worker(p,identity,O);identity=None
        assert p.returncode is not None and O.proc(p.pid) is None
    finally:
        if identity is not None:L.close_worker(p,identity,O)


def execute_fixture(tmp_path,monkeypatch,*,exit_code=0,fold=False,bad_status=False):
    a=args();a.plan='fold-001-plan.json' if fold else a.plan;monkeypatch.setattr(L,'OWNED',tmp_path/'owned');monkeypatch.setattr(L,'available',lambda:40*L.GIB)
    parent=os.getpid();original=L.establish_limits
    monkeypatch.setattr(L,'establish_limits',lambda:original() if os.getpid()!=parent else None)
    worker='''import argparse,json,time,os,resource\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');a=p.parse_args();out=Path(a.output);out.mkdir();time.sleep(.07)\n(out/'status.json').write_text(json.dumps(dict(status='resource_profile_completed',resource_probe_only=True,fold_qualified=False,certificate_passed=False)))\nprint('tiny complete',flush=True)\nraise SystemExit(EXIT)\n'''.replace('EXIT',str(exit_code))
    if fold and not bad_status:
        worker=worker.replace("dict(status='resource_profile_completed',resource_probe_only=True,fold_qualified=False,certificate_passed=False)", "dict(status='fold_qualified',resource_probe_only=False,passed=True,computational_completed=True,solver_history_complete=True,certificate_passed=False,passage_trajectories_evaluated=False)")
    blobs={L.OWNERSHIP_FILE:(REPO/L.OWNERSHIP_FILE).read_bytes(),'codex/hostile-fold-crosscheck/'+L.specification(a.plan)['worker']:worker.encode(),'codex/hostile-fold-crosscheck/'+a.plan:b'{}'}
    monkeypatch.setattr(L,'snapshot',lambda *x:({},blobs,'a'*40))
    return a

def test_success_receipt_retains_snapshot_originals_and_limits(tmp_path,monkeypatch):
    a=execute_fixture(tmp_path,monkeypatch);assert L.execute(a)==0
    attempt=L.OWNED/a.attempt;receipt=json.loads((attempt/'receipt.json').read_text());manifest=json.loads((attempt/'manifest.json').read_text())
    assert receipt['worker_termination_verified'] and not receipt['timed_out'] and receipt['exit_code']==0
    assert receipt['runtime_headroom_observation']['samples']>=1
    assert manifest['worker_identity']['nice']==10 and manifest['cpu_threads']==2
    assert set(x['path'] for x in receipt['artifacts'])=={str(p.relative_to(attempt))for p in attempt.rglob('*')if p.is_file() and p.name!='receipt.json'}
    assert all((attempt/x['path']).stat().st_mode&0o222==0 for x in receipt['artifacts'])

def test_worker_failure_is_retained_not_promoted(tmp_path,monkeypatch):
    a=execute_fixture(tmp_path,monkeypatch,exit_code=3);assert L.execute(a)==1
    p=L.OWNED/a.attempt;r=json.loads((p/'receipt.json').read_text())
    assert r['exit_code']==3 and r['worker_termination_verified'] and not r['certificate_passed'] and (p/'failure.json').exists()

def test_start_floor_refusal_preserved_before_snapshot(tmp_path,monkeypatch):
    a=execute_fixture(tmp_path,monkeypatch);monkeypatch.setattr(L,'available',lambda:27*L.GIB);monkeypatch.setattr(L,'snapshot',lambda *x:pytest.fail('no Git before headroom'))
    assert L.execute(a)==1
    r=json.loads((L.OWNED/a.attempt/'receipt.json').read_text());assert not r['worker_spawned'] and not r['certificate_passed']

def test_exclusive_attempt_not_reused(tmp_path,monkeypatch):
    a=execute_fixture(tmp_path,monkeypatch);(L.OWNED/a.attempt).mkdir(parents=True)
    with pytest.raises(FileExistsError):L.execute(a)


def test_fold_plan_has_only_frozen_resource_and_import_extension(tmp_path,monkeypatch):
    a,repo,blobs,plan=snapshot_fixture(tmp_path,monkeypatch,fold=True)
    spec=L.specification(a.plan)
    assert spec['caps']==dict(L.CAPS,timeout_seconds=1800)
    assert len(spec['workers'])==7 and not spec['resource_probe_only']
    actual,captured,remote=L.snapshot(repo,a,time.monotonic()+10,[])
    assert actual==plan and captured==blobs and remote==a.source

@pytest.mark.parametrize('change',[dict(timeout_seconds=300),dict(timeout_seconds=1800.0),dict(cpu_threads=True),dict(minimum_runtime_available_gib=12)])
def test_fold_wrong_resource_envelope_refused(tmp_path,monkeypatch,change):
    a,repo,blobs,plan=snapshot_fixture(tmp_path,monkeypatch,fold=True);plan['envelope'].update(change)
    p=repo/'codex/hostile-fold-crosscheck'/a.plan;p.write_text(json.dumps(plan));a.plan_sha256=L.sha(p)
    with pytest.raises(ValueError,match='plan required'):L.snapshot(repo,a,time.monotonic()+10,[])


def test_fold_cannot_omit_runtime_source(tmp_path,monkeypatch):
    a,repo,blobs,plan=snapshot_fixture(tmp_path,monkeypatch,fold=True)
    del plan['source_sha256']['codex/hostile-fold-crosscheck/fold_runtime.py']
    p=repo/'codex/hostile-fold-crosscheck'/a.plan;p.write_text(json.dumps(plan));a.plan_sha256=L.sha(p)
    with pytest.raises(ValueError,match='inventory'):L.snapshot(repo,a,time.monotonic()+10,[])


def test_fold_success_remains_numerical_not_certificate(tmp_path,monkeypatch):
    a=execute_fixture(tmp_path,monkeypatch,fold=True);assert L.execute(a)==0
    p=L.OWNED/a.attempt;r=json.loads((p/'receipt.json').read_text());m=json.loads((p/'manifest.json').read_text())
    assert r['status']=='fold_attempt_completed' and r['scientific_fold_result'] and not r['resource_probe_only'] and not r['certificate_passed']
    assert r['resource_envelope']['timeout_seconds']==1800 and m['command'][4].endswith('/fold_runtime.py')

@pytest.mark.parametrize('code,bad_status',[(2,False),(0,True)])
def test_fold_failed_gate_or_profile_status_never_passes(tmp_path,monkeypatch,code,bad_status):
    a=execute_fixture(tmp_path,monkeypatch,fold=True,exit_code=code,bad_status=bad_status);assert L.execute(a)==1
    r=json.loads((L.OWNED/a.attempt/'receipt.json').read_text())
    assert r['status']=='fold_attempt_failed' and not r['scientific_fold_result'] and r['worker_termination_verified']
