"""Task19 adapter/outer dispatch controls; no model, producer data or Git."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
L=load('task19_launcher',ROOT/'codex/return-map-crosscheck/launch.py')
O=load('task19_original_ownership',ROOT/L.OWNERSHIP_FILE)

def args(mode='trajectory'):
    return argparse.Namespace(attempt=mode+'-001',plan=mode+'-001-plan.json',source='a'*40,plan_sha256='b'*64)

def completed(mode):
    status=dict(status='completed',computational_completed=True,certificate_passed=False,
        experimental_claims=False,mode=mode,source_commit='a'*40)
    if mode=='trajectory':
        result=dict(transient_steps=1000,recorded_steps=2000,native_step_calls=3000,row_forwards=7000,seed=0,
            offsets=[1e-6,1e-9],post_asset_hashes_equal=True,
            disjoint_windows=[dict(start=i,stop=i+500) for i in range(0,2000,500)],
            running_means=[dict(stop=i) for i in range(100,2001,100)],
            numerical_repeating_text_chaos=False,primary_seven_phase_clock_matches=False)
    else:result=dict(all65_admitted=True,rows=[dict(index=i,class_='irregular')for i in range(65)])
    return status,result

@pytest.mark.parametrize('mode',['trajectory','saved'])
def test_fixed_mode_counts_and_negative_science_are_complete(mode):
    L.validate_arguments(args(mode));L.validate_completion(*completed(mode),mode,'a'*40)
    spec=L.specification(args(mode).plan)
    assert len(spec['workers'])==5 and spec['attempt']==args(mode).attempt
    assert spec['caps']['timeout_seconds']==(4200 if mode=='trajectory' else 300)
    assert spec['caps']['memory_gib']==(12 if mode=='trajectory' else 4)
    assert spec['caps']['cpu_threads']==(2 if mode=='trajectory' else 1)

def test_wrong_mode_attempt_and_source_refused():
    for change in [dict(attempt='saved-001'),dict(attempt='../escape'),dict(source='bad'),dict(plan='other.json')]:
        a=args();vars(a).update(change)
        with pytest.raises(ValueError):L.validate_arguments(a)

@pytest.mark.parametrize('mode',['trajectory','saved'])
def test_status_and_partial_rows_refused(mode):
    status,result=completed(mode);status['computational_completed']=False
    with pytest.raises(RuntimeError):L.validate_completion(status,result,mode,'a'*40)
    status,result=completed(mode)
    if mode=='trajectory':result['running_means'].pop()
    else:result['rows'][-1]['index']=63
    with pytest.raises(RuntimeError):L.validate_completion(status,result,mode,'a'*40)

@pytest.mark.parametrize('mode',['trajectory','saved'])
@pytest.mark.parametrize('initial',[0,10,11])
def test_exact_priority_and_mode_address_space(monkeypatch,mode,initial):
    nice=[initial];limits=[L.GIB,18*L.GIB];changes=[]
    monkeypatch.setattr(L.os,'getpriority',lambda *x:nice[0])
    monkeypatch.setattr(L.os,'nice',lambda n:nice.__setitem__(0,nice[0]+n))
    monkeypatch.setattr(L.resource,'getrlimit',lambda *x:tuple(limits))
    monkeypatch.setattr(L.resource,'setrlimit',lambda k,v:(changes.append(v),limits.__setitem__(slice(None),v)))
    if initial>10:
        with pytest.raises(RuntimeError):L.establish_limits(L.CAPS[mode])
        assert not changes
    else:
        L.establish_limits(L.CAPS[mode]);L.establish_limits(L.CAPS[mode])
        assert nice==[10] and changes==[(L.CAPS[mode]['memory_gib']*L.GIB,)*2]*2

@pytest.mark.parametrize('mode',['trajectory','saved'])
def test_hard_limit_and_headroom_before_git(monkeypatch,mode):
    caps=L.CAPS[mode];monkeypatch.setattr(L.os,'getpriority',lambda *x:10)
    monkeypatch.setattr(L.resource,'getrlimit',lambda *x:(L.GIB,(caps['memory_gib']-1)*L.GIB))
    monkeypatch.setattr(L.resource,'setrlimit',lambda *x:pytest.fail('cannot lift hard cap'))
    with pytest.raises(RuntimeError):L.establish_limits(caps)
    for startup,floor in [(True,caps['minimum_start_available_gib']),(False,16)]:
        monkeypatch.setattr(L,'available',lambda:(floor-1)*L.GIB)
        with pytest.raises(RuntimeError):L.guard(time.monotonic()+30,[],caps,startup=startup)

def snapshot_fixture(tmp_path,monkeypatch,mode):
    a=args(mode);spec=L.specification(a.plan);blobs={s:b'scientific fixture'for s in spec['workers']}
    blobs[L.OWNERSHIP_FILE]=(ROOT/L.OWNERSHIP_FILE).read_bytes()
    for name in ['launch.py',spec['card'],spec['dispatch']]:blobs['codex/return-map-crosscheck/'+name]=b'operational fixture'
    plan=dict(schema=spec['schema'],admission='reviewed',mode=mode,envelope=spec['caps'],
        source_sha256={s:L.hashlib.sha256(blobs[s]).hexdigest()for s in spec['workers']})
    rel='codex/return-map-crosscheck/'+a.plan;blobs[rel]=json.dumps(plan).encode();a.plan_sha256=L.hashlib.sha256(blobs[rel]).hexdigest()
    for name,b in blobs.items():p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    def git(command,**kwargs):
        assert kwargs['timeout']<=30
        if command[1]=='rev-parse':return (a.source+'\n').encode()
        if command[1]=='show':return blobs[command[2].split(':',1)[1]]
        if command[1]=='ls-remote':return (a.source+'\trefs/heads/main\n').encode()
        if command[1]=='merge-base':return b''
        raise AssertionError(command)
    monkeypatch.setattr(L.subprocess,'check_output',git);monkeypatch.setattr(L,'available',lambda:40*L.GIB)
    return a,blobs,plan

@pytest.mark.parametrize('mode',['trajectory','saved'])
def test_ten_file_capsule_and_prepared_refusal(tmp_path,monkeypatch,mode):
    a,blobs,plan=snapshot_fixture(tmp_path,monkeypatch,mode)
    actual,captured,remote=L.snapshot(tmp_path,a,time.monotonic()+30,[])
    assert len(captured)==10 and captured==blobs and actual==plan and remote==a.source
    plan['admission']='prepared';p=tmp_path/'codex/return-map-crosscheck'/a.plan;p.write_text(json.dumps(plan));a.plan_sha256=L.sha(p)
    with pytest.raises(ValueError,match='prepared'):L.snapshot(tmp_path,a,time.monotonic()+30,[])

def test_changed_dispatch_or_source_refused(tmp_path,monkeypatch):
    a,blobs,plan=snapshot_fixture(tmp_path,monkeypatch,'trajectory')
    (tmp_path/'codex/return-map-crosscheck/trajectory-001-dispatch.sh').write_text('changed')
    with pytest.raises(ValueError,match='committed'):L.snapshot(tmp_path,a,time.monotonic()+30,[])

@pytest.mark.parametrize('mode,exit_code',[('trajectory',0),('saved',3)])
def test_real_tiny_child_receipt_and_failure_retention(tmp_path,monkeypatch,mode,exit_code):
    a=args(mode);monkeypatch.setattr(L,'OWNED',tmp_path/'owned');monkeypatch.setattr(L,'available',lambda:40*L.GIB)
    parent=os.getpid();limits=L.establish_limits
    monkeypatch.setattr(L,'establish_limits',lambda caps:limits(caps) if os.getpid()!=parent else None)
    status,result=completed(mode)
    worker='''import argparse,json,time,os,resource
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');a=p.parse_args();out=Path(a.output);out.mkdir();time.sleep(.08)
(out/'status.json').write_text(STATUS)
(out/'result.json').write_text(RESULT)
print('tiny retained stdout',flush=True)
raise SystemExit(EXIT)
'''.replace('STATUS',repr(json.dumps(status))).replace('RESULT',repr(json.dumps(result))).replace('EXIT',str(exit_code))
    blobs={L.OWNERSHIP_FILE:(ROOT/L.OWNERSHIP_FILE).read_bytes(),
        'codex/return-map-crosscheck/runtime.py':worker.encode(),'codex/return-map-crosscheck/'+a.plan:b'{}'}
    monkeypatch.setattr(L,'snapshot',lambda *x:({},blobs,'a'*40))
    assert L.execute(a)==(0 if exit_code==0 else 1)
    out=L.OWNED/a.attempt;r=json.loads((out/'receipt.json').read_text());m=json.loads((out/'manifest.json').read_text())
    assert r['exit_code']==exit_code and r['worker_termination_verified'] and not r['uncertain_ownership']
    assert O.proc(m['owned_pid']) is None and m['cpu_threads']==L.CAPS[mode]['cpu_threads']
    assert 'tiny retained stdout' in (out/'stdout.log').read_text()
    assert (out/'failure.json').exists()==(exit_code!=0)

def test_real_owned_timeout_cleanup(tmp_path,monkeypatch):
    p=subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(30)'],start_new_session=True)
    identity=O.Identity(O.proc(p.pid));monkeypatch.setattr(L,'available',lambda:40*L.GIB)
    try:
        with pytest.raises(TimeoutError):L.monitor(p,identity,O,time.monotonic()+.02,[],dict(samples=0,minimum_available_bytes=40*L.GIB),L.CAPS['saved'])
        assert L.close_worker(p,identity,O);identity=None
        assert p.returncode is not None and O.proc(p.pid) is None
    finally:
        if identity is not None:L.close_worker(p,identity,O)

@pytest.mark.parametrize('mode',['trajectory','saved'])
def test_outer_forced_fork_wait_logs_and_exclusive_attempt(tmp_path,mode):
    repo=tmp_path/'repo';(repo/'codex/return-map-crosscheck').mkdir(parents=True)
    (repo/'codex/return-map-crosscheck/launch.py').write_text("import os,time,sys\nprint('owned',os.getpid(),'nice',os.getpriority(os.PRIO_PROCESS,0),flush=True)\nprint('retained error',file=sys.stderr,flush=True)\ntime.sleep(.05)\nraise SystemExit(7)\n")
    supervision=tmp_path/'supervision'
    text=(ROOT/f'codex/return-map-crosscheck/{mode}-001-dispatch.sh').read_text()
    assert 'setsid --wait' in text and 'nice -n' not in text and 'noclobber' in text
    text=text.replace('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp',str(repo))
    text=text.replace('/data2/neural-shrimp-data/codex/llm-return-map-crosscheck/'+mode+'-001-supervision',str(supervision))
    script=tmp_path/'dispatch.sh';script.write_text(text)
    # Force setsid to fork by making its inherited process the process-group leader.
    p=subprocess.Popen(['bash',str(script),'a'*40,'b'*64],preexec_fn=os.setpgrp,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout,stderr=p.communicate(timeout=5)
    assert p.returncode==7 and stdout==b'' and stderr==b''
    saved=(supervision/'launcher.stdout').read_text();pid=int(saved.split()[1])
    assert pid!=p.pid and 'nice 10' in saved and O.proc(pid) is None
    assert (supervision/'launcher.stderr').read_text().strip()=='retained error'
    rerun=subprocess.run(['bash',str(script),'a'*40,'b'*64],capture_output=True,timeout=5)
    assert rerun.returncode!=0 and (supervision/'launcher.stdout').read_text()==saved
