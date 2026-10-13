"""No Git/network/model calls: exact fake plans and tiny owned OS processes."""
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
spec = importlib.util.spec_from_file_location('derivative_campaign_under_test', ROOT/'codex/certify-llm/derivative_campaign.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
H = 'a'*64; SOURCE = 'b'*40


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(C.canonical(obj)+'\n'); return C.sha(path)


@pytest.fixture
def frozen_fixture(tmp_path):
    repo = tmp_path/'repo'; owned = tmp_path/'owned'; repo.mkdir(); owned.mkdir()
    sources = {}
    for rel in C.FILES:
        p = repo/rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# synthetic source '+rel+'\n'); sources[rel] = C.sha(p)
    definition = owned/'domain'/'definition.json'; dh = save(definition, {'synthetic': True})
    prefix = owned/'prefix'; prefix.mkdir()
    evaluators = {k: sources[k] for k in C.EVALUATOR_FILES}
    common = dict(schema='codex-llm-final-derivative-shard-plan-v1', definition_path=str(definition), definition_sha256=dh,
        map_sha256=H, backend_sha256=H, source_sha256=evaluators,
        evaluator_sha256=hashlib.sha256(C.canonical(dict(schema='codex-derivative-evaluator-source-bundle-v1', source_sha256=evaluators)).encode()).hexdigest(),
        prefix_attempt=str(prefix), prefix_receipt_sha256=H, prefix_map_source_sha256=H, prefix_cache_sha256=H,
        precision=1024, token_chunk=1024, envelope=C.ENVELOPE.copy())
    rows = []
    for p, s in C.ASSIGNMENTS:
        basename = f'p{p}-{s}.json'; path = repo/'codex/certify-llm'/basename
        h = save(path, dict(common, phase_index=p, start_column=s))
        rows.append(dict(attempt=f'shard-{p}-{s}', phase_index=p, start_column=s, plan={'name': basename, 'sha256': h}))
    first = dict(rows[0], receipt_sha256=H, audit={'path': str(owned/'audit'/'first.json'), 'sha256': H})
    r = dict(concurrency=2, cpu_total=4, aggregate_rss_bytes=8*1024**3, minimum_available_bytes=1,
        timeout_seconds=7200, registration_seconds=10, receipt_seconds=10, sample_seconds=1,
        raw_bytes=1024**3, minimum_disk_bytes=1,time_headroom_milli=1200,rss_headroom_milli=1200,storage_headroom_milli=1200)
    plan = dict(schema=C.SCHEMA, status='reviewed_measured', execution_checkout=str(repo), first=first, common=common, assignments=rows[1:], sources=sources, resources=r)
    return repo, owned, plan


def test_complete_canonical_contract_and_no_actual_plan_created(frozen_fixture):
    repo, owned, plan = frozen_fixture
    assert C.validate(plan, repo, owned) is plan
    assert len(plan['assignments']) == 111
    assert not (owned/plan['first']['attempt']).exists()


@pytest.mark.parametrize('fault', ['pending', 'duplicate', 'omitted', 'boolean', 'foreign_domain', 'foreign_source', 'cpu', 'existing', 'deadline'])
def test_plan_corruptions_fail_before_launch(frozen_fixture, fault):
    repo, owned, plan = frozen_fixture
    if fault == 'pending': plan['status'] = 'prepared'
    elif fault == 'duplicate': plan['assignments'][1] = plan['assignments'][0]
    elif fault == 'omitted': plan['assignments'].pop()
    elif fault == 'boolean': plan['assignments'][0]['phase_index'] = False
    elif fault == 'foreign_domain':
        row = plan['assignments'][3]; p = repo/'codex/certify-llm'/row['plan']['name']; obj=C.read(p); obj['map_sha256']='c'*64; row['plan']['sha256']=save(p,obj)
    elif fault == 'foreign_source': plan['sources']['codex/certify-llm/launch.py']='c'*64
    elif fault == 'cpu': plan['resources']['concurrency']=11
    elif fault == 'existing': (owned/plan['assignments'][1]['attempt']).mkdir()
    elif fault == 'deadline': plan['common']['envelope']['timeout_seconds']=3599
    with pytest.raises(ValueError): C.validate(plan, repo, owned)


def test_first_authority_requires_pinned_measured_independent_audit(frozen_fixture, monkeypatch):
    repo, owned, plan = frozen_fixture
    result = dict(elapsed_seconds=1.0, max_rss_kib=1024, bytes=4096)
    calls=[]
    monkeypatch.setattr(C, 'completed', lambda *a, **k: (calls.append((a,k)) or result))
    f=plan['first']; audit=dict(schema=C.AUDIT_SCHEMA, passed=True, certificate_passed=False, attempt=f['attempt'],
        receipt_sha256=f['receipt_sha256'], plan_sha256=f['plan']['sha256'], phase_index=0, start_column=0,
        **{k:plan['common'][k] for k in ('definition_sha256','map_sha256','backend_sha256','evaluator_sha256')},
        measured_elapsed_seconds=1.0, measured_max_rss_kib=1024, reviewer='independent synthetic reviewer', review_source_sha256=H)
    path=Path(f['audit']['path']); f['audit']['sha256']=save(path,audit)
    assert C.first_authority(plan, owned)['sampled_rss_is_hard_cap'] is False
    assert calls[0][1]['expected_receipt']==H
    audit['passed']=False; f['audit']['sha256']=save(path,audit)
    with pytest.raises(ValueError,match='passed'):C.first_authority(plan,owned)
    audit['passed']=True; audit['measured_elapsed_seconds']=.5; f['audit']['sha256']=save(path,audit)
    with pytest.raises(ValueError,match='cold'):C.first_authority(plan,owned)


def test_dirty_and_remote_source_check_is_readonly(frozen_fixture, monkeypatch):
    repo, owned, plan = frozen_fixture; path=repo/'codex/certify-llm/campaign.json'; h=save(path,plan); commands=[]
    dirty=[False]
    def git(_repo,*args):
        commands.append(args)
        if args[0]=='rev-parse':return SOURCE
        if args[0]=='status':return ' M tracked' if dirty[0] else ''
        if args[0]=='ls-remote':return SOURCE+'\trefs/heads/main'
        raise AssertionError(args)
    monkeypatch.setattr(C,'git',git)
    def checked(cmd,**kw):
        assert cmd[:2]==['git','show']; return (repo/cmd[2].split(':',1)[1]).read_bytes()
    monkeypatch.setattr(C.subprocess,'check_output',checked)
    monkeypatch.setattr(C.subprocess,'run',lambda cmd,**kw: commands.append(tuple(cmd)))
    assert C.frozen(repo,path,h,SOURCE,plan,remote=True)==SOURCE
    assert all('fetch' not in c and 'push' not in c and 'checkout' not in c for c in commands)
    dirty[0]=True
    with pytest.raises(ValueError,match='dirty'):C.frozen(repo,path,h,SOURCE,plan)


FAKE_LAUNCH = r'''
import json,os,resource,subprocess,sys,time
from pathlib import Path
attempt=Path(sys.argv[1]); mode=sys.argv[2]; basename=sys.argv[3]; duration=float(sys.argv[4])
attempt.mkdir(); source=attempt/'source/codex/certify-llm';source.mkdir(parents=True)
if mode=='before':time.sleep(20)
worker=source/'derivative_shard.py'
worker.write_text('import time,sys\ntime.sleep('+str(duration)+')\nsys.exit('+('7' if mode=='fail' else '0')+')\n')
if mode=='descendant':
    worker.write_text('import subprocess,sys,time\nsubprocess.Popen([sys.executable,\"-c\",\"import time; time.sleep(5)\"])\ntime.sleep(.3)\n')
(source/basename).write_text('{}')
command=[sys.executable,'-I','-B','-u',str(worker),'--output',str(attempt/'run'),'--plan',str(source/basename)]
def limits():
    resource.setrlimit(resource.RLIMIT_AS,(18*1024**3,)*2)
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
p=subprocess.Popen(command,start_new_session=True,preexec_fn=limits)
(attempt/'test-child.json').write_text(json.dumps({'pid':p.pid,'parent':os.getpid()}))
if mode=='orphan':os._exit(9)
if mode=='delayed_manifest':time.sleep(.7)
manifest={'source_commit':'b'*40,'threads':1,'memory_gib':18,'timeout_seconds':3600,'command':command,'owned_pid':p.pid}
(attempt/'manifest.json').write_text(json.dumps(manifest))
(attempt/'registered-at').write_text(str(time.monotonic()))
rc=p.wait()
(attempt/'receipt.json').write_text(json.dumps({'exit_code':rc,'timed_out':False,'artifacts':[]}))
sys.exit(0 if rc==0 else 1)
'''


def lifecycle(tmp_path, modes, durations=None, concurrency=2):
    owned=tmp_path/'owned';owned.mkdir(); script=tmp_path/'fake_launcher.py';script.write_text(FAKE_LAUNCH)
    rows=[dict(attempt=f'owned-{i}',phase_index=0,start_column=i*64,plan={'name':f'plan-{i}.json','sha256':H}) for i in range(len(modes))]
    plan=dict(assignments=rows, resources=dict(concurrency=concurrency,timeout_seconds=10,registration_seconds=1.0,
        receipt_seconds=1.5,sample_seconds=.015,aggregate_rss_bytes=1024**3,minimum_available_bytes=1,minimum_disk_bytes=1,raw_bytes=1024**2))
    def cmd(row):
        i=rows.index(row)
        return [sys.executable,'-I','-B','-u',str(script),str(owned/row['attempt']),modes[i],row['plan']['name'],str((durations or [.5]*len(modes))[i])]
    def complete(c):
        r=C.read(c['attempt']/'receipt.json'); assert r['exit_code']==0
        return {'receipt_sha256':C.sha(c['attempt']/'receipt.json')}
    return C.Dispatcher(plan,tmp_path/'campaign',SOURCE,owned=owned,repo=tmp_path,command=cmd,complete=complete)


def assert_closed(d):
    for c in d.children:
        assert c['p'].poll() is not None
        if c['worker']:
            q=C.proc(c['worker_identity']['pid'])
            assert q is None or q['state']=='Z' or q['start']!=c['worker_identity']['start']


def test_real_owned_overlap_sequential_registration_and_receipts(tmp_path):
    d=lifecycle(tmp_path,['ok','ok','ok']);r=d.run()
    assert r['status']=='campaign_completed' and r['completed']==3
    assert not r['certificate_passed'] and r['scientific_derivative_audit_pending']
    for a,b in zip(d.children,d.children[1:]):
        assert float((a['attempt']/'registered-at').read_text()) <= b['born']
    assert d.children[1]['born']<d.children[0]['born']+.5
    assert all(c['worker_termination_verified'] for c in d.children)
    assert_closed(d)


def test_failed_shard_stops_future_dispatch_and_keeps_peer_receipt(tmp_path):
    d=lifecycle(tmp_path,['fail','ok','ok'],[.25,5,.5]);r=d.run()
    assert r['status']=='campaign_failed' and r['started']==2 and r['undispatched']==1
    assert all((c['attempt']/'receipt.json').exists() for c in d.children)
    assert all(c['worker_termination_verified'] for c in d.children)
    assert_closed(d)


def test_memory_abort_does_not_signal_unrelated_process(tmp_path,monkeypatch):
    stranger=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True)
    identity=C.Identity(C.proc(stranger.pid))
    try:
        d=lifecycle(tmp_path,['ok','ok'],[5,5])
        monkeypatch.setattr(C,'available',lambda:0 if any(c['registered'] for c in d.children) else 2**40)
        result=d.run();assert 'MemAvailable' in result['stop_reason']
        assert result['started']==1 and stranger.poll() is None
        assert_closed(d)
    finally:
        identity.send(signal.SIGTERM);stranger.wait(timeout=3);identity.close()


def test_pre_registration_worker_recovered_without_killing_receipt_parent(tmp_path):
    d=lifecycle(tmp_path,['delayed_manifest','ok'],[5,.5]);d.r['registration_seconds']=.15;r=d.run()
    assert r['status']=='campaign_failed' and r['started']==1
    assert d.children[0]['worker_termination_verified']
    assert (d.children[0]['attempt']/'receipt.json').is_file()
    assert_closed(d)


def test_registration_before_fork_aborted_only_in_own_session(tmp_path):
    d=lifecycle(tmp_path,['before','ok']);d.r['registration_seconds']=.15;r=d.run()
    assert r['status']=='campaign_failed' and r['started']==1
    assert not (d.children[0]['attempt']/'test-child.json').exists()
    assert_closed(d)


def test_dead_launcher_before_manifest_is_explicitly_uncertain(tmp_path):
    d=lifecycle(tmp_path,['orphan','ok'],[.5,.5]);r=d.run()
    assert r['status']=='campaign_failed_uncertain_ownership' and r['started']==1
    assert not d.children[0]['worker_termination_verified']
    assert r['uncertain_ownership'] and not r['certificate_passed']
    # This test's tiny orphan exits by itself; the dispatcher never guesses its
    # identity or claims it terminated. No unverified process is signalled.
    pid=C.read(d.children[0]['attempt']/'test-child.json')['pid']
    end=time.monotonic()+2
    while (q:=C.proc(pid)) and q['state']!='Z' and time.monotonic()<end:time.sleep(.02)
    assert q is None or q['state']=='Z'


def test_whole_deadline_and_signal_stop_future_work(tmp_path):
    import threading
    d=lifecycle(tmp_path,['ok','ok'],[5,5]);d.plan['resources']['concurrency']=1
    timer=threading.Timer(.25,lambda:os.kill(os.getpid(),signal.SIGTERM));timer.start()
    try:r=d.run()
    finally:timer.cancel();timer.join()
    assert r['status']=='campaign_failed' and r['started']==1 and r['stop_reason']=='signal '+str(signal.SIGTERM)
    assert_closed(d)


def test_pidfd_identity_rejects_changed_birth_without_signalling(tmp_path):
    p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)'],start_new_session=True)
    value=C.proc(p.pid)
    try:
        bad=dict(value,start=value['start']+1)
        with pytest.raises(RuntimeError,match='changed'):C.Identity(bad)
        assert p.poll() is None
    finally:p.terminate();p.wait(timeout=3)


def test_whole_deadline_cancels_current_worker_without_next_launch(tmp_path):
    d=lifecycle(tmp_path,['ok','ok'],[5,5],concurrency=1)
    d.r['timeout_seconds']=.25
    r=d.run()
    assert r['status']=='campaign_failed' and r['started']==1 and 'deadline' in r['stop_reason']
    assert_closed(d)


def test_source_guard_failure_stops_before_next_spawn(tmp_path):
    d=lifecycle(tmp_path,['ok','ok'],[5,5])
    def guard():
        if d.children:raise ValueError('dirty source synthetic control')
    d.before_launch=guard;r=d.run()
    assert r['started']==1 and 'dirty source' in r['stop_reason']
    assert_closed(d)


def test_campaign_seal_preserves_uncertainty_and_readonly_completed_files(tmp_path):
    uncertain=tmp_path/'uncertain';uncertain.mkdir();save(uncertain/'log.json',{})
    C.seal(uncertain,{'status':'campaign_failed_uncertain_ownership','uncertain_ownership':[{'reason':'unknown'}]})
    assert not(uncertain/'receipt.json').exists() and (uncertain/'finalization-pending.json').exists()
    done=tmp_path/'done';done.mkdir();save(done/'result.json',{'certificate_passed':False})
    C.seal(done,{'status':'campaign_failed','uncertain_ownership':[]})
    assert C.read(done/'receipt.json')['status']=='campaign_failed'
    assert not((done/'result.json').stat().st_mode&0o222)


def test_first_measurement_uses_explicit_frozen_headroom_not_hardcoded_two(frozen_fixture,monkeypatch):
    repo,owned,plan=frozen_fixture
    result=dict(elapsed_seconds=1811.0,max_rss_kib=1024,bytes=4096)
    monkeypatch.setattr(C,'completed',lambda *a,**k:result)
    f=plan['first'];audit=dict(schema=C.AUDIT_SCHEMA,passed=True,certificate_passed=False,attempt=f['attempt'],
        receipt_sha256=f['receipt_sha256'],plan_sha256=f['plan']['sha256'],phase_index=0,start_column=0,
        **{k:plan['common'][k]for k in('definition_sha256','map_sha256','backend_sha256','evaluator_sha256')},
        measured_elapsed_seconds=1811.0,measured_max_rss_kib=1024,reviewer='synthetic independent',review_source_sha256=H)
    f['audit']['sha256']=save(Path(f['audit']['path']),audit);plan['resources']['timeout_seconds']=200000
    assert C.first_authority(plan,owned)['forecast_seconds']<200000
    plan['resources']['time_headroom_milli']=2000
    with pytest.raises(ValueError,match='time'):C.first_authority(plan,owned)


def test_popen_identity_failure_retained_and_cleaned(tmp_path,monkeypatch):
    d=lifecycle(tmp_path,['ok','ok'],[5,5]);original=C.Identity;count=[0]
    def fail_once(value):
        count[0]+=1
        if count[0]==1:raise OSError('synthetic pidfd registration failure')
        return original(value)
    monkeypatch.setattr(C,'Identity',fail_once)
    r=d.run()
    assert r['status']=='campaign_failed' and r['started']==1 and r['undispatched']==1
    assert 'pidfd registration' in r['stop_reason'] and d.children[0]['launcher_pid']>0
    assert_closed(d)


def test_exited_worker_leader_does_not_hide_owned_descendants(tmp_path):
    d=lifecycle(tmp_path,['descendant','ok'],[.5,.5],concurrency=1)
    r=d.run()
    assert r['status']=='campaign_failed' and r['started']==1
    assert 'group remains live' in r['stop_reason']
    assert d.children[0]['worker_termination_verified']
    assert not C.group(d.children[0]['worker'])
    assert_closed(d)


def test_stop_confirmation_timeout_always_continues_own_launcher(tmp_path,monkeypatch):
    d=lifecycle(tmp_path,['before']);c=d.start(d.plan['assignments'][0]);leader=c['leader']
    original=leader.live
    try:
        def hidden_stop():
            q=original()
            return dict(q,state='R') if q else q
        with monkeypatch.context() as m:
            m.setattr(leader,'live',hidden_stop)
            d.cancel_unregistered(c)
        deadline=time.monotonic()+1
        while (q:=original()) and q['state'] in ('T','t') and time.monotonic()<deadline:time.sleep(.01)
        assert q and q['state'] not in ('T','t')
        assert any(v['reason']=='launcher stop not confirmed' for v in d.uncertain)
    finally:
        leader.send(signal.SIGCONT);leader.send(signal.SIGTERM);c['p'].wait(timeout=3)
        leader.close();c['stdout'].close();c['stderr'].close()


def test_failed_log_write_still_cleans_every_owned_process(tmp_path,monkeypatch):
    d=lifecycle(tmp_path,['fail','ok'],[.2,5]);original=C.write
    def write(path,value):
        if Path(path).name=='failure.json':raise OSError('synthetic log disk failure')
        return original(path,value)
    monkeypatch.setattr(C,'write',write)
    r=d.run()
    assert r['status']=='campaign_failed_uncertain_ownership'
    assert any('log write' in v['reason'] for v in r['uncertain_ownership'])
    assert_closed(d)
