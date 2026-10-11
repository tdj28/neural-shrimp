"""Synthetic prospective audit controls only; no fixed outcome/data reads."""
import copy,importlib.util,json,math
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('restart_audit',ROOT/'tools/reviews/audit_cifar_restart_outcome.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
def event_fixture(n=50):
    events=[];attempts=dict.fromkeys(A.CAPS,0);returns=dict.fromkeys(A.CAPS,0)
    def op(name):
        attempts[name]+=1;events.append(dict(kind='operation_started',operation=name,index=attempts[name]))
        if name=='action':op('j2vp')
        if name=='j2vp':op('hvp');op('hvp')
        returns[name]+=1;events.append(dict(kind='operation_returned',operation=name,index=returns[name],seconds=.001))
    for _ in range(3):op('direct_gradient')
    op('hvp');op('j2vp')
    history=[]
    for i in range(n+2):
        op('action');events.append(dict(kind='matvec',index=i+1,seconds=.004))
        if i<n:history.append(.9-i*.0001);events.append(dict(kind='gmres_inner',iteration=i+1,estimated_relative_residual=history[-1]))
    trials=[]
    for i in range(4):
        op('direct_gradient');op('direct_gradient');trials.append(dict(index=i));events.append(dict(kind='trial_result',index=i))
    result=dict(status='diagnostic_complete',operation_attempted=attempts,operation_returned=returns,retained_prefix_values=history,trials=trials)
    return events,result
@pytest.mark.parametrize('n',[50,200])
def test_complete_counter_flow_early_and_full_cycle(n):
    events,result=event_fixture(n);a=A.counter_check(events,result);assert a['matvecs']==n+2 and not a['pending_operations']
@pytest.mark.parametrize('mutation',['callback_order','nested_direct','cap','count','unknown','missing_hvp'])
def test_counter_corruption_fails(mutation):
    events,result=event_fixture()
    if mutation=='callback_order':
        ids=[i for i,e in enumerate(events) if e['kind']=='gmres_inner'];events[ids[0]],events[ids[1]]=events[ids[1]],events[ids[0]]
    elif mutation=='nested_direct':
        i=next(i for i,e in enumerate(events) if e['kind']=='operation_started' and e['operation']=='action');events.insert(i+1,dict(kind='operation_started',operation='direct_gradient',index=4))
    elif mutation=='cap':result['operation_attempted']['action']=203
    elif mutation=='count':result['operation_returned']['hvp']-=1
    elif mutation=='unknown':events.append(dict(kind='invented'))
    else:
        i=next(i for i,e in enumerate(events) if e['kind']=='operation_started' and e['operation']=='j2vp');events.pop(i+1)
    with pytest.raises((ValueError,KeyError)):A.counter_check(events,result)
def test_interrupted_nested_operation_remains_pending():
    events,_=event_fixture();end=next(i for i,e in enumerate(events) if e['kind']=='operation_started' and e['operation']=='action')+3;events=events[:end]
    starts={k:sum(e['kind']=='operation_started' and e['operation']==k for e in events) for k in A.CAPS};ends={k:sum(e['kind']=='operation_returned' and e['operation']==k for e in events) for k in A.CAPS}
    result=dict(status='failed_or_interrupted',operation_attempted=starts,operation_returned=ends,retained_prefix_values=[])
    a=A.counter_check(events,result);assert [v['operation'] for v in a['pending_operations']]==['action','j2vp','hvp']
@pytest.mark.parametrize('n,qualified',[(49,False),(50,True),(200,True)])
def test_prefix_length_qualification(n,qualified):assert A.prefix_check([.9]*n,[.9]*200)['passed'] is qualified
def test_prefix_threshold_and_mismatch_retained():
    x=[.9]*50;x[49]+=.9e-10;assert A.prefix_check(x,[.9]*200)['passed'];x[49]+=.01;r=A.prefix_check(x,[.9]*200);assert not r['passed'] and r['first_mismatch']==50
@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-.1])
def test_invalid_prefix(bad):
    with pytest.raises(ValueError):A.prefix_check([bad],[.9]*200)
def linear_fixture(q=.4,info=1):
    r=np.array([1.,0.,0.]);p=np.array([.1,.2,.3]);ap=np.array([-1+q,0.,0.]);e=ap+r
    saved=dict(rhs_norm=1.,true_linear_residual_norm=abs(q),true_forcing_ratio=abs(q),merit_directional_derivative=-1+q,forcing_test_pass=abs(q)<=.5,descent_direction=q<1,requested_tolerance_met=abs(q)<=1e-10,gmres_info=info,gmres_reported_converged=info==0,restart=200,cycles=1,max_matvec=202,forcing_limit=.5,requested_linear_tolerance=1e-10)
    return r,p,ap,e,saved
@pytest.mark.parametrize('q',[.4,.5,.6])
def test_true_forcing_inclusive_boundary(q):assert A.linear_check(*linear_fixture(q))['forcing_test_pass'] is (q<=.5)
def test_solver_info_zero_does_not_imply_true_tolerance():
    r=A.linear_check(*linear_fixture(.4,0));assert not r['requested_tolerance_met'] and r['forcing_test_pass']
def test_exact_linear_sum_signed_zero_and_flag_corruption():
    args=list(linear_fixture());args[3][1]=-.0
    with pytest.raises(ValueError):A.linear_check(*args)
    args=list(linear_fixture());args[-1]['forcing_test_pass']=False
    with pytest.raises(ValueError):A.linear_check(*args)
def test_saved_reference_parity_failure_and_nonfinite():
    x=np.zeros(3);assert A.parity_check(x,x)['passed'];y=x.copy();y[0]=1e-5;assert not A.parity_check(y,x)['passed']
    y[0]=np.nan;assert not A.parity_check(y,x)['passed']
    with pytest.raises(ValueError):A.parity_check(y,x,dict(passed=True))
def trial_fixture(alpha=.5,q=.4):
    theta=np.array([1.,2.,3.]);r,p,ap,e,saved=linear_fixture(q);linear=A.linear_check(r,p,ap,e,saved);trial=theta+alpha*p;one=trial+np.array([.1,0.,0.]);res=np.array([.2,0.,0.]);two=trial+res;res=two-trial
    phi=.5*float(res@res);relative=float(np.linalg.norm(res)/np.linalg.norm(trial));sep=float(np.linalg.norm(one-trial)/np.linalg.norm(trial));upper=.5+1e-4*alpha*(-1+q)
    row=dict(alpha=alpha,armijo_upper=upper,relative_step_norm=float(alpha*np.linalg.norm(p)/np.linalg.norm(theta)),residual_norm=float(np.linalg.norm(res)),relative_residual=relative,merit=phi,merit_ratio=2*phi,relative_separation=sep,closure_screen=False,primitive_screen=True,merit_decreased=True,armijo_pass=True,finite=True,inexact_step_candidate=q<=.5,accepted_orbit=False,applied_to_continuation=False)
    return theta,r,p,alpha,dict(theta=trial,map1=one,map2=two,residual=res),row,linear
@pytest.mark.parametrize('q',[.4,.6])
def test_trial_absolute_merit_keeps_forcing_separate(q):assert A.trial_check(*trial_fixture(q=q))['inexact_step_candidate'] is (q<=.5)
@pytest.mark.parametrize('mutation',['different_base','residual','armijo','orbit'])
def test_trial_corruption(mutation):
    args=list(trial_fixture())
    if mutation=='different_base':args[4]['theta'][0]+=.1
    elif mutation=='residual':args[4]['residual'][0]+=.1
    elif mutation=='armijo':args[5]['armijo_pass']=False
    else:args[5]['accepted_orbit']=True
    with pytest.raises(ValueError):A.trial_check(*args)
def resources():
    ls=dict(started_utc='2026-10-08T12:00:00+00:00',group_pid=10);start=dict(started_utc='2026-10-08T12:00:01+00:00',runtime=dict(pid=11));result=dict(finished_utc='2026-10-08T12:00:10+00:00',elapsed_seconds=9.,max_rss_kib=1000);lr=dict(finished_utc='2026-10-08T12:00:11+00:00',elapsed_seconds=11.,max_group_rss_bytes=2000000,observed_owned_pids=[10,11]);return start,result,ls,lr
@pytest.mark.parametrize('mutation',['duration','worker_rss','group_rss','pid','utc'])
def test_observed_resource_contradictions(mutation):
    args=list(resources());assert A.resource_check(*args,True)
    if mutation=='duration':args[1]['elapsed_seconds']=51.;args[3]['elapsed_seconds']=50.
    elif mutation=='worker_rss':args[1]['max_rss_kib']=3*1024**2
    elif mutation=='group_rss':args[3]['max_group_rss_bytes']=3*1024**3
    elif mutation=='pid':args[0]['runtime']['pid']=12
    else:args[1]['finished_utc']='2026-10-08T11:00:00+00:00'
    with pytest.raises(ValueError):A.resource_check(*args,True)
def test_incomplete_rss_guard_is_recorded_not_made_complete():
    args=list(resources());args[1]['max_rss_kib']=3*1024**2;assert A.resource_check(*args,False)['worker_peak_rss_kib']==3*1024**2
@pytest.mark.parametrize('mutation',['extra','corrupt','symlink'])
def test_hash_inventory_fail_closed(tmp_path,mutation):
    p=tmp_path/'x';p.write_bytes(b'a');(tmp_path/'manifest.json').write_bytes(A.canonical(dict(files={'x':A.entry(p)})));pin=A.entry(tmp_path/'manifest.json')['sha256'];assert A.inventory(tmp_path,pin,True)
    if mutation=='extra':(tmp_path/'extra').write_bytes(b'')
    elif mutation=='corrupt':p.write_bytes(b'b')
    else:p.unlink();p.symlink_to(tmp_path/'manifest.json')
    with pytest.raises(ValueError):A.inventory(tmp_path,pin,True)
def test_nice_guard_before_output(tmp_path,monkeypatch):
    monkeypatch.setattr(A.os,'getpriority',lambda *_:0)
    with pytest.raises(ValueError):A.run(tmp_path/'out',tmp_path/'pins','0'*64,'a'*40)
    assert not (tmp_path/'out').exists()
@pytest.mark.parametrize('mode',['missing_directory','missing_manifest','malformed_manifest','contradictory_valid_manifest'])
def test_partial_inventory_classification_before_science(tmp_path,monkeypatch,mode):
    raw=tmp_path/'raw';launch=tmp_path/'launch';launch.mkdir();monkeypatch.setattr(A,'RAW',raw);monkeypatch.setattr(A,'LAUNCH',launch);monkeypatch.setattr(A,'source_check',lambda *_:None)
    if mode!='missing_directory':raw.mkdir();(raw/'partial.npy').write_bytes(b'partial unparsed bytes')
    pin=None
    if mode.endswith('manifest') and mode not in ('missing_manifest',):
        (raw/'manifest.json').write_bytes(b'{broken' if mode=='malformed_manifest' else A.canonical(dict(files={'partial.npy':A.entry(raw/'partial.npy')})))
        pin=A.entry(raw/'manifest.json')['sha256']
    rawrows=A.directory(raw) if raw.exists() else {};wrapper=b'synthetic orchestration'
    ls=dict(source_commit='a'*40,review_commit=A.SOURCE_REVIEW,sources={'tools/launch_cifar_restart.py':A.digest(wrapper)},out=str(raw),launcher=str(launch),nice=10,thread_env={k:'1' for k in A.THREADS},gpu_used=False,external_seconds=60,kill_grace_seconds=5,supervisor_seconds=65.,rss_cap_bytes=2*1024**3,store_bytes=31*1024**2,launcher_bytes=1024**2,log_bytes=128*1024,group_pid=10,started_utc='2026-10-08T12:00:00+00:00')
    ls['command']=['timeout','--signal=TERM','--kill-after=5s','60s','nice','-n','10','/data2/neural-shrimp/.venv/bin/python','tools/diagnose_cifar_restart.py','--out',str(raw),'--review-commit',A.SOURCE_REVIEW]
    lr=dict(source_commit='a'*40,review_commit=A.SOURCE_REVIEW,group_pid=10,elapsed_seconds=1.,finished_utc='2026-10-08T12:00:01+00:00',max_group_rss_bytes=1000,owned_live_pids_remaining=[],raw_inventory=rawrows,raw_file_count=len(rawrows),raw_bytes=sum(v['bytes'] for v in rawrows.values()),within_store_cap=True,raw_manifest_verified=False,successful=False)
    for name,v in [('runtime-start.json',ls),('receipt.json',lr)]:(launch/name).write_bytes(A.canonical(v))
    (launch/'launcher-source.py').write_bytes(wrapper);(launch/'stdout.log').write_bytes(b'');(launch/'manifest.json').write_bytes(A.canonical(dict(files=A.directory(launch))))
    pins=dict(execution_commit='a'*40,source_review_commit=A.SOURCE_REVIEW,raw_manifest_sha256=pin,launcher_manifest_sha256=A.entry(launch/'manifest.json')['sha256']);saved={}
    if mode=='contradictory_valid_manifest':
        with pytest.raises(ValueError):A.audit(pins,lambda n,v:saved.update({n:v}))
    else:
        result=A.audit(pins,lambda n,v:saved.update({n:v}));assert result['verdict']=='incomplete_inventory_only' and result['comparison_qualified'] is False and set(saved)=={'provenance.json'}

@pytest.mark.parametrize('failure',[True,False])
def test_audit_driver_retains_outputs_without_target_reads(tmp_path,monkeypatch,capsys,failure):
    root=tmp_path/'source';root.mkdir();monkeypatch.setattr(A,'ROOT',root)
    for name in A.AUDIT_FILES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic source')
    source='a'*40;pins={k:('b'*(40 if k.endswith('commit') else 64)) for k in A.PIN_KEYS};pp=tmp_path/'pins.json';pp.write_bytes(A.canonical(pins))
    def git(argv,**kwargs):
        if argv[1]=='rev-parse':return source+'\n'
        if argv[1]=='status':return ''
        return (root/argv[2].split(':',1)[1]).read_bytes()
    monkeypatch.setattr(A.subprocess,'check_output',git);monkeypatch.setattr(A.os,'getpriority',lambda *_:10)
    def fake(_pins,emit):
        emit('partial.json',dict(saved=True))
        if failure:raise TimeoutError('synthetic interruption')
        return dict(verdict='saved_outcome_audit_pass')
    monkeypatch.setattr(A,'audit',fake);out=tmp_path/'out';assert A.run(out,pp,A.entry(pp)['sha256'],source)==(2 if failure else 0)
    assert A.read(out/'result.json')['status']==('failed' if failure else 'finished');assert (out/'partial.json').exists();A.inventory(out,A.entry(out/'manifest.json')['sha256'])
    out.chmod(0o755)
    for p in out.iterdir():p.chmod(0o644)

def test_dirty_audit_checkout_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(A.os,'getpriority',lambda *_:10);monkeypatch.setattr(A.subprocess,'check_output',lambda args,**_:'a'*40 if args[1]=='rev-parse' else '?? dirty')
    with pytest.raises(ValueError,match='clean audit'):A.run(tmp_path/'out',tmp_path/'pins','0'*64,'a'*40)
    assert not (tmp_path/'out').exists()

def launcher_module():
    spec=importlib.util.spec_from_file_location('restart_audit_launcher',ROOT/'tools/reviews/launch_cifar_restart_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_audit_launcher_itself_requires_nice10(monkeypatch):
    l=launcher_module();monkeypatch.setattr(l.os,'getpriority',lambda *_:0)
    with pytest.raises(ValueError,match='launcher nice10'):l.main()

def test_owned_group_cleanup_after_leader_and_pipes_exit():
    import subprocess,sys
    l=launcher_module()
    code="import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(p.pid,flush=True)"
    p=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    try:
        out,_=p.communicate(timeout=3);child=int(out);assert p.returncode==0
        assert child in [r['pid'] for r in l.group_status(p.pid) if r['state']!='Z']
        assert l.cleanup_owned(p.pid)==[]
    finally:l.cleanup_owned(p.pid)

def test_final_bookkeeping_does_not_relabel_cooperative_deadline():
    args=list(resources());args[1]['elapsed_seconds']=45.01;args[3]['elapsed_seconds']=47.
    checked=A.resource_check(*args,True)
    assert checked['finished_record_over_internal_seconds']>0 and checked['cooperative_internal_seconds']==45.

def test_nonfinite_reductions_stay_guard_failures():
    with np.errstate(over='ignore',invalid='ignore'):
        checked=A.parity_check(np.array([1e308]),np.array([-1e308]))
        assert checked['passed'] is False and checked['error_overflow'] is True
    args=list(trial_fixture());args[4]['map2']=np.array([1e308,0.,0.]);args[4]['residual']=args[4]['map2']-args[4]['theta']
    args[5].update(finite=False,inexact_step_candidate=False)
    assert A.trial_check(*args)['finite'] is False

def test_launcher_guards_survive_python_optimization():
    import subprocess,sys
    code="import importlib.util; p='tools/reviews/launch_cifar_restart_audit.py'; s=importlib.util.spec_from_file_location('l',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.os.getpriority=lambda *_:0;m.main()"
    r=subprocess.run([sys.executable,'-O','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=3)
    assert r.returncode!=0 and 'launcher nice10 required' in r.stderr
