"""Independent saved-auditor controls; fabricated arrays/events, no target files."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('reviewed_restart_auditor', ROOT/'tools/reviews/audit_cifar_restart_outcome.py')
audit=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def complete_events():
    rows=[]; attempted=dict.fromkeys(audit.CAPS,0);returned=attempted.copy()
    def operation(name, children=()):
        attempted[name]+=1
        rows.append(dict(kind='operation_started',operation=name,index=attempted[name]))
        for child in children:child()
        returned[name]+=1
        rows.append(dict(kind='operation_returned',operation=name,index=returned[name],seconds=.001))
    grad=lambda:operation('direct_gradient')
    hvp=lambda:operation('hvp')
    j2=lambda:operation('j2vp',(hvp,hvp))
    action=lambda:operation('action',(j2,))
    grad();grad();grad();hvp();j2()
    history=[]
    for i in range(52):
        action();rows.append(dict(kind='matvec',index=i+1,seconds=.002))
        if i<50:
            value=.9-.001*i;history.append(value)
            rows.append(dict(kind='gmres_inner',iteration=i+1,estimated_relative_residual=value))
    trials=[]
    for i in range(4):
        grad();grad();trial=dict(index=i,alpha=(1.,.5,.25,.125)[i])
        trials.append(trial);rows.append(dict(kind='trial_result',**trial))
    result=dict(status='diagnostic_complete',operation_attempted=attempted,operation_returned=returned,retained_prefix_values=history,trials=trials)
    return rows,result


def reindex_operations(rows):
    seen={kind:dict.fromkeys(audit.CAPS,0) for kind in ('operation_started','operation_returned')}
    for row in rows:
        if row['kind'] in seen:
            seen[row['kind']][row['operation']]+=1;row['index']=seen[row['kind']][row['operation']]


def test_valid_early_converged_one_cycle_event_accounting():
    events,result=complete_events();checked=audit.counter_check(events,result)
    assert checked['attempted']==dict(action=52,j2vp=53,hvp=107,direct_gradient=11)
    assert checked['returned']==checked['attempted'] and checked['pending_operations']==[]


def test_callbacks_cannot_precede_their_operator_actions():
    events,result=complete_events()
    fabricated=[r for r in events if r['kind']=='gmres_inner']+[r for r in events if r['kind']!='gmres_inner']
    with pytest.raises(ValueError):audit.counter_check(fabricated,result)


def test_direct_gradients_cannot_be_nested_inside_hvp():
    events,result=complete_events()
    pair=events[:2];remaining=events[2:]
    location=next(i for i,r in enumerate(remaining) if r['kind']=='operation_started' and r['operation']=='hvp')+1
    fabricated=remaining[:location]+pair+remaining[location:]
    reindex_operations(fabricated)
    with pytest.raises(ValueError):audit.counter_check(fabricated,result)


def test_partial_operation_stack_and_attempts_survive():
    events,_=complete_events()
    end=next(i for i,r in enumerate(events) if r['kind']=='operation_started' and r['operation']=='action')
    events=events[:end+3]  # action, J2, first HVP have started but not returned.
    attempted=dict.fromkeys(audit.CAPS,0);returned=attempted.copy()
    for row in events:
        if row['kind']=='operation_started':attempted[row['operation']]+=1
        elif row['kind']=='operation_returned':returned[row['operation']]+=1
    result=dict(status='failed_or_interrupted',operation_attempted=attempted,operation_returned=returned,retained_prefix_values=[])
    checked=audit.counter_check(events,result)
    assert [frame['operation'] for frame in checked['pending_operations']]==['action','j2vp','hvp']
    assert checked['attempted']['action']==1 and checked['returned']['action']==0


def saved_linear(r,ap,e,info=1):
    rhs=float(np.linalg.norm(r));tn=float(np.linalg.norm(e));slope=float(r@ap)
    return dict(rhs_norm=rhs,true_linear_residual_norm=tn,true_forcing_ratio=tn/rhs,
        merit_directional_derivative=slope,forcing_test_pass=bool(tn/rhs<=.5),
        descent_direction=bool(slope<0),requested_tolerance_met=bool(tn<=1e-10*rhs),
        gmres_info=info,gmres_reported_converged=info==0,restart=200,cycles=1,max_matvec=202,
        forcing_limit=.5,requested_linear_tolerance=1e-10)


def test_info_zero_does_not_override_failed_strict_true_residual():
    r=np.array([1.,2.]);p=np.array([-.8,-1.6]);ap=p.copy();e=ap+r
    saved=saved_linear(r,ap,e,0);checked=audit.linear_check(r,p,ap,e,saved)
    assert checked['forcing_test_pass'] and not checked['requested_tolerance_met']
    corrupted=copy.deepcopy(saved);corrupted['requested_tolerance_met']=True
    with pytest.raises(ValueError):audit.linear_check(r,p,ap,e,corrupted)


def test_exact_residual_requires_signed_zero_bytes():
    r=np.array([1.,0.]);ap=np.array([-.8,-0.]);e=ap+r;p=ap.copy()
    saved=saved_linear(r,ap,e)
    e[1]=-0.
    with pytest.raises(ValueError):audit.linear_check(r,p,ap,e,saved)


def test_prefix_uses_original_absolute_relative_rule_and_full_fifty():
    baseline=[.9]*200
    assert audit.prefix_check([.9]*50,baseline)['passed']
    assert not audit.prefix_check([.9]*49,baseline)['passed']
    wrong=[.9]*50;wrong[-1]+=1e-6
    checked=audit.prefix_check(wrong,baseline)
    assert not checked['passed'] and checked['first_mismatch']==50


def test_actual_resource_limits_cannot_be_replaced_by_configured_limits():
    start=dict(started_utc='2026-10-08T12:00:01+00:00',runtime=dict(pid=11))
    result=dict(finished_utc='2026-10-08T12:00:11+00:00',elapsed_seconds=10.,max_rss_kib=1024)
    launch_start=dict(started_utc='2026-10-08T12:00:00+00:00',group_pid=10)
    launch_result=dict(finished_utc='2026-10-08T12:00:12+00:00',elapsed_seconds=12.,max_group_rss_bytes=4*1024**2,observed_owned_pids=[10,11])
    audit.resource_check(start,result,launch_start,launch_result,True)
    for target,key,value in [('result','elapsed_seconds',float('nan')),('result','max_rss_kib',3*1024**2),('launcher','max_group_rss_bytes',3*1024**3),('launcher','observed_owned_pids',[10]),('launcher','elapsed_seconds',67.),('result','finished_utc','2026-10-08T11:59:59+00:00')]:
        rr=copy.deepcopy(result);ll=copy.deepcopy(launch_result)
        (rr if target=='result' else ll)[key]=value
        with pytest.raises(ValueError):audit.resource_check(start,rr,launch_start,ll,True)


def test_manifest_rejects_extra_omitted_corrupted_and_symlink_inputs(tmp_path):
    folder=tmp_path/'raw';folder.mkdir();path=folder/'record.json';path.write_text('{}')
    manifest=folder/'manifest.json';manifest.write_bytes(audit.canonical(dict(files={'record.json':audit.entry(path)})))
    pin=audit.entry(manifest)['sha256'];audit.inventory(folder,pin,True)
    extra=folder/'extra';extra.write_text('x')
    with pytest.raises(ValueError):audit.inventory(folder,pin,True)
    extra.unlink();original=path.read_bytes();path.write_bytes(b'[]')
    with pytest.raises(ValueError):audit.inventory(folder,pin,True)
    path.write_bytes(original);link=folder/'link';link.symlink_to(path)
    with pytest.raises(ValueError):audit.inventory(folder,pin,True)
    link.unlink();path.unlink()
    with pytest.raises(ValueError):audit.inventory(folder,pin,True)


def test_absolute_armijo_blocks_apparent_normalized_closure_improvement():
    theta=np.zeros(2);r=np.ones(2);p=np.full(2,-100.);ap=-r;e=ap+r
    linear=audit.linear_check(r,p,ap,e,saved_linear(r,ap,e))
    trial=theta+p;one=trial+.01;two=trial+2.;res=two-trial
    sep=np.linalg.norm(one-trial)/np.linalg.norm(trial)
    row=dict(index=0,alpha=1.,armijo_upper=1.-.0002,relative_step_norm=float(np.linalg.norm(p)),
        residual_norm=float(np.linalg.norm(res)),relative_residual=float(np.linalg.norm(res)/np.linalg.norm(trial)),
        merit=4.,merit_ratio=4.,relative_separation=float(sep),closure_screen=False,primitive_screen=True,
        merit_decreased=False,armijo_pass=False,finite=True,inexact_step_candidate=False,
        accepted_orbit=False,applied_to_continuation=False)
    parts=dict(theta=trial,map1=one,map2=two,residual=res)
    checked=audit.trial_check(theta,r,p,1.,parts,row,linear)
    assert row['relative_residual']<np.linalg.norm(r) and not checked['inexact_step_candidate']
    row['armijo_pass']=True;row['inexact_step_candidate']=True
    with pytest.raises(ValueError):audit.trial_check(theta,r,p,1.,parts,row,linear)


def synthetic_complete_attempt(tmp_path,monkeypatch):
    """Complete fake file graph; only vector dimension/source lookup are replaced."""
    raw=tmp_path/'raw';launch=tmp_path/'launch';bundle=tmp_path/'bundle';old=tmp_path/'old';baseline=tmp_path/'baseline'
    for symbol,folder in [('RAW',raw),('LAUNCH',launch),('BUNDLE',bundle),('OLD',old),('BASELINE',baseline)]:
        folder.mkdir();monkeypatch.setattr(audit,symbol,folder)
    monkeypatch.setattr(audit,'source_check',lambda *_:None)
    original_vec=audit.vec
    monkeypatch.setattr(audit,'vec',lambda a,finite=True,n=98666:original_vec(a,finite,n=3))
    def write(folder,name,value):
        (folder/name).write_bytes(audit.canonical(value))
    def seal(folder,wrapped=False):
        members={p.name:audit.entry(p) for p in folder.iterdir() if p.name!='manifest.json'}
        write(folder,'manifest.json',dict(files=members) if wrapped else members)
        return audit.entry(folder/'manifest.json')['sha256']
    theta=np.array([1.,2.,3.]);r=np.ones(3);one=theta+.1;two=theta+r
    np.save(old/'state.npy',theta)
    state_sha=audit.entry(old/'state.npy')['sha256']
    x=np.zeros((2,2));y=np.ones((2,1))
    data_identity={name:dict(shape=list(v.shape),dtype=v.dtype.str,sha256=audit.digest(v.tobytes())) for name,v in [('x',x),('y',y)]}
    model=dict(d=3072,width=32,k=10)
    basis=dict(status='complete',source_commit='a3debab72195d5ac58bee5562cb59afd8b6466e8',source_sha256={},eta=.00827753,lam=.01,model=model,data_identity=data_identity)
    reference=dict(x=x,y=y,theta=theta,directions=np.ones((3,3)),gradient=np.zeros(3),hvp=np.zeros((3,3)),j2vp=np.zeros((3,3)))
    np.savez(bundle/'reference.npz',**reference);write(bundle,'result.json',basis)
    write(old,'orbit.json',dict(status='corrector_not_accepted',source_commit='7de5dcfb5eb50ed7829bc3b69b43c6e2c4f889cd',source_sha256={},last_corrector=dict(state_file='state.npy'),data_identity=data_identity))
    events,counters=complete_events();history=counters['retained_prefix_values'];base_linear=dict(restart=50,cycles=4,inner_iterations=200,matvec_calls=205,estimated_relative_residual_history=history+[.85]*150,true_forcing_ratio=.8)
    base_result=dict(source_commit='a9830e714d259597a53cee6ef7be41a9f07bb205',source_sha256={},status='diagnostic_complete',linear=base_linear,gmres_python_source_sha256=audit.GMRES_SHA,**audit.VERSIONS)
    write(baseline,'linear.json',base_linear);write(baseline,'result.json',base_result)
    initial_arrays={'initial-theta':theta,'initial-map1':one,'initial-map2':two,'initial-residual':r}
    for name,value in initial_arrays.items():np.save(baseline/(name+'.npy'),value);np.save(raw/(name+'.npy'),value)
    np.save(baseline/'true-linear-residual.npy',.8*r)
    pins_old=dict(bundle=seal(bundle),old=seal(old),baseline=seal(baseline),state=state_sha);monkeypatch.setattr(audit,'PINS',pins_old)
    p=-.8*r;ap=p.copy();e=ap+r;linear=saved_linear(r,ap,e)
    linear.update(inner_iterations=50,matvec_calls=52,estimated_relative_residual_history=history)
    for name,value in [('direction',p),('applied-direction',ap),('true-linear-residual',e)]:np.save(raw/(name+'.npy'),value)
    parity={}
    for name in ['gradient','hvp','j2vp']:
        np.save(raw/('reference-'+name+'.npy'),np.zeros(3));parity[name]=dict(max_absolute_error=0.,max_scaled_error=0.,passed=True)
    initial=dict(residual_norm=float(np.linalg.norm(r)),relative_residual=float(np.linalg.norm(r)/np.linalg.norm(theta)),relative_separation=float(np.linalg.norm(one-theta)/np.linalg.norm(theta)))
    trial_rows=[]
    for i,alpha in enumerate((1.,.5,.25,.125)):
        t=theta+alpha*p;first=t+.1;second=t+(r+alpha*ap);res=second-t
        phi=.5*float(res@res);phi0=.5*float(r@r);sep=float(np.linalg.norm(first-t)/np.linalg.norm(t));relative=float(np.linalg.norm(res)/np.linalg.norm(t));upper=phi0+1e-4*alpha*float(r@ap)
        for name,value in [('theta',t),('map1',first),('map2',second),('residual',res)]:np.save(raw/(f'trial-{i}-{name}.npy'),value)
        row=dict(index=i,alpha=alpha,accepted_orbit=False,applied_to_continuation=False,relative_step_norm=float(alpha*np.linalg.norm(p)/np.linalg.norm(theta)),armijo_upper=upper,residual_norm=float(np.linalg.norm(res)),relative_residual=relative,merit=phi,merit_ratio=phi/phi0,relative_separation=sep,closure_screen=relative<1e-8,primitive_screen=sep>=1e-7,merit_decreased=phi<phi0,armijo_pass=phi<=upper,finite=True,inexact_step_candidate=phi<=upper)
        trial_rows.append(row)
    events=[dict(kind='trial_result',**trial_rows[row['index']]) if row['kind']=='trial_result' else row for row in events]
    (raw/'events.jsonl').write_text(''.join(__import__('json').dumps(row)+'\n' for row in events))
    wrapper=b'synthetic launcher';sources={name:'c'*64 for name in audit.SOURCES};sources['tools/launch_cifar_restart.py']=audit.digest(wrapper)
    execution='a'*40
    runtime=dict(schema='cifar-restart-sensitivity-v1',status='running',evidence_rung=0,source_commit=execution,source_sha256=sources,review_commit=audit.SOURCE_REVIEW,settings=audit.SETTINGS,started_utc='2026-10-08T12:00:01+00:00',accepted_orbit=False,applied_to_continuation=False,spectrum_requested=False,comparison_qualified=False,
        runtime=dict(**audit.VERSIONS,gmres_python_source_sha256=audit.GMRES_SHA,device='cpu',gpu=False,nice=10,torch_threads=1,torch_interop_threads=1,thread_env={k:'1' for k in audit.THREADS},pid=11))
    identity=dict(bundle=str(bundle),bundle_manifest_sha256=pins_old['bundle'],old_corrector=str(old),old_manifest_sha256=pins_old['old'],baseline=str(baseline),baseline_manifest_sha256=pins_old['baseline'],baseline_source_commit=base_result['source_commit'],state_sha256=state_sha,data_identity=data_identity,model=model,eta=.00827753,lam=.01,reference_source_commit=basis['source_commit'])
    result=copy.deepcopy(runtime);result.update(counters,status='diagnostic_complete',comparison_qualified=True,input_identity=identity,initial=initial,reference_parity=parity,linear=linear,trials=trial_rows,candidate_trial_indices=list(range(4)),baseline_true_forcing_ratio=.8,elapsed_seconds=10.,max_rss_kib=1024,finished_utc='2026-10-08T12:00:11+00:00')
    for name,value in [('runtime-start.json',runtime),('input-identity.json',identity),('initial.json',initial),('parity.json',parity),('linear.json',linear),('returned-direction.json',dict(shape=[98666],finite=True,pending_independent_true_check=True,gmres_info=1)),('result.json',result)]:write(raw,name,value)
    rawpin=seal(raw,True);rawrows=audit.directory(raw)
    launch_start=dict(source_commit=execution,review_commit=audit.SOURCE_REVIEW,sources=sources,out=str(raw),launcher=str(launch),nice=10,thread_env={k:'1' for k in audit.THREADS},gpu_used=False,external_seconds=60,kill_grace_seconds=5,supervisor_seconds=65.,rss_cap_bytes=2*1024**3,store_bytes=31*1024**2,launcher_bytes=1024**2,log_bytes=128*1024,group_pid=10,started_utc='2026-10-08T12:00:00+00:00')
    launch_start['command']=['timeout','--signal=TERM','--kill-after=5s','60s','nice','-n','10','/data2/neural-shrimp/.venv/bin/python','tools/diagnose_cifar_restart.py','--out',str(raw),'--review-commit',audit.SOURCE_REVIEW]
    receipt=dict(source_commit=execution,review_commit=audit.SOURCE_REVIEW,group_pid=10,owned_live_pids_remaining=[],observed_owned_pids=[10,11],raw_inventory=rawrows,raw_file_count=len(rawrows),raw_bytes=sum(v['bytes'] for v in rawrows.values()),within_store_cap=True,raw_manifest_verified=True,scientific_complete=True,successful=True,process_exit=0,failure=None,log_truncated=False,elapsed_seconds=12.,finished_utc='2026-10-08T12:00:12+00:00',max_group_rss_bytes=4*1024**2)
    write(launch,'runtime-start.json',launch_start);write(launch,'receipt.json',receipt);(launch/'launcher-source.py').write_bytes(wrapper);(launch/'stdout.log').write_bytes(b'')
    return dict(execution_commit=execution,source_review_commit=audit.SOURCE_REVIEW,raw_manifest_sha256=rawpin,launcher_manifest_sha256=seal(launch,True))


def test_complete_synthetic_saved_graph_reaches_arithmetic_verdict(tmp_path,monkeypatch):
    pins=synthetic_complete_attempt(tmp_path,monkeypatch);stages={}
    result=audit.audit(pins,lambda name,value:stages.update({name:value}))
    assert result['verdict']=='saved_outcome_audit_pass' and result['comparison_qualified']
    assert set(stages)=={'provenance.json','events-audit.json','arithmetic-audit.json'}
    assert result['checks']['linear']['forcing_test_pass']
    assert len(result['checks']['trials'])==4


@pytest.mark.parametrize('mutation', ['candidate_aggregate', 'undeclared_file', 'linear_sum'])
def test_consistently_rehashed_complete_contradictions_fail(tmp_path,monkeypatch,mutation):
    import json
    pins=synthetic_complete_attempt(tmp_path,monkeypatch)
    raw=tmp_path/'raw';launch=tmp_path/'launch'
    if mutation=='candidate_aggregate':
        path=raw/'result.json';value=json.loads(path.read_bytes());value['candidate_trial_indices']=[];path.write_bytes(audit.canonical(value))
    elif mutation=='undeclared_file':
        (raw/'extra.json').write_bytes(b'{}')
    else:
        path=raw/'true-linear-residual.npy';value=np.load(path,allow_pickle=False);value[0]+=.01;np.save(path,value)
    members={p.name:audit.entry(p) for p in raw.iterdir() if p.name!='manifest.json'}
    (raw/'manifest.json').write_bytes(audit.canonical(dict(files=members)))
    receipt_path=launch/'receipt.json';receipt=json.loads(receipt_path.read_bytes())
    receipt['raw_inventory']=audit.directory(raw);receipt['raw_file_count']=len(receipt['raw_inventory']);receipt['raw_bytes']=sum(v['bytes'] for v in receipt['raw_inventory'].values());receipt_path.write_bytes(audit.canonical(receipt))
    (launch/'manifest.json').write_bytes(audit.canonical(dict(files={p.name:audit.entry(p) for p in launch.iterdir() if p.name!='manifest.json'})))
    pins['raw_manifest_sha256']=audit.entry(raw/'manifest.json')['sha256'];pins['launcher_manifest_sha256']=audit.entry(launch/'manifest.json')['sha256']
    with pytest.raises(ValueError):audit.audit(pins,lambda *_:None)
