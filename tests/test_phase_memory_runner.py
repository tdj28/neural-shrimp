"""Synthetic controller/codec/replay tests; no research ODEs, fits or network."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

CHECKPOINT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHECKPOINT/'tools'))
import run_phase_memory as run
import phase_memory as pm


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False))


def forbidden(*args,**kwargs):raise AssertionError('Synthetic fixture attempted production solve/build/network')


def toy_rhs(t,x,alpha,gamma):
    a,b,c=x;w=2*np.pi;r=1-a*a-b*b
    return np.array([-w*b+a*r,w*a+b*r,-c])


def toy_jac(t,x,alpha,gamma):
    a,b,_=x;w=2*np.pi;r=1-a*a-b*b
    return np.array([[r-2*a*a,-w-2*a*b,0.],[w-2*a*b,r-2*b*b,0.],[0.,0.,-1.]])


def analytic_orbit(epoch=0.):
    t=np.linspace(0.,1.,4097);theta=2*np.pi*(t+epoch)
    states=np.column_stack((np.cos(theta),np.sin(theta),np.zeros(len(t))))
    velocity=np.column_stack((-2*np.pi*np.sin(theta),2*np.pi*np.cos(theta),np.zeros(len(t))))
    r0=np.array([[np.cos(2*np.pi*epoch),-np.sin(2*np.pi*epoch),0.],[np.sin(2*np.pi*epoch),np.cos(2*np.pi*epoch),0.],[0.,0.,1.]])
    rotations=np.zeros((len(t),3,3));rotations[:,0,0]=np.cos(theta);rotations[:,0,1]=-np.sin(theta)
    rotations[:,1,0]=np.sin(theta);rotations[:,1,1]=np.cos(theta);rotations[:,2,2]=1.
    scales=np.zeros_like(rotations);scales[:,0,0]=np.exp(-2*t);scales[:,1,1]=1.;scales[:,2,2]=np.exp(-t)
    phi=rotations@scales@r0.T;phi[0]=np.eye(3)
    derivative=np.asarray([toy_jac(a,x,0,0)@p for a,x,p in zip(t,states,phi,strict=True)])
    norm=np.linalg.norm(velocity[0]);initial=states[0].copy()
    return dict(times=t,states=states,derivatives=velocity,phi=phi,phi_derivatives=derivative,
        initial=initial,monodromy=phi[-1].copy(),status='complete',solver=dict(run.acquire.FINE),
        relative_closure=float(np.linalg.norm(states[-1]-initial)/max(1.,np.linalg.norm(initial))),
        neutral_flow_relative_residual=float(np.linalg.norm(phi[-1]@velocity[0]-velocity[0])/norm),
        transported_flow_relative_residual=float(np.linalg.norm(phi[-1]@velocity[0]-velocity[-1])/norm))


def reference(setting):
    constants=run.acquire.SETTINGS[setting];orbit=analytic_orbit();w=2*np.pi;t=orbit['times'];theta=w*t
    seed=dict(setting=setting,**constants);seed['section_state']=np.asarray(seed['section_state'],dtype=np.float64)
    z=np.column_stack((-np.sin(theta),np.cos(theta),np.zeros(len(t))))/w
    derivative=np.column_stack((-np.cos(theta),-np.sin(theta),np.zeros(len(t))))
    neutral=pm.neutral_left_phase(orbit['monodromy'],orbit['derivatives'][0])
    # Match the declared single SVD normalization while retaining analytic samples.
    z[-1]=neutral['phase_gradient'];residual=np.einsum('ij,ij->i',z,orbit['derivatives'])-1.
    derivative[-1]=-toy_jac(1.,orbit['states'][-1],0,0).T@z[-1]
    cert=dict(alpha=constants['alpha'],gamma=constants['gamma'],period=1.,section_state=[1.,0.,0.],
        verified=True,attracting=True,primitive=True,peak_count=9,expected_peak_count=9,
        gates={name:True for name in run.acquire.CERTIFICATE_GATES})
    result=dict(setting=setting,alpha=constants['alpha'],gamma=constants['gamma'],period=1.,seed=seed,
        certificate=cert,status='complete',qualified=True,neutral=neutral,thresholds=dict(run.acquire.REFERENCE_GATES),
        gates=dict(original_certificate=True,fixed_duration_covers_eight_cycles=True,fine_orbit_closure=True,
                   adjoint_periodicity=True,adjoint_normalization=True,finite_adjoint_derivatives=True),orbit=orbit,
        adjoint=dict(times=t,values=z,derivatives=derivative,solver=dict(run.acquire.FINE),terminal_normalized_once=z[-1].copy(),normalization_count=1,
            normalization_residuals=residual,maximum_normalization_residual=float(np.max(np.abs(residual))),
            periodic_residual_vector=z[0]-z[-1],relative_periodic_residual=float(np.linalg.norm(z[0]-z[-1])/max(np.linalg.norm(z[0]),np.linalg.norm(z[-1])))),
        phases=[])
    for j in range(8):
        index=512*j;result['phases'].append(dict(phase_index=j,phase_epoch=j/8,initial=orbit['states'][index].copy(),
            flow=orbit['derivatives'][index].copy(),phase_gradient=z[index].copy(),**pm.pulse_directions(z[index],orbit['derivatives'][index])))
    return result


def propagator(ref,phase):
    orbit=analytic_orbit(phase/8)
    return dict(setting=ref['setting'],phase_index=phase,phase_epoch=phase/8,alpha=ref['alpha'],gamma=ref['gamma'],
        period=1.,qualified=True,gates=dict(relative_closure=True),**orbit)


def trace(task,ref,*,native_fn=None):
    context=run.acquire.case_context(task,ref);initial=context['initial'];times=run.acquire.TIMES.copy()
    r0=np.linalg.norm(initial[:2]);theta0=np.arctan2(initial[1],initial[0]);theta=theta0+2*np.pi*times
    radius=1/np.sqrt(1+(1/(r0*r0)-1)*np.exp(-2*times))
    states=np.column_stack((radius*np.cos(theta),radius*np.sin(theta),initial[2]*np.exp(-times)))
    states[0]=initial
    solver=(dict(id=task['method'],kind='RK4',dt=float(task['method'][4:]),steps=round(320/float(task['method'][4:])))
            if task['method'].startswith('rk4-') else dict(id='dop853',kind='DOP853',**run.acquire.FINE,success=True,nfev=0))
    return dict(task=task,**context,times=times,states=states,status='complete',complete=True,solver=solver,duration=320.,sample_dt=.025)


def plan():
    return dict(**copy.deepcopy(run.CONTRACT),date='2026-09-17',seeds=[dict(setting=k,**v) for k,v in run.acquire.SETTINGS.items()],
        historical_inputs=[dict(path=p,bytes=0,sha256='0'*64) for p in run.acquire.HISTORICAL],
        source_files=[dict(path=p,bytes=0,sha256='0'*64) for p in run.SOURCE_PATHS],tasks=run.acquire.build_tasks(),claim_boundary='Synthetic only')


def test_metadata_census_validation_never_calls_dynamics(monkeypatch):
    monkeypatch.setattr(run.acquire,'acquire_reference',forbidden);monkeypatch.setattr(run.acquire,'acquire_case',forbidden)
    value=plan();assert run.validate(value)==dict(references=2,propagators=16,traces=912,total=930)
    assert {k:len(v) for k,v in run.stages(value).items()}==dict(references=2,propagators=16,traces=912)
    changed=copy.deepcopy(value);changed['tasks'][0]['sign']=1
    with pytest.raises(ValueError,match='census'):run.validate(changed)


def test_codec_lossless_partial_roundtrip_hash_and_exclusivity(tmp_path):
    packet=dict(states=np.array([[1.,np.nan,2.],[np.inf,-np.inf,3.]],dtype=np.float64),
        stages=[dict(progress=np.array([1,2],dtype=np.int64))],residual=np.float64(np.nan),category=np.str_('failed'),complete=False)
    saved=run.save_packet(tmp_path,'partial',packet);restored=run.load_packet(saved['path'])
    assert restored['states'].dtype==np.float64 and np.array_equal(restored['states'],packet['states'],equal_nan=True)
    assert np.isnan(restored['residual']) and type(restored['category']) is str
    assert run.packet_digest(packet)==run.packet_digest(restored)
    with pytest.raises(FileExistsError):run.save_packet(tmp_path,'partial',packet)
    with pytest.raises(ValueError,match='Object'):run.save_packet(tmp_path,'bad',dict(array=np.array([{}],dtype=object)))
    meta=run.read(saved['path']);meta['arrays']['sha256']='0'*64;write(saved['path'],meta)
    with pytest.raises(ValueError,match='hash'):run.load_packet(saved['path'])


@pytest.mark.parametrize('mutation',['dtype','shape','escape','extra_array'])
def test_codec_rejects_schema_or_path_substitution(tmp_path,mutation):
    path=Path(run.save_packet(tmp_path,'p',dict(a=np.zeros(3)))['path']);meta=run.read(path)
    if mutation=='dtype':meta['tree']['a']['dtype']='<f4'
    elif mutation=='shape':meta['tree']['a']['shape']=[1,3]
    elif mutation=='escape':meta['arrays']['path']='../p.npz'
    else:
        raw=tmp_path/'p.npz';np.savez(raw,a000000=np.zeros(3),extra=np.ones(2))
        meta['arrays'].update(bytes=raw.stat().st_size,sha256=run.sha(raw))
    write(path,meta)
    with pytest.raises(ValueError):run.load_packet(path)


def setup_run(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir();monkeypatch.setattr(run,'ROOT',repo)
    path=repo/'experiments/hopfield/EXP-014-plan.json';value=plan();write(path,value)
    review=repo/'docs/EXP-014-local-review.json';write(review,dict(status='ready_for_exploratory_execution',plan_sha256=run.sha(path)))
    monkeypatch.setattr(run,'REVIEW',review)
    monkeypatch.setattr(run,'validate',lambda *a,**kw:dict(synthetic=True))
    monkeypatch.setattr(run,'source_guard',lambda *a:dict(source_commit='1'*40,pushed_branch='codex/synthetic',remote_verified_at='synthetic',review=run.binding(review,repo)))
    monkeypatch.setattr(run.importlib.metadata,'version',lambda k:run.CONTRACT['required_packages'][k])
    monkeypatch.setattr(run.platform,'platform',lambda:'synthetic')
    state=dict(available_ram_bytes=32*run.GIB,total_ram_bytes=64*run.GIB,disk_free_bytes=32*run.GIB)
    monkeypatch.setattr(run,'resources',lambda *a:state)
    monkeypatch.setattr(run,'resource_check',lambda *a,**kw:state)
    monkeypatch.setattr(run.signal,'setitimer',lambda *a:None)
    monkeypatch.setattr(run.work,'build_libraries',lambda *a:dict(trace=dict(path='synthetic-no-library')))
    monkeypatch.setattr(run.legacy,'load_native',lambda *a:object())
    monkeypatch.setattr(run.acquire,'load_seed_records',lambda *a:[dict(setting=k,**v,synthetic=True) for k,v in run.acquire.SETTINGS.items()])
    monkeypatch.setattr(run.acquire,'acquire_reference',lambda seed:reference(seed['setting']))
    monkeypatch.setattr(run.acquire,'acquire_phase_propagator',propagator)
    monkeypatch.setattr(run.acquire,'acquire_case',trace)
    monkeypatch.setattr(run.acquire.diagnostics,'hopfield_rhs',toy_rhs)
    monkeypatch.setattr(run.acquire.diagnostics,'hopfield_jacobian',toy_jac)
    monkeypatch.setattr(run.acquire,'solve_ivp',forbidden)
    monkeypatch.setattr(run.acquire,'integrate_trace',forbidden)
    def committed(command):
        if command[-1]=='1'*40+':experiments/hopfield/EXP-014-plan.json':return path.read_bytes()
        forbidden()
    monkeypatch.setattr(run.subprocess,'check_output',committed)
    return path


def test_actual_owned_worker_codec_seal_partial_cohort_and_saved_replay(tmp_path,monkeypatch):
    path=setup_run(tmp_path,monkeypatch);original_execute=run.work.execute
    def bounded_execute(specs,contexts,directory,libraries,**kwargs):
        # Exercise real fork/owned cleanup/worker/codec on13 tiny fabricated jobs.
        # The other preplanned cases remain explicit dependency failures.
        if directory.name=='references':selected=specs
        elif directory.name=='propagators':selected=[t for t in specs if t['case']['phase_index']==0]
        else:selected=[t for t in specs if t['case']['setting']=='core' and t['case']['phase_index']==0 and
            (t['case']['kind']=='sham' or (t['case']['dose_index']==0 and t['case']['direction']=='phase_write'))]
        return original_execute(selected,contexts,directory,libraries,**kwargs)
    monkeypatch.setattr(run.work,'execute',bounded_execute)
    summary=run.run(path,tmp_path/'data','attempt-001');directory=tmp_path/'data/EXP-014/attempt-001'
    assert summary['status_counts']['references']=={'complete':2}
    assert summary['status_counts']['propagators']=={'complete':2,'not_started_dependency_failure':14}
    assert summary['status_counts']['traces']=={'complete':9,'not_started_dependency_failure':903}
    assert summary['complete'] is False and summary['qualified'] is False
    assert all(summary['scientific']['required_counts'][k]==v for k,v in dict(trace_rows=912,trace_comparisons=608,paired_readout_cells=5184,balanced_accuracy_comparisons=3456).items())
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    seal=next(e for e in events if e['stage']=='reference_stage_sealed');start=next(e for e in events if e['stage']=='traces_started')
    assert seal['monotonic_ns']<start['monotonic_ns'] and seal['sha256']==start['reference_seal_sha256']
    packet=run.load_packet(directory/'analysis.json')
    assert len(packet['scored_records'])==9 and not packet['analysis_errors']
    assert len(packet['paired_dynamics'])==432 and sum(r['status']=='complete' for r in packet['paired_dynamics'])==3
    assert len(packet['summary']['signed_response_readouts'])==1728
    assert all(s['aggregates'] is None for s in packet['summary']['settings'].values())
    result=run.verify(directory)
    assert result['passed'] is True and result['tasks']==930 and result['new_dynamics'] is False
    for stage in ('references','propagators','traces'):
        cleanup=run.read(directory/stage/'cleanup.json')
        assert cleanup['all_original_identities_absent'] is True and cleanup['all_registered_workers_joined'] is True
    # Editing the pre-pulse seal, even with refreshed outer inventory, must fail.
    changed=run.load_packet(directory/'reference-seal.json');changed['tasks'][0]['sign']=1
    assert run.packet_digest(changed)!=seal['semantic_sha256']


def test_preflight_failure_retains_all930_and_every_readout(tmp_path,monkeypatch):
    path=setup_run(tmp_path,monkeypatch)
    monkeypatch.setattr(run,'resource_check',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('synthetic headroom failure')))
    monkeypatch.setattr(run.work,'execute',forbidden);monkeypatch.setattr(run.work,'build_libraries',forbidden)
    summary=run.run(path,tmp_path/'data','attempt-001');directory=tmp_path/'data/EXP-014/attempt-001'
    assert not summary['complete'] and not summary['qualified'] and 'headroom failure' in summary['error']
    records=run.read(directory/'all-terminal-records.json');assert sum(map(len,records.values()))==930
    assert all(r['status']=='not_started_dependency_failure' for rows in records.values() for r in rows)
    assert len(summary['scientific']['terminal_rows'])==912 and len(summary['scientific']['paired_readouts'])==5184
    assert run.verify(directory)['passed'] is True


@pytest.mark.parametrize('name',['../attempt-1','x','attempt-','attempt-a'])
def test_invalid_attempt_names_rejected(tmp_path,name):
    with pytest.raises(ValueError):run.external_attempt(tmp_path,name)


def test_output_exclusivity_and_repository_symlink_escape(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir();monkeypatch.setattr(run,'ROOT',repo)
    with pytest.raises(ValueError):run.external_attempt(repo,'attempt-001')
    directory=run.external_attempt(tmp_path/'data','attempt-001');assert directory.is_dir()
    with pytest.raises(FileExistsError):run.external_attempt(tmp_path/'data','attempt-001')
    escape=tmp_path/'escape';escape.mkdir();(escape/'EXP-014').symlink_to(repo,target_is_directory=True)
    with pytest.raises(ValueError):run.external_attempt(escape,'attempt-001')


def test_cooperative_adaptive_timeout_retains_unaccepted_rhs_packet(tmp_path,monkeypatch):
    import time
    task=next(t for t in run.acquire.build_tasks() if t['method']=='dop853')
    spec=dict(id=task['id'],kind='trace',case=task,method=dict(id='dop853'))
    ref=reference('core')
    def slow(fun,span,initial,**kwargs):
        while True:
            fun(.125,initial)
            time.sleep(.005)
    monkeypatch.setattr(run.acquire,'solve_ivp',slow)
    monkeypatch.setattr(run.acquire.diagnostics,'hopfield_rhs',toy_rhs)
    destination=tmp_path/'traces'
    rows=run.work.execute([spec],{task['id']:dict(reference=ref)},destination,{},workers=1,
        deadline=time.monotonic()+5.,per_task=.06,worker_fn=run.worker)
    assert rows[0]['status']=='task_timeout'
    child=run.read(destination/'records'/f'{task["id"]}.json')
    assert child['status']=='failed' and 'Owned child cancellation' in child['error']
    partial=run.load_packet(destination/child['packet_path'])
    assert partial['complete'] is False and partial['task']==task
    assert partial['partial_failure']['last_rhs']['accepted_solver_state'] is False
    assert partial['partial_failure']['last_rhs']['time']==.125
    assert 'TimeoutError' in partial['partial_failure']['error']
    retained=run._packets(destination,rows)[task['id']]
    assert retained['status']=='task_timeout' and retained['qualified'] is False
    assert run.read(destination/'cleanup.json')['all_original_identities_absent'] is True


def test_valid_seal_interrupted_before_replay_event_has_no_trace_dispatch(tmp_path,monkeypatch):
    path=setup_run(tmp_path,monkeypatch);original_execute=run.work.execute
    def bounded(specs,contexts,directory,libraries,**kwargs):
        if directory.name=='traces':forbidden()
        chosen=specs if directory.name=='references' else [t for t in specs if t['case']['phase_index']==0]
        return original_execute(chosen,contexts,directory,libraries,**kwargs)
    monkeypatch.setattr(run.work,'execute',bounded)
    replay=run.evaluation.replay_prepared_reference_outputs;calls=[]
    def fail_once(*args,**kwargs):
        calls.append(True)
        if len(calls)==1:raise RuntimeError('synthetic interruption just after reference seal')
        return replay(*args,**kwargs)
    monkeypatch.setattr(run.evaluation,'replay_prepared_reference_outputs',fail_once)
    summary=run.run(path,tmp_path/'data','attempt-001');directory=tmp_path/'data/EXP-014/attempt-001'
    assert 'just after reference seal' in summary['error']
    assert (directory/'reference-seal.json').exists() and not (directory/'reference-seal-replay.json').exists()
    assert not (directory/'traces').exists()
    assert run.verify(directory)['passed'] is True


@pytest.mark.parametrize('field,value',[('complete',True),('qualified',True),('error',None),('postflight_error','fabricated')])
def test_saved_replay_rejects_forged_top_level_conclusion(tmp_path,monkeypatch,field,value):
    path=setup_run(tmp_path,monkeypatch)
    monkeypatch.setattr(run,'resource_check',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('synthetic stop')))
    run.run(path,tmp_path/'data','attempt-001');directory=tmp_path/'data/EXP-014/attempt-001'
    summary=run.read(directory/'summary.json');summary[field]=value;write(directory/'summary.json',summary)
    receipt=run.read(directory/'receipt.json');receipt['artifacts']=run.work.inventory(directory)
    receipt['artifact_bytes']=sum(r['bytes'] for r in receipt['artifacts']);write(directory/'receipt.json',receipt)
    with pytest.raises(ValueError,match='conclusion|completion|qualification'):run.verify(directory)


@pytest.mark.parametrize('failure',[None,'dirty','branch','remote','review','committed'])
def test_exact_private_source_plan_review_guard_without_network(tmp_path,monkeypatch,failure):
    repo=tmp_path/'repo';repo.mkdir();monkeypatch.setattr(run,'ROOT',repo)
    sources=['tools/a.py','tests/test_a.py'];monkeypatch.setattr(run,'SOURCE_PATHS',sources)
    for rel in sources:
        file=repo/rel;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('synthetic\n')
    path=repo/'experiments/hopfield/EXP-014-plan.json';write(path,{'synthetic':True})
    review=repo/'docs/EXP-014-local-review.json';monkeypatch.setattr(run,'REVIEW',review)
    write(review,dict(status='ready_for_exploratory_execution',plan_sha256=run.sha(path) if failure!='review' else '0'*64))
    commit='1'*40;calls=[]
    def git(*args):
        calls.append(args)
        if args[0]=='status':return '?? unstaged' if failure=='dirty' else ''
        if args[0]=='rev-parse':return commit
        if args[0]=='branch':return 'main' if failure=='branch' else 'codex/synthetic'
        if args[0]=='ls-remote':return ('2'*40 if failure=='remote' else commit)+'\trefs/heads/codex/synthetic'
        forbidden()
    monkeypatch.setattr(run.legacy,'git',git)
    commands=[]
    def show(command):
        commands.append(command)
        rel=command[-1].split(':',1)[1]
        return b'changed' if failure=='committed' else (repo/rel).read_bytes()
    monkeypatch.setattr(run.subprocess,'check_output',show)
    if failure is not None:
        with pytest.raises(ValueError):run.source_guard({},path)
    else:
        result=run.source_guard({},path)
        assert result['source_commit']==commit
        assert [c[-1].split(':',1)[1] for c in commands]==sources+['experiments/hopfield/EXP-014-plan.json','docs/EXP-014-local-review.json']
        assert calls[-1]==('ls-remote','--exit-code','origin','refs/heads/codex/synthetic')


def test_parent_stop_bypasses_per_trace_diagnostic_catches(monkeypatch):
    task=run.acquire.build_tasks()[0]
    monkeypatch.setattr(run.acquire,'trace_derivatives',lambda *a:None)
    def stop(*args,**kwargs):raise run.ProducerStop('synthetic deadline')
    monkeypatch.setattr(run.evaluation,'score_trace',stop)
    with pytest.raises(run.ProducerStop,match='deadline'):
        run.analyze({task['setting']:{}},{(task['setting'],0):{}},{task['id']:{'complete':True}},{} )
