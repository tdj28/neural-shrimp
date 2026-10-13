"""Synthetic complete-parent and pending-pair tests; no target/model execution."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
import center_continue as driver
parent=driver.parent
saved=driver.saved
base=driver.base


@pytest.fixture(autouse=True)
def tiny(monkeypatch,tmp_path):
    old=ctx.prec,ctx.threads;ctx.prec=2048;ctx.threads=1
    monkeypatch.setattr(base,'PERIOD',2);monkeypatch.setattr(base,'DIMENSION',1)
    monkeypatch.setattr(base,'OWNED',tmp_path);monkeypatch.setattr(driver,'OWNED',tmp_path)
    yield
    ctx.prec,ctx.threads=old


def h(c):return [[c[0][0]/2-c[1][0]],[c[1][0]/2-c[0][0]]]
def centers(i):return [[arb(i)],[arb(i*i+i)]]
def A_B():return [arb_mat([[arb(1)/2]]) for _ in range(2)],arb_mat([[arb(5)/4]])
def provenance():return [dict(phase_index=i,input_sha256='a'*64,point_sha256='b'*64) for i in range(2)]
def state(i,cs=None):
    cs=centers(i) if cs is None else cs
    images=[[cs[0][0]/2],[cs[1][0]/2]]
    return parent.tuple_from_records(saved.center_records(cs),saved.center_records(images),saved.center_records(images),index=i,provenance=provenance())


def admitted(tmp_path,count=3):
    A,B=A_B();kernel=driver.inverse_secant_fast.InverseSecant(A,B,**driver.restore.CAPS)
    refs=[]
    for i in range(count):
        old,new=state(i),state(i+1)
        decision=kernel.update(old['centers'],old['residuals'],new['centers'],new['residuals'],evidence={'i':i})
        if decision['accepted']:refs.append(dict(attempt=str(tmp_path/'parent'),source_commit='a'*40,path=f'run/secant-{i}.json',sha256='d'*64))
        else:refs=[]
    prior=state(count)
    proposal=kernel.propose(prior['centers'],prior['residuals'],evidence={'final':count})
    if not kernel.history:refs=[]
    after=state(count+1,proposal['centers'])
    return dict(before=prior,after=after,history=kernel.history,factor_references=refs,
        pending_pair_sha256=parent.pending_identity(prior,after),archive_links=[],prior_phase_evaluations=2*(count+2),
        admission={'kind':'synthetic'})


def test_exactly_one_pending_pair_and_no_inverse_mutation(tmp_path):
    inherited=admitted(tmp_path);A,B=A_B();out=tmp_path/'run';out.mkdir();events=[]
    proposal,history,refs=driver.prepare_candidate(inherited,A,B,out,lambda name,**kw:events.append(name),source_commit='f'*40,attempt=tmp_path/'child')
    decision=saved.read_json(out/'pending-secant.json');factor=saved.read_json(out/'factor-state.json')
    assert decision['evidence']['old_tuple']==3 and decision['evidence']['new_tuple']==4
    assert decision['old_centers']==[v for row in inherited['before']['record']['centers'] for v in row]
    assert decision['new_centers']==[v for row in inherited['after']['record']['centers'] for v in row]
    assert decision['old_residuals']==[base.packed(v) for row in inherited['before']['residuals'] for v in row]
    assert factor['history_before_update']==len(inherited['history'])
    assert len(history)==len(refs)==factor['history_after_proposal']
    assert events==['single_pending_secant_started','single_candidate_saved']
    assert A[0][0,0]==arb(1)/2 and B[0,0]==arb(5)/4
    assert isinstance(driver.restore.install_factors(A,B,[]),driver.inverse_secant_fast.InverseSecant)


def test_full_history_restart_discards_pending_pair_without_retry(tmp_path):
    inherited=admitted(tmp_path,4);assert len(inherited['history'])==4
    out=tmp_path/'run';out.mkdir()
    _,history,refs=driver.prepare_candidate(inherited,*A_B(),out,lambda *a,**k:None,source_commit='f'*40,attempt=tmp_path/'child')
    decision=saved.read_json(out/'pending-secant.json');proposal=saved.read_json(out/'proposal.json')
    assert decision['reason']=='history_cap_restart' and not decision['accepted']
    assert proposal['reason']=='baseline_no_history' and history==refs==[]
    assert saved.read_json(out/'factor-state.json')['history_after_update']==0


def test_pending_pair_tamper_prevents_kernel(tmp_path,monkeypatch):
    inherited=admitted(tmp_path);inherited['pending_pair_sha256']='0'*64
    monkeypatch.setattr(driver.restore,'install_factors',lambda *a,**k:pytest.fail('kernel before pair binding'))
    with pytest.raises(ValueError,match='pending'):driver.prepare_candidate(inherited,*A_B(),tmp_path,lambda *a,**k:None,source_commit='f'*40,attempt=tmp_path/'child')


@pytest.mark.parametrize('fault',['index','radius','missing','narrowed','provenance','summary'])
def test_full_capsule_tuple_refusals(fault):
    row=copy.deepcopy(state(3)['record'])
    if fault=='index':row['sequence_index']=True
    if fault=='radius':row['centers'][0][0]['rad']=[1,-2000]
    if fault=='missing':row['images'].pop()
    if fault=='narrowed':row['original_images'][0][0]['rad']=[1,-2000]
    if fault=='provenance':row['phase_provenance'][0]['phase_index']=1
    if fault=='summary':row['residual']['inf_upper']={'mid':[0,0],'rad':[0,0]}
    with pytest.raises(ValueError):parent.read_tuple(row)


def test_original_radii_survive_capsule_and_new_residual_computation():
    cs=saved.center_records(centers(3));original=[[base.packed(arb(3,(1,-1800)))],[base.packed(arb(4,(1,-1790)))]]
    st=parent.tuple_from_records(cs,original,original,index=3,provenance=provenance())
    loaded=parent.read_tuple(st['record'])
    assert all(v.rad()>0 for row in loaded['images'] for v in row)
    assert all(v.rad()>0 for row in loaded['residuals'] for v in row)
    assert loaded['record']['original_images']==original


@pytest.mark.parametrize('fault',['shape','radius','witness_hash','escape','cap'])
def test_factor_capsule_refusals(tmp_path,fault):
    inherited=admitted(tmp_path,1);records=parent.factor_records(inherited['history'],inherited['factor_references'])
    assert records
    if fault=='shape':records[0]['u'].pop()
    if fault=='radius':records[0]['u'][0]['rad']=[1,-100]
    if fault=='witness_hash':records[0]['witness']['sha256']=''
    if fault=='escape':records[0]['witness']['path']='../other'
    if fault=='cap':
        records[0]['u']=[base.packed(arb(1<<30))]*2;records[0]['w']=[base.packed(arb(1))]*2
    with pytest.raises(ValueError):parent.read_factors(records)


def test_saved_factor_state_exactly_restores_without_old_arithmetic(tmp_path):
    inherited=admitted(tmp_path,3);records=parent.factor_records(inherited['history'],inherited['factor_references'])
    history,refs=parent.read_factors(records)
    assert history==inherited['history'] and refs==inherited['factor_references']


def plan_fixture(tmp_path,monkeypatch):
    for name in ('parent','prefix','seed'):(tmp_path/name).mkdir()
    monkeypatch.setattr(base,'validate_plan',lambda *a,**k:None)
    return dict(schema=driver.SCHEMA,parent_kind='center_secant',parent_attempt=str(tmp_path/'parent'),
        parent_receipt_sha256='b'*64,parent_source_sha256={k:'c'*64 for k in parent.IMMUTABLE},
        parent_audit={'path':'codex/certify-llm/audit.md','sha256':'d'*64},parent_capsule_sha256=None,
        base_plan={'prefix_attempt':str(tmp_path/'prefix'),'center_seed_attempt':str(tmp_path/'seed')},
        source_sha256={k:'e'*64 for k in driver.SOURCES},precision=2048,max_candidates=1,
        caps=copy.deepcopy(driver.restore.CAPS),envelope=copy.deepcopy(driver.ENVELOPE))


def test_initial_and_continuation_plan_scope(tmp_path,monkeypatch):
    plan=plan_fixture(tmp_path,monkeypatch);assert driver.validate_plan(plan)==tmp_path/'parent'
    plan.update(parent_kind='center_continue',parent_capsule_sha256='a'*64,
        parent_source_sha256={k:'c'*64 for k in driver.SOURCES})
    assert driver.validate_plan(plan)==tmp_path/'parent'


@pytest.mark.parametrize('fault',['workers','alias','timeout','caps','precision','source_missing','audit','capsule','kind','candidate'])
def test_plan_refuses_tampered_scope(tmp_path,monkeypatch,fault):
    plan=plan_fixture(tmp_path,monkeypatch)
    if fault=='workers':plan['envelope']['workers']=8
    if fault=='alias':plan['envelope']['cpu_threads']=True
    if fault=='timeout':plan['envelope']['timeout_seconds']=3600
    if fault=='caps':plan['caps']['max_history']=5
    if fault=='precision':plan['precision']=1024
    if fault=='source_missing':plan['source_sha256'].pop(next(iter(plan['source_sha256'])))
    if fault=='audit':plan['parent_audit']['sha256']=''
    if fault=='capsule':plan['parent_capsule_sha256']='a'*64
    if fault=='kind':plan['parent_kind']='center_resume'
    if fault=='candidate':plan['max_candidates']=2
    with pytest.raises(ValueError):driver.validate_plan(plan)


def test_isolated_help_does_not_load_model():
    result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'codex/certify-llm/center_continue.py'),'--help'],capture_output=True,text=True,timeout=10)
    assert result.returncode==0 and '--plan' in result.stdout


def parallel_fixture(tmp_path,monkeypatch,fault=None):
    import parallel_phase_pass as scheduling
    monkeypatch.setattr(parent.worker,'LAYERS',2)
    candidate=tmp_path/'candidate';candidate.mkdir();metadata=candidate/'parallel-workers';metadata.mkdir()
    centers=state(3)['centers'];raw=saved.center_records(centers)
    expected=dict(source_commit='a'*40,source_sha256={scheduling.WORKER_KEY:'b'*64},
        map={'exact':'synthetic'},map_sha256=parent.canonical({'exact':'synthetic'}),
        backend={'precision':2048},backend_sha256=parent.canonical({'precision':2048}),
        prefix_attempt=str(tmp_path/'prefix'),prefix_receipt_sha256='c'*64,prefix_map_source_sha256='d'*64,
        prefix_cache_sha256='e'*64,precision=2048)
    request=dict(**copy.deepcopy(expected),centers=raw,centers_sha256=parent.canonical(raw))
    if fault=='foreign_map':request.update(map={'exact':'different'},map_sha256=parent.canonical({'exact':'different'}))
    if fault=='foreign_backend':request.update(backend={'precision':1024},backend_sha256=parent.canonical({'precision':1024}))
    if fault=='foreign_source':request['source_commit']='f'*40
    if fault=='foreign_prefix':request['prefix_cache_sha256']='f'*64
    base.write_json(candidate/'request.json',request);request_sha=base.digest(candidate/'request.json')
    report=dict(status='completed',request_sha256=request_sha,unstarted_phases=[],source_commit=request['source_commit'],
        max_workers=4,worker_timeout_seconds=600,reserve_seconds=600,aggregate_rss_bytes=32*1024**3,minimum_available_bytes=12*1024**3,
        scheduler_sha256=base.digest(Path(scheduling.__file__)),sampled_rss_is_hard_cap=False,scientific_endpoint_audit_performed=False,
        worker_sha256=request['source_sha256'][scheduling.WORKER_KEY],certificate_passed=False,phase_receipts=[],
        termination_verification=[dict(phase_index=i,leader_exit_code=0,verified_termination=True,remaining_observed_owned_processes=[]) for i in range(2)])
    for i in range(2):
        out=candidate/f'phase-{i:02d}';out.mkdir();(out/'request.json').write_bytes((candidate/'request.json').read_bytes())
        inputs=dict(request_sha256=request_sha,source_commit=request['source_commit'],phase_index=i,
            map_sha256=request['map_sha256'],backend_sha256=request['backend_sha256'],source_sha256=request['source_sha256'],certificate_passed=False)
        base.write_json(out/'inputs.json',inputs)
        base.write_json(out/'input.json',dict(phase_index=i,center=raw[i],next_center=raw[(i+1)%2]))
        live=copy.deepcopy(raw[i])
        if fault=='noncanonical_live' and i==0:
            m,e=live[0]['mid'];live[0]['mid']=[m*2,e-1]
        if fault=='live_radius' and i==0:live[0]['rad']=[1,-2100]
        zero=[base.packed(arb(0))]
        base.write_json(out/'live-input.json',dict(phase_index=i,center=live,tangent=zero))
        value=[base.packed(arb(i+1,(1,-1800)))];tangent=copy.deepcopy(zero)
        if fault=='nonzero_tangent' and i==0:tangent=[base.packed(arb(1))]
        base.write_json(out/'point.json',dict(value=value,tangent=tangent))
        base.write_json(out/'prefix-binding.json',dict(receipt_sha256=request['prefix_receipt_sha256'],graph_sha256=request['prefix_map_source_sha256'],prefix_sha256=request['prefix_cache_sha256'],precision=2048,receiver_construction_checked=True))
        events=[dict(event=name,layer=j) for j in range(2) for name in ('step_start','step_end')]
        if fault=='missing_layer' and i==0:events.pop()
        (out/'events.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in events))
        status=dict(status='phase_point_completed',phase_index=i,request_sha256=request_sha,certificate_passed=False,
            input_sha256=base.digest(out/'input.json'),live_input_sha256=base.digest(out/'live-input.json'),point_sha256=base.digest(out/'point.json'))
        base.write_json(out/'status.json',status)
        receipt=dict(status='completed',phase_index=i,exit_code=0,verified_exit=True,request_sha256=request_sha,
            worker_sha256=request['source_sha256'][scheduling.WORKER_KEY],certificate_passed=False,artifacts=scheduling.inventory(out))
        if fault=='wrong_receipt_phase' and i==0:receipt['phase_index']=1
        if fault=='wrong_receipt_request' and i==0:receipt['request_sha256']='0'*64
        base.write_json(metadata/f'phase-{i:02d}.receipt.json',receipt)
        report['phase_receipts'].append(dict(phase_index=i,status='completed',receipt_sha256=base.digest(metadata/f'phase-{i:02d}.receipt.json')))
    if fault=='missing_termination':report['termination_verification'].pop()
    if fault=='duplicate_termination':report['termination_verification'][1]['phase_index']=0
    if fault=='false_exit_alias':report['termination_verification'][0]['leader_exit_code']=False
    if fault=='remaining_process':report['termination_verification'][0]['remaining_observed_owned_processes']=[{'pid':123}]
    base.write_json(candidate/'parallel-pass.json',report)
    return candidate,centers,expected


def test_parallel_collection_original_radii_and_all_phases(tmp_path,monkeypatch):
    candidate,centers,expected=parallel_fixture(tmp_path,monkeypatch)
    images,provenance=parent.collect_parallel_images(candidate,centers,expected_identity=expected)
    assert len(images)==len(provenance)==2
    assert all(saved.dyadic(row[0]['rad'])>0 for row in images)
    assert [r['phase_index'] for r in provenance]==[0,1]


@pytest.mark.parametrize('fault',['foreign_map','foreign_backend','foreign_source','foreign_prefix',
    'noncanonical_live','live_radius','nonzero_tangent','missing_layer','wrong_receipt_phase','wrong_receipt_request',
    'missing_termination','duplicate_termination','false_exit_alias','remaining_process'])
def test_parallel_coherent_but_invalid_evidence_refused(tmp_path,monkeypatch,fault):
    candidate,centers,expected=parallel_fixture(tmp_path,monkeypatch,fault)
    with pytest.raises(ValueError):parent.collect_parallel_images(candidate,centers,expected_identity=expected)


def test_complete_parent_inventory_authentication_and_extra_file_refusal(tmp_path):
    attempt=tmp_path/'attempt';source=attempt/'source';source.mkdir(parents=True)
    (source/'synthetic.py').write_text('# no model\n')
    row=dict(path='synthetic.py',bytes=(source/'synthetic.py').stat().st_size,sha256=base.digest(source/'synthetic.py'))
    base.write_json(attempt/'manifest.json',dict(source_commit='a'*40,source=[row]))
    rows=[dict(path=str(p.relative_to(attempt)),bytes=p.stat().st_size,sha256=base.digest(p)) for p in [source/'synthetic.py',attempt/'manifest.json']]
    base.write_json(attempt/'receipt.json',dict(source_commit='a'*40,exit_code=0,timed_out=False,artifacts=rows))
    sha=base.digest(attempt/'receipt.json')
    _,files=parent.authenticated_attempt(attempt,sha);assert len(files)==2
    (attempt/'unlisted').write_text('retained but not inventoried')
    with pytest.raises(ValueError,match='inventory'):parent.authenticated_attempt(attempt,sha)


def test_failure_after_update_keeps_pending_record_and_original_inputs(tmp_path,monkeypatch):
    inherited=admitted(tmp_path);out=tmp_path/'run';out.mkdir()
    def broken(self,*a,**k):raise ValueError('synthetic proposal failure')
    monkeypatch.setattr(driver.inverse_secant_fast.InverseSecant,'propose',broken)
    with pytest.raises(ValueError,match='synthetic'):driver.prepare_candidate(inherited,*A_B(),out,lambda *a,**k:None,source_commit='f'*40,attempt=tmp_path/'child')
    assert (out/'admitted-parent.json').is_file() and (out/'pending-secant.json').is_file()
    assert (out/'residual-before.json').is_file() and not (out/'centers-after.json').exists()


def capsule_fixture(tmp_path):
    inherited=admitted(tmp_path,2);before,after=inherited['before'],inherited['after']
    plan=dict(base_plan={'immutable':'base'},parent_capsule_sha256='a'*64)
    status=dict(status='proposal_continuation_completed',candidates=1,new_phase_evaluations=2,capsule_sha256='a'*64)
    cap=dict(schema=parent.CAPSULE_SCHEMA,certificate_passed=False,caps=copy.deepcopy(driver.restore.CAPS),
        base_plan=plan['base_plan'],map={'map':'fixed'},backend={'precision':2048},
        before=before['record'],after=after['record'],
        post_proposal_factors=parent.factor_records(inherited['history'],inherited['factor_references']),
        active_history_after_proposal=len(inherited['history']),pending_pair_sha256=inherited['pending_pair_sha256'],
        cumulative_phase_evaluations=inherited['prior_phase_evaluations'],
        archive_links=[dict(attempt=str(tmp_path/'ancestor'),receipt_sha256='b'*64,
            external_audit={'path':'codex/certify-llm/audit.md','sha256':'c'*64},capsule_sha256=None)],
        trust_boundary='Explicit external audit required; older bytes not rehashed.')
    return cap,plan,status


def decode_capsule(cap,plan,status):
    return parent.decode_capsule(cap,plan,status,expected_map={'map':'fixed'},expected_backend={'precision':2048})


def test_future_capsule_restores_exact_post_proposal_state_without_old_archive_reads(tmp_path,monkeypatch):
    cap,plan,status=capsule_fixture(tmp_path)
    monkeypatch.setattr(parent,'authenticated_attempt',lambda *a,**k:pytest.fail('pure capsule decoder reread ancestry'))
    before,after,history,refs=decode_capsule(cap,plan,status)
    assert after['record']['sequence_index']==before['record']['sequence_index']+1
    assert parent.pending_identity(before,after)==cap['pending_pair_sha256']
    assert parent.factor_records(history,refs)==cap['post_proposal_factors']


@pytest.mark.parametrize('fault',['pending_hash','nonconsecutive','wrong_map','wrong_backend','wrong_base','wrong_caps',
    'preproposal_count','factor_radius','capsule_status_hash','missing_archive','missing_audit','partial','certificate'])
def test_future_capsule_tamper_refused(tmp_path,fault):
    cap,plan,status=capsule_fixture(tmp_path)
    if fault=='pending_hash':cap['pending_pair_sha256']='0'*64
    if fault=='nonconsecutive':cap['after']['sequence_index']+=1
    if fault=='wrong_map':cap['map']={'map':'different'}
    if fault=='wrong_backend':cap['backend']={'precision':1024}
    if fault=='wrong_base':cap['base_plan']={'immutable':'changed'}
    if fault=='wrong_caps':cap['caps']['max_history']=8
    if fault=='preproposal_count':cap['active_history_after_proposal']+=1
    if fault=='factor_radius':cap['post_proposal_factors'][0]['u'][0]['rad']=[1,-2048]
    if fault=='capsule_status_hash':status['capsule_sha256']='f'*64
    if fault=='missing_archive':cap['archive_links']=[]
    if fault=='missing_audit':cap['archive_links'][0]['external_audit']['sha256']=''
    if fault=='partial':status['status']='proposal_continuation_failed'
    if fault=='certificate':cap['certificate_passed']=True
    with pytest.raises(ValueError):decode_capsule(cap,plan,status)


def test_self_reported_capsule_validated_flag_never_substitutes_external_hash(tmp_path,monkeypatch):
    path=tmp_path/'capsule.json';path.write_text('{}')
    monkeypatch.setattr(parent,'common_parent',lambda *a,**k:({}, {'run/continuation-capsule.json':path},{},{}))
    with pytest.raises(ValueError,match='external admitted capsule pin'):
        parent.load_capsule({'parent_capsule_sha256':'0'*64},{},[],None,{})


@pytest.mark.parametrize('field',['source_sha256','map_sha256','backend_sha256','prefix_attempt',
    'prefix_receipt_sha256','prefix_map_source_sha256','precision'])
def test_every_remaining_immediate_request_identity_field_checked(tmp_path,monkeypatch,field):
    candidate,centers,expected=parallel_fixture(tmp_path,monkeypatch)
    original=copy.deepcopy(expected)
    if field=='source_sha256':expected[field]={next(iter(expected[field])):'0'*64}
    elif field=='precision':expected[field]=1024
    elif field=='prefix_attempt':expected[field]=str(tmp_path/'another-prefix')
    else:expected[field]='0'*64
    assert expected!=original
    with pytest.raises(ValueError):parent.collect_parallel_images(candidate,centers,expected_identity=expected)


@pytest.mark.parametrize('key,value',[('scheduler_sha256','0'*64),('max_workers',8),('worker_timeout_seconds',601),
    ('reserve_seconds',0),('aggregate_rss_bytes',64*1024**3),('minimum_available_bytes',1),
    ('sampled_rss_is_hard_cap',True),('scientific_endpoint_audit_performed',True)])
def test_collector_rejects_contradictory_scheduler_scope(tmp_path,monkeypatch,key,value):
    candidate,centers,expected=parallel_fixture(tmp_path,monkeypatch)
    report=saved.read_json(candidate/'parallel-pass.json');report[key]=value;base.write_json(candidate/'parallel-pass.json',report)
    with pytest.raises(ValueError):parent.collect_parallel_images(candidate,centers,expected_identity=expected)


def test_proposal_binding_cannot_relabel_step_fallback_or_midpoint(tmp_path):
    inherited=admitted(tmp_path,3)
    A,B=A_B();kernel=driver.restore.install_factors(A,B,inherited['history'])
    before=inherited['after'];proposal=kernel.propose(before['centers'],before['residuals'],evidence={'synthetic':True})
    after=state(before['record']['sequence_index']+1,proposal['centers'])
    parent.validate_proposal_binding(proposal['record'],before,after,len(kernel.history))
    # The full capsule loader independently binds next_centers; this pure helper
    # reconstructs the saved chosen correction against the caller's exact tuple.
    altered={**after,'centers':[list(row) for row in after['centers']]};altered['centers'][0][0]=arb(1<<100)
    with pytest.raises(ValueError,match='midpoint'):parent.validate_proposal_binding(proposal['record'],before,altered,len(kernel.history))
    changed=copy.deepcopy(proposal['record']);changed['active_history']=5
    with pytest.raises(ValueError):parent.validate_proposal_binding(changed,before,after,len(kernel.history))
