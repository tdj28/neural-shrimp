"""Full synthetic point/capsule/prefix admission; no model or checkpoint calls.

Uses the existing complete-capsule writers and worker-artifact fixture helpers.
No authentication routine is replaced for success tests. Only dimensions,
owned paths and runtime metadata are fixture-sized.
"""
import copy
from fractions import Fraction
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from flint import arb,arb_mat,ctx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
import point_evidence as pe
import test_codex_llm_capsule_admission as captest
import test_codex_llm_derivative_evidence as detest
base,parent,driver=pe.base,pe.parent,pe.continuation
saved,worker=driver.saved,driver.worker
write,read=captest.write,saved.read_json
ASSETS,IDS,TV=captest.ASSETS,captest.PREFIX_IDS,captest.TORCH_VERSION


def prefix_attempt(path,precision,source):
    (path/'run').mkdir(parents=True)
    detest.snapshot(path,['codex/certify-llm/ball_qwen.py'])
    mapping=worker.map_spec(ASSETS,IDS)
    inputs=dict(source_commit=source,versions=parent.backend_versions(TV),precision=precision,
        arb_context=dict(precision=precision,threads=1),skip_box=precision==2048,
        **{k:mapping[k]for k in ('pinned_assets','prefix_ids','alpha_ratio','log10_temperature_ratio','temperature','rotary')})
    write(path/'run/inputs.json',inputs)
    values=[[[[[detest.pack(i+j+k)]]for k in IDS]for j in range(2)]for i in range(2)]
    with gzip.open(path/'run/prefix-cache.json.gz','wt')as f:json.dump(values,f)
    cache=base.digest(path/'run/prefix-cache.json.gz')
    detest.events(path/'run/events.jsonl',[dict(event=e,layer=i)for i in range(2)for e in ('prefix_start','prefix_end')]+[dict(event='prefix_saved',sha256=cache)])
    write(path/'run/status.json',dict(status='point_setup_completed'if precision==2048 else 'setup_failed',
        phase='rigorous_box_jvp',certificate_passed=False))
    write(path/'manifest.json',dict(source_commit=source))
    pin=detest.seal(path,exit_code=0 if precision==2048 else 1)
    return dict(**pin,map_source_sha256=base.digest(ROOT/'codex/certify-llm/ball_qwen.py'),cache_sha256=cache,
        completion='completed_prefix_point'if precision==2048 else 'retained_prefix_before_box_failure')


def seed_attempt(path):
    (path/'run').mkdir(parents=True)
    names=['codex/certify-llm/center_seed.py','codex/crosscheck/qwen_map.py']
    detest.snapshot(path,names)
    write(path/'run/inputs.json',dict(source_commit='2'*40,pinned_assets=ASSETS,prefix_ids=IDS,
        alpha_ratio=list(base.ALPHA.as_integer_ratio()),log10_temperature_ratio=list(base.LOG10_TEMPERATURE.as_integer_ratio()),
        driver_sha256=base.digest(ROOT/names[0]),independent_map_sha256=base.digest(ROOT/names[1]),
        period=8,dimension=1,parameter_array_order=['alpha','log10_temperature']))
    arrays=dict(A=np.full((8,1,1),.5),B=np.full((1,1),1.25),centers=np.zeros((8,1)),
        parameters=np.array([base.ALPHA,base.LOG10_TEMPERATURE]),product=np.zeros((1,1)),
        left_inverse_defect=np.zeros((1,1)),right_inverse_defect=np.zeros((1,1)))
    np.savez(path/'run/preconditioner.npz',**arrays)
    phases=[]
    for i in range(8):
        p=path/'run'/f'phase-{i:02d}.npz';np.savez(p,proposal=arrays['A'][i])
        row=dict(index=i,passed=True,path=p.name,sha256=base.digest(p));phases.append(row)
        write(path/'run'/f'phase-{i:02d}.json',row)
    sha=base.digest(path/'run/preconditioner.npz')
    write(path/'run/preconditioner.json',dict(phase_artifacts=phases,certificate_passed=False,arrays_sha256=sha))
    write(path/'run/status.json',dict(status='proposal_completed',certificate_passed=False,phase_artifacts=phases,preconditioner_sha256=sha))
    write(path/'manifest.json',dict(source_commit='2'*40));pin=detest.seal(path)
    return pin,sha,base.digest(ROOT/names[0]),base.digest(ROOT/names[1])


@pytest.fixture
def full(tmp_path,monkeypatch):
    old=ctx.prec,ctx.threads;ctx.prec,ctx.threads=2048,1
    for module in (base,worker,pe.derivative):
        monkeypatch.setattr(module,'PERIOD',8);monkeypatch.setattr(module,'DIMENSION',1)
    for module in (worker,pe.derivative):monkeypatch.setattr(module,'LAYERS',2)
    for module in (base,worker,driver,pe.derivative,pe.evidence):monkeypatch.setattr(module,'OWNED',tmp_path)
    # Fixture dimensions only, not implementation/authentication substitution.
    monkeypatch.setattr(base.validate_plan,'__kwdefaults__',dict(owned=tmp_path))
    monkeypatch.setattr(base.validate_preconditioner_arrays,'__kwdefaults__',dict(period=8,dimension=1))
    monkeypatch.setenv('CERT_SOURCE_COMMIT',captest.COMMIT)
    pp=prefix_attempt(tmp_path/'prefix',2048,'1'*40)
    dp=prefix_attempt(tmp_path/'derivative-prefix',1024,'3'*40)
    seed,arr,sd,sm=seed_attempt(tmp_path/'seed')
    bp=dict(schema='codex-llm-center-refine-v1',alpha=base.ALPHA,log10_temperature=base.LOG10_TEMPERATURE,
        period=8,dimension=1,precision=2048,max_corrections=1,evaluation_passes=2,
        envelope=dict(cpu_threads=1,nice=10,memory_gib=18,timeout_seconds=3600),prefix_attempt=pp['attempt'],
        prefix_receipt_sha256=pp['receipt_sha256'],prefix_map_source_sha256=pp['map_source_sha256'],
        center_seed_attempt=seed['attempt'],center_seed_receipt_sha256=seed['receipt_sha256'],
        center_seed_driver_sha256=sd,center_seed_map_source_sha256=sm,preconditioner_sha256=arr)
    attempt=tmp_path/'continued';out=attempt/'run';out.mkdir(parents=True)
    ancestor=tmp_path/'ancestor';ancestor.mkdir()
    pins={name:base.digest(ROOT/name)for name in driver.SOURCES}
    old_plan=dict(schema=driver.SCHEMA,parent_kind='center_secant',parent_attempt=str(ancestor),
        parent_receipt_sha256='c'*64,parent_capsule_sha256=None,
        parent_audit={'path':'codex/certify-llm/synthetic-audit.md','sha256':'d'*64},base_plan=bp,source_sha256=pins)
    before=captest.synthetic_state(3,[[arb(2*i+1)]for i in range(8)])
    after=captest.synthetic_state(4,[[arb(3*i+2)]for i in range(8)])
    admitted=dict(before=before,after=after,history=[],factor_references=[],pending_pair_sha256=parent.pending_identity(before,after),
        archive_links=[],prior_phase_evaluations=40,admission={'kind':'synthetic-complete-older-history'})
    proposal,history,refs=driver.prepare_candidate(admitted,[arb_mat([[arb(1)/2]])for _ in range(8)],arb_mat([[arb(5)/4]]),out,
        lambda *a,**kw:None,source_commit=captest.COMMIT,attempt=attempt)
    request,rsha=driver.freeze_request(old_plan,proposal['centers'],out/'candidate-01',ASSETS,IDS,TV,captest.COMMIT)
    captest.save_synthetic_worker_pass(out/'candidate-01',request)
    write(out/'request-binding.json',dict(request_sha256=rsha,centers_sha256=request['centers_sha256']))
    comparison,capsule=driver.finish_candidate(admitted,proposal,history,refs,out,old_plan,ASSETS,IDS,TV,lambda *a,**kw:None)
    detest.snapshot(attempt,driver.SOURCES)
    planpath=attempt/'source/codex/certify-llm/synthetic-parent-plan.json';write(planpath,old_plan)
    _,binding=base.load_bound_inputs(bp,ASSETS,IDS)
    write(out/'inputs.json',dict(source_commit=captest.COMMIT,precision=2048,threads=1,certificate_passed=False,
        pinned_assets=ASSETS,prefix_ids=IDS,versions=parent.backend_versions(TV),plan=old_plan,plan_sha256=base.digest(planpath),
        driver_sha256=pins['codex/certify-llm/center_continue.py'],source_binding=binding))
    write(out/'status.json',dict(status='proposal_continuation_completed',certificate_passed=False,candidates=1,new_phase_evaluations=8,
        cumulative_phase_evaluations=48,comparison=comparison,capsule_sha256=base.digest(out/'continuation-capsule.json')))
    # A temporary local review/source view supplies the external audit file;
    # source bytes themselves remain identical to the reviewed implementations.
    repo=tmp_path/'reviewed-reader';repo.mkdir()
    for name in driver.SOURCES:
        p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    audit='codex/certify-llm/independent-synthetic-audit.md';(repo/audit).write_text('Synthetic external review trust anchor.\n')
    monkeypatch.setattr(driver,'ROOT',repo)
    current=dict(schema=driver.SCHEMA,parent_kind='center_continue',parent_attempt=str(attempt),parent_source_sha256=pins,
        base_plan=bp,parent_audit={'path':audit,'sha256':base.digest(repo/audit)},source_sha256=pins,
        precision=2048,max_candidates=1,caps=dict(driver.restore.CAPS),envelope=dict(driver.ENVELOPE))
    fixture=captest.CompleteFixture(attempt,current,binding);fixture.reseal()
    mapping=pe.derivative.map_definition(ASSETS,IDS)
    definition=dict(schema=pe.derivative.DEFINITION_SCHEMA,role=pe.coverage.ROLE,period=8,dimension=1,
        centers=capsule['after']['centers'],radii=[detest.pack(Fraction(1,2**2000))for _ in range(8)],
        map=mapping,map_sha256=pe.domain.canonical_hash(mapping),selection_provenance={'synthetic_proposed_domain':True})
    definition_path=tmp_path/'candidate.json';write(definition_path,definition)
    admission_path=tmp_path/'admission.json';write(admission_path,current)
    backend=dict(pe.derivative.backend_definition(TV),precision=1024,threads=1)
    args=dict(admission_plan_path=admission_path,admission_plan_sha256=base.digest(admission_path),
        definition_path=definition_path,definition_sha256=base.digest(definition_path),expected_assets=ASSETS,prefix_ids=IDS,
        torch_version=TV,expected_source_sha256={name:base.digest(ROOT/name)for name in pe.SOURCES},
        expected_derivative_backend=backend,expected_derivative_prefix=dp)
    yield args,fixture
    ctx.prec,ctx.threads=old


def refresh(args,fixture):
    fixture.reseal();write(args['admission_plan_path'],fixture.plan)
    args['admission_plan_sha256']=base.digest(args['admission_plan_path'])
    args['definition_sha256']=base.digest(args['definition_path'])


@pytest.mark.parametrize('precision',[64,1024,2048])
def test_complete_original_images_and_context_preserved(full,precision):
    args,fixture=full
    ctx.prec=precision
    centers,images,report=pe.authenticate_points(**args)
    assert ctx.prec==precision and ctx.threads==1
    assert report['artifact_authentication_completed'] and not report['certificate_passed']
    assert report['point_precision']==2048 and report['derivative_precision']==1024
    assert report['assembly_precision']==precision and report['older_ancestry_rehashed']is False
    assert report['point_prefix']['receipt_sha256']!=report['derivative_prefix']['receipt_sha256']
    assert report['derivative_prefix']['overall_attempt_succeeded']is False
    assert report['point_map_sha256']!=report['derivative_map_sha256']
    assert report['model_evaluations']==report['optimizer_actions']==0
    assert not report['git_commit_blob_verification_performed']
    for i in range(8):
        assert pe.domain.pack_ball(centers[i][0])==report['original_centers'][i][0]
        m,r=pe.domain._record_parts(report['original_images'][i][0]);u,v=pe.domain._record_parts(pe.domain.pack_ball(images[i][0]))
        assert r>0 and u-v<=m-r and u+v>=m+r


@pytest.mark.parametrize('fault',['low_bit','rotation','noncanonical','center_radius','missing_phase','missing_coordinate'])
def test_original_center_mismatch_refused_before_any_decoder(full,monkeypatch,fault):
    args,_=full;definition=read(args['definition_path'])
    if fault=='low_bit':
        m,r=pe.domain._record_parts(definition['centers'][0][0]);definition['centers'][0][0]=detest.pack(m+Fraction(1,2**2400))
    elif fault=='rotation':definition['centers']=definition['centers'][1:]+definition['centers'][:1]
    elif fault=='noncanonical':
        pair=definition['centers'][0][0]['mid'];pair[0]*=2;pair[1]-=1
    elif fault=='center_radius':definition['centers'][0][0]['rad']=[1,-2000]
    elif fault=='missing_phase':definition['centers'].pop()
    else:definition['centers'][0].pop()
    write(args['definition_path'],definition);args['definition_sha256']=base.digest(args['definition_path'])
    def forbidden(*a,**kw):pytest.fail('Arb decoder or legacy admission ran before original center check')
    monkeypatch.setattr(pe.coverage,'_decode',forbidden);monkeypatch.setattr(parent,'load_capsule',forbidden)
    with pytest.raises(ValueError):pe.authenticate_points(**args)


@pytest.mark.parametrize('fault',['prototype','map','source','reader_source','backend','assets','prefix_ids','point_prefix_precision',
    'derivative_prefix_precision','parent_failed','original_radius','capsule_pin','definition_bytes','plan_bytes','audit_bytes'])
def test_external_admission_failures(full,fault):
    args,fixture=full
    if fault in ('prototype','map'):
        value=read(args['definition_path'])
        if fault=='prototype':value['role']='prototype_box'
        else:value['map']['alpha_ratio']=[1,2];value['map_sha256']=pe.domain.canonical_hash(value['map'])
        write(args['definition_path'],value);args['definition_sha256']=base.digest(args['definition_path'])
    elif fault=='source':args['expected_source_sha256'].pop('codex/certify-llm/secant_parent.py')
    elif fault=='reader_source':args['expected_source_sha256']['codex/certify-llm/secant_parent.py']='0'*64
    elif fault=='backend':args['expected_derivative_backend']['precision']=2048
    elif fault=='assets':args['expected_assets']={'synthetic_weights':'0'*64}
    elif fault=='prefix_ids':args['prefix_ids']=[1,3]
    elif fault=='point_prefix_precision':
        p=Path(fixture.plan['base_plan']['prefix_attempt']);captest.mutate(p/'run/inputs.json',lambda v:v.update(precision=1024))
        fixture.plan['base_plan']['prefix_receipt_sha256']=detest.seal(p)['receipt_sha256'];refresh(args,fixture)
    elif fault=='derivative_prefix_precision':
        p=Path(args['expected_derivative_prefix']['attempt']);captest.mutate(p/'run/inputs.json',lambda v:v.update(precision=2048))
        args['expected_derivative_prefix']['receipt_sha256']=detest.seal(p,exit_code=1)['receipt_sha256']
    elif fault=='parent_failed':
        captest.mutate(fixture.out/'status.json',lambda v:v.update(status='proposal_continuation_failed'));refresh(args,fixture)
    elif fault=='original_radius':
        captest.mutate(fixture.out/'continuation-capsule.json',lambda v:v['after']['original_images'][0][0].update(rad=[0,0]));refresh(args,fixture)
    elif fault=='capsule_pin':fixture.plan['parent_capsule_sha256']='0'*64;write(args['admission_plan_path'],fixture.plan);args['admission_plan_sha256']=base.digest(args['admission_plan_path'])
    elif fault=='definition_bytes':args['definition_path'].write_bytes(args['definition_path'].read_bytes()+b' ')
    elif fault=='plan_bytes':args['admission_plan_path'].write_bytes(args['admission_plan_path'].read_bytes()+b' ')
    elif fault=='audit_bytes':(driver.ROOT/fixture.plan['parent_audit']['path']).write_text('Changed external review.')
    ctx.prec=80
    with pytest.raises(ValueError):pe.authenticate_points(**args)
    assert ctx.prec==80 and ctx.threads==1
