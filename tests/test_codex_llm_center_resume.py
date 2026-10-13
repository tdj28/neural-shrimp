"""Tiny complete-parent continuations; no checkpoint or target calls."""
import copy
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import flint
from flint import arb, arb_mat, ctx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'codex/certify-llm'))
import center_resume as resume
base = resume.base


@pytest.fixture(autouse=True)
def tiny_scope(monkeypatch):
    monkeypatch.setattr(base, 'PERIOD', 2); monkeypatch.setattr(base, 'DIMENSION', 2)
    with ctx.workprec(2048): yield


def point(value): return base.packed(arb(value))


class TinyMap:
    def __init__(self, *, fail_at=None): self.calls = 0; self.fail_at = fail_at
    def forward(self, center):
        self.calls += 1
        if self.calls == self.fail_at: raise ValueError('synthetic interrupted phase')
        return [arb(1)+arb((1, -1700)) for _ in center], [arb(0) for _ in center]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); base.write_json(path, value)


def seal(attempt):
    receipt = dict(source_commit='a'*40, exit_code=0, timed_out=False, elapsed_seconds=1.5,
        artifacts=[dict(path=str(p.relative_to(attempt)), bytes=p.stat().st_size, sha256=base.digest(p))
                   for p in sorted(attempt.rglob('*')) if p.is_file() and p.name != 'receipt.json'])
    write(attempt/'receipt.json', receipt)
    return base.digest(attempt/'receipt.json')


def base_plan(owned):
    return dict(schema='codex-llm-center-refine-v1', alpha=.071674, log10_temperature=-.115,
        period=2, dimension=2, precision=2048, max_corrections=1, evaluation_passes=2,
        envelope=dict(cpu_threads=1, nice=10, memory_gib=18, timeout_seconds=3600),
        prefix_attempt=str(owned/'prefix'), center_seed_attempt=str(owned/'seed'),
        **{key:'b'*64 for key in base.HASH_KEYS})


def source_hashes(attempt, kind):
    names = list(resume.COMMON_SOURCES)
    if kind == 'center_resume': names += ['codex/certify-llm/center_resume.py']
    result = {}
    for name in names:
        path = attempt/'source'/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes()); result[name] = base.digest(path)
    return result


def shared_metadata(attempt, plan, source_hashes, kind, assets, binding):
    import torch
    source_plan = attempt/'source/codex/certify-llm/frozen-plan.json'
    write(source_plan, plan)
    write(attempt/'run/inputs.json', dict(plan=plan, plan_sha256=base.digest(source_plan), source_commit='a'*40,
        driver_sha256=source_hashes['codex/certify-llm/'+kind+'.py'], pinned_assets=assets,
        precision=2048, threads=1, source_binding=binding,
        versions=dict(python=sys.version, python_flint=flint.__version__, flint=flint.__FLINT_VERSION__, numpy=np.__version__, torch=torch.__version__)))
    bp = plan if kind == 'center_refine' else plan['base_plan']
    write(attempt/'run/prefix-binding.json', dict(receipt_sha256=bp['prefix_receipt_sha256'],
        graph_sha256=bp['prefix_map_source_sha256'], source_commit=binding['prefix_source'], precision=2048,
        receiver_construction_checked=True, steering_layer=12,
        prefix_sha256=base.digest(Path(bp['prefix_attempt'])/'run/prefix-cache.json.gz')))


def prepare_base(tmp_path, monkeypatch):
    owned = tmp_path/'owned'; owned.mkdir()
    for name in ('prefix', 'seed', 'initial'): (owned/name).mkdir()
    prefix = owned/'prefix/run/prefix-cache.json.gz'; prefix.parent.mkdir(); prefix.write_bytes(b'synthetic')
    monkeypatch.setattr(resume, 'OWNED', owned)
    # Pure base validation defaults to its module-owned root; preserve the
    # production validation logic while using this fixture's owned directory.
    real_validate = base.validate_plan
    monkeypatch.setattr(base, 'validate_plan', lambda p, *, owned=owned: real_validate(p, owned=owned))
    bp = base_plan(owned); attempt = owned/'initial'; out = attempt/'run'; out.mkdir()
    centers = [[arb(2), arb(3)], [arb(-2), arb(4)]]
    derivatives = [arb_mat(2, 2), arb_mat(2, 2)]; inverse = arb_mat([[1,0],[0,1]])
    outcomes = []; model = TinyMap()
    before_images = base.evaluate_pass(model, centers, out, 'before', lambda *a, **k:None, outcomes)
    before_upper, before_lower, before = base.residual_summary(centers, before_images)
    write(out/'centers-before.json', resume.center_records(centers)); write(out/'residual-before.json', before)
    proposal = base.correct(centers, before_images, derivatives, inverse)
    write(out/'correction-witnesses.json', proposal['record']); centers = proposal['centers']
    write(out/'centers-after.json', resume.center_records(centers))
    images = base.evaluate_pass(model, centers, out, 'after', lambda *a, **k:None, outcomes)
    upper, lower, after = base.residual_summary(centers, images); write(out/'residual-after.json', after)
    comparison = dict(before_inf_upper=base.packed(before_upper), before_inf_lower=base.packed(before_lower),
        after_inf_upper=base.packed(upper), after_inf_lower=base.packed(lower),
        upper_bound_decreased=bool(upper<before_upper), strict_residual_norm_decrease_established=bool(upper<before_lower))
    write(out/'residual-comparison.json', comparison)
    write(out/'status.json', dict(status='proposal_setup_completed', certificate_passed=False, corrections=1,
        phase_outcomes=outcomes, comparison=comparison))
    hashes = source_hashes(attempt, 'center_refine'); assets = {'synthetic':'c'*64}
    binding = dict(prefix_source='b'*40, seed_source='c'*40)
    shared_metadata(attempt, bp, hashes, 'center_refine', assets, binding)
    plan = dict(schema=resume.SCHEMA, base_plan=bp, parent_attempt=str(attempt), parent_kind='center_refine',
        parent_source_sha256=hashes, parent_receipt_sha256=seal(attempt), max_corrections=3, envelope=bp['envelope'])
    return owned, plan, assets, binding, derivatives, inverse


def load(plan, assets, binding):
    return resume.load_completed_parent(plan, assets, (1,2), binding=binding)


def test_complete_parent_exact_input_admitted_without_evaluations(tmp_path, monkeypatch):
    _, plan, assets, binding, _, _ = prepare_base(tmp_path, monkeypatch)
    got = load(plan, assets, binding)
    assert got['cumulative_phase_evaluations'] == 4
    assert got['cumulative_elapsed_seconds'] == 1.5
    assert all(v == arb(1)+arb((1,-1700)) and v.is_exact() for row in got['centers'] for v in row)
    assert len(got['reuse']) == 2 and len(got['lineage']) == 1


@pytest.mark.parametrize('key,value', [('max_corrections',0),('max_corrections',4),('max_corrections',True),
    ('schema','other'),('parent_kind','partial'),('parent_receipt_sha256','a'*63),
    ('parent_source_sha256',{}),('envelope',dict(cpu_threads=4,nice=10,memory_gib=18,timeout_seconds=3600))])
def test_wrong_scope_refused(tmp_path, monkeypatch, key, value):
    _, plan, _, _, _, _ = prepare_base(tmp_path, monkeypatch); plan[key] = value
    with pytest.raises(ValueError): resume.validate_plan(plan)


def test_unknown_plan_and_same_upstream_attempt_refused(tmp_path, monkeypatch):
    _, plan, _, _, _, _ = prepare_base(tmp_path, monkeypatch)
    for change in ({'acceleration':'anderson'}, {'parent_attempt':plan['base_plan']['prefix_attempt']}):
        bad = copy.deepcopy(plan); bad.update(change)
        with pytest.raises(ValueError): resume.validate_plan(bad)


@pytest.mark.parametrize('fault', ['incomplete','wrong-input','missing-coordinate','nonzero-tangent','radius-center',
    'wrong-backend','wrong-params','changed-map','failed-receipt','timeout','missing-phase','wrong-comparison','wrong-correction-images'])
def test_bad_complete_parent_refused_even_after_new_receipt_hash(tmp_path, monkeypatch, fault):
    _, plan, assets, binding, _, _ = prepare_base(tmp_path, monkeypatch)
    attempt = Path(plan['parent_attempt']); out = attempt/'run'
    if fault == 'incomplete':
        value = resume.read_json(out/'status.json'); value['status']='proposal_setup_failed'; write(out/'status.json', value)
    elif fault in ('wrong-input','missing-coordinate','radius-center'):
        path = out/'after-phase-00-input.json'; value = resume.read_json(path)
        if fault == 'wrong-input': value['center'][0]=point(99)
        elif fault == 'missing-coordinate': value['center'].pop()
        else: value['center'][0]['rad']=[1,-1900]
        write(path, value)
        rows = resume.read_json(out/'phase-outcomes.json'); rows[2]['input_sha256']=base.digest(path)
        write(out/'phase-outcomes.json', rows)
    elif fault == 'nonzero-tangent':
        path=out/'after-phase-00-point.json';value=resume.read_json(path);value['tangent'][0]=point(1);write(path,value)
        rows=resume.read_json(out/'phase-outcomes.json');rows[2]['point_sha256']=base.digest(path);write(out/'phase-outcomes.json',rows)
    elif fault in ('wrong-backend','wrong-params'):
        value=resume.read_json(out/'inputs.json')
        if fault=='wrong-backend': value['versions']['flint']='wrong'
        else: value['plan']['alpha']=.2
        write(out/'inputs.json',value)
    elif fault=='wrong-correction-images':
        path=out/'correction-witnesses.json';value=resume.read_json(path);value['images'][0][0]=point(999);write(path,value)
    elif fault=='changed-map':
        (attempt/'source/codex/certify-llm/ball_qwen.py').write_text('changed')
        plan['parent_source_sha256']['codex/certify-llm/ball_qwen.py']=base.digest(attempt/'source/codex/certify-llm/ball_qwen.py')
    elif fault=='missing-phase':
        rows=resume.read_json(out/'phase-outcomes.json');rows.pop();write(out/'phase-outcomes.json',rows)
    elif fault=='wrong-comparison':
        value=resume.read_json(out/'residual-comparison.json');value['upper_bound_decreased']=False;write(out/'residual-comparison.json',value)
    plan['parent_receipt_sha256']=seal(attempt)
    if fault in ('failed-receipt','timeout'):
        receipt=resume.read_json(attempt/'receipt.json')
        if fault=='failed-receipt':receipt['exit_code']=1
        else:receipt['timed_out']=True
        write(attempt/'receipt.json',receipt);plan['parent_receipt_sha256']=base.digest(attempt/'receipt.json')
    with pytest.raises(ValueError): load(plan,assets,binding)


def test_reconstructed_image_radius_can_widen_never_shrink():
    original={'mid':[3,-1],'rad':[(1<<29)+1,-300]}
    restored=resume.decoded_vector([original],1)[0]; record=base.packed(restored)
    assert resume.dyadic(record['rad'])>=resume.dyadic(original['rad'])
    assert restored.mid()==arb(3)/2
    with pytest.raises(ValueError):resume.decoded_vector([original],1,exact=True)
    with pytest.raises(ValueError):resume.decoded_vector([{'mid':[1.0,0],'rad':[0,0]}],1)


def test_all_three_nonimproving_corrections_retained(tmp_path):
    centers=[[arb(1)+arb((1,-1700))]*2 for _ in range(2)]; images=[list(row) for row in centers]
    model=TinyMap();iterations=[]
    got,_=resume.run_corrections(model,centers,images,[arb_mat(2,2)]*2,arb_mat([[1,0],[0,1]]),tmp_path,3,
        lambda *a,**k:None,iterations)
    assert model.calls==6 and len(iterations)==3
    assert all(v['status']=='completed' and not v['comparison']['upper_bound_decreased'] for v in iterations)
    assert all(v==arb(1)+arb((1,-1700)) for row in got for v in row)
    assert len(list(tmp_path.glob('correction-*/*-point.json')))==6


def test_later_failure_retains_prior_complete_iteration_and_partial_input(tmp_path):
    centers=[[arb(2)]*2 for _ in range(2)];model=TinyMap(fail_at=4);iterations=[]
    with pytest.raises(ValueError):resume.run_corrections(model,centers,centers,[arb_mat(2,2)]*2,
        arb_mat([[1,0],[0,1]]),tmp_path,3,lambda *a,**k:None,iterations)
    assert iterations[0]['status']=='completed' and iterations[1]['status']=='running'
    assert (tmp_path/'correction-02/after-phase-00-point.json').is_file()
    assert (tmp_path/'correction-02/after-phase-01-input.json').is_file()
    assert not (tmp_path/'correction-02/after-phase-01-point.json').exists()


@pytest.mark.parametrize('wrong_images',[False,True])
def test_complete_resume_can_be_parent_with_full_lineage(tmp_path, monkeypatch,wrong_images):
    owned, plan, assets, binding, derivatives, inverse=prepare_base(tmp_path,monkeypatch)
    inherited=load(plan,assets,binding);attempt=owned/'resume-01';out=attempt/'run';out.mkdir(parents=True)
    iterations=[];resume.run_corrections(TinyMap(),inherited['centers'],inherited['images'],derivatives,inverse,out,3,
        lambda *a,**k:None,iterations)
    rows=[dict(correction_index=v['correction_index'],**row) for v in iterations for row in v['phase_outcomes']]
    write(out/'status.json',dict(status='proposal_resume_completed',certificate_passed=False,corrections=3,
        iterations=iterations,phase_outcomes=rows))
    write(out/'reuse-manifest.json',dict(parent_attempt=plan['parent_attempt'],parent_receipt_sha256=plan['parent_receipt_sha256'],
        phases=inherited['reuse'],lineage=inherited['lineage'],exact_centers=resume.center_records(inherited['centers']),
        cumulative_prior_phase_evaluations=inherited['cumulative_phase_evaluations'],
        cumulative_prior_elapsed_seconds=inherited['cumulative_elapsed_seconds'],residual_recomputed=True,no_image_radius_discarded=True))
    hashes=source_hashes(attempt,'center_resume');shared_metadata(attempt,plan,hashes,'center_resume',assets,binding)
    next_plan=dict(schema=resume.SCHEMA,base_plan=plan['base_plan'],parent_attempt=str(attempt),parent_kind='center_resume',
        parent_source_sha256=hashes,parent_receipt_sha256=seal(attempt),max_corrections=1,envelope=plan['envelope'])
    if wrong_images:
        path=out/'correction-01/correction-witnesses.json';value=resume.read_json(path)
        value['images'][0][0]=point(0);write(path,value);next_plan['parent_receipt_sha256']=seal(attempt)
        with pytest.raises(ValueError,match='source phase images'):load(next_plan,assets,binding)
        return
    got=load(next_plan,assets,binding)
    assert got['cumulative_phase_evaluations']==10 and got['cumulative_elapsed_seconds']==3.
    assert len(got['lineage'])==2 and got['lineage'][0]['attempt']==plan['parent_attempt']
    assert len(got['reuse'])==2 and got['reuse'][0]['point_path'].startswith('run/correction-03/')


def test_saved_residual_narrowing_is_refused():
    centers=[[arb(0)],[arb(0)]];image={'mid':[1,0],'rad':[1,-50]}
    record=dict(residuals=[[point(1)],[point(1)]],inf_upper=point(2),inf_lower=point(0))
    with pytest.raises(ValueError):resume._residual_record(record,centers,[[image],[image]])


@pytest.mark.parametrize('failure',[None,4])
def test_whole_resume_driver_tiny_mock_retains_all_work_and_never_changes_parent(tmp_path,monkeypatch,failure):
    import torch
    sys.path.insert(0,str(ROOT/'codex/crosscheck'))
    import qwen_map
    owned,plan,assets,binding,_,_=prepare_base(tmp_path,monkeypatch)
    parent=Path(plan['parent_attempt']);before={str(p):base.digest(p) for p in parent.rglob('*') if p.is_file()}
    source=tmp_path/'source';source.mkdir();plan_path=source/'plan.json';write(plan_path,plan)
    out=owned/'new-resume'/'run'
    arrays=dict(A=np.zeros((2,2,2)),B=np.eye(2))
    model=SimpleNamespace(config={'synthetic':True},weights={'w':torch.zeros(1,dtype=torch.float64)},
        vector=torch.zeros(2,dtype=torch.float64),steering_layer=12)
    calls=[]
    class FakeBall(TinyMap):
        def __init__(self,*args,**kwargs):super().__init__(fail_at=failure)
        def forward(self,center):calls.append(list(center));return super().forward(center)
    monkeypatch.setattr(resume,'HERE',source)
    monkeypatch.setattr(base,'load_bound_inputs',lambda *a:(arrays,binding))
    monkeypatch.setattr(base,'BallQwen',FakeBall)
    monkeypatch.setattr(base,'load_prefix',lambda *a,**k:resume.read_json(parent/'run/prefix-binding.json'))
    monkeypatch.setattr(qwen_map,'PINNED_SHA256',assets)
    monkeypatch.setattr(qwen_map.ManualQwenSoftMap,'from_pinned_files',lambda: model)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    monkeypatch.setenv('CERT_SOURCE_COMMIT','a'*40)
    monkeypatch.setattr(sys,'argv',['center_resume.py','--output',str(out),'--plan',str(plan_path)])
    if failure is None:resume.main()
    else:
        with pytest.raises(ValueError,match='synthetic interrupted'):resume.main()
    status=resume.read_json(out/'status.json')
    assert status['certificate_passed'] is False
    assert {str(p):base.digest(p) for p in parent.rglob('*') if p.is_file()}==before
    assert all(v==arb(1)+arb((1,-1700)) and v.is_exact() for row in calls for v in row)
    if failure is None:
        assert len(calls)==6 and status['status']=='proposal_resume_completed'
        assert status['new_phase_evaluations']==6 and status['cumulative_phase_evaluations']==10
        assert len(status['iterations'])==3 and all(not r['comparison']['upper_bound_decreased'] for r in status['iterations'])
    else:
        assert len(calls)==4 and status['status']=='proposal_resume_failed'
        assert status['iterations'][0]['status']=='completed'
        assert status['iterations'][1]['status']=='failed'
        assert status['iterations'][1]['phase_outcomes'][0]['status']=='completed'
        assert status['iterations'][1]['phase_outcomes'][1]['status']=='failed'
        assert (out/'correction-02/after-phase-01-input.json').is_file()
        assert not (out/'correction-02/after-phase-01-point.json').exists()


def test_parent_output_path_refused_before_any_write(tmp_path,monkeypatch):
    _,plan,_,_,_,_=prepare_base(tmp_path,monkeypatch)
    parent=Path(plan['parent_attempt']);before={str(p):base.digest(p) for p in parent.rglob('*') if p.is_file()}
    source=tmp_path/'source';source.mkdir();path=source/'plan.json';write(path,plan)
    monkeypatch.setattr(resume,'HERE',source)
    monkeypatch.setattr(sys,'argv',['center_resume.py','--output',str(parent/'new-output'),'--plan',str(path)])
    with pytest.raises(ValueError):resume.main()
    assert {str(p):base.digest(p) for p in parent.rglob('*') if p.is_file()}==before


def test_wrong_reconstruction_context_refused(tmp_path,monkeypatch):
    _,plan,assets,binding,_,_=prepare_base(tmp_path,monkeypatch)
    with ctx.workprec(1024):
        with pytest.raises(ValueError,match='2048-bit'):load(plan,assets,binding)


def test_all_planned_corrections_are_a_bounded_integer(tmp_path):
    for count in (0,4,True,1.0):
        with pytest.raises(ValueError):resume.run_corrections(None,[],[],[],None,tmp_path,count,lambda *a,**k:None,[])


def test_isolated_launcher_import_and_help_need_no_model():
    import subprocess
    result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'codex/certify-llm/center_resume.py'),'--help'],
        capture_output=True,text=True,check=False,timeout=15)
    assert result.returncode==0, result.stderr
    assert '--plan' in result.stdout and '--output' in result.stdout


@pytest.mark.parametrize('field,value',[('cpu_threads',True),('cpu_threads',1.0),('nice',10.0),
    ('memory_gib',18.0),('timeout_seconds',3600.0)])
def test_resource_envelope_rejects_boolean_and_float_aliases(tmp_path,monkeypatch,field,value):
    _,plan,_,_,_,_=prepare_base(tmp_path,monkeypatch)
    plan['envelope']=dict(plan['envelope']);plan['envelope'][field]=value
    with pytest.raises(ValueError):resume.validate_plan(plan)
