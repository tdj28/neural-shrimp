"""Synthetic full-admission lineage and mock driver tests; no target calls."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import flint
from flint import arb, arb_mat, ctx
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
import streamed_pilot as driver
base=driver.base


@pytest.fixture(autouse=True)
def precision():
    old=ctx.threads;ctx.threads=1
    with ctx.workprec(1024):yield
    ctx.threads=old


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);base.write_json(path,value)


def seal(attempt):
    sources=[dict(path=str(p.relative_to(attempt/'source')),bytes=p.stat().st_size,sha256=base.sha(p))
             for p in sorted((attempt/'source').rglob('*')) if p.is_file()]
    write(attempt/'manifest.json',dict(source_commit='a'*40,source=sources))
    artifacts=[dict(path=str(p.relative_to(attempt)),bytes=p.stat().st_size,sha256=base.sha(p))
               for p in sorted(attempt.rglob('*')) if p.is_file() and p.name!='receipt.json']
    write(attempt/'receipt.json',dict(source_commit='a'*40,exit_code=0,timed_out=False,elapsed_seconds=2.,artifacts=artifacts))
    return base.sha(attempt/'receipt.json')


def saved_step(n,count):
    return base.capture_step([arb(0)]*n,arb_mat(n,count),dimension=n,count=count)


def fixture(tmp_path,monkeypatch):
    import torch
    owned=tmp_path/'owned';owned.mkdir()
    for name in ('arb-pilot-002','signed-pilot-001','box-derivative-001'):(owned/name).mkdir()
    prefix=owned/'arb-pilot-002/run/prefix-cache.json.gz';prefix.parent.mkdir();prefix.write_bytes(b'synthetic cache')
    bp=dict(schema='codex-llm-box-derivative-pilot-v1',prefix_attempt=str(owned/'arb-pilot-002'),
        comparison_attempt=str(owned/'signed-pilot-001'),precision=1024,alpha=.071674,log10_temperature=-.115,
        period=8,dimension=896,vocabulary=151936,phase_index=0,nominal_radius_power=-800,
        direction_indices=list(range(32)),envelope=dict(cpu_threads=1,nice=10,memory_gib=18,timeout_seconds=3600),**base.PINS)
    attempt=owned/'box-derivative-001';out=attempt/'run';out.mkdir()
    hashes={}
    for relative in driver.BASE_SOURCES:
        path=attempt/'source'/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/relative).read_bytes());hashes[relative]=base.sha(path)
    plan_path=attempt/'source/codex/certify-llm/box-plan.json';write(plan_path,bp)
    n=64;points=np.zeros((9,n));prior8=saved_step(n,8)
    original=dict(prefix_artifacts_checked=1,comparison_artifacts_checked=1,prefix_source_commit='b'*40,comparison_source_commit='c'*40)
    assets={'synthetic':'d'*64};prefix_ids=(1,2)
    monkeypatch.setattr(driver,'OWNED',owned);monkeypatch.setattr(driver,'DIMENSION',n)
    monkeypatch.setattr(base,'load_inputs',lambda *a:(points,prior8,original))
    inputs=dict(plan=bp,plan_sha256=base.sha(plan_path),source_commit='a'*40,
        driver_sha256=hashes['codex/certify-llm/box_derivative_pilot.py'],pinned_assets=assets,
        prefix_ids=prefix_ids,precision=1024,threads=1,certificate_passed=False,upstream_binding=original,
        versions=dict(python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,numpy=np.__version__,torch=str(torch.__version__)))
    write(out/'inputs.json',inputs)
    artifact,state,domain=base.domain_inputs(points[0]);write(out/'phase-domain.json',artifact);write(out/'actual-inputs.json',domain)
    write(out/'domain-origin.json',dict(proposal_sha256=base.PINS['proposal_sha256'],plan_sha256=inputs['plan_sha256'],
        driver_source_sha256=inputs['driver_sha256'],constructor_source_sha256=artifact['helper_source_sha256'],
        generated_artifact_sha256=artifact['artifact_sha256'],artifact_file_sha256=base.sha(out/'phase-domain.json'),
        source_orbit_period=8,source_phase_index=0,selected_phase_count=1))
    _,directions=driver.basis_record(n,32);write(out/'direction-inputs.json',directions)
    step=saved_step(n,32);step.update(seconds=1.,phase_index=0,actual_balls_sha256=domain['actual_balls_sha256'],
        actual_inputs_file_sha256=base.sha(out/'actual-inputs.json'),direction_inputs_file_sha256=base.sha(out/'direction-inputs.json'))
    write(out/'boxed-step.json',step)
    comparison=base.compare_saved_steps(step,prior8,dimension=n,count=32,compared=8)
    comparison.update(comparison_receipt_sha256=base.PINS['comparison_receipt_sha256'],comparison_step_sha256=base.PINS['comparison_step_sha256'])
    write(out/'comparison.json',comparison)
    write(out/'prefix-binding.json',dict(receipt_sha256=base.PINS['prefix_receipt_sha256'],graph_sha256=base.PINS['prefix_map_source_sha256'],
        source_commit=original['prefix_source_commit'],precision=1024,receiver_construction_checked=True,steering_layer=12,prefix_sha256=base.sha(prefix)))
    write(out/'status.json',dict(status='boxed_derivative_setup_completed',certificate_passed=False,
        columns_completed=32,omitted_columns=864,comparison_rows_checked=8,full_state_dimension=896,
        full_vocabulary=151936,phase_index=0,precision=1024,arb_seconds=1.,elapsed_seconds=2.))
    plan=dict(schema=driver.SCHEMA,baseline_plan=bp,baseline_attempt=str(attempt),baseline_source_sha256=hashes,
        baseline_receipt_sha256=seal(attempt),baseline_step_sha256=base.sha(out/'boxed-step.json'),
        baseline_actual_inputs_sha256=base.sha(out/'actual-inputs.json'),baseline_actual_balls_sha256=domain['actual_balls_sha256'],
        precision=1024,direction_indices=list(range(64)),token_chunk=1024,envelope=bp['envelope'])
    return owned,plan,assets,prefix_ids


def test_complete_baseline_all_rows_and_identical_actual_domain_admitted(tmp_path,monkeypatch):
    owned,plan,assets,ids=fixture(tmp_path,monkeypatch)
    points,step,binding,artifact,state,domain=driver.load_baseline(plan,assets,ids)
    assert points.shape==(9,64) and len(step['columns'])==32
    assert binding['baseline_rows_rechecked']==32 and binding['baseline_point_rows_rechecked']==8
    assert domain['actual_balls_sha256']==plan['baseline_actual_balls_sha256']
    assert all(v.rad()>=arb((1,-800)) for v in state)


@pytest.mark.parametrize('field,value',[('schema','other'),('precision',2048),('precision',True),
    ('token_chunk',64),('token_chunk',1024.),('direction_indices',list(range(32))),
    ('direction_indices',[False]+list(range(1,64))),('baseline_receipt_sha256','x'*64),
    ('baseline_step_sha256',None),('baseline_source_sha256',{}),
    ('envelope',dict(cpu_threads=True,nice=10,memory_gib=18,timeout_seconds=3600))])
def test_plan_admission_refuses_changed_scope_or_missing_external_pins(tmp_path,monkeypatch,field,value):
    _,plan,_,_=fixture(tmp_path,monkeypatch);plan[field]=value
    with pytest.raises(ValueError):driver.validate_plan(plan)


@pytest.mark.parametrize('fault',['status','timeout','failure','source','plan','backend','domain','domainhash','origin',
    'direction','last_column','last_comparison','prefix','extra_file','missing_phase_file','step_pin'])
def test_tampered_or_incomplete_baseline_refused_after_resealing(tmp_path,monkeypatch,fault):
    _,plan,assets,ids=fixture(tmp_path,monkeypatch);attempt=Path(plan['baseline_attempt']);out=attempt/'run'
    def change(name,edit):
        value=driver.read_json(out/name);edit(value);write(out/name,value)
    if fault=='status':change('status.json',lambda v:v.update(status='boxed_derivative_setup_failed'))
    elif fault=='source':
        path=attempt/'source/codex/certify-llm/ball_jet.py';path.write_text('different')
        plan['baseline_source_sha256']['codex/certify-llm/ball_jet.py']=base.sha(path)
    elif fault=='plan':change('inputs.json',lambda v:v['plan'].update(alpha=.1))
    elif fault=='backend':change('inputs.json',lambda v:v['versions'].update(flint='different'))
    elif fault=='domain':change('actual-inputs.json',lambda v:v['actual_balls'][0][0].update(rad=[1,-799]))
    elif fault=='domainhash':plan['baseline_actual_balls_sha256']='a'*64
    elif fault=='origin':change('domain-origin.json',lambda v:v.update(source_phase_index=1))
    elif fault=='direction':change('direction-inputs.json',lambda v:v['columns'][31][63].update(mid=[1,0]))
    elif fault=='last_column':change('boxed-step.json',lambda v:v['columns'][31].pop())
    elif fault=='last_comparison':change('comparison.json',lambda v:v['columns'][31].update(finite_entries=1))
    elif fault=='prefix':change('prefix-binding.json',lambda v:v.update(receiver_construction_checked=False))
    elif fault=='missing_phase_file':(out/'boxed-step.json').unlink()
    plan['baseline_receipt_sha256']=seal(attempt)
    if fault=='extra_file':(attempt/'extra.txt').write_text('unaccounted')
    if fault in ('timeout','failure'):
        receipt=driver.read_json(attempt/'receipt.json')
        receipt.update(timed_out=fault=='timeout',exit_code=1 if fault=='failure' else 0)
        write(attempt/'receipt.json',receipt);plan['baseline_receipt_sha256']=base.sha(attempt/'receipt.json')
    if fault in ('domain',):plan['baseline_actual_inputs_sha256']=base.sha(out/'actual-inputs.json')
    if fault in ('last_column',):plan['baseline_step_sha256']=base.sha(out/'boxed-step.json')
    if fault=='step_pin':plan['baseline_step_sha256']='a'*64
    with pytest.raises((ValueError,KeyError)):driver.load_baseline(plan,assets,ids)


def test_all32_overlap_rows_and_all64_finite_rows_checked(monkeypatch):
    monkeypatch.setattr(driver,'DIMENSION',2)
    current=saved_step(2,64);prior=saved_step(2,32)
    result=driver.compare_steps(current,prior)
    assert result['comparison_passed'] and result['compared_columns']==32 and len(result['columns'])==64
    assert sum(v['compared_to_baseline_box'] for v in result['columns'])==32
    prior['columns'][31][1]=base.pack_ball(arb(1))
    failed=driver.compare_steps(current,prior)
    assert not failed['comparison_passed'] and failed['columns'][31]['nonoverlap_coordinates']==[1]


@pytest.mark.parametrize('failure',[None,'comparison','forward'])
def test_mock_driver_retains64_columns_before_comparison_and_preserves_baseline(tmp_path,monkeypatch,failure):
    import torch
    sys.path.insert(0,str(ROOT/'codex/crosscheck'));import qwen_map
    owned,plan,assets,ids=fixture(tmp_path,monkeypatch)
    parent=Path(plan['baseline_attempt']);oldhash={str(p):base.sha(p) for p in parent.rglob('*') if p.is_file()}
    source=tmp_path/'source';source.mkdir();planfile=source/'plan.json';write(planfile,plan)
    (source/'streamed_head.py').write_bytes((ROOT/'codex/certify-llm/streamed_head.py').read_bytes())
    out=owned/'streamed-001/run'
    monkeypatch.setattr(driver,'HERE',source);monkeypatch.setattr(qwen_map,'PINNED_SHA256',assets);monkeypatch.setattr(qwen_map,'PREFIX_IDS',ids)
    model=SimpleNamespace(d=64,config={'vocab_size':151936},weights={'w':torch.zeros(1,dtype=torch.float64)},vector=torch.zeros(64,dtype=torch.float64),steering_layer=12)
    monkeypatch.setattr(qwen_map.ManualQwenSoftMap,'from_pinned_files',lambda:model)
    monkeypatch.setattr(base,'load_prefix',lambda *a,**k:{'synthetic':True})
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None);monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    calls=[]
    class Tiny:
        def __init__(self,*args,**kwargs):
            assert kwargs['direction_count']==64 and kwargs['token_chunk']==1024
            self.progress=kwargs['progress'];self.head_progress=kwargs['head_progress']
            self.last_head_diagnostics=dict(vocabulary_count=151936,competitor_count=151935,certificate_passed=False,strict_margin_established=False)
        def forward(self,state,directions):
            calls.append(1);assert len(state)==64 and all(v.rad()>=arb((1,-800)) for v in state)
            assert all(directions[i,k]==int(i==k) for i in range(64) for k in range(64))
            self.progress('step_start',0,None);self.progress('step_end',23,None)
            self.head_progress('head_logits',0,1024)
            if failure=='forward':raise ValueError('synthetic forward failure')
            return [arb(0)]*64,arb_mat(64,64)
    monkeypatch.setattr(driver,'StreamedHeadQwenJet',Tiny)
    compare=driver.compare_steps
    def checked_compare(*args):
        retained=driver.read_json(out/'streamed-step.json');assert len(retained['columns'])==64 and len(retained['value'])==64
        result=compare(*args)
        if failure=='comparison':result['comparison_passed']=False
        return result
    monkeypatch.setattr(driver,'compare_steps',checked_compare)
    monkeypatch.setenv('CERT_SOURCE_COMMIT','a'*40);monkeypatch.setattr(sys,'argv',['streamed_pilot.py','--output',str(out),'--plan',str(planfile)])
    if failure:
        with pytest.raises(ValueError):driver.main()
    else:driver.main()
    status=driver.read_json(out/'status.json')
    assert status['certificate_passed'] is False and calls==[1]
    assert status['status']==('streamed_derivative_setup_completed' if failure is None else 'streamed_derivative_setup_failed')
    assert {str(p):base.sha(p) for p in parent.rglob('*') if p.is_file()}==oldhash
    assert (out/'actual-inputs.json').is_file() and (out/'direction-inputs.json').is_file()
    if failure!='forward':
        assert (out/'streamed-step.json').is_file() and (out/'comparison.json').is_file()
        assert (out/'head-diagnostics.json').is_file() and (out/'timing.json').is_file()
    else:assert not (out/'streamed-step.json').exists()


def test_output_inside_baseline_refused_before_writes(tmp_path,monkeypatch):
    owned,plan,_,_=fixture(tmp_path,monkeypatch);parent=Path(plan['baseline_attempt'])
    before={str(p):base.sha(p) for p in parent.rglob('*') if p.is_file()}
    source=tmp_path/'source';source.mkdir();planfile=source/'plan.json';write(planfile,plan)
    monkeypatch.setattr(driver,'HERE',source);monkeypatch.setattr(sys,'argv',['streamed_pilot.py','--output',str(parent/'run'),'--plan',str(planfile)])
    with pytest.raises(ValueError,match='immutable'):driver.main()
    assert {str(p):base.sha(p) for p in parent.rglob('*') if p.is_file()}==before


def test_isolated_help_requires_no_model():
    import subprocess
    result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'codex/certify-llm/streamed_pilot.py'),'--help'],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert '--plan' in result.stdout
