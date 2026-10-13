"""Bounded engineering-driver controls with tiny synthetic maps only."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from flint import arb,arb_mat,ctx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
import box_derivative_pilot as driver


@pytest.fixture(autouse=True)
def precision():
    with ctx.workprec(1024):yield


def make_plan(tmp_path):
    owned=tmp_path/'owned';owned.mkdir()
    for name in ('arb-pilot-002','signed-pilot-001'):(owned/name).mkdir()
    plan=dict(schema='codex-llm-box-derivative-pilot-v1',prefix_attempt=str(owned/'arb-pilot-002'),
        comparison_attempt=str(owned/'signed-pilot-001'),precision=1024,alpha=.071674,log10_temperature=-.115,
        period=8,dimension=896,vocabulary=151936,phase_index=0,nominal_radius_power=-800,
        direction_indices=list(range(32)),envelope=dict(cpu_threads=1,nice=10,memory_gib=18,timeout_seconds=3600),**driver.PINS)
    return owned,plan


def test_exact_scope_and_immutable_pins_admitted(tmp_path):
    owned,plan=make_plan(tmp_path)
    assert driver.validate_plan(plan,owned=owned)==(owned/'arb-pilot-002',owned/'signed-pilot-001')


@pytest.mark.parametrize('key,value',[
    ('schema','other'),('precision',2048),('precision',True),('phase_index',1),('phase_index',False),
    ('alpha',.1),('alpha',0),('log10_temperature',-.116),('period',4),('dimension',895),
    ('vocabulary',32000),('nominal_radius_power',-80),('nominal_radius_power',-800.),
    ('direction_indices',list(range(8))),('direction_indices',[True]+list(range(1,32))),
    ('prefix_receipt_sha256','0'*64),('comparison_step_sha256','0'*64),
    ('envelope',dict(cpu_threads=2,nice=10,memory_gib=18,timeout_seconds=3600)),
    ('envelope',dict(cpu_threads=True,nice=10,memory_gib=18,timeout_seconds=3600)),
])
def test_wrong_scope_and_pin_refused(tmp_path,key,value):
    owned,plan=make_plan(tmp_path);plan[key]=value
    with pytest.raises(ValueError):driver.validate_plan(plan,owned=owned)


def test_unknown_missing_and_unowned_plan_fields_refused(tmp_path):
    owned,plan=make_plan(tmp_path)
    variants=[]
    v=copy.deepcopy(plan);v['do_full_jacobian']=True;variants.append(v)
    v=copy.deepcopy(plan);v.pop('proposal_sha256');variants.append(v)
    v=copy.deepcopy(plan);v['prefix_attempt']=str(tmp_path);variants.append(v)
    for bad in variants:
        with pytest.raises(ValueError):driver.validate_plan(bad,owned=owned)


def fixture_attempt(tmp_path):
    attempt=tmp_path/'attempt';(attempt/'source').mkdir(parents=True)
    source=attempt/'source/test.py';source.write_text('# synthetic source\n')
    data=attempt/'data.json';data.write_text('{}\n')
    manifest=dict(source_commit='a'*40,source=[dict(path='test.py',bytes=source.stat().st_size,sha256=driver.sha(source))])
    driver.write_json(attempt/'manifest.json',manifest)
    receipt=dict(source_commit='a'*40,exit_code=1,timed_out=False,artifacts=[])
    for path in (source,data,attempt/'manifest.json'):
        receipt['artifacts'].append(dict(path=str(path.relative_to(attempt)),bytes=path.stat().st_size,sha256=driver.sha(path)))
    driver.write_json(attempt/'receipt.json',receipt)
    return attempt,receipt


def test_all_artifact_and_source_authentication_keeps_failure_classification_separate(tmp_path):
    attempt,receipt=fixture_attempt(tmp_path)
    got,files=driver.authenticate_attempt(attempt,driver.sha(attempt/'receipt.json'))
    assert got==receipt and got['exit_code']==1 and len(files)==3
    (attempt/'data.json').write_text('changed')
    with pytest.raises(ValueError):driver.authenticate_attempt(attempt,driver.sha(attempt/'receipt.json'))


@pytest.mark.parametrize('fault',['hash','duplicate','escape','extra','source','source_inventory'])
def test_inventory_and_source_tamper_refused(tmp_path,fault):
    attempt,receipt=fixture_attempt(tmp_path)
    if fault=='duplicate':receipt['artifacts'].append(receipt['artifacts'][0])
    if fault=='escape':receipt['artifacts'][0]['path']='../outside'
    if fault=='extra':(attempt/'unaccounted.txt').write_text('extra')
    if fault in ('source','source_inventory'):
        manifest=json.loads((attempt/'manifest.json').read_text())
        if fault=='source':manifest['source_commit']='b'*40
        else:manifest['source']=[]
        driver.write_json(attempt/'manifest.json',manifest)
        row=next(r for r in receipt['artifacts'] if r['path']=='manifest.json')
        row.update(bytes=(attempt/'manifest.json').stat().st_size,sha256=driver.sha(attempt/'manifest.json'))
    driver.write_json(attempt/'receipt.json',receipt)
    expected='0'*64 if fault=='hash' else driver.sha(attempt/'receipt.json')
    with pytest.raises(ValueError):driver.authenticate_attempt(attempt,expected)


def saved_step(n=3,k=32,value=None,column=None):
    value=driver.pack_ball(arb(0)) if value is None else value
    column=driver.pack_ball(arb(1)) if column is None else column
    return dict(value=[copy.deepcopy(value) for _ in range(n)],
                columns=[[copy.deepcopy(column) for _ in range(n)] for _ in range(k)],direction_indices=list(range(k)))


def test_all_eight_overlap_rows_and_all_32_radius_rows_checked():
    current=saved_step(column=driver.pack_ball(arb(0,1)))
    old=saved_step(k=8,column=driver.pack_ball(arb(1,1)))
    result=driver.compare_saved_steps(current,old,dimension=3)
    assert result['comparison_passed']
    assert len(result['columns'])==32 and sum(v['compared_to_point'] for v in result['columns'])==8
    assert all(v['overlap_count']==3 for v in result['columns'][:8])
    assert all(v['finite_entries']==3 and 'maximum_radius' in v for v in result['columns'])
    # Intervals overlap but neither contains the other: containment is not required.
    assert not driver.unpack(current['columns'][0][0]).contains(driver.unpack(old['columns'][0][0]))
    assert not driver.unpack(old['columns'][0][0]).contains(driver.unpack(current['columns'][0][0]))


def test_failure_in_eighth_comparison_row_is_not_hidden():
    current=saved_step();old=saved_step(k=8)
    old['columns'][7][2]=driver.pack_ball(arb(9))
    result=driver.compare_saved_steps(current,old,dimension=3)
    assert not result['comparison_passed']
    assert result['columns'][7]['nonoverlap_coordinates']==[2]
    assert all(v['overlap_passed'] for v in result['columns'][:7])


def test_saved_exact_endpoints_prevent_decode_widening_false_overlap():
    previous={'mid':[0,0],'rad':[566971103,-502]}
    fresh={'mid':[2*566971103+1,-503],'rad':[0,0]}
    assert driver.unpack(previous).contains(driver.unpack(fresh))  # one radius-ULP widening
    current=saved_step(n=1);old=saved_step(n=1,k=8)
    current['columns'][0][0]=fresh;old['columns'][0][0]=previous
    result=driver.compare_saved_steps(current,old,dimension=1)
    assert not result['comparison_passed'] and result['columns'][0]['overlap_count']==0


def test_primal_nonoverlap_is_retained_and_fails():
    current=saved_step();old=saved_step(k=8);old['value'][0]=driver.pack_ball(arb(5))
    result=driver.compare_saved_steps(current,old,dimension=3)
    assert not result['comparison_passed'] and result['primal_nonoverlap_coordinates']==[0]
    assert len(result['columns'])==32


@pytest.mark.parametrize('fault',['value_shape','column_shape','omitted_column','bad_direction','float_record','negative_radius','last_column'])
def test_saved_full_outputs_fail_loudly_when_invalid(fault):
    step=saved_step()
    if fault=='value_shape':step['value'].pop()
    if fault=='column_shape':step['columns'][10].pop()
    if fault=='omitted_column':step['columns'].pop()
    if fault=='bad_direction':step['direction_indices'][0]=True
    if fault=='float_record':step['columns'][0][0]={'mid':[1.,0],'rad':[0,0]}
    if fault=='negative_radius':step['columns'][0][0]={'mid':[1,0],'rad':[-1,-100]}
    if fault=='last_column':step['columns'][31][2]=float('nan')
    with pytest.raises(ValueError):driver.validate_saved_step(step,dimension=3,count=32)


def test_capture_requires_full_finite_matrix_and_primal():
    step=driver.capture_step([arb(1),arb(2)],arb_mat(2,32),dimension=2,count=32)
    assert len(step['columns'])==32
    for values,matrix in [([arb('nan'),arb(2)],arb_mat(2,32)),([arb(1)],arb_mat(2,32)),
                          ([arb(1),arb(2)],arb_mat(2,31)),([arb(1),arb(2)],np.zeros((2,32)))]:
        with pytest.raises(ValueError):driver.capture_step(values,matrix,dimension=2,count=32)
    bad=arb_mat(2,32);bad[1,31]=arb('inf')
    with pytest.raises(ValueError):driver.capture_step([arb(1),arb(2)],bad,dimension=2,count=32)


def test_domain_contains_actual_nominal_input_after_extra_reconstruction():
    artifact,state,checked=driver.domain_inputs(np.array([.1,-2.],dtype=np.float64))
    assert artifact['period']==1 and artifact['dimension']==2
    assert checked['full_nominal_domain_contained']
    assert checked['actual_balls_sha256']==driver.canonical_hash([[driver.pack_ball(v) for v in state]])
    for i,value in enumerate(state):
        assert value.contains(driver.exact([.1,-2.][i])) and value.rad()>=arb((1,-800))
        assert not value.is_exact()
    assert artifact['artifact_sha256']==driver.canonical_hash({k:v for k,v in artifact.items() if k!='artifact_sha256'})


@pytest.mark.parametrize('comparison_failure',[False,True])
def test_mock_orchestration_retains_all_outputs_before_comparison(tmp_path,monkeypatch,comparison_failure):
    import torch
    sys.path.insert(0,str(ROOT/'codex/crosscheck'))
    import qwen_map
    owned,plan=make_plan(tmp_path);source=tmp_path/'source';source.mkdir()
    plan_path=source/'box-plan.json';driver.write_json(plan_path,plan)
    out=owned/'new-attempt/run';calls=[]
    monkeypatch.setattr(driver,'OWNED',owned);monkeypatch.setattr(driver,'HERE',source);monkeypatch.setattr(driver,'DIMENSION',32)
    model=SimpleNamespace(d=32,config={'vocab_size':151936},weights={'fake':torch.zeros(1,dtype=torch.float64)},
                          vector=torch.zeros(32,dtype=torch.float64),steering_layer=12)
    monkeypatch.setattr(qwen_map.ManualQwenSoftMap,'from_pinned_files',lambda:model)
    monkeypatch.setattr(driver,'load_inputs',lambda *a:(np.zeros((9,32)),saved_step(n=32,k=8,column=driver.pack_ball(arb(0))),{'synthetic':True}))
    monkeypatch.setattr(driver,'load_prefix',lambda *a,**k:{'synthetic':True})
    class TinyJet:
        def __init__(self,*a,**kw):assert kw['direction_count']==32
        def forward(self,state,directions):
            calls.append(1)
            assert len(state)==32 and all(v.rad()>=arb((1,-800)) for v in state)
            assert directions.nrows()==directions.ncols()==32
            for i in range(32):
                for j in range(32):assert directions[i,j]==int(i==j)
            return [arb(0)]*32,arb_mat(32,32)
    monkeypatch.setattr(driver,'BallQwenJet',TinyJet)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None);monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    real_compare=driver.compare_saved_steps
    def checked_compare(*a,**kw):
        assert (out/'boxed-step.json').is_file()
        saved=json.loads((out/'boxed-step.json').read_text());assert len(saved['columns'])==32 and len(saved['value'])==32
        result=real_compare(*a,**kw)
        if comparison_failure:result['comparison_passed']=False
        return result
    monkeypatch.setattr(driver,'compare_saved_steps',checked_compare)
    monkeypatch.setenv('CERT_SOURCE_COMMIT','a'*40)
    monkeypatch.setattr(sys,'argv',['box_derivative_pilot.py','--output',str(out),'--plan',str(plan_path)])
    if comparison_failure:
        with pytest.raises(ValueError):driver.main()
    else:driver.main()
    status=json.loads((out/'status.json').read_text())
    assert status['certificate_passed'] is False and calls==[1]
    assert status['status']==('boxed_derivative_setup_failed' if comparison_failure else 'boxed_derivative_setup_completed')
    assert all((out/name).is_file() for name in ['phase-domain.json','domain-origin.json','actual-inputs.json','direction-inputs.json','boxed-step.json','comparison.json'])
