"""Refinement admission/retention controls with synthetic data only."""
from pathlib import Path
import copy
import json
import sys
import numpy as np
import pytest
from flint import arb, ctx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'codex/certify-llm'))
import center_refine as driver


@pytest.fixture(autouse=True)
def precision():
    with ctx.workprec(256):
        yield


def plan_fixture(tmp_path):
    owned=tmp_path/'owned';owned.mkdir()
    for name in ('prefix','seed'):(owned/name).mkdir()
    plan=dict(schema='codex-llm-center-refine-v1',alpha=.071674,log10_temperature=-.115,
              period=8,dimension=896,precision=2048,max_corrections=1,evaluation_passes=2,
              envelope=dict(cpu_threads=1,nice=10,memory_gib=18,timeout_seconds=3600),
              prefix_attempt=str(owned/'prefix'),center_seed_attempt=str(owned/'seed'))
    plan.update({key:'a'*64 for key in driver.HASH_KEYS})
    return owned,plan


def test_exact_planned_scope_admitted(tmp_path):
    owned,plan=plan_fixture(tmp_path)
    assert driver.validate_plan(plan,owned=owned)==(owned/'prefix',owned/'seed')


@pytest.mark.parametrize('key,value',[('schema','wrong'),('alpha',.1),('log10_temperature',0.),
    ('period',4),('dimension',8),('precision',1024),('max_corrections',2),('evaluation_passes',1),
    ('max_corrections',True),('prefix_receipt_sha256','a'*63),('center_seed_driver_sha256','A'*64),
    ('envelope',dict(cpu_threads=4,nice=10,memory_gib=18,timeout_seconds=3600))])
def test_wrong_scope_and_hashes_refused(tmp_path,key,value):
    owned,plan=plan_fixture(tmp_path);plan[key]=value
    with pytest.raises(ValueError):driver.validate_plan(plan,owned=owned)


def test_missing_unknown_and_unowned_inputs_refused(tmp_path):
    owned,plan=plan_fixture(tmp_path)
    variants=[]
    missing=copy.deepcopy(plan);missing.pop('preconditioner_sha256');variants.append(missing)
    unknown=copy.deepcopy(plan);unknown['quietly_do_two_corrections']=True;variants.append(unknown)
    external=copy.deepcopy(plan);external['prefix_attempt']=str(tmp_path);variants.append(external)
    same=copy.deepcopy(plan);same['prefix_attempt']=same['center_seed_attempt'];variants.append(same)
    for bad in variants:
        with pytest.raises(ValueError):driver.validate_plan(bad,owned=owned)


def arrays_fixture():
    return dict(A=np.zeros((2,2,2)),B=np.eye(2),centers=np.zeros((2,2)),
                parameters=np.array([.071674,-.115]),product=np.zeros((2,2)),
                left_inverse_defect=np.zeros((2,2)),right_inverse_defect=np.zeros((2,2)))


def test_exact_preconditioner_inventory_and_dyadic_admission():
    arrays=arrays_fixture()
    assert driver.validate_preconditioner_arrays(arrays,period=2,dimension=2) is arrays
    value=np.array([[.1,1.+2**-52],[-.25,0.]])
    matrix=driver.exact_matrix(value)
    for i in range(2):
        for j in range(2):
            assert matrix[i,j].is_exact() and matrix[i,j]==driver.exact(value[i,j])


@pytest.mark.parametrize('fault',['missing','extra','float32','shape','nan','parameter'])
def test_bad_preconditioner_refused(fault):
    arrays=arrays_fixture()
    if fault=='missing':arrays.pop('product')
    if fault=='extra':arrays['extra']=np.ones(2)
    if fault=='float32':arrays['A']=arrays['A'].astype(np.float32)
    if fault=='shape':arrays['B']=np.ones((2,1))
    if fault=='nan':arrays['right_inverse_defect'][0,0]=np.nan
    if fault=='parameter':arrays['parameters'][0]=.2
    with pytest.raises(ValueError):driver.validate_preconditioner_arrays(arrays,period=2,dimension=2)


def receipt_fixture(tmp_path):
    attempt=tmp_path/'attempt';attempt.mkdir();file=attempt/'data.json';file.write_text('{}')
    receipt=dict(source_commit='b'*40,exit_code=0,timed_out=False,
                 artifacts=[dict(path='data.json',bytes=file.stat().st_size,sha256=driver.digest(file))])
    driver.write_json(attempt/'receipt.json',receipt)
    return attempt,receipt


def test_complete_receipt_binding_and_mutation_refusal(tmp_path):
    attempt,receipt=receipt_fixture(tmp_path);sha=driver.digest(attempt/'receipt.json')
    got,files=driver.checked_attempt(attempt,sha)
    assert got==receipt and files=={'data.json':attempt/'data.json'}
    (attempt/'data.json').write_text('changed')
    with pytest.raises(ValueError):driver.checked_attempt(attempt,sha)


@pytest.mark.parametrize('fault',['hash','timeout','failure','source','duplicate','escape'])
def test_invalid_upstream_receipts_refused(tmp_path,fault):
    attempt,receipt=receipt_fixture(tmp_path)
    if fault=='timeout':receipt['timed_out']=True
    if fault=='failure':receipt['exit_code']=1
    if fault=='source':receipt['source_commit']='unknown'
    if fault=='duplicate':receipt['artifacts']*=2
    if fault=='escape':receipt['artifacts'][0]['path']='../outside'
    driver.write_json(attempt/'receipt.json',receipt)
    sha='0'*64 if fault=='hash' else driver.digest(attempt/'receipt.json')
    with pytest.raises(ValueError):driver.checked_attempt(attempt,sha)


def test_full_precision_phase_inputs_and_immediate_outputs(tmp_path):
    # 1+2^-180 would become 1 if the driver rounded the centre to binary64.
    centers=[[arb(1)+arb((1,-180))],[arb(-2)+arb((1,-160))]]
    class TinyMap:
        def forward(self,state):
            assert state[0].is_exact()
            return [state[0]/2],[arb(0)]
    events=[];outcomes=[]
    images=driver.evaluate_pass(TinyMap(),centers,tmp_path,'before',lambda *a,**k:events.append((a,k)),outcomes)
    assert images[0][0]!=arb(1)/2
    assert len(outcomes)==2 and all(v['status']=='completed' for v in outcomes)
    saved=json.loads((tmp_path/'before-phase-00-input.json').read_text())['center'][0]
    assert saved['rad']==[0,0] and arb(tuple(saved['mid']))==centers[0][0]
    assert len(events)==4 and all((tmp_path/v['point_path']).is_file() for v in outcomes)
    upper,lower,record=driver.residual_summary(centers,images)
    assert upper>=lower and len(record['residuals'])==2


def test_failure_retains_prior_phase_and_failed_phase_input(tmp_path):
    class FailingMap:
        calls=0
        def forward(self,state):
            self.calls+=1
            if self.calls==2:raise ValueError('synthetic positivity failure')
            return [arb(0)],[arb(0)]
    outcomes=[]
    with pytest.raises(ValueError):
        driver.evaluate_pass(FailingMap(),[[arb(1)],[arb(2)]],tmp_path,'after',lambda *a,**k:None,outcomes)
    assert outcomes[0]['status']=='completed' and outcomes[1]['status']=='running'
    assert (tmp_path/'after-phase-00-point.json').is_file()
    assert (tmp_path/'after-phase-01-input.json').is_file()
    assert not (tmp_path/'after-phase-01-point.json').exists()


def test_nonpoint_center_and_bad_image_refused(tmp_path):
    class InvalidMap:
        def forward(self,state):return [arb('nan')],[arb(0)]
    with pytest.raises(ValueError):driver.evaluate_pass(InvalidMap(),[[arb(1,(1,-60))]],tmp_path,'before',lambda *a,**k:None,[])
    with pytest.raises(ValueError):driver.evaluate_pass(InvalidMap(),[[arb(1)]],tmp_path,'before',lambda *a,**k:None,[])


def test_entire_driver_with_tiny_fake_map_runs_one_correction_and_16_phases(tmp_path,monkeypatch):
    """Exercise orchestration, never the checkpoint loader or target evaluator."""
    import torch
    sys.path.insert(0,str(ROOT/'codex/crosscheck'))
    import qwen_map
    from types import SimpleNamespace
    owned,plan=plan_fixture(tmp_path)
    source=tmp_path/'source';source.mkdir();plan_path=source/'synthetic-plan.json'
    driver.write_json(plan_path,plan)
    output=owned/'new-attempt'/'run'
    arrays=dict(A=np.zeros((8,2,2)),B=np.eye(2),centers=np.full((8,2),2.))
    model=SimpleNamespace(config={'synthetic':True},weights={'w':torch.zeros(1,dtype=torch.float64)},
                          vector=torch.zeros(2,dtype=torch.float64),steering_layer=12)
    loader_calls=[];forward_calls=[]
    def fake_loader(*args,**kwargs):
        loader_calls.append(1);return model
    class TinyBall:
        def __init__(self,config,weights,prefix_ids,vector,**kwargs):
            self.config=config;self.weights=weights;self.vector=vector
        def forward(self,center):
            forward_calls.append(list(center))
            return [arb(1)+arb((1,-200)) for _ in center],[arb(0) for _ in center]
    real_validate=driver.validate_plan
    monkeypatch.setattr(driver,'OWNED',owned);monkeypatch.setattr(driver,'HERE',source)
    monkeypatch.setattr(driver,'validate_plan',lambda p:real_validate(p,owned=owned))
    monkeypatch.setattr(driver,'load_bound_inputs',lambda *a: (arrays,{'synthetic_only':True}))
    monkeypatch.setattr(driver,'BallQwen',TinyBall)
    monkeypatch.setattr(driver,'load_prefix',lambda *a,**k:{'synthetic_only':True})
    monkeypatch.setattr(qwen_map.ManualQwenSoftMap,'from_pinned_files',fake_loader)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    monkeypatch.setenv('CERT_SOURCE_COMMIT','a'*40)
    monkeypatch.setattr(sys,'argv',['center_refine.py','--output',str(output),'--plan',str(plan_path)])
    driver.main()
    status=json.loads((output/'status.json').read_text())
    assert status['status']=='proposal_setup_completed' and status['certificate_passed'] is False
    assert status['corrections']==1 and len(status['phase_outcomes'])==16
    assert len(loader_calls)==1 and len(forward_calls)==16
    assert all(v==arb(2) for row in forward_calls[:8] for v in row)
    assert all(v==arb(1)+arb((1,-200)) for row in forward_calls[8:] for v in row)
    assert status['comparison']['upper_bound_decreased'] is True
    assert status['comparison']['strict_residual_norm_decrease_established'] is True
    assert (output/'correction-witnesses.json').is_file()
    assert len(list(output.glob('*-phase-*-point.json')))==16
