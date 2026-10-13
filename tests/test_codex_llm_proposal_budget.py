"""Synthetic formula/admission/retention tests; never load the target seed/model."""
from pathlib import Path
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'codex/certify-llm'))
import proposal_budget as driver


def toy_matrices(period=4):
    patterns = np.array([[[.125,.0625],[-.25,.125]],
                         [[.25,-.125],[.0625,.1875]],
                         [[-.125,.25],[.1875,.0625]],
                         [[.25,.125],[-.0625,-.125]]], dtype=np.float64)
    A = np.stack([patterns[i%4] for i in range(period)])
    product = np.eye(2)
    for matrix in A:
        product = matrix @ product
    return A, np.linalg.inv(np.eye(2)-product)


def explicit_cyclic(A):
    p,n,_ = A.shape
    H = np.zeros((p*n,p*n))
    for i in range(p):
        H[i*n:(i+1)*n,i*n:(i+1)*n] += A[i]
        j = (i+1)%p
        H[i*n:(i+1)*n,j*n:(j+1)*n] -= np.eye(n)
    return H


@pytest.mark.parametrize('period',[1,2,4,8])
def test_factored_estimates_cover_explicit_cyclic_inverse_blocks(period):
    A,B = toy_matrices(period)
    report = driver.estimate_budget(A,B)
    inverse = np.linalg.inv(explicit_cyclic(A))
    prefixes=[np.eye(2)]
    for matrix in A: prefixes.append(matrix @ prefixes[-1])
    suffixes=[]
    for j in range(period):
        product=np.eye(2)
        for k in range(j+1,period): product=A[k] @ product
        suffixes.append(product)
    D=np.eye(2)-B @ (np.eye(2)-prefixes[-1])
    assert report['phase_pair_count']==period*period
    assert len(report['phase_rows'])==period
    assert report['certificate_passed'] is False
    assert report['rigorous_enclosures'] is False
    for i,row in enumerate(report['phase_rows']):
        assert row['phase_index']==i and len(row['pairs'])==period
        assert row['inverse_defect_constant']==pytest.approx(np.linalg.norm(prefixes[i]@D,np.inf))
        for j,pair in enumerate(row['pairs']):
            segment_matrix=np.zeros((2,2))
            segment_norm=0.
            if j<i:
                segment_matrix=np.eye(2);segment_norm=1.
                for k in range(j+1,i):
                    segment_matrix=A[k]@segment_matrix
                    segment_norm*=np.linalg.norm(A[k],np.inf)
            block=-prefixes[i] @ B @ suffixes[j]-segment_matrix
            np.testing.assert_allclose(block,inverse[i*2:(i+1)*2,j*2:(j+1)*2],atol=2e-15,rtol=2e-15)
            expected=np.linalg.norm(prefixes[i],np.inf)*np.linalg.norm(B,np.inf)*np.linalg.norm(suffixes[j],np.inf)+segment_norm
            assert pair['row_phase']==i and pair['column_phase']==j
            assert pair['R_factor_estimate']==pytest.approx(expected)
            assert pair['R_factor_estimate']+2e-15>=np.linalg.norm(block,np.inf)
            assert pair['inverse_defect_constant']==(row['inverse_defect_constant'] if j==0 else 0.)
        assert row['epsilon_coefficient']==pytest.approx(sum(v['R_factor_estimate'] for v in row['pairs']))
        assert row['inverse_defect_constant']+row['epsilon_coefficient']*row['epsilon_threshold']==pytest.approx(1.)
    assert report['nominal_uniform_epsilon_threshold']==min(r['epsilon_threshold'] for r in report['phase_rows'])
    json.dumps(report,allow_nan=False)


def test_signed_nonnormal_cancellation_precedes_norms():
    matrix=np.array([[1.,1.],[-1.,-1.]])
    A=np.stack([matrix,matrix]);B=np.eye(2)
    report=driver.estimate_budget(A,B)
    assert np.any(np.abs(matrix) @ np.abs(matrix))
    assert report['cycle_product_norm_estimate']==0.
    assert report['inverse_defect_norm_estimate']==0.
    assert report['propagated_inverse_defect_estimates']==[0.,0.]
    assert [r['epsilon_coefficient'] for r in report['phase_rows']]==[3.,7.]
    assert report['nominal_uniform_epsilon_threshold']==1./7.


def test_defect_formula_matches_explicit_I_minus_RH_for_imperfect_inverse():
    A,B=toy_matrices();B=B+np.array([[2**-10,0.],[0.,-2**-11]])
    p,n,_=A.shape;R=np.zeros((p*n,p*n));Q=np.eye(n)
    for j in range(p-1,-1,-1):
        block=-B@Q
        for i in range(p):
            R[i*n:(i+1)*n,j*n:(j+1)*n]=block
            block=A[i]@block-(np.eye(n) if i==j else 0.)
        if j:Q=Q@A[j]
    defect=np.eye(p*n)-R@explicit_cyclic(A)
    report=driver.estimate_budget(A,B)
    for i,row in enumerate(report['phase_rows']):
        assert row['inverse_defect_constant']==pytest.approx(np.linalg.norm(defect[i*n:(i+1)*n,:n],np.inf),abs=2e-15)
        assert np.max(np.abs(defect[i*n:(i+1)*n,n:]))<2e-15


@pytest.mark.parametrize('B', [np.eye(2)*-1., np.zeros((2,2))])
def test_negative_zero_thresholds_fail_and_retain_all_eight_rows(B):
    A=np.zeros((8,2,2))
    with pytest.raises(driver.PlanningFailure) as caught:
        driver.estimate_budget(A,B)
    report=caught.value.report
    assert report['status']=='planning_failed'
    assert len(report['phase_rows'])==8
    assert all(len(row['pairs'])==8 for row in report['phase_rows'])
    assert report['invalid_threshold_rows']
    assert 'nominal_uniform_epsilon_threshold' not in report
    if B[0,0]<0:
        assert report['phase_rows'][0]['epsilon_threshold']<0
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('fault',['float32','nonsquare','empty','nan','Bshape','Bfloat32','Binf','list','overflow'])
def test_invalid_or_nonfinite_arithmetic_refused(fault):
    A,B=toy_matrices()
    if fault=='float32':A=A.astype(np.float32)
    if fault=='nonsquare':A=np.zeros((4,2,1))
    if fault=='empty':A=np.empty((0,2,2))
    if fault=='nan':A[0,0,0]=np.nan
    if fault=='Bshape':B=np.zeros((2,1))
    if fault=='Bfloat32':B=B.astype(np.float32)
    if fault=='Binf':B[0,0]=np.inf
    if fault=='list':A=A.tolist()
    if fault=='overflow':A=np.full((4,2,2),1e308)
    with pytest.raises(driver.PlanningFailure) as caught:driver.estimate_budget(A,B)
    assert caught.value.report['status']=='planning_failed'
    json.dumps(caught.value.report,allow_nan=False)


def plan_fixture(tmp_path):
    owned=tmp_path/'owned';owned.mkdir()
    for name in ('prefix','seed'):(owned/name).mkdir()
    bp=dict(schema='codex-llm-center-refine-v1',alpha=.071674,log10_temperature=-.115,
        period=8,dimension=896,precision=2048,max_corrections=1,evaluation_passes=2,
        envelope=dict(cpu_threads=1,nice=10,memory_gib=18,timeout_seconds=3600),
        prefix_attempt=str(owned/'prefix'),center_seed_attempt=str(owned/'seed'))
    bp.update({key:'a'*64 for key in driver.base.HASH_KEYS})
    plan=dict(schema=driver.SCHEMA,base_plan=bp,envelope=dict(driver.ENVELOPE),
              source_sha256={name:driver.base.digest(ROOT/name) for name in driver.SOURCE_PATHS})
    return owned,plan


def test_plan_admission_and_bound_loader_delegation(tmp_path,monkeypatch):
    owned,plan=plan_fixture(tmp_path)
    monkeypatch.setattr(driver,'OWNED',owned)
    seen=[];arrays={'A':'synthetic','B':'synthetic'}
    def fake_loader(bp,assets,prefix):
        seen.append((bp,assets,prefix));return arrays,{'checked':'synthetic'}
    monkeypatch.setattr(driver.base,'load_bound_inputs',fake_loader)
    assert driver.validate_plan(plan)==plan
    got=driver.load_inputs(plan,{'fake':'hash'},(1,2))
    assert got[0] is arrays and seen==[(plan['base_plan'],{'fake':'hash'},(1,2))]
    plan['source_sha256'][driver.SOURCE_PATHS[0]]='0'*64
    with pytest.raises(ValueError,match='source mismatch'):driver.load_inputs(plan,{},())
    assert len(seen)==1


@pytest.mark.parametrize('fault',['schema','missing','extra','source_missing','source_upper','source_extra',
                                 'base_missing','base_parameter','threads_bool','threads_float','seconds','memory'])
def test_wrong_plan_refused(tmp_path,fault):
    owned,plan=plan_fixture(tmp_path)
    if fault=='schema':plan['schema']='wrong'
    if fault=='missing':plan.pop('source_sha256')
    if fault=='extra':plan['evaluate_target']=True
    if fault=='source_missing':plan['source_sha256'].pop(driver.SOURCE_PATHS[0])
    if fault=='source_upper':plan['source_sha256'][driver.SOURCE_PATHS[0]]='A'*64
    if fault=='source_extra':plan['source_sha256']['unknown']='a'*64
    if fault=='base_missing':plan['base_plan'].pop('preconditioner_sha256')
    if fault=='base_parameter':plan['base_plan']['alpha']=.1
    if fault=='threads_bool':plan['envelope']['cpu_threads']=True
    if fault=='threads_float':plan['envelope']['cpu_threads']=1.
    if fault=='seconds':plan['envelope']['timeout_seconds']=121
    if fault=='memory':plan['envelope']['memory_gib']=19
    with pytest.raises(ValueError):driver.validate_plan(plan,owned=owned)


@pytest.mark.parametrize('fails',[False,True])
def test_driver_retains_all_rows_without_model_access(tmp_path,monkeypatch,fails):
    owned,plan=plan_fixture(tmp_path)
    source=tmp_path/'source';source.mkdir()
    path=source/'plan.json';driver.base.write_json(path,plan)
    out=owned/'new'/'run'
    A,B=toy_matrices(period=8)
    if fails:A=np.zeros_like(A);B=-np.eye(2)
    monkeypatch.setattr(driver,'OWNED',owned);monkeypatch.setattr(driver,'HERE',source)
    monkeypatch.setattr(driver,'load_inputs',lambda p,a,i:({'A':A,'B':B},{'synthetic_only':True}))
    monkeypatch.setenv('CERT_SOURCE_COMMIT','f'*40)
    monkeypatch.setattr(sys,'argv',['proposal_budget','--output',str(out),'--plan',str(path)])
    monkeypatch.setattr(os,'getpriority',lambda *a:10)
    # Only metadata is used by the production driver; this fake module has no
    # model constructor or tensor/checkpoint API to accidentally call.
    import types
    metadata=types.ModuleType('qwen_map');metadata.PINNED_SHA256={'synthetic':'a'*64};metadata.PREFIX_IDS=(1,2)
    monkeypatch.setitem(sys.modules,'qwen_map',metadata)
    if fails:
        with pytest.raises(driver.PlanningFailure):driver.main()
    else:driver.main()
    inputs=json.loads((out/'inputs.json').read_text())
    status=json.loads((out/'status.json').read_text())
    report=json.loads((out/'estimates.json').read_text())
    assert inputs['source_commit']=='f'*40 and inputs['plan']==plan
    assert inputs['plan_sha256']==driver.base.digest(path)
    assert inputs['source_binding']=={'synthetic_only':True}
    assert len(report['phase_rows'])==8 and sum(len(r['pairs']) for r in report['phase_rows'])==64
    assert status['certificate_passed'] is False and status['rigorous_enclosures'] is False
    assert status['status']==('proposal_budget_failed' if fails else 'proposal_budget_completed')
    if not fails:assert status['estimates_sha256']==driver.base.digest(out/'estimates.json')


def test_isolated_help_needs_no_target_or_plan():
    result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'codex/certify-llm/proposal_budget.py'),'--help'],
                          capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert '--plan' in result.stdout
