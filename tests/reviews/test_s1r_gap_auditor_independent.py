"""Independent adversarial saved-only controls. No model/decomposition execution.

Reviewer authored the gap producer. The auditor's six-role fixture is reused only
as fabricated input; targeted expectations and perturbations are reviewer-owned.
"""
import copy
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('gap_review_fixture',ROOT/'tests/reviews/test_s1r_lower_gap_outcome_audit.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
A=F.A


def journal_prefix(folder,stages):
    rows=[]
    for i,(kind,payload) in enumerate(stages):
        name=f'{i:05d}-{kind}.json'
        (folder/name).write_text(json.dumps(dict(kind=kind,sequence=i,payload=payload)))
        rows.append(dict(kind=kind,sequence=i,file=name))
    (folder/'journal.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return rows


def test_attempt_before_emit_is_one_next_operation_only(tmp_path):
    journal_prefix(tmp_path,[('bound_inputs',{}),('saved_line_search_reduction',[])])
    result=dict(status='failed_or_interrupted',operation_attempted=dict(maps=1,jacobians=1,svds=0),operation_returned=dict(maps=0,jacobians=0,svds=0))
    A.records(tmp_path,A.directory(tmp_path),False,result)
    for attempted,returned in [(dict(maps=2,jacobians=2,svds=0),dict(maps=0,jacobians=0,svds=0)),(dict(maps=1,jacobians=1,svds=0),dict(maps=1,jacobians=1,svds=0)),(dict(maps=0,jacobians=0,svds=1),dict(maps=0,jacobians=0,svds=0))]:
        with pytest.raises(ValueError):A.records(tmp_path,A.directory(tmp_path),False,dict(result,operation_attempted=attempted,operation_returned=returned))


def prefix_through(values,last):
    sequence=[k for k in A.expected_sequence() if k!='operation_started']
    return {k:values[k] for k in sequence[:sequence.index(last)+1]}


@pytest.mark.parametrize('later_role',[False,True])
def test_completed_or_superseded_role_corruption_never_partial_abstention(later_role):
    values,original,result,identity=F.six_role_fixture();role=A.ROLES[0]
    if later_role:
        values=prefix_through(values,A.ROLES[1]+'_first_phase_analytic_return')
        # Omit the role-complete marker only to challenge the independent arithmetic
        # helper; records() separately rejects such a non-prefix journal.
        del values[role+'_complete'];result['completed_roles']=[]
    else:
        values=prefix_through(values,role+'_complete');result['completed_roles']=[role]
    values[role+'_assembly']['A'][0,0]=.01;result['status']='failed_or_interrupted'
    with pytest.raises(ValueError):A.validate_arithmetic(values,original,False,result,identity)


@pytest.mark.parametrize('bad',[None,True,float('nan')])
def test_completed_loss_is_finite_numeric(bad):
    values,original,result,identity=F.six_role_fixture();role=A.ROLES[0]
    values[role+'_first_phase_analytic_return']['loss']=bad
    values[role+'_first_phase']['loss']=bad
    with pytest.raises(ValueError):A.validate_arithmetic(values,original,True,result,identity)


def test_invalid_terminal_return_is_unvalidated_not_completed():
    values,original,result,identity=F.six_role_fixture();role=A.ROLES[0]
    values=prefix_through(values,role+'_first_phase')
    values[role+'_first_phase']['jacobian'][0,0]=np.nan
    result.update(status='failed_or_interrupted',completed_roles=[])
    checked=A.validate_arithmetic(values,original,False,result,identity)
    assert checked['validated_roles']==[]
    assert [r['role'] for r in checked['unvalidated_partial_roles']]==[role]


def test_nontrivial_scaled_null_conversion_and_phase_exchange():
    # raw B=[diag(2,3),(-2,0)] and D=(2,3), hscale=1/8.
    # D^-1 B C has orthogonal rows (2,0,-1/8), (0,3,0).
    raw=np.array([[2.,0.,-2.],[0.,3.,0.]])
    columns=np.array([2.,3.,.125]);matrix=raw*columns/np.array([2.,3.])[:,None]
    length=math.sqrt(2.**2+.125**2)
    v=np.array([.125,0.,2.])/length
    Vh=np.array([[0.,1.,0.],[2.,0.,-.125],v]);Vh[1]/=length
    info=dict(status='full_row_rank_resolved')
    secant=np.array([-1.,0.,-1.])
    result=A.tangent(matrix,Vh,info,columns,secant,raw)
    np.testing.assert_allclose(result['tangent'],-np.array([1.,0.,1.])/math.sqrt(2),atol=1e-14)
    assert result['secant_cosine']>0 and result['raw_residual']<1e-14
    parent=np.array([[1.,2.],[-1.,-2.]])
    child=np.array([[-1.25,-2.5],[1.5,3.]])
    secants=A.secants(parent,child,.2,.1,1)
    np.testing.assert_allclose(secants['parent',0],[.5,1.,-.1])
    assert A.exact(secants['child',1],secants['parent',0])


def test_saved_trial_bits_and_infinity_merit_not_l2():
    state=np.zeros(2);step=np.zeros(2);residual=np.array([1.,1.])
    iteration=dict(call=473,iteration=3,state=state,residual=residual,step=step)
    trials=[dict(call=473,iteration=3,halving=k,state=state.copy(),residual=np.array([1.1,0.]),accepted=False) for k in range(13)]
    assert all(not r['accepted'] for r in A.trial_check(iteration,trials))
    trials[0]['state'][0]=-0.
    with pytest.raises(ValueError,match='exact saved trial state'):A.trial_check(iteration,trials)


def test_forcing_is_not_promoted_when_saved_residual_disagrees():
    A0=np.diag([2.,3.]);r=np.array([.01,.03]);p=np.array([-.005,-.01])
    result=A.linear_check(A0,r,np.ones(2),p,r+np.array([1e-6,0.]))
    assert result['true_relative_forcing']==0.
    assert result['verdict']=='arithmetic_resolution_limited'


def test_review_controls_forbid_numerical_decompositions(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('forbidden fresh decomposition')
    for name in ('svd','eig','eigh','solve','lstsq','inv','qr'):
        monkeypatch.setattr(np.linalg,name,forbidden)
    values,original,result,identity=F.six_role_fixture()
    checked=A.validate_arithmetic(values,original,True,result,identity)
    assert len(checked['validated_roles'])==6
