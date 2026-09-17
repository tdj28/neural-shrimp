"""Audit-only fabricated fixtures. No target files, fits, ODEs or network."""
import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import ndtr

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'tools')]
import audit_phase_memory as independent
import test_phase_memory_runner as fixtures
import run_phase_memory as storage
import phase_memory as producer_math
import phase_memory_evaluation as producer_evaluation


def test_independent_full_census_matches_fixed_contract_without_target_access():
    rows=independent.tasks()
    assert len(rows)==len({r['id'] for r in rows})==912
    assert sum(r['kind']=='sham' for r in rows)==48
    assert rows==producer_evaluation.task_census()
    assert len({r['id'].rsplit('--',1)[0] for r in rows})==304


def test_independent_codec_decodes_all_arrays_and_preserves_failed_nonfinite(tmp_path):
    value=dict(states=np.array([[1.,np.nan,np.inf]],dtype=np.float64),progress=np.array([3,1]),
               error=-float('inf'),status='failed',nested=[np.float64(.5),np.str_('a')])
    stored=storage.save_packet(tmp_path,'packet',value);audit=independent.Audit()
    decoded=independent.decode_packet(stored['path'],audit)
    assert not audit.failures and np.array_equal(decoded['states'],value['states'],equal_nan=True)
    assert decoded['error']==-float('inf') and decoded['nested']==[.5,'a']
    with (tmp_path/'packet.npz').open('ab') as f:f.write(b'extra')
    changed=independent.Audit();independent.decode_packet(stored['path'],changed)
    assert any('packet bytes' in x['check'] for x in changed.failures)


def test_scipy_hermite_matches_analytic_cubic_and_periodic_derivative_units():
    t=np.array([0.,.2,.8,1.])
    values=np.column_stack((t**3,2*t*t,3*t+2));derivatives=np.column_stack((3*t*t,4*t,np.full(4,3.)))
    interpolation=independent.spline(t,values,derivatives)
    q=np.linspace(0.,1.,17)
    assert np.allclose(interpolation(q),np.column_stack((q**3,2*q*q,3*q+2)),atol=1e-15)
    assert np.allclose(interpolation(q,1),np.column_stack((3*q*q,4*q,np.full(17,3.))),atol=1e-14)
    period=2.;knots=np.linspace(0.,period,4097);angle=np.pi*knots
    states=np.column_stack((np.cos(angle),np.sin(angle),np.zeros(len(knots))))
    velocity=np.column_stack((-np.pi*np.sin(angle),np.pi*np.cos(angle),np.zeros(len(knots))))
    periodic=independent.spline(knots,states,velocity,period=period)
    assert np.max(np.abs(periodic(q,1)-np.column_stack((-np.pi*np.sin(np.pi*q),np.pi*np.cos(np.pi*q),np.zeros(len(q))))))<2e-10
    assert np.array_equal(periodic(np.array([0.])),periodic(np.array([period])))


@pytest.mark.parametrize('plus,minus,orientation',[(0.,0.,1),(.001,-.001,1),(-.001,.001,1),(.001,.001,1),(-.001,.001,-1)])
def test_erf_gaussian_noise_accuracy_independent_of_producer_ndtr(plus,minus,orientation):
    for sigma in independent.NOISES:
        expected=float((ndtr(orientation*plus/(sigma*2))+ndtr(-orientation*minus/(sigma*2)))/2)
        actual=independent.accuracy(plus,minus,2.,orientation,sigma)
        assert abs(actual-expected)<=2e-16
    assert independent.accuracy(0.,0.,1.,1,1e-4)==.5
    assert independent.accuracy(-.001,.001,1.,1,1e-4)<.5


def test_adjoint_null_direction_uses_covector_not_flow():
    z=np.array([1.,2.,4.]);flow=np.array([1.,0.,0.])
    directions,index=independent.direction_table(z)
    assert abs(z@directions['phase_null'])<2e-15
    assert abs(flow@directions['phase_null'])>.5
    assert all(abs(np.linalg.norm(v)-1)<1e-15 for v in directions.values())
    assert index==0
    assert independent.direction_table(np.array([0.,0.,1.]))[1]==0


def test_independent_reference_equations_on_analytic_circle_only(monkeypatch):
    reference=fixtures.reference('core')
    # The analytic-circle fixture is not a nine-peak Hopfield certificate.
    # Isolate the saved fine/adjoint equation checks in this unit test.
    monkeypatch.setattr(independent,'certificate_audit',lambda *a:True)
    monkeypatch.setattr(independent,'rhs',lambda x,a,g:np.asarray([fixtures.toy_rhs(0.,row,a,g) for row in x]))
    monkeypatch.setattr(independent,'jac',lambda x,a,g:np.asarray([fixtures.toy_jac(0.,row,a,g) for row in x]))
    audit=independent.Audit();result=independent.reference_audit(reference,audit)
    assert result['qualified'] and not audit.failures
    changed=copy.deepcopy(reference);changed['adjoint']['values'][50]*=1.001
    audit=independent.Audit();result=independent.reference_audit(changed,audit)
    assert not result['qualified'] and audit.failures


def test_independent_clock_and_signed_score_against_synthetic_saved_producer_packet():
    references={key:fixtures.reference(key) for key in independent.SETTINGS}
    props={(key,0):fixtures.propagator(ref,0) for key,ref in references.items()}
    prepared=producer_evaluation.prepare_reference_outputs(references,props)
    task=independent.tasks()[0];reference=references['core'];trace=fixtures.trace(task,reference)
    # The synthetic waveform is only an array fixture. Both readers reconstruct
    # prescribed Hopfield RHS derivatives from it; neither integrates a system.
    derivative=independent.rhs(trace['states'],*independent.SETTINGS['core'])
    saved=producer_evaluation.score_trace(task,trace,reference,props[('core',0)],state_derivatives=derivative,prepared=prepared)
    audit=independent.Audit();fresh=independent.score(task,trace,reference,props[('core',0)],prepared,audit)
    assert saved['status']=='complete' and not audit.failures
    for delay,row in fresh['readouts'].items():
        assert abs(row['q']-saved['readouts'][str(delay)]['q'])<1e-13
        assert abs(row['gain']-saved['readouts'][str(delay)]['noise_gain'])<1e-13
    forged=copy.deepcopy(trace);forged['realized_delta'][0]+=.1
    audit=independent.Audit();independent.score(task,forged,reference,props[('core',0)],prepared,audit)
    assert any('exact realization' in x['check'] for x in audit.failures)


def test_all_reference_only_cases_and_paired_nulls_rebuilt_without_trace_access():
    references={key:fixtures.reference(key) for key in independent.SETTINGS}
    props={(key,0):fixtures.propagator(ref,0) for key,ref in references.items()}
    prepared=producer_evaluation.prepare_reference_outputs(references,props)
    audit=independent.Audit();result=independent.prepared_predictions(prepared,references,{f'{s}--p{j}':p for (s,j),p in props.items()},audit)
    assert result==dict(independently_rebuilt_cases=38,independently_rebuilt_null_ba_cells=216)
    assert not audit.failures
    changed=copy.deepcopy(prepared);changed['paired_nulls'][0]['phase_only_balanced_accuracy']=.123
    audit=independent.Audit();independent.prepared_predictions(changed,references,{f'{s}--p{j}':p for (s,j),p in props.items()},audit)
    assert audit.failures


@pytest.mark.parametrize('field', ['phase_gradient', 'flow', 'phase_index', 'phase_order'])
def test_saved_phase_identity_is_independently_bound(monkeypatch,field):
    reference=fixtures.reference('core')
    monkeypatch.setattr(independent,'certificate_audit',lambda *a:True)
    monkeypatch.setattr(independent,'rhs',lambda x,a,g:np.asarray([fixtures.toy_rhs(0.,row,a,g) for row in x]))
    monkeypatch.setattr(independent,'jac',lambda x,a,g:np.asarray([fixtures.toy_jac(0.,row,a,g) for row in x]))
    if field in ('phase_gradient','flow'):
        reference['phases'][0][field]=reference['phases'][0][field]+np.array([.1,.2,.3])
    elif field=='phase_index':
        reference['phases'][0][field]=False
    else:
        reference['phases'][0],reference['phases'][1]=reference['phases'][1],reference['phases'][0]
    audit=independent.Audit();result=independent.reference_audit(reference,audit)
    assert not result['qualified'] and audit.failures


def test_report_numpy_scalars_serialize_without_changing_values():
    import json
    report=dict(gates=dict(unique=np.bool_(True)),residual=np.float64(1e-12),count=np.int64(912),shape=np.array([3,4]))
    result=json.loads(json.dumps(report,default=independent.json_default,allow_nan=False))
    assert result==dict(gates=dict(unique=True),residual=1e-12,count=912,shape=[3,4])
    with pytest.raises(ValueError):
        json.dumps(dict(invalid=np.float64(np.nan)),default=independent.json_default,allow_nan=False)
