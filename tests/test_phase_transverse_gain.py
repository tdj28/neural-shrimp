"""Synthetic only: no real gain cells, integrations, fits or target files."""
from pathlib import Path
import copy
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import phase_transverse_gain as gain
import phase_memory as pm
import phase_memory_evaluation as old
from test_phase_memory_evaluation import synthetic_inputs


def toy(**changes):
    t=np.linspace(0,1,32);psi=np.zeros((32,3,3));psi[:,0,0]=1;psi[:,1,1]=np.exp(-t);psi[:,2,2]=np.exp(-2*t)
    flow=np.tile([1.,0.,0.],(32,1));weights=flow/32
    values=dict(psi=psi,states=np.zeros((32,3)),template_velocity=flow.copy(),physical_flow=flow.copy(),initial_flow=np.array([1.,0.,0.]),gradient=np.array([1.,0.,0.]),weights=weights,direction=np.array([1.,1.,1.])/np.sqrt(3))
    values.update(changes);return values


def test_census_exact_complete_and_ordered():
    rows=gain.cell_census();assert len(rows)==192 and len({r['id'] for r in rows})==192
    assert rows[0]==dict(id='core--p0--phase_write--k1',setting='core',phase_index=0,direction='phase_write',delay_cycles=1)
    assert rows[-1]['id']=='failed-near-edge--p7--neuron_one--k8'
    assert gain.CONTRACT['compatibility_rows']==1152 and gain.CONTRACT['unit_noise_rows']==576 and gain.CONTRACT['realized_noise_rows']==3456


def test_exact_partition_and_ideal_flow():
    values=toy();result=gain.decompose(**values)
    assert all(x['passed'] for x in result['algebraic_checks'].values())
    assert np.allclose(result['state']['phase_algebraic']+result['state']['transverse'],result['state']['total'])
    assert result['observable']['transverse']==0
    assert result['observable']['phase_algebraic']==pytest.approx(values['direction'][0])
    assert all(x['passed'] for x in gain._approximation(result).values())


def test_nonnormal_projector_not_orthogonal_energy_split():
    z=np.array([1.,2.,0.]);result=gain.decompose(**toy(gradient=z))
    P=result['projector'];assert not np.array_equal(P,P.T)
    assert result['reference_defects']['normalization']==0
    assert all(x['passed'] for x in result['algebraic_checks'].values())
    a=result['state']['phase_algebraic'];b=result['state']['transverse'];full=result['state']['total']
    assert not np.isclose(np.sum(a*a)+np.sum(b*b),np.sum(full*full))


def test_normalization_defect_preserved_without_repair():
    z=np.array([1.+1e-5,2.,0.]);result=gain.decompose(**toy(gradient=z))
    assert result['reference_defects']['normalization']>gain.GATES['normalization_atol']
    assert result['reference_defects']['projector_idempotence']['maximum_absolute']>0
    assert result['reference_defects']['phase_null_leakage']['maximum_absolute']>0
    assert all(x['passed'] for x in result['algebraic_checks'].values())
    assert np.array_equal(result['projector'],np.eye(3)-np.outer([1.,0.,0.],z))


def test_transport_and_clock_defects_do_not_cancel_gate():
    v=toy()['template_velocity'];physical=v*2
    result=gain.decompose(**toy(physical_flow=physical,direction=np.array([.01,0.,0.])))
    assert np.array_equal(result['state']['transport_defect'],-result['state']['clock_flow_defect'])
    gates=gain._approximation(result)
    assert not gates['transport_defect']['passed'] and not gates['clock_flow_defect']['passed']
    assert gates['combined_ideal_clock_defect']['passed']
    assert all(x['passed'] for x in result['algebraic_checks'].values())


def test_noise_scaling_and_orientation_are_signed():
    result=gain.decompose(**toy(direction=np.array([-.01,0.,0.])))
    rows=gain._noise_rows(result,-1)
    assert len(rows)==3 and all(r['signed_standardized']['total']>0 for r in rows)
    assert rows[0]['signed_standardized']['total']==pytest.approx(10*rows[1]['signed_standardized']['total'])


def test_zero_physical_flow_ratio_unavailable_not_repaired():
    result=gain.decompose(**toy(physical_flow=np.zeros((32,3))))
    assert result['reference_defects']['maximum_relative_physical_flow_transport'] is None
    assert result['reference_defects']['zero_reference_flow_samples']==32


@pytest.mark.parametrize('field,value',[('psi',np.zeros((31,3,3))),('weights',np.zeros((32,3))),('direction',np.zeros(3,dtype=np.float32)),('gradient',np.array([np.nan,0.,0.])),('states',np.full((32,3),np.inf))])
def test_invalid_math_inputs_reject(field,value):
    with pytest.raises(ValueError):gain.decompose(**toy(**{field:value}))


def test_noncommuting_elapsed_matrix_order_and_old_direct_match():
    t=np.linspace(0,1,4097);A=np.array([[0.,1.,0.],[0.,0.,0.],[0.,0.,-.25]])
    B=np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,0.]])
    phi=np.eye(3)[None,:,:]+t[:,None,None]*A+(t*(1-t))[:,None,None]*B
    derivative=np.broadcast_to(A,phi.shape)+(1-2*t)[:,None,None]*B
    segment=pm.HermiteSegment(t,phi,derivative);M=phi[-1];query=2+(np.arange(32)+.5)/32
    matrix=gain.propagated_matrices(segment,M,query)
    assert np.allclose(matrix,segment.evaluate(query-2)@np.linalg.matrix_power(M,2))
    assert not np.allclose(matrix,np.linalg.matrix_power(M,2)@segment.evaluate(query-2))
    d=np.array([1.,.2,-.1]);direct=pm.full_linear_response(segment,M,d,query)['state_delta']
    assert gain._close(np.einsum('nij,j->ni',matrix,d),direct)['passed']


@pytest.fixture(scope='module')
def fixture():
    references,props=synthetic_inputs();cell=gain.cell_census()[13]
    reference=references[cell['setting']];prop=props[cell['setting'],cell['phase_index']]
    first=old._TASKS_BY_ID[gain._six_ids(cell)[0]+'--'+old.METHODS[0]]
    context,mathematics=old._prepared_context(first,reference,prop)
    cases={ident:old._prepared_case(old._TASKS_BY_ID[ident+'--'+old.METHODS[0]],context,mathematics) for ident in gain._six_ids(cell)}
    return cell,dict(reference=reference,propagator=prop,sealed_context=context,sealed_cases=cases)


def test_serialized_pipeline_compatible_and_no_mutation(fixture):
    cell,inputs=fixture;before=copy.deepcopy(inputs)
    result=gain.evaluate_cell(cell,**inputs)
    assert result['status']=='complete' and result['complete'] and result['qualified_algebraic']
    # Analytic circle fixture is deliberately not a solution of fixed Hopfield RHS.
    assert not result['qualified_ideal_neutral'] and not result['qualified']
    assert len(result['compatibility'])==6 and all(r['complete'] for r in result['compatibility'])
    assert all(all(c['passed'] for c in r['checks'].values()) for r in result['compatibility'])
    assert gain._equal(inputs,before)


def test_projection_omits_unneeded_dense_diagnostics(fixture):
    cell,inputs=fixture;v=copy.deepcopy(inputs)
    for name in ('diagnostic_times','reference_states','reference_velocity','reference_adjoint'):v['sealed_context'].pop(name)
    for c in v['sealed_cases'].values():
        for name in ('full_linear_state_delta','full_linear_activation_delta','phase_only_activations'):c.pop(name)
    assert gain.evaluate_cell(cell,**v)['qualified_algebraic']


@pytest.mark.parametrize('mutation',[lambda v:v['sealed_context'].__setitem__('phase_index',True),lambda v:v['sealed_context']['readouts']['2']['weights'].__setitem__((0,0),123.),lambda v:v['propagator'].__setitem__('qualified',False),lambda v:v['reference']['orbit'].__setitem__('states',v['reference']['orbit']['states'].astype(np.float32)),lambda v:v['sealed_cases'].pop(next(iter(v['sealed_cases'])))])
def test_bound_input_failures_keep_all_six_rows(fixture,mutation):
    cell,inputs=fixture;v=copy.deepcopy(inputs);mutation(v);row=gain.evaluate_cell(cell,**v)
    assert row['status']=='failed' and not row['qualified'] and len(row['compatibility'])==6
    assert row['error']


def test_numeric_compatibility_failure_stays_complete(fixture):
    cell,inputs=fixture;v=copy.deepcopy(inputs)
    v['sealed_cases'][next(iter(v['sealed_cases']))]['readouts']['2']['full_linear_q']+=1e-6
    row=gain.evaluate_cell(cell,**v)
    assert row['complete'] and not row['qualified_algebraic']
    assert not row['compatibility'][0]['checks']['q']['passed']


def test_realized_and_requested_not_interchangeable(fixture):
    cell,inputs=fixture;v=copy.deepcopy(inputs);first=next(iter(v['sealed_cases']))
    assert not np.array_equal(v['sealed_cases'][first]['pulse']['realized_delta'],v['sealed_cases'][first]['pulse']['requested_delta'])
    v['sealed_cases'][first]['pulse']['realized_delta']=v['sealed_cases'][first]['pulse']['requested_delta'].copy()
    row=gain.evaluate_cell(cell,**v)
    assert not row['complete'] and 'pulse identity' in row['error']


def test_invalid_cell_metadata_fails_without_access():
    cell=gain.cell_census()[0];cell['phase_index']=False
    with pytest.raises(ValueError):gain.evaluate_cell(cell,reference=None,propagator=None,sealed_context=None,sealed_cases=None)


def test_no_file_solver_or_target_scoring_access(fixture,monkeypatch):
    cell,inputs=fixture
    import builtins,scipy.integrate
    def forbidden(*a,**k):pytest.fail('Unexpected file, integration or target-scoring access')
    monkeypatch.setattr(builtins,'open',forbidden);monkeypatch.setattr(scipy.integrate,'solve_ivp',forbidden);monkeypatch.setattr(old,'score_trace',forbidden)
    assert gain.evaluate_cell(cell,**inputs)['qualified_algebraic']


def test_all_192_synthetic_cells_and_1152_compatibility_rows():
    references,props=synthetic_inputs();prepared=old.prepare_reference_outputs(references,props)
    rows=[]
    for cell in gain.cell_census():
        context=prepared['contexts'][f'{cell["setting"]}--p{cell["phase_index"]}']
        rows.append(gain.evaluate_cell(cell,reference=references[cell['setting']],propagator=props[cell['setting'],cell['phase_index']],sealed_context=context,sealed_cases={i:prepared['cases'][i] for i in gain._six_ids(cell)}))
    assert len(rows)==192 and all(r['complete'] and r['qualified_algebraic'] for r in rows)
    assert sum(len(r['compatibility']) for r in rows)==1152
    assert sum(len(r['unit']['noise_rows']) for r in rows)==576
    assert sum(len(c['noise_rows']) for r in rows for c in r['compatibility'])==3456


def test_nonfinite_phase_only_comparator_retains_failed_row(fixture):
    cell,inputs=fixture;v=copy.deepcopy(inputs)
    v['sealed_cases'][next(iter(v['sealed_cases']))]['readouts']['2']['phase_only_q']=float('nan')
    row=gain.evaluate_cell(cell,**v)
    assert row['status']=='failed' and len(row['compatibility'])==6


def test_noise_underflow_fails_without_infinite_output():
    with pytest.raises(ValueError):gain._noise_rows({'noise_gain':1e-320,'observable':{'total':1.}},1)


@pytest.mark.parametrize('name',['reference','propagator','sealed_context','sealed_cases'])
def test_nonmapping_inputs_return_failed_cell(fixture,name):
    cell,inputs=fixture;values=dict(inputs);values[name]=None
    result=gain.evaluate_cell(cell,**values)
    assert result['status']=='failed' and len(result['compatibility'])==6


@pytest.mark.parametrize('part',['clock','pulse'])
def test_nonmapping_nested_packet_returns_failed_cell(fixture,part):
    cell,inputs=fixture;values=copy.deepcopy(inputs)
    if part=='clock':values['sealed_context']['readouts']['2']=None
    else:values['sealed_cases'][next(iter(values['sealed_cases']))]=None
    result=gain.evaluate_cell(cell,**values)
    assert result['status']=='failed' and len(result['compatibility'])==6


def test_signed_normalization_residual_is_retained():
    result=gain.decompose(**toy(gradient=np.array([1.-1e-5,2.,0.])))
    defects=result['reference_defects']
    assert defects['normalization_signed']<0
    assert defects['normalization']==abs(defects['normalization_signed'])
