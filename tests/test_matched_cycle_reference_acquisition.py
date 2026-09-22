"""New six-slot/eight-phase adapter tests; analytic ODE injection only."""
from pathlib import Path
from collections import Counter
import copy
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import matched_cycle_reference_acquisition as acq
import periodic_reference_refinement as original
from phase_memory_acquisition import AcquisitionFailure
import test_periodic_reference_refinement_integration as oscillator


def slot(cycle_id='core-rho025'):
    anchor=cycle_id.startswith('anchor-')
    source=cycle_id[len('anchor-'):] if anchor else cycle_id.rsplit('-rho',1)[0]
    target=None if anchor else (.25 if cycle_id.endswith('025') else .5)
    return dict(id=cycle_id,cycle_id=cycle_id,role='anchor' if anchor else 'candidate',source_setting=source,
        target_rho=target,branch_id=None if anchor else source+'-direction-+1',node_index=None if anchor else 3,
        node_id=None if anchor else source+'-direction-+1--node-003',available=True,status='selected',reasons=[],
        alpha=.123,gamma=4.567,period=float(oscillator.PERIOD),first_maximum_state=oscillator.SEED.copy(),
        seed_provenance=dict(input_sha256='a'*64,original_role='canonical_global_maximum' if anchor else 'historical_maximum'))


def binding(cycle_id='core-rho025'):
    return dict(experiment_id='EXP-022',source_commit='b'*40,plan_sha256='c'*64,
        reference_id=cycle_id+'--reference-pair',packet_sha256='d'*64,packet_semantic_sha256='e'*64)


@pytest.fixture(scope='module')
def data():
    seed=acq.prepare_seed(slot());rows=acq.phase_rows()
    shots={m:acq.refine_cycle(seed,m,rhs=oscillator.rhs,jacobian=oscillator.jacobian,solver=oscillator.exact_solver) for m in acq.METHODS}
    pair=acq.crossverify_reference(seed,shots,rows,rhs=oscillator.rhs,jacobian=oscillator.jacobian,solver=oscillator.exact_solver)
    assert pair['complete'] and pair['qualified']
    return seed,rows,shots,pair


def context(data,index=3):
    return acq.prepare_context(data[3],data[1][index],source_binding=binding(),rhs=oscillator.rhs)


def test_exact_114_job_and_528_ivp_census():
    jobs=acq.job_census();ivps=acq.ivp_census()
    assert len(jobs)==len({r['id'] for r in jobs})==114
    assert Counter(r['kind'] for r in jobs)==dict(shoot=12,reference=6,tangent=96)
    assert len(ivps)==len({r['id'] for r in ivps})==528
    assert Counter(r['kind'] for r in ivps)==dict(shoot_trial=300,verification=24,adjoint=12,tangent_segment=192)
    assert len({r['cycle_id'] for r in jobs})==6
    assert not any('event' in r['kind'] for r in jobs)
    assert acq.contract()['finite_write_jobs']==0


def test_phase_seed_freshness_and_unsorted_order():
    expected=np.random.Generator(np.random.PCG64(2026091702)).random(8)
    rows=acq.phase_rows();assert [r['split'] for r in rows]==['fresh']*8
    np.testing.assert_array_equal([r['phase_fraction'] for r in rows],expected)
    assert np.any(np.diff(expected)<0)
    metadata=acq.phase_metadata(rows,7.,'anchor-core')
    assert [r['phase_index'] for r in metadata]==list(range(8))
    assert [r['phase_fraction'] for r in metadata]==expected.tolist()
    assert acq.phase_metadata(metadata,7.,'anchor-core')==metadata


@pytest.mark.parametrize('change',['count','duplicate','bool_index','split','epoch'])
def test_invalid_phase_census_rejected_without_solve(change):
    rows=acq.phase_rows()
    if change=='count':rows=rows*3
    elif change=='duplicate':rows[1]['phase_fraction']=rows[0]['phase_fraction']
    elif change=='bool_index':rows[0]['phase_index']=False
    elif change=='split':rows[0]['split']='development'
    else:rows=acq.phase_metadata(rows,7.,'anchor-core');rows[0]['phase_epoch']+=1e-8
    with pytest.raises(ValueError):acq.phase_metadata(rows,7.,'anchor-core')


@pytest.mark.parametrize('cycle_id',acq.CYCLE_IDS)
def test_seed_admission_is_not_new_cycle_qualification(cycle_id):
    selected=slot(cycle_id);seed=acq.prepare_seed(selected)
    assert seed['setting']==seed['cycle_id']==cycle_id
    assert seed['status']=='admitted_historical_seed' and seed['admitted_historical_seed']
    assert seed['qualified'] is False and seed['new_reference_qualified'] is False
    assert seed['alpha']==selected['alpha'] and seed['gamma']==selected['gamma']
    assert seed['certificate']['seed_interface_only'] and seed['orbit']['seed_interface_only']
    seed['certificate']['section_state'][0]+=1
    np.testing.assert_array_equal(selected['first_maximum_state'],oscillator.SEED)


@pytest.mark.parametrize('change',['missing','id','role','source','rho','index','nonfinite','provenance'])
def test_malformed_or_unavailable_seed_never_silently_admitted(change):
    selected=slot()
    if change=='missing':selected.update(available=False,status='no_historical_match',first_maximum_state=None)
    elif change=='id':selected['id']='core'
    elif change=='role':selected['role']='anchor'
    elif change=='source':selected['source_setting']='invented'
    elif change=='rho':selected['target_rho']=.7
    elif change=='index':selected['node_index']=True
    elif change=='nonfinite':selected['first_maximum_state'][0]=np.nan
    else:selected['seed_provenance']={}
    with pytest.raises(ValueError):acq.prepare_seed(selected)


def test_generic_shooting_retains_truthful_ids_parameters_and_all_trial_slots(data):
    seed,rows,shots,pair=data
    for method,p in shots.items():
        assert p['id']=='core-rho025--'+method and p['setting']==p['cycle_id']=='core-rho025'
        assert p['alpha']==.123 and p['gamma']==4.567 and p['source_setting']=='core'
        assert p['complete'] and p['converged'] and len(p['trials'])==25
        assert sum(r['ivp_started'] for r in p['trials'])==1
    assert seed['qualified'] is False and seed['status']=='admitted_historical_seed'
    assert original.EXPECTED_PEAKS==9


def test_forward_solver_failure_retains_trial_and_public_admission_metadata():
    seed=acq.prepare_seed(slot())
    def broken(fun,span,initial,**kwargs):
        fun(span[0],initial)
        raise RuntimeError('synthetic owned solve failure')
    with pytest.raises(AcquisitionFailure) as caught:
        acq.refine_cycle(seed,'tight-dop853',rhs=oscillator.rhs,jacobian=oscillator.jacobian,solver=broken)
    p=caught.value.payload
    assert p['cycle_id']=='core-rho025' and p['source_setting']=='core'
    assert len(p['trials'])==25 and p['trials'][0]['ivp_started'] and not p['trials'][0]['ivp_completed']
    assert 'partial_failure' in p['trials'][0]
    assert all(not r['ivp_started'] for r in p['trials'][1:])
    assert seed['qualified'] is False


def test_eight_exact_dense_phases_and_one_global_gradient(data):
    seed,rows,shots,pair=data
    assert pair['cycle_id']==pair['setting']=='core-rho025'
    assert pair['canonical']['normalization_count']==1
    assert len(pair['verifications'])==4 and len(pair['adjoints'])==2
    z=pair['canonical']['terminal_adjoint']
    for base in pair['bases'].values():
        assert len(base['phases'])==len(base['phase_rows'])==8
        assert base['exact_phase_arrays']['states'].shape==(8,3)
        assert base['exact_phase_arrays']['phi'].shape==(8,3,3)
        assert base['new_normalizations']==0
        np.testing.assert_array_equal(base['raw_endpoints']['final_z'],z)
        assert [r['phase_fraction'] for r in base['phase_rows']]==[r['phase_fraction'] for r in rows]
        assert [r['split'] for r in base['phase_rows']]==['fresh']*8
        assert base['cycle_id']=='core-rho025'
    assert all(r['certificate']['peak_count']==9 for r in pair['verifications'].values())


def test_crosschecks_continue_after_one_failure_and_block_both_adjoints(data):
    seed,rows,shots,pair=data;calls=0
    def one_failure(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('synthetic second verifier failure')
        return oscillator.exact_solver(*args,**kwargs)
    p=acq.crossverify_reference(seed,shots,rows,rhs=oscillator.rhs,jacobian=oscillator.jacobian,solver=one_failure)
    assert calls==4 and p['status']=='blocked_crossverification' and not p['qualified']
    assert all(r['ivp_started'] for r in p['verifications'].values())
    assert sum(r['complete'] for r in p['verifications'].values())==3
    assert all(not r['ivp_started'] and r['status']=='blocked_prerequisite' for r in p['adjoints'].values())
    assert not p['bases']


def test_wrong_parameter_shoot_or_phase_count_rejected_before_cross_solves(data):
    seed,rows,shots,pair=data
    changed=copy.deepcopy(shots);changed['tight-rk45']['gamma']+=1e-5
    def forbidden(*args,**kwargs):pytest.fail('Invalid prerequisite launched a solver')
    with pytest.raises(ValueError):acq.crossverify_reference(seed,changed,rows,rhs=oscillator.rhs,jacobian=oscillator.jacobian,solver=forbidden)
    with pytest.raises(ValueError):acq.crossverify_reference(seed,shots,rows*3,rhs=oscillator.rhs,jacobian=oscillator.jacobian,solver=forbidden)


def test_context_exact_common_canonical_phase_and_no_local_gradient_repair(data):
    result=context(data,7);phase=data[3]['bases']['tight-dop853']['phases'][7]
    assert result['id']=='core-rho025--p07' and result['cycle_id']==result['setting']=='core-rho025'
    assert result['split']=='fresh' and result['new_normalizations']==0
    np.testing.assert_array_equal(result['initial'],phase['state'])
    np.testing.assert_array_equal(result['phase_gradient'],phase['z'])
    np.testing.assert_array_equal(result['nominal_event_phase_gradient'],phase['z'])
    assert abs(result['phase_gradient']@result['direction'])<1e-12
    changed=copy.deepcopy(data[3]);base=changed['bases']['tight-dop853']
    base['phases'][7]['z']*=1.01;base['exact_phase_arrays']['z'][7]*=1.01
    modified=acq.prepare_context(changed,data[1][7],source_binding=binding(),rhs=oscillator.rhs)
    np.testing.assert_array_equal(modified['phase_gradient'],base['phases'][7]['z'])
    assert modified['normalization_residual']==pytest.approx(.01,abs=2e-12)


def test_bad_phase_or_current_source_binding_rejected(data):
    pair=copy.deepcopy(data[3]);pair['bases']['tight-dop853']['phases'][1]['state'][0]+=1e-8
    with pytest.raises(ValueError):acq.prepare_context(pair,data[1][1],source_binding=binding(),rhs=oscillator.rhs)
    bad=binding();bad['reference_id']='core--reference-pair'
    with pytest.raises(ValueError):acq.prepare_context(data[3],data[1][1],source_binding=bad,rhs=oscillator.rhs)
    bad=binding();bad['receipt_sha256']='f'*64
    with pytest.raises(ValueError):acq.prepare_context(data[3],data[1][1],source_binding=bad,rhs=oscillator.rhs)


def test_tangent_two_segments_have_fresh_identity_and_correct_physical_clock(data):
    ctx=context(data,0);calls=[];physical=[]
    def solver(fun,span,initial,**kwargs):
        calls.append((span,initial.copy(),kwargs['method']))
        return oscillator.exact_solver(fun,span,initial,**kwargs)
    def rhs(t,x,a,g):
        assert a==.123 and g==4.567;physical.append(t)
        return oscillator.rhs(t,x,a,g)
    p=acq.acquire_tangent(ctx,'dop853',rhs=rhs,jacobian=oscillator.jacobian,solver=solver)
    T=ctx['period'];assert p['id']=='core-rho025--p00--reference--dop853'
    assert calls[0][0]==(0.,T) and calls[1][0]==(0.,7*T)
    assert max(physical)==8*T
    for span,initial,method in calls:np.testing.assert_array_equal(initial[3:].reshape(3,3),np.eye(3))
    np.testing.assert_array_equal(calls[0][1][:3],ctx['initial'])
    np.testing.assert_array_equal(calls[1][1][:3],p['pre']['states'][-1])
    assert all(r['ivp_started'] and r['ivp_returned'] and r['ivp_completed'] for r in p['ivp_slots'].values())
    sealed=acq.seal_context(ctx,p,rhs=oscillator.rhs)
    np.testing.assert_array_equal(sealed['event_phase_gradient'],ctx['phase_gradient'])
    assert abs(sealed['event_normalization_residual'])<1e-10


def test_post_failure_preserves_completed_pre_and_exact_two_slot_ledger(data):
    ctx=context(data,2);calls=0
    def solver(fun,span,initial,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:
            fun(0.,initial)
            raise RuntimeError('synthetic post solver failure')
        return oscillator.exact_solver(fun,span,initial,**kwargs)
    with pytest.raises(AcquisitionFailure) as caught:
        acq.acquire_tangent(ctx,'rk45',rhs=oscillator.rhs,jacobian=oscillator.jacobian,solver=solver)
    p=caught.value.payload
    assert p['failed_stage']=='post' and p['pre']['states'].shape==(4097,3)
    assert p['ivp_slots']['pre']['ivp_completed'] and p['ivp_slots']['post']['ivp_started']
    assert not p['ivp_slots']['post']['ivp_completed']
    assert p['partial_failure']['last_rhs']['global_time']==ctx['period']
    assert len(p['ivp_slots'])==2


def test_derivative_failure_retains_successful_ivp_completion_flag(data):
    ctx=context(data,2);rhs_calls=0
    def rhs(t,x,a,g):
        nonlocal rhs_calls
        rhs_calls+=1
        if rhs_calls==3:raise RuntimeError('synthetic derivative reconstruction failure')
        return oscillator.rhs(t,x,a,g)
    with pytest.raises(AcquisitionFailure) as caught:
        acq.acquire_tangent(ctx,'dop853',rhs=rhs,jacobian=oscillator.jacobian,solver=oscillator.exact_solver)
    p=caught.value.payload
    assert p['failed_stage']=='pre' and p['ivp_slots']['pre']['ivp_completed']
    assert p['partial_failure']['states'].shape==(4097,3)
    assert not p['ivp_slots']['post']['ivp_started']


@pytest.mark.parametrize('field',['alpha','gamma','period','state','provenance'])
def test_seed_mutation_cannot_escape_attached_historical_selection(field):
    seed=acq.prepare_seed(slot())
    if field in ('alpha','gamma'):seed[field]+=1e-8
    elif field=='period':seed['period']+=1e-8;seed['certificate']['period']=seed['period']
    elif field=='state':seed['certificate']['section_state'][0]+=1e-8;seed['orbit']['states'][0]=seed['certificate']['section_state']
    else:seed['seed_provenance']['input_sha256']='0'*64
    def forbidden(*args,**kwargs):pytest.fail('Altered historical seed launched an IVP')
    with pytest.raises(ValueError):acq.refine_cycle(seed,'tight-dop853',rhs=oscillator.rhs,jacobian=oscillator.jacobian,solver=forbidden)
