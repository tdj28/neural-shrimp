"""Bounded refinement and failure retention on an analytic synthetic cycle."""
from pathlib import Path
from types import SimpleNamespace
import copy
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import periodic_reference_refinement as acq
from phase_memory_acquisition import AcquisitionFailure

PERIOD = 2*np.pi
SEED = np.array([0., 0., 1.])
RADIAL, TRANSVERSE = .4, .7


def rhs(t, x, alpha, gamma):
    radius = np.hypot(x[0], x[2])
    radial = -RADIAL*(radius-1.)/radius
    return np.array([-x[2]+radial*x[0], -TRANSVERSE*x[1], x[0]+radial*x[2]])


def jacobian(t, x, alpha, gamma):
    plane = x[[0, 2]]
    radius = np.linalg.norm(plane)
    matrix = np.zeros((3, 3))
    matrix[np.ix_([0, 2], [0, 2])] = np.array([[0., -1.], [1., 0.]])-RADIAL*((1.-1./radius)*np.eye(2)+np.outer(plane, plane)/radius**3)
    matrix[1, 1] = -TRANSVERSE
    return matrix


def exact_flow(initial, elapsed):
    elapsed = np.asarray(elapsed)
    radius0 = np.hypot(initial[0], initial[2])
    theta0 = np.arctan2(initial[2], initial[0])
    theta = theta0+elapsed
    radius = 1.+(radius0-1.)*np.exp(-RADIAL*elapsed)
    normal = np.array([np.cos(theta), np.zeros_like(theta), np.sin(theta)])
    tangent = np.array([-np.sin(theta), np.zeros_like(theta), np.cos(theta)])
    normal0 = np.array([np.cos(theta0), 0., np.sin(theta0)])
    tangent0 = np.array([-np.sin(theta0), 0., np.cos(theta0)])
    states = radius*normal
    states[1] = initial[1]*np.exp(-TRANSVERSE*elapsed)
    phi = np.einsum('i...,j,...->...ij', normal, normal0, np.exp(-RADIAL*elapsed))
    phi += np.einsum('i...,j,...->...ij', tangent, tangent0, radius/radius0)
    phi[..., 1, 1] = np.exp(-TRANSVERSE*elapsed)
    return states, phi


def exact_solver(fun, span, initial, *, t_eval, **kwargs):
    origin = float(span[0])
    def dense(query):
        query = np.asarray(query)
        if len(initial) == 12:
            states, phi = exact_flow(initial[:3], query-origin)
            phi = phi@initial[3:].reshape(3, 3)
            result = np.concatenate([states.reshape(3, -1), phi.reshape(-1, 9).T])
        else:
            theta = np.pi/2+query
            normal = np.array([np.cos(theta), np.zeros_like(theta), np.sin(theta)])
            tangent = np.array([-np.sin(theta), np.zeros_like(theta), np.cos(theta)])
            nt = np.array([np.cos(np.pi/2+origin), 0., np.sin(np.pi/2+origin)])
            tt = np.array([-np.sin(np.pi/2+origin), 0., np.cos(np.pi/2+origin)])
            result = ((initial@nt)*np.exp(RADIAL*(query-origin))*normal+(initial@tt)*tangent).reshape(3, -1)
            result[1] = initial[1]*np.exp(TRANSVERSE*(query-origin)).reshape(-1)
        result[:, query.reshape(-1) == origin] = initial[:, None]
        return result[:, 0] if query.ndim == 0 else result.reshape((len(initial),)+query.shape)
    fun(span[0], initial)
    values = dense(t_eval)
    fun(span[1], values[:, -1])
    result = SimpleNamespace(t=t_eval.copy(), y=values, sol=dense, success=True,
        message='analytic stable oscillator', nfev=2, njev=0, nlu=0)
    if 'events' in kwargs:
        te = np.array([0., span[1]])
        result.t_events, result.y_events = [te], [dense(te).T]
    return result


def reference(perturbed=False):
    state = SEED.copy()
    period = PERIOD
    if perturbed:
        state += np.array([2e-8, -3e-8, 4e-8])
        period *= 1.+2e-8
    return dict(status='complete', qualified=True, setting='synthetic', alpha=0., gamma=0., period=period,
        certificate=dict(section_state=state.copy(), period=period), orbit=dict(states=state[None].copy()))


def phases():
    return [dict(phase_index=i, phase_fraction=(i+.37)/24., split='development' if i<8 else 'heldout') for i in range(24)]


def shoots(old):
    return {m:acq.refine_cycle(old, m, rhs=rhs, jacobian=jacobian, solver=exact_solver) for m in acq.METHODS}


@pytest.fixture(scope='module')
def candidates():
    return shoots(reference())


@pytest.fixture(scope='module')
def bundle(candidates):
    original = acq.EXPECTED_PEAKS
    acq.EXPECTED_PEAKS = 1
    try:
        return acq.crossverify_reference(reference(), candidates, phases(), rhs=rhs, jacobian=jacobian, solver=exact_solver)
    finally:
        acq.EXPECTED_PEAKS = original


def test_exact_bordered_jacobian_matches_four_variable_finite_differences():
    old=reference(True);state=old['certificate']['section_state'];period=old['period']
    end,phi=exact_flow(state,period)
    scales=np.maximum(1.,abs(state));scale=max(1.,np.linalg.norm(jacobian(0.,state,0.,0.)[2]*scales))
    metrics=acq.shooting_metrics(state,period,dict(states=np.array([state,end]),phi=np.array([np.eye(3),phi])),
        state,period,scales,scale,0.,0.,rhs=rhs,jacobian=jacobian)
    def residual(unknown):
        return np.r_[exact_flow(unknown[:3],unknown[3])[0]-unknown[:3],rhs(0.,unknown[:3],0.,0.)[2]]
    h=1e-6;unknown=np.r_[state,period]
    numerical=np.column_stack([(residual(unknown+h*e)-residual(unknown-h*e))/(2*h) for e in np.eye(4)])
    np.testing.assert_allclose(metrics['raw_bordered'],numerical,rtol=2e-8,atol=3e-10)
    np.testing.assert_array_equal(metrics['scaled_bordered'],metrics['raw_bordered']*np.r_[scales,period][None,:]/np.r_[scales,scale][:,None])
    assert metrics['raw_bordered'][3,3] == 0.


@pytest.mark.parametrize('method', list(acq.METHODS))
def test_refinement_uses_original_seed_and_preserves_fixed_trial_census(method):
    old=reference(True);saved=copy.deepcopy(old)
    packet=acq.refine_cycle(old,method,rhs=rhs,jacobian=jacobian,solver=exact_solver)
    assert packet['complete'] and packet['converged'] and packet['qualified']
    np.testing.assert_allclose(packet['candidate']['state'],SEED,rtol=0.,atol=1e-12)
    assert abs(packet['candidate']['period']-PERIOD)<1e-12
    np.testing.assert_array_equal(old['orbit']['states'],saved['orbit']['states'])
    assert len(packet['trials'])==25 and len({row['id'] for row in packet['trials']})==25
    assert 1<packet['ivp_evaluations']<=25
    assert packet['section_scale']==max(1.,np.linalg.norm(jacobian(0.,saved['certificate']['section_state'],0.,0.)[2]*packet['state_scales']))
    for row in packet['trials']:
        if row['ivp_attempted']:
            assert row['ivp_started'] and row['ivp_completed']
            assert row['forward']['solver']['method_spec']==acq.method_spec(method,row['period'])
        else:
            assert not row['ivp_started'] and not row['ivp_completed']
            assert row['reason']!='not_reached'


def test_real_scipy_refines_the_analytic_cycle_without_a_target_network():
    packet=acq.refine_cycle(reference(True),acq.PRIMARY,rhs=rhs,jacobian=jacobian)
    assert packet['converged']
    assert packet['candidate']['metrics']['relative_closure']<=1e-12
    assert packet['candidate']['metrics']['section_residual']<=1e-12
    np.testing.assert_allclose(packet['candidate']['state'],SEED,atol=3e-12,rtol=0.)


def test_neighborhood_bound_keeps_unreachable_cycle_unqualified():
    old=reference();old['certificate']['section_state'][2]+=2e-6
    old['orbit']['states'][0]=old['certificate']['section_state']
    packet=acq.refine_cycle(old,acq.PRIMARY,rhs=rhs,jacobian=jacobian,solver=exact_solver)
    assert packet['complete'] and not packet['converged'] and not packet['qualified']
    rejected=[row for row in packet['trials'] if row['status']=='outside_neighborhood']
    assert rejected and all(not row['ivp_started'] for row in rejected)
    assert packet['candidate']['metrics']['seed_scaled_distance']<=acq.STATE_RADIUS
    assert len(packet['iterations'])<=6 and packet['ivp_evaluations']<=25


def test_equilibrium_cannot_pass_via_zero_closure_and_zero_section():
    old=reference();old['certificate']['section_state'][:]=0.;old['orbit']['states'][:]=0.
    def linear_rhs(t,x,a,g):return -x
    def linear_jacobian(t,x,a,g):return -np.eye(3)
    def linear_solver(fun,span,initial,*,t_eval,**kwargs):
        def dense(query):
            query=np.asarray(query);decay=np.exp(-(query-span[0]))
            out=np.concatenate([(initial[:3,None]*decay.reshape(-1)),
                (initial[3:,None]*decay.reshape(-1))])
            return out[:,0] if query.ndim==0 else out.reshape((12,)+query.shape)
        fun(span[0],initial)
        return SimpleNamespace(t=t_eval.copy(),y=dense(t_eval),sol=dense,success=True,
            message='exact stable equilibrium',nfev=1,njev=0,nlu=0)
    packet=acq.refine_cycle(old,acq.PRIMARY,rhs=linear_rhs,jacobian=linear_jacobian,solver=linear_solver)
    metrics=packet['candidate']['metrics']
    assert metrics['relative_closure']==0. and metrics['section_residual']==0.
    assert metrics['scaled_condition'] is None
    assert packet['complete'] and not packet['converged'] and not packet['qualified']
    assert packet['stop_reason']=='singular_or_ill_conditioned_bordered_system'
    assert packet['ivp_evaluations']==1 and not packet['iterations']


def test_repeated_cycle_is_rejected_by_full_state_primitive_return(monkeypatch):
    def repeated_solver(fun,span,initial,*,t_eval,**kwargs):
        out=exact_solver(fun,span,initial,t_eval=t_eval,**kwargs)
        event_times=np.array([0.,PERIOD,2*PERIOD])
        out.t_events=[event_times];out.y_events=[out.sol(event_times).T]
        return out
    forward,_=acq._forward(SEED.copy(),2*PERIOD,0.,0.,acq.PRIMARY,rhs=rhs,
        jacobian=jacobian,solver=repeated_solver,events=True)
    # Isolate primitive return from the independent production nine-peak gate.
    monkeypatch.setattr(acq,'EXPECTED_PEAKS',2)
    certificate=acq._certificate(forward,0.,0.,rhs=rhs,jacobian=jacobian)
    assert certificate['gates']['expected_peak_count'] and certificate['gates']['closure']
    assert certificate['interior_return_distances'].shape==(1,)
    assert certificate['interior_return_distances'][0]<1e-14
    assert not certificate['gates']['primitive'] and not certificate['qualified']


def test_zero_newton_step_does_not_replace_residual_convergence(monkeypatch):
    # An injected stalled linear solve must exhaust the four Armijo factors,
    # never qualify merely because the correction is tiny.
    monkeypatch.setattr(acq.np.linalg,'solve',lambda matrix,value:np.zeros_like(value))
    packet=acq.refine_cycle(reference(True),acq.PRIMARY,rhs=rhs,jacobian=jacobian,solver=exact_solver)
    assert packet['complete'] and not packet['converged'] and not packet['qualified']
    assert packet['stop_reason']=='line_search_no_decrease'
    assert packet['ivp_evaluations']==5 and len(packet['iterations'])==1
    assert packet['iterations'][0]['trial_indices']==[1,2,3,4]
    assert all(not row['accepted'] for row in packet['trials'][1:5])
    assert packet['candidate']['source_trial_index']==0
    assert packet['candidate']['metrics']['relative_closure']>1e-12


def test_exact_six_update_ceiling_retains_unconverged_final_candidate(monkeypatch):
    # Slow but sufficient Armijo decrease exercises every update slot without
    # changing the actual analytic forward trajectory or residual arithmetic.
    original_solve=np.linalg.solve
    monkeypatch.setattr(acq.np.linalg,'solve',lambda matrix,value:1e-3*original_solve(matrix,value))
    packet=acq.refine_cycle(reference(True),acq.PRIMARY,rhs=rhs,jacobian=jacobian,solver=exact_solver)
    assert packet['complete'] and not packet['converged'] and not packet['qualified']
    assert packet['stop_reason']=='maximum_updates'
    assert len(packet['iterations'])==6 and packet['ivp_evaluations']==7
    assert [it['trial_indices'] for it in packet['iterations']]==[[1],[5],[9],[13],[17],[21]]
    assert packet['candidate']['source_trial_index']==21
    assert packet['candidate']['metrics']['relative_closure']>1e-12
    assert all(not row['attempted'] and row['reason']=='line_search_accepted' for row in packet['trials'][22:])


def test_all_four_verifications_precede_two_shared_adjoint_reconstructions(candidates,monkeypatch):
    calls=[]
    def logged(*args,**kwargs):
        calls.append((len(args[2]),kwargs['method']))
        return exact_solver(*args,**kwargs)
    monkeypatch.setattr(acq,'EXPECTED_PEAKS',1)
    packet=acq.crossverify_reference(reference(),candidates,phases(),rhs=rhs,jacobian=jacobian,solver=logged)
    assert packet['complete'] and packet['qualified']
    assert calls==[(12,'DOP853'),(12,'RK45'),(12,'DOP853'),(12,'RK45'),(3,'DOP853'),(3,'RK45')]
    assert len(packet['candidate_method_agreements'])==2
    canonical=packet['canonical'];assert canonical['normalization_count']==1
    assert canonical['source_verification']=='tight-dop853--tight-dop853'
    np.testing.assert_allclose(canonical['terminal_adjoint'],canonical['unnormalized_svd_vector']*canonical['normalization_applied_scale'],atol=0.,rtol=0.)
    for base in packet['bases'].values():
        assert len(base['phases'])==24 and base['new_normalizations']==0
        np.testing.assert_array_equal(base['raw_endpoints']['final_z'],canonical['terminal_adjoint'])
        for key,fine in base['grids']['8192'].items():
            np.testing.assert_array_equal(base['grids']['4096'][key],fine[::2])
    assert all(row['ivp_started'] and row['ivp_completed'] for row in [*packet['verifications'].values(),*packet['adjoints'].values()])


def test_production_peak_gate_rejects_one_peak_cycle(candidates):
    assert acq.EXPECTED_PEAKS==9
    packet=acq.crossverify_reference(reference(),candidates,phases(),rhs=rhs,jacobian=jacobian,solver=exact_solver)
    assert not packet['qualified'] and not packet['complete']
    assert all(row['complete'] and not row['certificate']['gates']['expected_peak_count'] for row in packet['verifications'].values())
    assert 'canonical' not in packet
    assert all(row['status']=='blocked_prerequisite' and not row['ivp_started'] for row in packet['adjoints'].values())


def test_unconverged_primary_cannot_be_replaced_by_alternate(candidates):
    bad=copy.deepcopy(candidates);bad[acq.PRIMARY]['converged']=False
    def forbidden(*args,**kwargs):pytest.fail('invalid prerequisite launched an IVP')
    with pytest.raises(ValueError,match='converged'):
        acq.crossverify_reference(reference(),bad,phases(),rhs=rhs,jacobian=jacobian,solver=forbidden)


def test_event_uses_exact_phase_start_fresh_identity_and_physical_epoch(bundle):
    assert bundle['qualified']
    base=bundle['bases'][acq.PRIMARY];phase=base['phases'][11];times=[]
    def timed_rhs(t,x,a,g):
        times.append(t);return rhs(t,x,a,g)
    event=acq.acquire_refined_event(base,11,rhs=timed_rhs,jacobian=jacobian,solver=exact_solver)
    assert event['complete'] and event['ivp_started'] and event['ivp_completed']
    assert event['physical_event_time']==phase['phase_epoch']+base['period']
    assert min(times)==phase['phase_epoch'] and max(times)==event['physical_event_time']
    np.testing.assert_array_equal(event['initial'],phase['state'])
    np.testing.assert_array_equal(event['trajectory']['phi'][0],np.eye(3))
    np.testing.assert_array_equal(event['event']['phase_gradient'],phase['z'])
    assert event['new_normalizations']==0
    assert abs(event['event']['normalization_residual'])<1e-12


def test_event_does_not_repair_a_scaled_phase_gradient(bundle):
    base=copy.deepcopy(bundle['bases'][acq.PRIMARY]);index=4
    base['phases'][index]['z']*=1.01;base['exact_phase_arrays']['z'][index]*=1.01
    packet=acq.acquire_refined_event(base,index,rhs=rhs,jacobian=jacobian,solver=exact_solver)
    assert packet['complete']
    assert packet['event']['normalization_residual']==pytest.approx(.01,abs=1e-12)


def exploding_solver(fun,span,initial,**kwargs):
    fun(span[0],initial)
    raise RuntimeError('synthetic interruption')


def test_initial_solver_failure_keeps_all_slots_and_last_rhs():
    with pytest.raises(AcquisitionFailure) as caught:
        acq.refine_cycle(reference(),acq.PRIMARY,rhs=rhs,jacobian=jacobian,solver=exploding_solver)
    packet=caught.value.payload;row=packet['trials'][0]
    assert not packet['complete'] and len(packet['trials'])==25
    assert row['ivp_started'] and not row['ivp_completed']
    assert row['partial_failure']['partial_failure']['last_rhs']['time']==0.
    assert all(not row['ivp_started'] for row in packet['trials'][1:])


def test_one_verification_failure_keeps_other_three_and_blocks_adjoints(candidates,monkeypatch):
    count=0
    def once(fun,span,initial,**kwargs):
        nonlocal count
        count+=1
        return exploding_solver(fun,span,initial,**kwargs) if count==2 else exact_solver(fun,span,initial,**kwargs)
    monkeypatch.setattr(acq,'EXPECTED_PEAKS',1)
    packet=acq.crossverify_reference(reference(),candidates,phases(),rhs=rhs,jacobian=jacobian,solver=once)
    assert count==4 and not packet['complete']
    assert sum(row['complete'] for row in packet['verifications'].values())==3
    failed=packet['verifications']['tight-dop853--tight-rk45']
    assert failed['ivp_started'] and not failed['ivp_completed'] and 'partial_failure' in failed
    assert not any(row['ivp_started'] for row in packet['adjoints'].values())


def test_event_failure_retains_phase_binding_and_partial_rhs(bundle):
    base=bundle['bases'][acq.PRIMARY]
    with pytest.raises(AcquisitionFailure) as caught:
        acq.acquire_refined_event(base,8,rhs=rhs,jacobian=jacobian,solver=exploding_solver)
    packet=caught.value.payload
    np.testing.assert_array_equal(packet['initial'],base['phases'][8]['state'])
    np.testing.assert_array_equal(packet['phase_gradient'],base['phases'][8]['z'])
    assert packet['ivp_started'] and not packet['ivp_completed']
    assert packet['partial_failure']['partial_failure']['last_rhs']['physical_time']==base['phases'][8]['phase_epoch']


def test_completed_solver_flag_survives_derivative_reconstruction_failure(monkeypatch):
    def fail(*args,**kwargs):raise ValueError('synthetic derivative failure')
    monkeypatch.setattr(acq.prior,'_derivatives',fail)
    with pytest.raises(AcquisitionFailure) as caught:
        acq.refine_cycle(reference(),acq.PRIMARY,rhs=rhs,jacobian=jacobian,solver=exact_solver)
    row=caught.value.payload['trials'][0]
    assert row['ivp_started'] and row['ivp_completed']
    assert row['partial_failure']['raw_solver']['values'].shape==(8193,12)


@pytest.mark.parametrize('mutation', ['period','state','status'])
def test_seed_metadata_rejected_before_solver(mutation):
    old=reference()
    if mutation=='period':old['certificate']['period']+=1e-9
    elif mutation=='state':old['orbit']['states'][0,0]=1e-9
    else:old['qualified']=False
    with pytest.raises(ValueError):
        acq.refine_cycle(old,acq.PRIMARY,rhs=rhs,jacobian=jacobian,solver=exploding_solver)


@pytest.mark.parametrize('method', ['DOP853','fine-dop853','tight-other'])
def test_only_prospectively_fixed_methods_are_accepted(method):
    with pytest.raises(ValueError,match='Unknown'):
        acq.method_spec(method,PERIOD)
