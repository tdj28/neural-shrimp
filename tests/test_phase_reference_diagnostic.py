"""Analytic circle and injected solver failures only; no historical/target data."""
from pathlib import Path
import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import phase_reference_diagnostic as acq
from phase_memory_acquisition import AcquisitionFailure

J = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]])


def rhs(t,x,a,g): return J@x
def jac(t,x,a,g): return J.copy()


def rotation(t):
    c,s=np.cos(t),np.sin(t)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def phases():
    return [dict(phase_index=i,phase_fraction=(i+.37)/24.,
        split='development' if i<8 else 'heldout') for i in range(24)]


def reference(zscale=1.):
    period=2*np.pi; times=acq.build_grids(period)['4096']
    states=np.column_stack([np.cos(times),np.sin(times),np.zeros_like(times)])
    flow=states@J.T;z=zscale*flow
    # The inherited terminal vector is a fixed boundary, not recomputed here.
    terminal=np.array([0.,zscale,0.]);z[-1]=terminal
    return dict(status='complete',qualified=True,setting='synthetic',alpha=0.,gamma=0.,period=period,
        certificate=dict(section_state=np.array([1.,0.,0.]),period=period),
        orbit=dict(times=times,states=states,derivatives=flow),
        adjoint=dict(times=times,values=z,derivatives=z@J.T,
            terminal_normalized_once=terminal,normalization_count=1))


def circle_solver(fun,span,initial,*,t_eval,**kw):
    assert kw['dense_output'] is True
    def dense(t):
        query=np.asarray(t);flat=query.reshape(-1);result=[]
        for local in flat:
            r=rotation(local-span[0])
            result.append(np.r_[r@initial[:3],(r@initial[3:].reshape(3,3)).ravel()]
                if len(initial)==12 else r@initial)
        values=np.asarray(result).T
        return values[:,0] if query.ndim==0 else values.reshape((len(initial),)+query.shape)
    y=dense(t_eval);y[:,0]=initial
    fun(float(span[0]),initial);fun(float(span[1]),y[:,-1])
    return SimpleNamespace(t=t_eval.copy(),y=y,sol=dense,success=True,
        message='analytic circle',nfev=2,njev=0,nlu=0)


@pytest.fixture(scope='module')
def base():
    return acq.acquire_base(reference(),phases(),'fine-dop853',rhs=rhs,jacobian=jac,solver=circle_solver)


@pytest.mark.parametrize('period',[37.33149958853496,37.63637538160568,2*np.pi,.7])
def test_exact_nested_grids_and_distinct_specs(period):
    grid=acq.build_grids(period)
    assert len(grid['4096'])==4097 and len(grid['8192'])==8193
    assert np.array_equal(grid['4096'],grid['8192'][::2])
    assert grid['8192'][0]==0. and grid['8192'][-1]==period
    assert np.all(np.diff(grid['8192'])>0)
    assert acq.method_spec('baseline-dop853',period)['max_step']==.025
    assert acq.method_spec('fine-dop853',period)['max_step']==period/8192
    assert acq.method_spec('fine-rk45',period)['kind']=='RK45'


@pytest.mark.parametrize('method',list(acq.METHODS))
def test_two_dense_solves_forward_then_backward_fixed_terminal(method):
    old=reference(1.03);calls=[]
    def spy(fun,span,initial,**kw):
        calls.append((span,initial.copy(),kw.copy()))
        return circle_solver(fun,span,initial,**kw)
    result=acq.acquire_base(old,phases(),method,rhs=rhs,jacobian=jac,solver=spy)
    assert result['complete'] and result['status']=='complete' and len(calls)==2
    assert result['id']=='synthetic--'+method
    assert calls[0][0]==(0.,2*np.pi) and calls[1][0]==(2*np.pi,0.)
    assert np.array_equal(calls[0][1][:3],old['certificate']['section_state'])
    assert np.array_equal(calls[0][1][3:].reshape(3,3),np.eye(3))
    assert np.array_equal(calls[1][1],old['adjoint']['terminal_normalized_once'])
    assert np.array_equal(result['raw_endpoints']['final_z'],calls[1][1])
    assert result['inherited']['new_normalizations']==0
    for _,_,kw in calls:
        assert kw['method']==acq.METHODS[method]['kind']
        assert kw['rtol']==1e-12 and kw['atol']==1e-14
        assert kw['max_step']==result['method_spec']['max_step']
    # Deliberately wrong normalization is retained, never repaired or gated.
    assert np.allclose(result['grids']['8192']['normalization_residuals'],.03,atol=2e-15)
    assert len(result['phases'])==24


def test_actual_adaptive_circle_normalization_and_direct_phase_samples():
    old=reference()
    result=acq.acquire_base(old,phases(),'baseline-dop853',rhs=rhs,jacobian=jac)
    assert result['complete']
    fine=result['grids']['8192']
    assert np.max(np.abs(fine['normalization_residuals']))<2e-12
    assert np.max(np.abs(fine['phi']-np.asarray([rotation(t) for t in fine['times']])))<2e-12
    assert np.array_equal(result['raw_endpoints']['final_z'],old['adjoint']['terminal_normalized_once'])
    for point in result['phases']:
        assert np.allclose(point['state'],[np.cos(point['phase_epoch']),np.sin(point['phase_epoch']),0.],atol=2e-12)
        assert np.isclose(point['z']@point['flow'],1.,atol=2e-12)


def test_phase_grid_sampling_old_template_and_raw_endpoint_distinguished(base):
    p=base['phases'][7];t=p['phase_epoch']
    assert not np.any(base['grids']['8192']['times']==t)
    assert np.array_equal(p['state'],base['exact_phase_arrays']['states'][7])
    assert np.allclose(p['state'],rotation(t)@np.array([1.,0.,0.]),atol=1e-15)
    assert p['old_hermite']['state_derivative'].shape==(3,)
    assert base['raw_endpoints']['final_state'][1]!=0.  # No periodic endpoint overwrite.
    assert not np.array_equal(base['raw_endpoints']['initial_state'],base['raw_endpoints']['final_state'])


def test_old_hermite_off_orbit_perturbation_does_not_change_new_dense_reference():
    old=reference();old['orbit']['states'][1:-1]*=1.001
    result=acq.acquire_base(old,phases(),'fine-dop853',rhs=rhs,jacobian=jac,solver=circle_solver)
    for p in result['phases']:
        assert abs(p['normalization_residual'])<1e-14
        assert abs(p['old_hermite']['normalization_residual'])>1e-4
        assert np.linalg.norm(p['state']-p['old_hermite']['state'])>1e-4


def test_event_uses_own_dense_start_and_fresh_tangent_with_physical_clock(base):
    observed=[]
    def tracked_rhs(t,x,a,g):
        observed.append(t);return rhs(t,x,a,g)
    packet=acq.acquire_event(base,20,rhs=tracked_rhs,jacobian=jac,solver=circle_solver)
    phase=base['phases'][20];tr=packet['trajectory'];event=packet['event']
    assert packet['complete'] and packet['id']=='synthetic--p20--fine-dop853'
    assert packet['base_id']==base['id'] and packet['event_time']==base['period']
    assert min(observed)==phase['phase_epoch']
    assert max(observed)==phase['phase_epoch']+base['period']
    assert np.array_equal(packet['initial'],phase['state'])
    assert np.array_equal(tr['states'][0],phase['state'])
    assert np.array_equal(tr['phi'][0],np.eye(3))
    assert np.array_equal(event['phase_gradient'],phase['z'])
    assert np.array_equal(event['Q'],np.outer(event['flow'],phase['z']))
    assert np.array_equal(event['P'],np.eye(3)-event['Q'])
    assert np.max(np.abs(event['return_difference']))<2e-15
    assert abs(event['normalization_residual'])<2e-15


def test_unqualified_normalization_does_not_skip_any_event_or_repair_z():
    wrong=acq.acquire_base(reference(1.03),phases(),'fine-rk45',rhs=rhs,jacobian=jac,solver=circle_solver)
    packet=acq.acquire_event(wrong,23,rhs=rhs,jacobian=jac,solver=circle_solver)
    assert packet['complete']
    assert packet['method']=='fine-rk45'
    assert np.isclose(packet['event']['normalization_residual'],.03,atol=2e-15)
    assert np.max(np.abs(packet['event']['Q_idempotence_residual']))>1e-3
    assert np.array_equal(packet['phase_gradient'],wrong['phases'][23]['z'])


@pytest.mark.parametrize('mutation',[
    lambda r:r['certificate'].update(period=7.),
    lambda r:r['certificate'].update(section_state=np.zeros(3)),
    lambda r:r['adjoint'].update(terminal_normalized_once=np.ones(3)),
    lambda r:r['adjoint'].update(normalization_count=2),
    lambda r:r.update(qualified=False),
])
def test_inherited_binding_error_never_calls_solver(mutation):
    old=reference();mutation(old)
    def forbidden(*a,**kw):raise AssertionError('solver called')
    with pytest.raises(ValueError):acq.acquire_base(old,phases(),'fine-dop853',rhs=rhs,jacobian=jac,solver=forbidden)


@pytest.mark.parametrize('mutation',[
    lambda p:p.pop(),lambda p:p[0].update(phase_index=True),
    lambda p:p[1].update(phase_index=0),lambda p:p[0].update(phase_fraction=1.),
    lambda p:p[0].update(phase_fraction=float('nan')),lambda p:p[8].update(split='development'),
])
def test_phase_census_rejected_before_solver(mutation):
    rows=phases();mutation(rows)
    def forbidden(*a,**kw):raise AssertionError('solver called')
    with pytest.raises(ValueError):acq.acquire_base(reference(),rows,'fine-dop853',rhs=rhs,jacobian=jac,solver=forbidden)


@pytest.mark.parametrize('mutation',[
    lambda b:b.update(complete=False),lambda b:b.update(id='wrong'),
    lambda b:b['method_spec'].update(rtol=1e-8),lambda b:b['phases'][0].update(state=np.zeros(3)),
    lambda b:b['phases'][0].update(z=np.ones(3)),lambda b:b['phases'][0].update(phase_epoch=.1),
])
def test_event_binding_error_never_calls_solver(base,mutation):
    packet=copy.deepcopy(base);mutation(packet)
    def forbidden(*a,**kw):raise AssertionError('solver called')
    with pytest.raises(ValueError):acq.acquire_event(packet,0,rhs=rhs,jacobian=jac,solver=forbidden)


def test_solver_exception_retains_unaccepted_stage_and_fixed_boundary():
    calls=0
    def broken(fun,span,initial,**kw):
        nonlocal calls
        calls+=1
        if calls==2:
            fun(span[0],initial);raise RuntimeError('synthetic backward failure')
        return circle_solver(fun,span,initial,**kw)
    with pytest.raises(AcquisitionFailure) as found:
        acq.acquire_base(reference(),phases(),'fine-rk45',rhs=rhs,jacobian=jac,solver=broken)
    result=found.value.payload
    assert result['failed_stage']=='backward' and not result['complete']
    assert result['grids']['8192']['states'].shape==(8193,3)
    assert result['partial_failure']['last_rhs']['accepted_solver_state'] is False
    assert np.array_equal(result['partial_failure']['initial'],result['inherited']['terminal_adjoint'])


def test_returned_partial_forward_trajectory_is_preserved():
    def partial(fun,span,initial,**kw):
        answer=circle_solver(fun,span,initial,**kw)
        answer.t=answer.t[:17];answer.y=answer.y[:,:17];answer.success=False
        return answer
    with pytest.raises(AcquisitionFailure) as found:
        acq.acquire_base(reference(),phases(),'fine-rk45',rhs=rhs,jacobian=jac,solver=partial)
    part=found.value.payload['partial_failure']
    assert part['values'].shape==(17,12) and part['times'].shape==(17,)
    assert found.value.payload['failed_stage']=='forward'


def test_dense_phase_failure_retains_both_completed_grids():
    calls=0
    def broken(fun,span,initial,**kw):
        nonlocal calls
        calls+=1;answer=circle_solver(fun,span,initial,**kw)
        if calls==2:
            def dense_error(*a):raise RuntimeError('synthetic phase output failure')
            answer.sol=dense_error
        return answer
    with pytest.raises(AcquisitionFailure) as found:
        acq.acquire_base(reference(),phases(),'fine-rk45',rhs=rhs,jacobian=jac,solver=broken)
    result=found.value.payload
    assert result['failed_stage']=='exact_phases'
    assert set(result['grids'])=={'4096','8192'}
    assert result['grids']['8192']['z'].shape==(8193,3)
    assert result['exact_phase_arrays']['states'].shape==(24,3)


def test_nonfinite_derivative_retains_accepted_solver_values():
    calls=0
    def fail_rhs(t,x,a,g):
        nonlocal calls
        calls+=1
        return rhs(t,x,a,g) if calls<=2 else np.full(3,np.nan)
    with pytest.raises(AcquisitionFailure) as found:
        acq.acquire_base(reference(),phases(),'fine-rk45',rhs=fail_rhs,jacobian=jac,solver=circle_solver)
    result=found.value.payload
    assert result['failed_stage']=='grid_derivatives'
    assert result['forward']['values'].shape==(8193,12)
    assert np.isnan(result['grids']['8192']['derivatives']).any()


def test_normalization_uses_scalar_dot_arithmetic_under_cancellation():
    z=np.array([22948.355843495923,9813.03225440923,2320.4609674939607])
    flow=np.array([-15302.190785720848,-22248.802510455473,0.])
    flow[2]=(1.-z[0]*flow[0]-z[1]*flow[1])/z[2]
    data=dict(times=np.array([0.,1.]),states=np.zeros((2,3)),
        phi=np.repeat(np.eye(3)[None],2,axis=0),z=np.repeat(z[None],2,axis=0))
    acq._derivatives(data,0.,0.,lambda *a:flow.copy(),lambda *a:np.zeros((3,3)))
    expected=np.array([float(z@flow-1.)]*2)
    assert np.array_equal(data['normalization_residuals'],expected)
    assert np.array_equal(data['z'],np.repeat(z[None],2,axis=0))


def test_event_partial_failure_has_causal_offset_and_no_made_up_endpoint(base):
    def broken(fun,span,initial,**kw):
        fun(.3,initial);raise RuntimeError('synthetic event failure')
    with pytest.raises(AcquisitionFailure) as found:
        acq.acquire_event(base,8,rhs=rhs,jacobian=jac,solver=broken)
    result=found.value.payload
    assert result['failed_stage']=='forward' and 'event' not in result
    assert result['partial_failure']['last_rhs']['physical_time']==base['phases'][8]['phase_epoch']+.3
    assert np.array_equal(result['initial'],base['phases'][8]['state'])
