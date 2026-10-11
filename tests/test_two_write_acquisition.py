"""Analytic/synthetic only: no Hopfield trajectory or historical arrays."""
from pathlib import Path
import copy
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import two_write_acquisition as acq
from phase_memory_acquisition import AcquisitionFailure


def flow(t, x, a, g): return np.array([1., 0., 0.])
def jac(t, x, a, g): return np.zeros((3, 3))


def context(period=37.33149958853496):
    return dict(id='synthetic--p00', setting='synthetic', period=period,
        alpha=0., gamma=0., initial=np.zeros(3), phase_gradient=np.array([1., 0., 0.]),
        flow=np.array([1., 0., 0.]), direction=np.array([0., 1., 0.]),
        nominal_event_phase_gradient=np.array([1., 0., 0.]))


def exact_drift_solver(fun, span, initial, *, t_eval, **kwargs):
    times=t_eval.copy(); values=np.repeat(initial[:, None], len(times), axis=1)
    values[0]+=times
    fun(0., initial)
    fun(float(times[-1]), values[:, -1])
    return SimpleNamespace(t=times, y=values, success=True, message='analytic drift', nfev=2, njev=0, nlu=0)


def native_drift(alpha, gamma, initial, dt, steps, stride, count, output, final, progress):
    x=np.ctypeslib.as_array(initial, shape=(3,)).copy()
    states=np.ctypeslib.as_array(output, shape=(count*3,)).reshape(count, 3)
    states[:]=x; states[:, 0]+=np.arange(count)*(stride*dt)
    np.ctypeslib.as_array(final, shape=(3,))[:]=states[-1]
    np.ctypeslib.as_array(progress, shape=(2,))[:]=[steps, count]
    return count


def sealed(period=37.33149958853496):
    c=context(period)
    reference=acq.acquire_tangent(c, rhs=flow, jacobian=jac, solver=exact_drift_solver)
    return acq.seal_context(c, reference, rhs=flow), reference


def task(method='rk4-fine', first=1, second=-1, arm='identity'):
    return dict(id='synthetic-trace', context_id='synthetic--p00', epsilon=1e-4,
                first_sign=first, second_sign=second, arm=arm, method=method)


@pytest.fixture
def short_grid(monkeypatch):
    # Keep all 32 immediate midpoint samples aligned; solver settings unchanged.
    monkeypatch.setattr(acq, 'SAMPLES_PER_CYCLE', 512)


@pytest.mark.parametrize('period', [37.33149958853496, 37.63637538160568, .7])
def test_period_locked_grid_and_native_exact_endpoints(period):
    grid=acq.build_grid(period)
    assert grid['pre'].shape==(4097,) and grid['post'].shape==(28673,)
    assert grid['pre'][-1]==period and grid['post'][-1]==7*period
    assert np.all(np.diff(grid['post'])>0)
    for steps in [32768,65536]:
        for key in ['pre','post']:
            times, states=acq.integrate_trace(native_drift,0.,0.,np.zeros(3),period/steps,
                duration=float(grid[key][-1]),sample_dt=period/4096)
            assert np.array_equal(times,grid[key])
            assert np.array_equal(states[0],np.zeros(3))


@pytest.mark.parametrize('method', ['dop853','rk45'])
def test_direct_tangent_reset_and_offsets(short_grid, method):
    c=context(); calls=[]
    def spy(fun, span, initial, **kw):
        calls.append((span,initial.copy(),kw.copy()))
        return exact_drift_solver(fun,span,initial,**kw)
    result=acq.acquire_tangent(c,method,rhs=flow,jacobian=jac,solver=spy)
    assert result['complete'] and len(calls)==2
    assert np.array_equal(calls[0][1][3:].reshape(3,3),np.eye(3))
    assert np.array_equal(calls[1][1][3:].reshape(3,3),np.eye(3))
    assert calls[1][1][0]==c['period']
    assert result['post']['offset']==c['period']
    assert np.array_equal(result['pre']['states'][-1],result['post']['states'][0])
    assert np.array_equal(result['post']['phi'][-1],np.eye(3))
    assert calls[0][2]['method']==acq.TANGENT_METHODS[method]['kind']
    assert calls[0][2]['max_step']==c['period']/acq.TANGENT_METHODS[method]['max_step_period_divisor']


@pytest.mark.parametrize('method',list(acq.METHODS))
@pytest.mark.parametrize('arm,expected', [('identity',0.),('timing',-1e-4),('transverse',0.)])
def test_exact_retain_then_second_write(short_grid,method,arm,expected):
    c,reference=sealed()
    result=acq.acquire_trace(task(method,arm=arm),c,native_fn=native_drift,rhs=flow,solver=exact_drift_solver)
    assert result['complete'] and result['event']['time']==c['period']
    event=result['event']; assert event['before'][1]==1e-4
    assert event['after'][1]==expected
    assert np.array_equal(result['post']['states'][0],event['after'])
    assert np.array_equal(result['pre']['states'][-1],event['before'])
    if arm=='identity': assert np.array_equal(event['before'],event['after_retention'])
    assert np.array_equal(event['second_pulse']['realized_delta'],event['after']-event['after_retention'])
    assert event['operation_order']=='retention_then_second_write'


def test_absent_writes_and_readout_nodes(short_grid):
    c,ref=sealed(); result=acq.acquire_trace(task(first=0,second=0),c,native_fn=native_drift,rhs=flow)
    assert np.array_equal(result['first_pulse']['requested_delta'],np.zeros(3))
    assert np.array_equal(result['event']['second_pulse']['requested_delta'],np.zeros(3))
    for window in ['immediate','k2','k4','k8']:
        data=acq.readout_samples(result,window)
        assert data['states'].shape==(32,3)
        assert np.array_equal(data['activations'],np.tanh(data['states']))
        assert np.array_equal(data['states'],result[data['segment']]['states'][data['indices']])
        assert np.all(data['times']<c['period']/8) if window=='immediate' else np.all(data['times']>c['period'])
    assert acq.readout_samples(result,'immediate')['indices'][0]>0
    assert acq.readout_samples(result,'k2')['indices'][0]>0


def test_event_uses_unrepaired_physical_projector(short_grid):
    c=context();c['nominal_event_phase_gradient']=np.array([1.+1e-8,0.,0.])
    primary=acq.acquire_tangent(c,rhs=flow,jacobian=jac,solver=exact_drift_solver)
    q=acq.seal_context(c,primary,rhs=flow)
    assert q['Q'][0,0]==1.+1e-8
    assert q['event_normalization_residual']!=0
    assert np.max(np.abs(q['event_projector_idempotence_residual']))>0
    assert 'Q' not in c


def test_reference_template_context_preserves_phase_not_rounded_to_old_eight():
    period=2*np.pi;times=np.linspace(0,period,4097)
    states=np.column_stack([np.sin(times),np.cos(times),np.zeros_like(times)])
    derivatives=np.column_stack([np.cos(times),-np.sin(times),np.zeros_like(times)])
    ref=dict(status='complete',qualified=True,setting='synthetic',alpha=0.,gamma=0.,period=period,
        orbit=dict(times=times,states=states,derivatives=derivatives),
        adjoint=dict(times=times,values=derivatives,derivatives=-states),
        certificate=dict(transverse_multipliers=[dict(real=-.8,imag=0.,modulus=.8),dict(real=.1,imag=0.,modulus=.1)]))
    def circle(t,x,a,g): return np.array([x[1],-x[0],0.])
    result=acq.prepare_context(ref,.23456,phase_index=10,rhs=circle)
    assert result['id']=='synthetic--p10' and result['split']=='heldout'
    assert result['phase_epoch']==.23456*period and result['rho']==-.8
    assert abs(result['phase_gradient']@result['direction'])<1e-12
    assert np.isclose(np.linalg.norm(result['direction']),1.)
    assert np.allclose(result['initial'],[np.sin(result['phase_epoch']),np.cos(result['phase_epoch']),0.],atol=1e-12)
    assert result['provenance']['reference_qualified'] is True


def test_post_solver_failure_retains_pre_and_event(short_grid):
    c,ref=sealed();calls=0
    def failing(fun,span,initial,**kw):
        nonlocal calls
        calls+=1
        if calls==2:
            fun(0.,initial)
            raise RuntimeError('synthetic failure after event')
        return exact_drift_solver(fun,span,initial,**kw)
    with pytest.raises(AcquisitionFailure) as found:
        acq.acquire_trace(task('dop853-independent'),c,rhs=flow,solver=failing)
    packet=found.value.payload
    assert packet['failed_stage']=='post' and 'pre' in packet and 'event' in packet
    assert packet['partial_failure']['last_rhs']['accepted_solver_state'] is False
    assert packet['partial_failure']['last_rhs']['global_time']==c['period']


def test_incomplete_tangent_retains_returned_arrays(short_grid):
    def incomplete(fun,span,initial,**kw):
        value=exact_drift_solver(fun,span,initial,**kw)
        value.t=value.t[:3];value.y=value.y[:,:3];value.success=False
        return value
    with pytest.raises(AcquisitionFailure) as found:
        acq.acquire_tangent(context(),rhs=flow,jacobian=jac,solver=incomplete)
    part=found.value.payload['partial_failure']
    assert part['values'].shape==(3,12) and part['times'].shape==(3,)
    assert found.value.payload['failed_stage']=='pre'


def test_native_failure_retains_progress(short_grid):
    c,ref=sealed()
    def fail_native(alpha,gamma,initial,dt,steps,stride,count,output,final,progress):
        native_drift(alpha,gamma,initial,dt,steps,stride,count,output,final,progress)
        return -2
    with pytest.raises(AcquisitionFailure) as found:
        acq.acquire_trace(task(),c,native_fn=fail_native,rhs=flow)
    part=found.value.payload['partial_failure']
    assert part['native_status']==-2 and len(part['progress'])==2
    assert part['states'].shape[1]==3


@pytest.mark.parametrize('edit',[
    lambda t:t.update(first_sign=True),lambda t:t.update(second_sign=2),
    lambda t:t.update(epsilon=0.),lambda t:t.update(arm='orthogonal'),
    lambda t:t.update(method='nearest-event'),lambda t:t.update(context_id='wrong')])
def test_invalid_task_never_calls_solver(short_grid,edit):
    c,ref=sealed(); spec=task();edit(spec)
    def forbidden(*a,**kw): raise AssertionError('solver called')
    with pytest.raises(ValueError): acq.acquire_trace(spec,c,native_fn=forbidden,rhs=flow,solver=forbidden)


def test_tampered_projector_rejected_before_solver(short_grid):
    c,ref=sealed();c['Q'][0,0]=.9
    with pytest.raises(ValueError,match='projectors'):
        acq.acquire_trace(task(),c,native_fn=native_drift,rhs=flow)


def test_seal_rejects_nonidentity_second_tangent(short_grid):
    c=context();r=acq.acquire_tangent(c,rhs=flow,jacobian=jac,solver=exact_drift_solver)
    r['post']['phi'][0,0,0]=2.
    with pytest.raises(ValueError,match='identity'):acq.seal_context(c,r,rhs=flow)


def test_readout_does_not_cross_jump(short_grid):
    c,ref=sealed();r=acq.acquire_trace(task(first=1,second=1),c,native_fn=native_drift,rhs=flow)
    assert np.all(acq.readout_samples(r,'immediate')['states'][:,1]==1e-4)
    assert np.all(acq.readout_samples(r,'k2')['states'][:,1]==2e-4)


def test_grid_failure_preserves_accepted_trace(short_grid,monkeypatch):
    c,ref=sealed()
    original=acq.integrate_trace
    def invalid_grid(*args,**kwargs):
        times,states=original(*args,**kwargs);times=times.copy();times[1]+=1e-5
        return times,states
    monkeypatch.setattr(acq,'integrate_trace',invalid_grid)
    with pytest.raises(AcquisitionFailure) as found:
        acq.acquire_trace(task(),c,native_fn=native_drift,rhs=flow)
    part=found.value.payload['partial_failure']
    assert part['states'].shape==(513,3) and part['times'].shape==(513,)


def test_reference_derivative_failure_preserves_accepted_arrays(short_grid):
    def bad_rhs(t,x,a,g): raise RuntimeError('reference derivative failed')
    def no_rhs_solver(fun,span,initial,*,t_eval,**kw):
        # Analytic payload returned without invoking RHS; failure occurs while
        # reconstructing the derivative arrays of the accepted reference.
        values=np.repeat(initial[:,None],len(t_eval),axis=1)
        return SimpleNamespace(t=t_eval.copy(),y=values,success=True,message='synthetic',nfev=0,njev=0,nlu=0)
    with pytest.raises(AcquisitionFailure) as found:
        acq.acquire_tangent(context(),rhs=bad_rhs,jacobian=jac,solver=no_rhs_solver)
    part=found.value.payload['partial_failure']
    assert part['states'].shape==(513,3) and part['phi'].shape==(513,3,3)


def test_native_trace_has_no_redundant_rhs_or_derivative_storage(short_grid):
    c,ref=sealed()
    def forbidden(*args):raise AssertionError('native trace should not reconstruct RHS')
    result=acq.acquire_trace(task(),c,native_fn=native_drift,rhs=forbidden)
    assert 'derivatives' not in result['pre'] and 'derivatives' not in result['post']


def test_real_scipy_on_analytic_linear_system(short_grid,monkeypatch):
    # A real ODE solve of an analytically soluble non-neural linear system.
    monkeypatch.setitem(acq.TANGENT_METHODS,'dop853',dict(kind='DOP853',rtol=1e-12,atol=1e-14,max_step_period_divisor=8))
    matrix=np.diag([0.,-.3,-.7]);c=context(.2);c['initial']=np.array([1.,2.,3.])
    def linear(t,x,a,g):return matrix@x
    def linear_jac(t,x,a,g):return matrix
    r=acq.acquire_tangent(c,rhs=linear,jacobian=linear_jac)
    for name in ['pre','post']:
        segment=r[name];global_times=segment['times']+segment['offset']
        expected=c['initial'][None,:]*np.exp(global_times[:,None]*np.diag(matrix))
        expected_phi=np.eye(3)[None,:,:]*np.exp(segment['times'][:,None,None]*np.diag(matrix)[None,None,:])
        assert np.allclose(segment['states'],expected,rtol=1e-11,atol=1e-13)
        assert np.allclose(segment['phi'],expected_phi,rtol=1e-11,atol=1e-13)
