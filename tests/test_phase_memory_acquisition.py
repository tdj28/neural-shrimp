"""Draft acquisition tests: synthetic metadata, mocked traces, analytic toy ODE.

The only integrated ODE is a one-unit attracting analytic circular oscillator.
The Hopfield RHS/Jacobian/certificate are replaced before those tests. No
historical trajectories, Hopfield solves, native builds or pulse jobs run.
"""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

CHECKPOINT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHECKPOINT/'tools'))
import phase_memory_acquisition as acquisition


def seed(setting='core'):
    source = acquisition.SETTINGS[setting]
    return dict(setting=setting, **{k:v for k,v in source.items() if k != 'section_state'},
                section_state=np.asarray(source['section_state'], dtype=np.float64))


def certificate(setting='core', period=1.):
    return dict(alpha=acquisition.SETTINGS[setting]['alpha'], gamma=acquisition.SETTINGS[setting]['gamma'],
        verified=True, attracting=True, primitive=True, expected_peak_count=9, peak_count=9,
        period=period, section_state=[1.,0.,0.], gates={k:True for k in acquisition.CERTIFICATE_GATES})


def toy_rhs(t, x, alpha, gamma):
    # Analytic attracting unit circle, period one, transverse rates -2 and -1.
    a,b,c = x; radial = 1-a*a-b*b; omega=2*np.pi
    return np.array([-omega*b+a*radial, omega*a+b*radial, -c])


def toy_jacobian(t, x, alpha, gamma):
    a,b,_ = x; radial=1-a*a-b*b; omega=2*np.pi
    return np.array([[radial-2*a*a,-omega-2*a*b,0.],
                     [omega-2*a*b,radial-2*b*b,0.],[0.,0.,-1.]])


def patch_toy(monkeypatch):
    monkeypatch.setattr(acquisition.diagnostics, 'hopfield_rhs', toy_rhs)
    monkeypatch.setattr(acquisition.diagnostics, 'hopfield_jacobian', toy_jacobian)
    monkeypatch.setattr(acquisition.diagnostics, 'certify', lambda *args,**kw: certificate())


@pytest.fixture(scope='module')
def toy_reference():
    with pytest.MonkeyPatch.context() as monkeypatch:
        patch_toy(monkeypatch)
        return acquisition.acquire_reference(seed())


def test_exact_census_and_disjoint_methods():
    tasks=acquisition.build_tasks()
    assert len(tasks)==len({t['id'] for t in tasks})==912
    assert Counter(t['method'] for t in tasks)==dict.fromkeys(acquisition.METHODS,304)
    assert Counter(t['kind'] for t in tasks)=={'pulse':864,'sham':48}
    assert Counter(t['setting'] for t in tasks)==dict.fromkeys(acquisition.SETTINGS,456)
    assert tasks[0]['id']=='core--p0--d0--phase_write--s-1--rk4-0.0025'
    assert tasks[-1]['id']=='failed-near-edge--p7--sham--dop853'
    for method in acquisition.METHODS:
        per=[t for t in tasks if t['method']==method]
        assert sum(t['kind']=='pulse' for t in per)==288
        assert sum(t['kind']=='sham' for t in per)==16
        assert all(t['epsilon'] is None and t['direction'] is None and t['sign'] is None
                   for t in per if t['kind']=='sham')
    assert len(acquisition.TIMES)==12801 and acquisition.TIMES[-1]==320.
    assert not acquisition.TIMES.flags.writeable


def test_seed_loader_binds_exact_five_files_and_receipt(tmp_path, monkeypatch):
    data={
        'experiments/hopfield/EXP-005-plan.json': {'synthetic':True},
        'evidence/EXP-005/attempt-001/manifest.json': {'synthetic':True},
    }
    prefix='evidence/EXP-005/attempt-001/'
    artifacts=[]
    for name, expected in acquisition.SETTINGS.items():
        cert=certificate(name, expected['period']); cert['section_state']=expected['section_state']
        point=dict(name=name, alpha=expected['alpha'], gamma=expected['gamma'], certificate=cert)
        data[prefix+f'points/{name}.json']=point
        raw=json.dumps(point).encode()
        artifacts.append(dict(path=f'points/{name}.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    data[prefix+'receipt.json']=dict(artifacts=artifacts)
    bindings={}
    for rel,value in data.items():
        path=tmp_path/rel; path.parent.mkdir(parents=True,exist_ok=True)
        raw=json.dumps(value).encode();path.write_bytes(raw)
        bindings[rel]=(len(raw),hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(acquisition, 'HISTORICAL', bindings)
    records=acquisition.load_seed_records(tmp_path)
    assert [r['setting'] for r in records]==list(acquisition.SETTINGS)
    assert all(len(r['historical_bindings'])==5 for r in records)
    assert all(set(r)=={'setting','alpha','gamma','period','section_state','historical_bindings'} for r in records)
    (tmp_path/prefix/'points/core.json').write_text('{}')
    with pytest.raises(ValueError,match='bytes changed'):
        acquisition.load_seed_records(tmp_path)


@pytest.mark.parametrize('key,value',[('alpha',1.),('gamma',1.),('period',1.),('setting','replacement')])
def test_no_seed_substitution(key,value):
    changed=seed();changed[key]=value
    with pytest.raises(ValueError):acquisition.rerun_certificate(changed)


def test_certificate_call_keeps_original_settings_and_expected_peaks(monkeypatch):
    calls=[]
    def fake(*args,**kwargs):
        calls.append((args,kwargs));args[2][0]=999.
        return {'verified':False,'reason':'synthetic root failure'}
    monkeypatch.setattr(acquisition.diagnostics,'certify',fake)
    original=seed();saved=original['section_state'].copy()
    result=acquisition.rerun_certificate(original)
    args,kwargs=calls[0]
    assert args[:2]==(1.064,10.260722222222222) and args[3]==37.63637538160568
    assert kwargs=={'expected_peak_count':9}
    assert np.array_equal(original['section_state'],saved)
    assert result['reason']=='synthetic root failure'


@pytest.mark.parametrize('field', ['verified','attracting','primitive','peak_count','gates'])
def test_failed_original_certificate_retained_without_fine_solve(monkeypatch,field):
    cert=certificate();cert[field]=8 if field=='peak_count' else {} if field=='gates' else False
    monkeypatch.setattr(acquisition.diagnostics,'certify',lambda *a,**kw:cert)
    monkeypatch.setattr(acquisition,'_variational',lambda *a,**kw:pytest.fail('fine solve after failed certificate'))
    with pytest.raises(acquisition.AcquisitionFailure) as caught:acquisition.acquire_reference(seed())
    packet=caught.value.payload
    assert packet['certificate']==cert and packet['qualified'] is False and packet['failed_stage']=='certificate'
    assert 'orbit' not in packet


def test_eight_period_horizon_blocks_without_extension(monkeypatch):
    monkeypatch.setattr(acquisition.diagnostics,'certify',lambda *a,**kw:certificate(period=40.0001))
    monkeypatch.setattr(acquisition,'_variational',lambda *a,**kw:pytest.fail('forbidden extension'))
    with pytest.raises(acquisition.AcquisitionFailure) as caught:acquisition.acquire_reference(seed())
    assert caught.value.payload['gates']['fixed_duration_covers_eight_cycles'] is False


def test_analytic_circle_reference_variational_and_backward_adjoint(toy_reference):
    ref=toy_reference;orbit=ref['orbit'];adjoint=ref['adjoint'];t=orbit['times'];omega=2*np.pi
    assert ref['qualified'] is True and all(ref['gates'].values())
    assert len(t)==4097 and np.array_equal(t,np.linspace(0.,1.,4097))
    expected=np.column_stack((np.cos(omega*t),np.sin(omega*t),np.zeros(len(t))))
    expected_z=np.column_stack((-np.sin(omega*t),np.cos(omega*t),np.zeros(len(t))))/omega
    assert np.max(np.abs(orbit['states']-expected))<2e-11
    assert np.max(np.abs(adjoint['values']-expected_z))<2e-11
    assert np.max(np.abs(adjoint['normalization_residuals']))<2e-10
    assert np.allclose(orbit['monodromy'],np.diag([np.exp(-2),1.,np.exp(-1)]),rtol=1e-10,atol=1e-11)
    assert np.array_equal(orbit['phi'][0],np.eye(3))
    assert adjoint['normalization_count']==1
    assert np.array_equal(adjoint['values'][-1],adjoint['terminal_normalized_once'])
    assert adjoint['orbit_for_adjoint']=='raw_DOP853_dense_output_no_periodic_wrap'
    assert orbit['relative_closure']<1e-10 and orbit['transported_flow_relative_residual']<1e-10
    assert len(ref['phases'])==8
    for j,phase in enumerate(ref['phases']):
        assert phase['phase_epoch']==j/8 and phase['phase_index']==j
        assert np.array_equal(phase['initial'],orbit['states'][j*512])
        assert abs(phase['phase_gradient']@phase['directions']['phase_null'])<1e-12


def test_raw_endpoint_preserved_in_derived_periodic_template(toy_reference):
    reference=copy.deepcopy(toy_reference)
    reference['orbit']['states'][-1,0]+=.01
    state,adjoint=acquisition.reference_templates(reference)
    assert state.closure()['relative_norm']>.009
    assert np.array_equal(state.evaluate(np.array(1.)),reference['orbit']['states'][0])
    assert np.array_equal(state.raw_endpoint,reference['orbit']['states'][-1])
    assert np.array_equal(adjoint.raw_endpoint,reference['adjoint']['values'][-1])


def test_direct_phase_propagation_starts_identity_and_matches_rotated_analytic_map(monkeypatch,toy_reference):
    patch_toy(monkeypatch)
    prop=acquisition.acquire_phase_propagator(toy_reference,2)
    assert prop['qualified'] is True and prop['phase_epoch']==.25
    assert np.array_equal(prop['phi'][0],np.eye(3))
    assert np.array_equal(prop['initial'],toy_reference['phases'][2]['initial'])
    assert np.allclose(prop['monodromy'],np.diag([1.,np.exp(-2),np.exp(-1)]),rtol=1e-10,atol=1e-11)
    assert prop['neutral_flow_relative_residual']<1e-10


@pytest.mark.parametrize('mutation',['phase','state','gradient','direction'])
def test_dependent_case_rejects_changed_phase_binding(toy_reference,mutation):
    ref=copy.deepcopy(toy_reference);task=acquisition.build_tasks()[0]
    if mutation=='phase':ref['phases'][0]['phase_epoch']=.1
    elif mutation=='state':ref['phases'][0]['initial'][0]+=.1
    elif mutation=='gradient':ref['phases'][0]['phase_gradient'][0]+=.1
    else:ref['phases'][0]['directions']['phase_write'][0]+=.1
    with pytest.raises(ValueError):acquisition.case_context(task,ref)


@pytest.mark.parametrize('key,value',[('phase_index',False),('sign',True),('dose_index',False),('epsilon',1),('method','other')])
def test_task_identity_is_strict_before_solver(toy_reference,key,value):
    task=acquisition.build_tasks()[0];task[key]=value
    with pytest.raises(ValueError):acquisition.case_context(task,toy_reference)


def test_pulse_and_sham_use_exact_rounded_addition_and_activation_gain(toy_reference):
    pulse=acquisition.build_tasks()[0]
    context=acquisition.case_context(pulse,toy_reference)
    phase=toy_reference['phases'][0]
    requested=-1e-4*phase['directions']['phase_write']
    assert np.array_equal(context['requested_delta'],requested)
    assert np.array_equal(context['initial'],phase['initial']+requested)
    assert np.array_equal(context['realized_delta'],context['initial']-phase['initial'])
    assert np.array_equal(context['addition_rounding_residual'],context['realized_delta']-requested)
    assert np.array_equal(context['initial_activation_displacement'],np.tanh(context['initial'])-np.tanh(phase['initial']))
    sham=next(t for t in acquisition.build_tasks() if t['kind']=='sham')
    unchanged=acquisition.case_context(sham,toy_reference)
    assert np.array_equal(unchanged['initial'],phase['initial']) and unchanged['realized_norm']==0.


@pytest.mark.parametrize('method,steps',[('rk4-0.0025',128000),('rk4-0.00125',256000)])
def test_native_trace_uses_unchanged_wrapper_and_exact_grid(monkeypatch,toy_reference,method,steps):
    calls=[];owned=object()
    def fake(fn,alpha,gamma,initial,dt,**kwargs):
        calls.append((fn,alpha,gamma,initial.copy(),dt,kwargs))
        return acquisition.TIMES.copy(),np.tile(initial,(12801,1))
    monkeypatch.setattr(acquisition,'integrate_trace',fake)
    task=next(t for t in acquisition.build_tasks() if t['method']==method)
    packet=acquisition.acquire_case(task,toy_reference,native_fn=owned)
    assert packet['complete'] is True and packet['states'].shape==(12801,3)
    assert packet['solver']['steps']==steps and packet['solver']['id']==method
    assert calls[0][0] is owned and calls[0][4]==float(method[4:])
    assert calls[0][5]==dict(duration=320.,sample_dt=.025)
    assert np.array_equal(packet['times'],acquisition.TIMES)


def test_native_partial_failure_and_previous_input_evidence_retained(monkeypatch,toy_reference):
    task=acquisition.build_tasks()[0]
    partial=dict(times=acquisition.TIMES.copy(),states=np.full((12801,3),np.nan),
                 final=np.array([1.,2.,3.]),progress=np.array([12.,2.]),native_status=-2)
    partial['states'][:2]=0.
    def fail(*a,**kw):raise acquisition.ProbeFailure('synthetic native failure',partial)
    monkeypatch.setattr(acquisition,'integrate_trace',fail)
    with pytest.raises(acquisition.AcquisitionFailure) as caught:acquisition.acquire_case(task,toy_reference,native_fn=object())
    packet=caught.value.payload
    assert packet['status']=='failed' and packet['complete'] is False
    assert packet['native_failure'] is partial and np.isnan(packet['states'][2:]).all()
    assert packet['task']==task and 'requested_delta' in packet


def test_independent_trace_contract_without_integration(monkeypatch,toy_reference):
    calls=[]
    def fake(fun,span,initial,**kwargs):
        calls.append((span,initial.copy(),kwargs))
        return SimpleNamespace(t=kwargs['t_eval'].copy(),y=np.tile(initial[:,None],(1,12801)),
            sol=None,success=True,message='synthetic',nfev=0,njev=0,nlu=0)
    monkeypatch.setattr(acquisition,'solve_ivp',fake)
    task=next(t for t in acquisition.build_tasks() if t['method']=='dop853')
    packet=acquisition.acquire_case(task,toy_reference)
    assert packet['complete'] is True and packet['solver']['success'] is True
    assert packet['solver']['id']=='dop853' and packet['solver']['kind']=='DOP853'
    span,initial,kwargs=calls[0]
    assert span==(0.,320.) and np.array_equal(initial,packet['initial'])
    assert {k:kwargs[k] for k in acquisition.FINE}==acquisition.FINE
    assert np.array_equal(kwargs['t_eval'],acquisition.TIMES)


def test_scipy_partial_and_exception_stages_are_not_labeled_complete(monkeypatch):
    times=np.array([0.,1.]);initial=np.array([1.,2.,3.])
    def partial(*a,**kw):return SimpleNamespace(t=times[:1],y=initial[:,None],sol=None,
        success=False,message='synthetic step failure',nfev=17,njev=0,nlu=0)
    monkeypatch.setattr(acquisition,'solve_ivp',partial)
    with pytest.raises(acquisition.AcquisitionFailure) as caught:
        acquisition._solve(lambda t,x:x,(0.,1.),initial,times,stage='test')
    assert caught.value.payload['values'].shape==(3,1) and caught.value.payload['nfev']==17
    def interrupted(fun,span,state,**kw):
        fun(.125,state)
        raise TimeoutError('synthetic deadline')
    monkeypatch.setattr(acquisition,'solve_ivp',interrupted)
    with pytest.raises(acquisition.AcquisitionFailure) as caught:
        acquisition._solve(lambda t,x:x,(0.,1.),initial,times,stage='test')
    assert caught.value.payload['last_rhs']['accepted_solver_state'] is False
    assert caught.value.payload['last_rhs']['time']==.125
    assert 'TimeoutError' in caught.value.payload['error']


def test_later_reference_failure_keeps_certificate_and_full_raw_orbit(monkeypatch):
    patch_toy(monkeypatch)
    def fail(*a,**kw):raise ValueError('synthetic neutral failure')
    monkeypatch.setattr(acquisition,'neutral_left_phase',fail)
    with pytest.raises(acquisition.AcquisitionFailure) as caught:acquisition.acquire_reference(seed())
    packet=caught.value.payload
    assert packet['failed_stage']=='neutral_left_phase' and packet['certificate']['verified'] is True
    assert packet['orbit']['states'].shape==(4097,3) and packet['orbit']['phi'].shape==(4097,3,3)
    assert packet['qualified'] is False


def test_completed_variational_arrays_survive_derivative_failure(monkeypatch):
    times=np.linspace(0.,1.,4097)
    values=np.tile(np.r_[np.array([1.,0.,0.]),np.eye(3).ravel()][:,None],(1,4097))
    def fake(*args,**kw):return dict(times=times,values=values,status='complete'),None
    monkeypatch.setattr(acquisition,'_solve',fake)
    monkeypatch.setattr(acquisition.diagnostics,'hopfield_rhs',lambda *a:np.array([0.,1.,0.]))
    monkeypatch.setattr(acquisition.diagnostics,'hopfield_jacobian',lambda *a:(_ for _ in ()).throw(RuntimeError('synthetic derivative failure')))
    with pytest.raises(acquisition.AcquisitionFailure) as caught:
        acquisition._variational(1.,1.,np.array([1.,0.,0.]),1.,stage='synthetic')
    partial=caught.value.payload
    assert partial['states'].shape==(4097,3) and partial['phi'].shape==(4097,3,3)
    assert partial['derivatives'].shape==(4097,3)
    assert partial['status']=='derivative_reconstruction_failed'


def test_saved_trace_derivative_reconstruction_never_integrates(monkeypatch,toy_reference):
    task=acquisition.build_tasks()[0]
    context=acquisition.case_context(task,toy_reference)
    packet=dict(task=task,**context,status='complete',complete=True,times=acquisition.TIMES,
                states=np.tile(context['initial'],(12801,1)))
    monkeypatch.setattr(acquisition,'solve_ivp',lambda *a,**kw:pytest.fail('saved derivative called solver'))
    monkeypatch.setattr(acquisition,'integrate_trace',lambda *a,**kw:pytest.fail('saved derivative called native'))
    calls=[]
    def rhs(t,x,alpha,gamma):
        calls.append(t)
        assert alpha==1.064 and gamma==10.260722222222222
        return -x
    monkeypatch.setattr(acquisition.diagnostics,'hopfield_rhs',rhs)
    result=acquisition.trace_derivatives(packet)
    assert np.array_equal(result,-packet['states']) and len(calls)==12801
    packet['complete']=False
    with pytest.raises(ValueError,match='Complete'):acquisition.trace_derivatives(packet)


def test_phase_closure_failure_keeps_full_propagator(monkeypatch,toy_reference):
    orbit=copy.deepcopy(toy_reference['orbit']);orbit['relative_closure']=1.0001e-8
    monkeypatch.setattr(acquisition,'_variational',lambda *a,**kw:(orbit,None))
    with pytest.raises(acquisition.AcquisitionFailure) as caught:
        acquisition.acquire_phase_propagator(toy_reference,0)
    partial=caught.value.payload
    assert partial['qualified'] is False and partial['gates']['relative_closure'] is False
    assert partial['phi'].shape==(4097,3,3) and np.array_equal(partial['monodromy'],orbit['monodromy'])
