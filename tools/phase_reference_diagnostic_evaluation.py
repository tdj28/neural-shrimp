"""EXP-017 reference-only saved-array diagnostics; no solver, fit, or file IO.

Every planned method is retained. This diagnoses inherited phase-reference
arithmetic; it neither requalifies EXP-016 nor creates a new cycle certificate.
Pointwise RHS/Jacobian calls check supplied arrays and interpolation defects.
Adjoints are never normalized or repaired, and no winning method is selected.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

import numpy as np

from hopfield_diagnostics import hopfield_rhs, hopfield_jacobian
from phase_memory import PeriodicHermiteTemplate

SETTINGS=('core','failed-near-edge')
METHODS=('baseline-dop853','fine-dop853','fine-rk45')
FINE_METHODS=METHODS[1:]
GRIDS=('4096','8192')
STATE_LIMIT=1e-6
ACTIVATION_LIMIT=1e-7
NORMALIZATION_LIMIT=1e-7
EPSILON=1e-4
CLOSURE_RELATIVE_LIMIT=1e-8
ADJOINT_PERIODICITY_RELATIVE_LIMIT=1e-7


def contract():
    return dict(settings=list(SETTINGS),methods=list(METHODS),event_methods=list(FINE_METHODS),grids=list(GRIDS),
        bases=6,events=96,contexts=48,base_pairs=6,event_pairs=48,
        thresholds=dict(state_max=STATE_LIMIT,activation_max=ACTIVATION_LIMIT,normalization=NORMALIZATION_LIMIT,
            epsilon=EPSILON,relative_closure=CLOSURE_RELATIVE_LIMIT,adjoint_relative_periodicity=ADJOINT_PERIODICITY_RELATIVE_LIMIT),
        original_study_requalified=False,diagnostic_only=True,renormalization=False,winning_method_selection=False)


def _array(value,name,shape=None):
    if (type(value) is not np.ndarray or value.dtype!=np.float64 or (shape is not None and value.shape!=shape)
            or not value.size or not np.isfinite(value).all()):raise ValueError(f'{name}: finite float64 array of shape {shape} required')
    return value


def _number(value,name,positive=False):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value):raise ValueError(name+': finite real scalar required')
    value=float(value)
    if positive and value<=0:raise ValueError(name+': positive value required')
    return value


def _equal(actual,expected,name):
    if isinstance(expected,np.ndarray):
        if not np.array_equal(_array(actual,name,expected.shape),expected):raise ValueError(name+': array differs')
    elif type(expected) is int:
        if type(actual) is not int or actual!=expected:raise ValueError(name+': integer differs')
    elif isinstance(expected,float):
        if _number(actual,name)!=expected:raise ValueError(name+': scalar differs')
    elif type(actual) is not type(expected) or actual!=expected:raise ValueError(name+': identity differs')


def difference(left,right):
    left=_array(left,'left');right=_array(right,'right',left.shape);delta=left-right
    if not np.isfinite(delta).all():raise ValueError('Unrepresentable difference')
    position=np.unravel_index(int(np.argmax(np.abs(delta))),delta.shape)
    return dict(max_abs=float(np.max(np.abs(delta))),rms=float(np.sqrt(np.mean(delta**2))),
        location=[int(x) for x in position],signed_at_max=float(delta[position]),
        component_max=np.max(np.abs(delta),axis=tuple(range(delta.ndim-1))) if delta.ndim>1 else np.abs(delta))


def normalization(z,flow):
    z=_array(z,'phase gradient');flow=_array(flow,'flow',z.shape)
    if z.shape[-1]!=3:raise ValueError('Three phase/flow coordinates required')
    residual=np.asarray([float(a@b-1.) for a,b in zip(z.reshape(-1,3),flow.reshape(-1,3),strict=True)],dtype=np.float64).reshape(z.shape[:-1])
    if not np.isfinite(residual).all():raise ValueError('Nonfinite normalization residual')
    return dict(residuals=residual,max_abs=float(np.max(np.abs(residual))),
        pass_bound=bool(np.max(np.abs(residual))<=NORMALIZATION_LIMIT))


def _strict(saved,expected,name):
    actual=_array(saved,name,np.asarray(expected).shape)
    if not np.allclose(actual,expected,rtol=1e-12,atol=1e-14):raise ValueError(name+': saved arithmetic differs')


def _phase_metadata(row,setting,index,period):
    _equal(row.get('context_id'),f'{setting}--p{index:02d}','phase context')
    _equal(row.get('phase_index'),index,'phase index')
    _equal(row.get('split'),'development' if index<8 else 'heldout','phase split')
    fraction=_number(row['phase_fraction'],'phase fraction')
    if not 0<=fraction<1:raise ValueError('Phase fraction outside [0,1)')
    _equal(row['phase_epoch'],fraction*period,'phase epoch')


def _samples(record,period,n,*,offset=0.,adjoint=False,rhs=hopfield_rhs,jacobian=hopfield_jacobian,alpha=None,gamma=None):
    # Acquisition's exact grid is validated by its canonical n+1 times; scalar
    # endpoint overrides are retained to avoid accidental endpoint overshoot.
    times=np.arange(n+1,dtype=np.float64)*(period/n);times[-1]=period
    _equal(record['times'],times,'sample grid')
    _array(record['states'],'states',(n+1,3));_array(record['phi'],'phi',(n+1,3,3))
    _array(record['derivatives'],'derivatives',(n+1,3));_array(record['phi_derivatives'],'phi derivatives',(n+1,3,3))
    _equal(record['phi'][0],np.eye(3),'initial Phi')
    physical_flow=np.asarray([rhs(offset+t,x,alpha,gamma) for t,x in zip(times,record['states'],strict=True)],dtype=np.float64)
    physical_phi=np.asarray([jacobian(offset+t,x,alpha,gamma)@p for t,x,p in zip(times,record['states'],record['phi'],strict=True)],dtype=np.float64)
    _strict(record['derivatives'],physical_flow,'physical state derivatives')
    _strict(record['phi_derivatives'],physical_phi,'physical Phi derivatives')
    if adjoint:
        _array(record['z'],'adjoint',(n+1,3));_array(record['z_derivatives'],'adjoint derivatives',(n+1,3))
        expected=np.asarray([-jacobian(offset+t,x,alpha,gamma).T@z for t,x,z in zip(times,record['states'],record['z'],strict=True)],dtype=np.float64)
        _strict(record['z_derivatives'],expected,'physical adjoint derivatives')
        _strict(record['normalization_residuals'],normalization(record['z'],physical_flow)['residuals'],'node normalization')


def _method(method,period):
    if method not in METHODS:raise ValueError('Unknown method')
    return dict(id=method,kind='RK45' if method=='fine-rk45' else 'DOP853',rtol=1e-12,atol=1e-14,
        max_step=.025 if method=='baseline-dop853' else period/8192,
        **({} if method=='baseline-dop853' else dict(max_step_period_divisor=8192)))


def _solver(record,method,period,name):
    wanted=_method(method,period)
    for key,value in wanted.items():_equal(record['method_spec'].get(key),value,name+'.'+key)
    solver=record['solver']
    if solver.get('success') is not True or type(solver.get('nfev')) is not int or solver['nfev']<0:raise ValueError(name+': successful solver and nonnegative integer nfev required')


def _base_seed(packet):
    inherited=packet['inherited'];period=_number(packet['period'],'period',True)
    for key in ('section_state','terminal_adjoint'):_array(inherited[key],key,(3,))
    if type(inherited.get('normalization_count_from_old')) is not int or inherited['normalization_count_from_old']!=1:raise ValueError('Original one-time normalization provenance required')
    for key in ('alpha','gamma'):_number(packet[key],key)
    return period


def interpolation_diagnostics(base,phase,grid,*,rhs=hopfield_rhs):
    """Unrepaired periodic Hermite versus solver-dense values at a fixed phase."""
    record=base['grids'][grid];T=base['period'];q=np.array(phase['phase_epoch'],dtype=np.float64)
    state=PeriodicHermiteTemplate(record['times'][:-1],record['states'][:-1],record['derivatives'][:-1],T,raw_endpoint=record['states'][-1])
    adjoint=PeriodicHermiteTemplate(record['times'][:-1],record['z'][:-1],record['z_derivatives'][:-1],T,raw_endpoint=record['z'][-1])
    x=state.evaluate(q);z=adjoint.evaluate(q);velocity=state.evaluate(q,derivative=True)
    f=_array(np.asarray(rhs(float(q),x,base['alpha'],base['gamma']),dtype=np.float64),'interpolated-state physical flow',(3,))
    return dict(grid=int(grid),state=x,z=z,physical_flow=f,template_velocity=velocity,
        state_error=difference(x,phase['state']),activation_error=difference(np.tanh(x),np.tanh(phase['state'])),
        adjoint_error=difference(z,phase['z']),physical_flow_error=difference(f,phase['flow']),
        template_velocity_error=difference(velocity,phase['flow']),
        dense_normalization=normalization(phase['z'],phase['flow']),
        interpolated_z_dense_flow=normalization(z,phase['flow']),
        interpolated_z_interpolated_state_physical_flow=normalization(z,f),
        interpolated_z_template_velocity=normalization(z,velocity),
        periodic_wrap_uses_initial_sample=True,raw_endpoint_retained=True)


def _diagnostic_tree(saved,expected,name):
    if not isinstance(saved,Mapping) or set(saved)!=set(expected):raise ValueError(name+': diagnostic keys differ')
    for key,value in expected.items():
        if isinstance(value,np.ndarray):_strict(saved[key],value,name+'.'+key)
        elif not math.isclose(_number(saved[key],name+'.'+key),float(value),rel_tol=1e-12,abs_tol=1e-14):raise ValueError(name+'.'+key+': arithmetic differs')


def base_diagnostics_from_arrays(base):
    fine=base['grids']['8192'];x0,xT=fine['states'][0],fine['states'][-1]
    f0,fT=fine['derivatives'][0],fine['derivatives'][-1];M=fine['phi'][-1]
    z0,zT=fine['z'][0],fine['z'][-1];terminal=base['inherited']['terminal_adjoint']
    fnorm=float(np.linalg.norm(f0));znorm=float(np.linalg.norm(terminal));zscale=max(float(np.linalg.norm(z0)),float(np.linalg.norm(zT)))
    if min(fnorm,znorm,zscale)<=0:raise ValueError('Nonzero physical flow and inherited adjoint required')
    return dict(endpoint_difference=xT-x0,relative_closure=float(np.linalg.norm(xT-x0)/max(1.,np.linalg.norm(x0))),
        neutral_flow_relative_residual=float(np.linalg.norm(M@f0-f0)/fnorm),
        transported_flow_relative_residual=float(np.linalg.norm(M@f0-fT)/fnorm),
        inherited_terminal_left_relative_residual=float(np.linalg.norm(M.T@terminal-terminal)/znorm),
        inherited_terminal_normalization_start=float(terminal@f0-1.),inherited_terminal_normalization_end=float(terminal@fT-1.),
        adjoint_periodic_difference=z0-zT,adjoint_relative_periodicity=float(np.linalg.norm(z0-zT)/zscale),
        maximum_grid_normalization_residual=normalization(fine['z'],fine['derivatives'])['max_abs'])


def event_diagnostics_from_arrays(packet,base):
    phase=base['phases'][packet['phase_index']];tr=packet['trajectory'];x=tr['states'][-1];f=tr['derivatives'][-1];f0=tr['derivatives'][0]
    z=phase['z'];oldz=phase['old_hermite']['z'];Q=np.outer(f,z);P=np.eye(3)-Q
    return dict(state=x,flow=f,phase_gradient=z,Q=Q,P=P,normalization_residual=float(z@f-1.),
        initial_normalization_residual=float(z@f0-1.),event_minus_initial_normalization=float(z@(f-f0)),
        initial_flow=f0,return_difference=x-phase['state'],old_z_initial_normalization=float(oldz@f0-1.),old_z_event_normalization=float(oldz@f-1.),
        Q_idempotence_residual=Q@Q-Q,P_idempotence_residual=P@P-P,QP_residual=Q@P,phase_leakage=z@P,
        Q_operator_norm=float(np.linalg.norm(Q,ord=2)),P_operator_norm=float(np.linalg.norm(P,ord=2)))


def validate_base_packet(packet,expected_id=None,*,rhs=hopfield_rhs,jacobian=hopfield_jacobian):
    """Validate saved identity, seed preservation and pointwise derivative arithmetic."""
    if not isinstance(packet,Mapping) or packet.get('complete') is not True or packet.get('status')!='complete':raise ValueError('Complete base packet required')
    setting,method=packet.get('setting'),packet.get('method')
    if setting not in SETTINGS or method not in METHODS:raise ValueError('Known base setting/method required')
    ident=f'{setting}--{method}'
    _equal(packet.get('id'),ident,'base id');_equal(packet.get('kind'),'base','base kind')
    if expected_id is not None:_equal(ident,expected_id,'planned base id')
    T=_base_seed(packet);wanted=_method(method,T)
    for k,v in wanted.items():_equal(packet['method_spec'].get(k),v,'base method_spec.'+k)
    _equal(packet.get('schema'),'phase-reference-base-v1','base schema')
    _equal(packet['inherited'].get('new_normalizations'),0,'no new normalization')
    for name in ('forward','backward'):_solver(packet[name],method,T,name)
    _equal(packet['forward']['initial'],np.r_[packet['inherited']['section_state'],np.eye(3).ravel()],'forward initial')
    _equal(packet['backward']['initial'],packet['inherited']['terminal_adjoint'],'backward seed')
    _equal(packet['forward']['span'],np.array([0.,T]),'forward span')
    _equal(packet['backward']['span'],np.array([T,0.]),'backward span')
    _equal(packet['forward']['offset'],0.,'base offset');_equal(packet['backward']['offset'],0.,'adjoint offset')
    if set(packet['grids'])!=set(GRIDS):raise ValueError('Both exact base grids required')
    for grid in GRIDS:
        data=packet['grids'][grid];_samples(data,T,int(grid),adjoint=True,rhs=rhs,jacobian=jacobian,alpha=packet['alpha'],gamma=packet['gamma'])
        _equal(data['states'][0],packet['inherited']['section_state'],'unchanged section initial')
        _equal(data['z'][-1],packet['inherited']['terminal_adjoint'],'unchanged adjoint terminal')
    fine=packet['grids']['8192'];coarse=packet['grids']['4096']
    for key in ('times','states','phi','derivatives','phi_derivatives','z','z_derivatives','normalization_residuals'):_equal(coarse[key],fine[key][::2],'nested grids '+key)
    for key,expected in dict(initial_state=fine['states'][0],final_state=fine['states'][-1],initial_phi=fine['phi'][0],final_phi=fine['phi'][-1],initial_z=fine['z'][0],final_z=fine['z'][-1]).items():_equal(packet['raw_endpoints'][key],expected,'raw endpoint '+key)
    if not isinstance(packet['phases'],list) or len(packet['phases'])!=24 or len(packet['phase_rows'])!=24:raise ValueError('Exactly24 phase records required')
    exact=packet['exact_phase_arrays']
    for key,shape in dict(times=(24,),states=(24,3),phi=(24,3,3),derivatives=(24,3),phi_derivatives=(24,3,3),z=(24,3),z_derivatives=(24,3),normalization_residuals=(24,)).items():_array(exact[key],'exact '+key,shape)
    for i,row in enumerate(packet['phases']):
        _phase_metadata(row,setting,i,T)
        for key in ('context_id','phase_index','phase_fraction','split','phase_epoch'):_equal(packet['phase_rows'][i][key],row[key],'phase row '+key)
        exact=packet['exact_phase_arrays']
        _equal(exact['times'][i],row['phase_epoch'],'exact phase time')
        _equal(exact['normalization_residuals'][i],row['normalization_residual'],'exact phase normalization')
        for key,name in [('state','states'),('phi','phi'),('flow','derivatives'),('phi_derivative','phi_derivatives'),('z','z'),('z_derivative','z_derivatives')]:_equal(exact[name][i],row[key],'exact phase array '+key)
        for key in ('state','flow','z','z_derivative'):_array(row[key],key,(3,))
        for key in ('phi','phi_derivative'):_array(row[key],key,(3,3))
        flow=np.asarray(rhs(row['phase_epoch'],row['state'],packet['alpha'],packet['gamma']),dtype=np.float64)
        jac=np.asarray(jacobian(row['phase_epoch'],row['state'],packet['alpha'],packet['gamma']),dtype=np.float64)
        _strict(row['flow'],flow,'dense physical flow');_strict(row['phi_derivative'],jac@row['phi'],'dense physical Phi derivative')
        _strict(row['z_derivative'],-jac.T@row['z'],'dense physical adjoint derivative')
        if not math.isclose(_number(row['normalization_residual'],'dense normalization'),float(row['z']@flow-1),rel_tol=1e-12,abs_tol=1e-14):raise ValueError('Dense normalization arithmetic differs')
        old=row['old_hermite']
        for key in ('state','flow','z','state_derivative','z_derivative'):_array(old[key],'old Hermite '+key,(3,))
    _diagnostic_tree(packet['diagnostics'],base_diagnostics_from_arrays(packet),'base diagnostics')
    return packet


def validate_event_packet(packet,base,expected_id=None,*,rhs=hopfield_rhs,jacobian=hopfield_jacobian):
    """An event starts from its own method's dense phase state and fresh Phi=I."""
    if not isinstance(packet,Mapping) or packet.get('complete') is not True or packet.get('status')!='complete':raise ValueError('Complete event packet required')
    setting,method=packet.get('setting'),packet.get('method');index=packet.get('phase_index')
    if setting not in SETTINGS or method not in FINE_METHODS or type(index) is not int or index not in range(24):raise ValueError('Known event setting/method/phase required')
    ident=f'{setting}--p{index:02d}--{method}'
    _equal(packet.get('id'),ident,'event id');_equal(packet.get('kind'),'event','event kind')
    if expected_id is not None:_equal(ident,expected_id,'planned event id')
    for key in ('setting','method','period','alpha','gamma'):_equal(packet[key],base[key],'event/base '+key)
    _equal(packet['base_id'],base['id'],'event base id')
    phase=base['phases'][index];T=base['period'];_phase_metadata(packet,setting,index,T)
    for key in ('phase_fraction','phase_epoch','split'):_equal(packet[key],phase[key],'event phase '+key)
    _equal(packet['initial'],phase['state'],'event own dense initial');_equal(packet['phase_gradient'],phase['z'],'event own dense phase gradient')
    _equal(packet['event_time'],T,'event elapsed time')
    trajectory=packet['trajectory'];_samples(trajectory,T,8192,offset=phase['phase_epoch'],rhs=rhs,jacobian=jacobian,alpha=base['alpha'],gamma=base['gamma'])
    _equal(packet.get('schema'),'phase-reference-event-v1','event schema')
    _equal(packet['physical_event_time'],phase['phase_epoch']+T,'physical event time')
    _equal(trajectory['offset'],phase['phase_epoch'],'event trajectory offset')
    _solver(packet['forward'],method,T,'event solver')
    for key,value in _method(method,T).items():_equal(packet['method_spec'].get(key),value,'event method spec '+key)
    _equal(packet['forward']['initial'],np.r_[phase['state'],np.eye(3).ravel()],'event solver initial')
    _equal(packet['forward']['span'],np.array([0.,T]),'event solver span')
    _equal(packet['forward']['offset'],phase['phase_epoch'],'event solver offset')
    _equal(packet['provenance'].get('new_normalizations'),0,'event no new normalization')
    _equal(trajectory['states'][0],phase['state'],'event trajectory initial')
    for key,value in phase['old_hermite'].items():_equal(packet['old_hermite'][key],value,'event old Hermite '+key)
    event=packet['event'];_equal(event['state'],trajectory['states'][-1],'event final state')
    _equal(event['flow'],trajectory['derivatives'][-1],'event final flow');_equal(event['phase_gradient'],phase['z'],'event unchanged phase z')
    _equal(event['Q'],np.outer(event['flow'],phase['z']),'unrepaired event Q');_equal(event['P'],np.eye(3)-event['Q'],'unrepaired event P')
    if not math.isclose(_number(event['normalization_residual'],'event norm'),float(phase['z']@event['flow']-1),rel_tol=1e-12,abs_tol=1e-14):raise ValueError('Event normalization arithmetic differs')
    _diagnostic_tree(packet['event'],event_diagnostics_from_arrays(packet,base),'event diagnostics')
    return packet


def _base_diagnostics(base):
    result=dict(id=base['id'],setting=base['setting'],method=base['method'],acquired_complete=True,arithmetic_replay_pass=True,complete=True,status='complete',grids={},phases=[],neutral_and_closure_diagnostics=base_diagnostics_from_arrays(base))
    for grid in GRIDS:
        data=base['grids'][grid];state_delta=data['states'][-1]-data['states'][0];z_delta=data['z'][0]-data['z'][-1]
        closure=float(np.linalg.norm(state_delta)/max(1.,float(np.linalg.norm(data['states'][0]))))
        scale=max(float(np.linalg.norm(data['z'][0])),float(np.linalg.norm(data['z'][-1])))
        periodicity=None if scale==0 else float(np.linalg.norm(z_delta)/scale)
        norm=normalization(data['z'],data['derivatives'])
        result['grids'][grid]=dict(state_closure_vector=state_delta,relative_closure=closure,
            closure_diagnostic_pass=closure<=CLOSURE_RELATIVE_LIMIT,adjoint_periodicity_vector=z_delta,
            relative_adjoint_periodicity=periodicity,adjoint_periodicity_diagnostic_pass=periodicity is not None and periodicity<=ADJOINT_PERIODICITY_RELATIVE_LIMIT,
            normalization=norm,raw_endpoint_state=data['states'][-1],raw_endpoint_adjoint=data['z'][-1])
    result['phases']=[dict(context_id=p['context_id'],phase_index=p['phase_index'],phase_fraction=p['phase_fraction'],normalization=normalization(p['z'],p['flow'])) for p in base['phases']]
    result['all_grid_diagnostics_pass']=all(g['closure_diagnostic_pass'] and g['adjoint_periodicity_diagnostic_pass'] and g['normalization']['pass_bound'] for g in result['grids'].values())
    return result


def array_agreement(left,right,*,adjoint=False):
    """Pulse-scaled tangent relevance; raw operator differences remain visible."""
    if not np.array_equal(left['times'],right['times']):raise ValueError('Comparison time grid differs')
    state=difference(left['states'],right['states']);activation=difference(np.tanh(left['states']),np.tanh(right['states']))
    raw_phi=difference(left['phi'],right['phi']);scaled_phi=difference(EPSILON*left['phi'],EPSILON*right['phi'])
    dl=(1-np.tanh(left['states'])**2)[..., :,None];dr=(1-np.tanh(right['states'])**2)[..., :,None]
    scaled_activation_phi=difference(EPSILON*(dl*left['phi']),EPSILON*(dr*right['phi']))
    gates=dict(state=state['max_abs']<=STATE_LIMIT,activation=activation['max_abs']<=ACTIVATION_LIMIT,
        pulse_scaled_tangent=scaled_phi['max_abs']<=STATE_LIMIT,pulse_scaled_activation_tangent=scaled_activation_phi['max_abs']<=ACTIVATION_LIMIT)
    result=dict(state=state,activation=activation,raw_phi=raw_phi,pulse_scaled_phi=scaled_phi,pulse_scaled_activation_phi=scaled_activation_phi,
        gates=gates,agreement_pass=all(gates.values()))
    if adjoint:result['adjoint_difference']=difference(left['z'],right['z'])
    return result


def _old_context(oldseal,cid):
    if not isinstance(oldseal,Mapping) or oldseal.get('context_id')!=cid or oldseal.get('status')!='complete':raise ValueError('Complete unchanged old context seal required')
    context=oldseal['acquisition']
    _equal(context['id'],cid,'old acquisition id')
    for key in ('initial','flow','phase_gradient','nominal_event_phase_gradient','reference_at_event','event_flow','event_phase_gradient'):_array(context[key],'old '+key,(3,))
    return context


def _phase_comparison(left,right):
    for key in ('context_id','phase_index','phase_fraction','phase_epoch','split'):_equal(left[key],right[key],'compared phase '+key)
    return array_agreement(dict(times=np.zeros(1),states=left['state'][None],phi=left['phi'][None],z=left['z'][None]),
        dict(times=np.zeros(1),states=right['state'][None],phi=right['phi'][None],z=right['z'][None]),adjoint=True)


def summarize(basesdict,eventsDict,oldsealsdict,*,rhs=hopfield_rhs,jacobian=hopfield_jacobian):
    """Retain six bases, 96 events, all method pairs and48 original contexts.

    No reference failure repairs the old study. Missing or invalid packets
    remain explicit unavailable entries and cannot reduce any denominator.
    """
    expected_bases={f'{s}--{m}' for s in SETTINGS for m in METHODS}
    expected_events={f'{s}--p{p:02d}--{m}' for s in SETTINGS for p in range(24) for m in FINE_METHODS}
    expected_contexts={f'{s}--p{p:02d}' for s in SETTINGS for p in range(24)}
    supplied=[('bases',basesdict,expected_bases),('events',eventsDict,expected_events),('oldseals',oldsealsdict,expected_contexts)]
    input_errors=[]
    for name,values,expected in supplied:
        if not isinstance(values,Mapping):raise ValueError(name+' must be mapping')
        if set(values)-expected:input_errors.append(dict(scope=name,error='Unexpected identities',ids=sorted(set(values)-expected)))
    bases=[];events=[];contexts=[];base_pairs=[];valid_bases={};valid_events={};event_diagnostics={}
    for setting in SETTINGS:
        for method in METHODS:
            ident=f'{setting}--{method}';record=dict(id=ident,setting=setting,method=method,acquired_complete=isinstance(basesdict.get(ident),Mapping) and basesdict[ident].get('complete') is True,arithmetic_replay_pass=False,complete=False,status='unavailable',grids={},phases=[])
            try:
                base=validate_base_packet(basesdict.get(ident),ident,rhs=rhs,jacobian=jacobian)
                record=_base_diagnostics(base);valid_bases[ident]=base
            except (ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError) as error:record['error']=f'{type(error).__name__}: {error}'
            bases.append(record)
    for setting in SETTINGS:
        for index in range(24):
            for method in FINE_METHODS:
                ident=f'{setting}--p{index:02d}--{method}';base_id=f'{setting}--{method}'
                record=dict(id=ident,context_id=f'{setting}--p{index:02d}',setting=setting,phase_index=index,method=method,base_id=base_id,acquired_complete=isinstance(eventsDict.get(ident),Mapping) and eventsDict[ident].get('complete') is True,arithmetic_replay_pass=False,complete=False,status='unavailable')
                try:
                    if base_id not in valid_bases:raise ValueError('Own base unavailable')
                    packet=validate_event_packet(eventsDict.get(ident),valid_bases[base_id],ident,rhs=rhs,jacobian=jacobian)
                    phase=valid_bases[base_id]['phases'][index];event=packet['event']
                    initial=normalization(phase['z'],phase['flow']);at_event=normalization(phase['z'],event['flow'])
                    record.update(complete=True,arithmetic_replay_pass=True,status='complete',initial_normalization=initial,event_normalization=at_event,
                        independently_replayed_event_diagnostics=event_diagnostics_from_arrays(packet,valid_bases[base_id]),
                        dense_initial_event_normalization_pass=initial['pass_bound'] and at_event['pass_bound'],
                        after_one_period_state_difference=difference(event['state'],phase['state']),
                        after_one_period_activation_difference=difference(np.tanh(event['state']),np.tanh(phase['state'])),
                        Q_operator_norm=float(np.linalg.norm(event['Q'],2)),P_operator_norm=float(np.linalg.norm(event['P'],2)),
                        event_flow=event['flow'],phase_gradient=phase['z'])
                    valid_events[ident]=packet
                except (ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError) as error:record['error']=f'{type(error).__name__}: {error}'
                events.append(record);event_diagnostics[ident]=record
    for setting in SETTINGS:
        for left,right in itertools.combinations(METHODS,2):
            ids=[f'{setting}--{m}' for m in (left,right)];record=dict(id=ids[0]+'--vs--'+right,setting=setting,methods=[left,right],base_ids=ids,complete=False,grids={},phases=[])
            try:
                a,b=(valid_bases[k] for k in ids)
                for key in ('period','alpha','gamma'):_equal(a[key],b[key],'base pair '+key)
                for key in ('section_state','terminal_adjoint'):_equal(a['inherited'][key],b['inherited'][key],'base pair inherited '+key)
                record['grids']={g:array_agreement(a['grids'][g],b['grids'][g],adjoint=True) for g in GRIDS}
                record['phases']=[dict(phase_index=i,**_phase_comparison(a['phases'][i],b['phases'][i])) for i in range(24)]
                record.update(complete=True,agreement_pass=all(g['agreement_pass'] for g in record['grids'].values()) and all(p['agreement_pass'] for p in record['phases']))
            except (ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError) as error:record['error']=f'{type(error).__name__}: {error}'
            base_pairs.append(record)
    for setting in SETTINGS:
        for index in range(24):
            cid=f'{setting}--p{index:02d}'
            record=dict(context_id=cid,setting=setting,phase_index=index,complete=False,old=None,methods={},fine_pair=None,dense_fine_pair_diagnostic_pass=False)
            try:
                old=_old_context(oldsealsdict.get(cid),cid)
                record['old']=dict(initial_state=old['initial'],phase_gradient=old['phase_gradient'],initial_flow=old['flow'],
                    initial_normalization=normalization(old['phase_gradient'],old['flow']),
                    event_state=old['reference_at_event'],event_flow=old['event_flow'],event_phase_gradient=old['event_phase_gradient'],
                    event_normalization=normalization(old['event_phase_gradient'],old['event_flow']),
                    original_qualified=oldsealsdict[cid]['qualified'],original_gates=oldsealsdict[cid]['evaluation']['gates'])
                for method in METHODS:
                    item=dict(method=method,complete=False,interpolation={},event=None);record['methods'][method]=item
                    try:
                        base=valid_bases[f'{setting}--{method}'];phase=base['phases'][index]
                        for key in ('period','alpha','gamma'):_equal(base[key],old[key],'new/old '+key)
                        for key in ('phase_fraction','phase_epoch','split'):_equal(phase[key],old[key],'new/old phase '+key)
                        for newkey,oldkey in [('state','initial'),('z','phase_gradient'),('flow','flow')]:_equal(phase['old_hermite'][newkey],old[oldkey],'original Hermite '+newkey)
                        item.update(complete=True,dense_state=phase['state'],dense_z=phase['z'],dense_flow=phase['flow'],
                            dense_normalization=normalization(phase['z'],phase['flow']),
                            old_vs_dense=dict(state=difference(old['initial'],phase['state']),activation=difference(np.tanh(old['initial']),np.tanh(phase['state'])),
                                adjoint=difference(old['phase_gradient'],phase['z']),
                                old_z_dense_flow=normalization(old['phase_gradient'],phase['flow']),dense_z_old_flow=normalization(phase['z'],old['flow'])),
                            interpolation={g:interpolation_diagnostics(base,phase,g,rhs=rhs) for g in GRIDS})
                        if method in FINE_METHODS:
                            ident=cid+'--'+method;event=valid_events[ident]['event'];item['event']=event_diagnostics[ident]
                            item['event_cross_pairing']=dict(old_nominal_z_new_event_flow=normalization(old['nominal_event_phase_gradient'],event['flow']),
                                dense_z_old_event_flow=normalization(phase['z'],old['event_flow']),
                                old_vs_new_event_state=difference(old['reference_at_event'],event['state']),
                                old_vs_new_event_activation=difference(np.tanh(old['reference_at_event']),np.tanh(event['state'])))
                    except (ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError) as error:item.update(complete=False,error=f'{type(error).__name__}: {error}')
                a,b=(valid_events[cid+'--'+m] for m in FINE_METHODS)
                record['fine_pair']=array_agreement(a['trajectory'],b['trajectory'])
                record['fine_pair']['dense_initial']=_phase_comparison(valid_bases[f'{setting}--{FINE_METHODS[0]}']['phases'][index],valid_bases[f'{setting}--{FINE_METHODS[1]}']['phases'][index])
                record['complete']=all(v['complete'] for v in record['methods'].values())
                record['dense_fine_pair_diagnostic_pass']=bool(record['complete'] and record['fine_pair']['agreement_pass'] and record['fine_pair']['dense_initial']['agreement_pass']
                    and all(event_diagnostics[cid+'--'+m]['dense_initial_event_normalization_pass'] for m in FINE_METHODS))
            except (ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError) as error:record['error']=f'{type(error).__name__}: {error}'
            contexts.append(record)
    complete=not input_errors and all(r['complete'] for rows in (bases,events,base_pairs,contexts) for r in rows)
    return dict(complete=complete,diagnostic_complete=complete,diagnostic_only=True,original_study_requalified=False,contract=contract(),input_errors=input_errors,
        counts=dict(bases_planned=6,bases_complete=sum(r['complete'] for r in bases),events_planned=96,events_complete=sum(r['complete'] for r in events),
            base_pairs_planned=6,base_pairs_complete=sum(r['complete'] for r in base_pairs),contexts_planned=48,contexts_complete=sum(r['complete'] for r in contexts),
            dense_fine_pair_diagnostic_pass=sum(r['dense_fine_pair_diagnostic_pass'] for r in contexts)),
        bases=bases,events=events,base_pairs=base_pairs,contexts=contexts,
        all_dense_fine_pairs_pass=complete and all(r['dense_fine_pair_diagnostic_pass'] for r in contexts),
        scope='Outcome-selected numerical reference diagnosis; no pulse/observer outcomes, original requalification or method selection')
