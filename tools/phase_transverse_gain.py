"""EXP015 pure saved-reference mathematics; no I/O, solve, fit or target access.

Algebraic partition and ideal physical/clock neutral approximations are distinct.
All failed cells and all six realized-pulse compatibility rows are retained.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
import phase_memory as pm
import phase_memory_evaluation as old
from hopfield_diagnostics import hopfield_rhs

DIRECTIONS=old.DIRECTIONS
DELAYS=pm.DELAYS
DOSES=old.DOSES
NOISES=old.NOISES
GATES=dict(algebra_rtol=1e-12,algebra_atol=1e-14,normalization_atol=1e-7,
    potential_max=1e-6,activation_max=1e-7,standardized_readout=.01,primary_noise=1e-4)
CONTRACT=dict(schema_version='phase-transverse-gain-v1',settings=list(old.SETTINGS),phases=8,
    directions=list(DIRECTIONS),delays=list(DELAYS),doses=list(DOSES),noise_levels=list(NOISES),
    cells=192,realized_pulses_per_cell=6,compatibility_rows=1152,clock_samples=32,
    unit_noise_rows=576,realized_noise_rows=3456,gates=GATES,
    no_ODE=True,no_fit=True,setting_means=False,original_study_requalified=False,
    relative_flow_transport='descriptive only; not a certified accumulated-error bound',
    physical_flow='unchanged Hopfield RHS evaluated on saved periodic Hermite states',
    clock_velocity='derivative of the original periodic Hermite template')


def cell_census():
    return [dict(id=f'{setting}--p{phase}--{direction}--k{delay}',setting=setting,
        phase_index=phase,direction=direction,delay_cycles=delay)
        for setting in old.SETTINGS for phase in range(8) for direction in DIRECTIONS for delay in DELAYS]


def _cell(value):
    if not isinstance(value,dict):raise ValueError('Exact cell mapping required')
    expected=next((x for x in cell_census() if x['id']==value.get('id')),None)
    if expected is None or expected!=value or any(type(value[k]) is not type(expected[k]) for k in expected):
        raise ValueError('Unplanned or type-substituted cell')
    return expected


def _array(value,name,shape):
    a=np.asarray(value)
    if a.dtype!=np.float64 or a.shape!=shape or not np.isfinite(a).all():
        raise ValueError(f'{name}: finite float64 {shape} required')
    return a


def _equal(a,b):
    if isinstance(a,np.generic):a=a.item()
    if isinstance(b,np.generic):b=b.item()
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b,strict=True))
    return a==b


def _stats(a):
    a=np.asarray(a,dtype=np.float64)
    if not a.size or not np.isfinite(a).all():raise ValueError('Finite nonempty diagnostics required')
    location=np.unravel_index(int(np.argmax(np.abs(a))),a.shape);peak=float(np.max(np.abs(a)))
    scaled=a/peak if peak else a
    return dict(maximum_absolute=peak,rms=float(peak*np.sqrt(np.mean(scaled*scaled))),
        maximum_location=[int(i) for i in location],value_at_maximum=float(a[location]))


def _close(actual,expected):
    a=np.asarray(actual);b=np.asarray(expected)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite or incompatible arithmetic comparison')
    residual=a-b;limit=GATES['algebra_atol']+GATES['algebra_rtol']*np.abs(b)
    return dict(passed=bool(np.all(np.abs(residual)<=limit)),residual=_stats(residual),
        maximum_tolerance_ratio=float(np.max(np.abs(residual)/limit)))


def clock_digest(clock):
    h=hashlib.sha256(json.dumps({k:clock[k] for k in ('derivative_energy','phase_epoch','period','delay_cycles')},sort_keys=True,separators=(',',':')).encode())
    for k in ('times','reference_activations','weights'):h.update(k.encode());h.update(clock[k].tobytes(order='C'))
    return h.hexdigest()


def propagated_matrices(segment,monodromy,times):
    """Independent matrix form Phi(s) M**k, preserving noncommuting order."""
    M=_array(monodromy,'monodromy',(3,3));t=_array(times,'clock times',(32,))
    if not isinstance(segment,pm.HermiteSegment) or segment.values.shape!=(4097,3,3):raise ValueError('Original 4097-node tangent segment required')
    if not np.array_equal(segment.values[0],np.eye(3)) or not np.array_equal(segment.values[-1],M):raise ValueError('Tangent endpoint binding differs')
    period=float(segment.times[-1]);cycles=np.floor(t/period).astype(np.int64)
    if np.any(t<0) or np.any(cycles>8):raise ValueError('Clock outside frozen elapsed horizon')
    remainder=t-cycles*period;phi=segment.evaluate(remainder);result=np.empty((32,3,3),dtype=np.float64)
    for k in np.unique(cycles):
        mask=cycles==k;result[mask]=phi[mask]@np.linalg.matrix_power(M,int(k))
    return _array(result,'elapsed propagator',(32,3,3))


def decompose(*,psi,states,template_velocity,physical_flow,initial_flow,gradient,weights,direction):
    """Pure directional mathematics, also usable with synthetic supplied arrays."""
    psi=_array(psi,'psi',(32,3,3));x=_array(states,'reference states',(32,3))
    v=_array(template_velocity,'template velocity',(32,3));f=_array(physical_flow,'physical flow',(32,3))
    f0=_array(initial_flow,'initial flow',(3,));z=_array(gradient,'phase gradient',(3,))
    w=_array(weights,'clock weights',(32,3));d=_array(direction,'input direction',(3,))
    noise_gain=float(np.linalg.norm(w))
    if not np.isfinite(noise_gain) or noise_gain<=0:raise ValueError('Positive finite clock noise gain required')
    D=1.-np.tanh(x)**2;Q=np.outer(f0,z);P=np.eye(3)-Q;amplitude=float(z@d)
    total=np.einsum('nij,j->ni',psi,d)
    transported=np.einsum('nij,j->ni',psi,f0)
    arrays=dict(total=total,phase_algebraic=transported*amplitude,
        transverse=np.einsum('nij,j->ni',psi,P@d),phase_physical=f*amplitude,
        phase_clock=v*amplitude,transport_defect=(transported-f)*amplitude,
        clock_flow_defect=(f-v)*amplitude)
    activation={k:D*value for k,value in arrays.items()}
    q={k:float(np.sum(w*value)) for k,value in activation.items()}
    L=np.einsum('ni,ni,nij->j',w,D,psi)
    observable=dict(q,initial_time_shift=amplitude,clock_normalization_defect=q['phase_clock']-amplitude,
        ideal_accounting_residual=q['total']-(amplitude+q['transverse']+q['transport_defect']+q['clock_flow_defect']+(q['phase_clock']-amplitude)))
    checks=dict(state_partition=_close(arrays['phase_algebraic']+arrays['transverse'],total),
        activation_partition=_close(activation['phase_algebraic']+activation['transverse'],activation['total']),
        readout_partition=_close(q['phase_algebraic']+q['transverse'],q['total']),
        row_operator=_close(float(L@d),q['total']),
        ideal_accounting=_close(amplitude+q['transverse']+q['transport_defect']+q['clock_flow_defect']+(q['phase_clock']-amplitude),q['total']))
    norms=np.linalg.norm(f,axis=1);defect=np.linalg.norm(transported-f,axis=1)
    relative=None if np.any(norms==0) else float(np.max(defect/norms))
    return dict(state=arrays,activation=activation,observable=observable,
        state_gain={k:_stats(value) for k,value in arrays.items()},activation_gain={k:_stats(value) for k,value in activation.items()},
        row_operator=L.copy(),projector=P.copy(),noise_gain=noise_gain,algebraic_checks=checks,
        reference_defects=dict(normalization=abs(float(z@f0)-1.),normalization_signed=float(z@f0)-1.,projector_idempotence=_stats(P@P-P),
            phase_null_leakage=_stats(z@P),flow_transport=_stats(transported-f),clock_flow=_stats(f-v),
            maximum_relative_physical_flow_transport=relative,zero_reference_flow_samples=int(np.sum(norms==0)),
            clock_weight_derivative_inner_product=float(np.sum(w*D*v)),
            clock_weight_derivative_normalization=abs(float(np.sum(w*D*v))-1.)))


def _noise_rows(mathematics,orientation):
    rows=[]
    for sigma in NOISES:
        sd=sigma*mathematics['noise_gain']
        if not np.isfinite(sd) or sd<=0:raise ValueError('Nonfinite or zero noise scale')
        values={k:orientation*v/sd for k,v in mathematics['observable'].items()}
        if not np.isfinite(sd) or sd<=0 or not all(np.isfinite(v) for v in values.values()):raise ValueError('Nonfinite noise-normalized readout')
        rows.append(dict(sigma=sigma,noise_sd=sd,signed_standardized=values))
    return rows


def _approximation(mathematics):
    result={}
    # No cancellation across transport and interpolation defects. Combined
    # physical/clock alternatives are also checked directly against algebraic.
    for name,state in [('transport_defect',mathematics['state']['transport_defect']),
                       ('clock_flow_defect',mathematics['state']['clock_flow_defect']),
                       ('combined_ideal_clock_defect',mathematics['state']['phase_algebraic']-mathematics['state']['phase_clock'])]:
        if name=='combined_ideal_clock_defect':
            activation=mathematics['activation']['phase_algebraic']-mathematics['activation']['phase_clock'];q=mathematics['observable']['phase_algebraic']-mathematics['observable']['phase_clock']
        else:activation=mathematics['activation'][name];q=mathematics['observable'][name]
        sx=float(np.max(np.abs(state)));su=float(np.max(np.abs(activation)));sq=abs(q)/(GATES['primary_noise']*mathematics['noise_gain'])
        result[name]=dict(state_max=sx,activation_max=su,primary_standardized_readout=sq,
            passed=bool(sx<=GATES['potential_max'] and su<=GATES['activation_max'] and sq<=GATES['standardized_readout']))
    return result


def _six_ids(cell):
    return [f'{cell["setting"]}--p{cell["phase_index"]}--d{dose}--{cell["direction"]}--s{sign:+d}' for dose in range(3) for sign in (-1,1)]


def evaluate_cell(cell,*,reference,propagator,sealed_context,sealed_cases):
    cell=_cell(cell);identities=_six_ids(cell)
    result=dict(cell=cell.copy(),status='failed',complete=False,qualified=False,qualified_algebraic=False,
        qualified_ideal_neutral=False,error=None,stage='input_validation',clock_sha256=None,unit=None,
        compatibility=[dict(id=i,status='failed',complete=False,error='dependency unavailable',checks=None,mathematics=None,noise_rows=None,approximation=None) for i in identities],
        gates=dict(input_identity=False,algebraic=False,old_linear_compatibility=False,normalization=False,ideal_approximation=False),
        scope='Reference-only local linear mechanism; no original qualification, finite-pulse accuracy, setting mean or geometry claim.')
    try:
        if not all(isinstance(v,dict) for v in (reference,propagator,sealed_context,sealed_cases)):
            raise ValueError('Reference, propagator, context and cases must be mappings')
        task=old._TASKS_BY_ID[identities[0]+'--'+old.METHODS[0]]
        # Raw input arrays remain strictly typed; old helper validates original
        # identities/epochs/endpoints/directions and finite qualified prerequisites.
        for packet,fields in [(reference['orbit'],('times','states','derivatives')),(reference['adjoint'],('times','values','derivatives')),(propagator,('times','phi','phi_derivatives','monodromy','initial'))]:
            for name in fields:
                if not isinstance(packet[name],np.ndarray) or packet[name].dtype!=np.float64 or not np.isfinite(packet[name]).all():raise ValueError('Original packet array dtype/finite identity differs')
        template,_,phase,segment,M=old._reference(task,reference,propagator)
        expected_context=dict(id=f'{cell["setting"]}--p{cell["phase_index"]}',setting=cell['setting'],phase_index=cell['phase_index'],status='complete',period=template.period,phase_epoch=phase['phase_epoch'],initial=phase['initial'],flow=phase['flow'],phase_gradient=phase['phase_gradient'],directions=phase['directions'],classifier_orientations=phase['classifier_orientations'])
        if any(not _equal(sealed_context.get(k),v) for k,v in expected_context.items()):raise ValueError('Sealed context identity differs')
        if set(sealed_cases)!=set(identities):raise ValueError('Exactly six original realized pulse identities required')
        clock=pm.clock_readout(template,phase['phase_epoch'],cell['delay_cycles'])
        saved_clock=sealed_context['readouts'][str(cell['delay_cycles'])]
        if not isinstance(saved_clock,dict):raise ValueError('Saved clock mapping required')
        for name in ('times','reference_activations','weights','derivative_energy','phase_epoch','period','delay_cycles'):
            if not _equal(saved_clock.get(name),getattr(clock,name)):raise ValueError('Original immutable clock differs: '+name)
        result['clock_sha256']=clock_digest(saved_clock);result['gates']['input_identity']=True
        result['stage']='propagation';psi=propagated_matrices(segment,M,clock.times)
        states=template.evaluate(clock.times+phase['phase_epoch']);velocity=template.evaluate(clock.times+phase['phase_epoch'],derivative=True)
        physical=np.asarray([hopfield_rhs(t,x,reference['alpha'],reference['gamma']) for t,x in zip(clock.times,states,strict=True)],dtype=np.float64)
        common=dict(psi=psi,states=states,template_velocity=velocity,physical_flow=physical,
            initial_flow=phase['flow'],gradient=phase['phase_gradient'],weights=clock.weights)
        direction=_array(phase['directions'][cell['direction']],'unit direction',(3,));orientation=phase['classifier_orientations'][cell['direction']]
        if type(orientation) is not int or orientation not in (-1,1) or abs(float(np.linalg.norm(direction))-1)>1e-12:raise ValueError('Direction/orientation differs')
        unit=decompose(**common,direction=direction)
        direct=pm.full_linear_response(segment,M,direction,clock.times)['state_delta']
        unit['algebraic_checks']['old_direct_unit_propagation']=_close(unit['state']['total'],direct)
        unit['noise_rows']=_noise_rows(unit,orientation);unit['orientation']=orientation;result['unit']=unit
        result['stage']='six_realized_compatibility'
        for row in result['compatibility']:
            packet=sealed_cases[row['id']]
            if not isinstance(packet,dict):raise ValueError('Saved pulse mapping required')
            oldtask=old._TASKS_BY_ID[row['id']+'--'+old.METHODS[0]];pulse=old._pulse(oldtask,phase)
            expected=dict(id=row['id'],context_id=expected_context['id'],status='complete',case={k:v for k,v in oldtask.items() if k not in ('id','method')},pulse=pulse,orientation=orientation,
                requested_time_displacement=float(phase['phase_gradient']@pulse['requested_delta']),realized_time_displacement=float(phase['phase_gradient']@pulse['realized_delta']),
                time_displacement_rounding_difference=float(phase['phase_gradient']@(pulse['realized_delta']-pulse['requested_delta'])),pulse_used_for_phase_only='requested_delta',pulse_used_for_full_linear='realized_delta')
            if any(not _equal(packet.get(k),v) for k,v in expected.items()):raise ValueError('Original pulse identity differs: '+row['id'])
            value=decompose(**common,direction=pulse['realized_delta']);oldreadout=packet['readouts'][str(cell['delay_cycles'])]
            checks=dict(state=_close(value['state']['total'],_array(oldreadout['full_linear_state_delta'],'old full linear state',(32,3))),
                activation=_close(value['activation']['total'],_array(oldreadout['full_linear_activation_delta'],'old full linear activation',(32,3))),
                q=_close(value['observable']['total'],oldreadout['full_linear_q']),
                direct_propagation=_close(value['state']['total'],pm.full_linear_response(segment,M,pulse['realized_delta'],clock.times)['state_delta']),
                noise_gain=_close(value['noise_gain'],oldreadout['noise_gain']))
            # Original finite requested-shift emulator is retained as a comparator,
            # never called the linear neutral component or refit to a pulse trace.
            row.update(status='complete',complete=True,error=None,checks=checks,mathematics=value,
                requested_delta=pulse['requested_delta'].copy(),realized_delta=pulse['realized_delta'].copy(),
                original_finite_phase_only_q=pm._scalar(oldreadout['phase_only_q'],'original finite-shift phase-only q'),noise_rows=_noise_rows(value,orientation),approximation=_approximation(value))
        result['gates']['algebraic']=all(v['passed'] for v in unit['algebraic_checks'].values()) and all(v['passed'] for row in result['compatibility'] for v in row['mathematics']['algebraic_checks'].values())
        result['gates']['old_linear_compatibility']=all(v['passed'] for row in result['compatibility'] for v in row['checks'].values())
        result['gates']['normalization']=unit['reference_defects']['normalization']<=GATES['normalization_atol']
        result['gates']['ideal_approximation']=all(v['passed'] for row in result['compatibility'] for v in row['approximation'].values())
        result['qualified_algebraic']=bool(result['gates']['input_identity'] and result['gates']['algebraic'] and result['gates']['old_linear_compatibility'])
        result['qualified_ideal_neutral']=bool(result['qualified_algebraic'] and result['gates']['normalization'] and result['gates']['ideal_approximation'])
        result.update(status='complete',complete=True,qualified=result['qualified_ideal_neutral'],stage='complete')
    except (ValueError,KeyError,TypeError,IndexError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as error:
        result['error']=f'{type(error).__name__}: {error}'
    return result
