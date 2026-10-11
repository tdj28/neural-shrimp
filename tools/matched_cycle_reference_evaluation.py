"""Six-cycle, reference-only evaluation. No IO, ODE integration or target writes.

Adapted source copies below preserve EXP018 reference validation and EXP016
observer mathematics while replacing their fixed two-cycle/24-phase census.
All matching and mode resolution rules are prospective descriptive diagnostics.
"""
from __future__ import annotations
import copy
import itertools
import math
from fractions import Fraction
from collections.abc import Mapping
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import periodic_reference_refinement_evaluation as legacy_reference
import phase_reference_diagnostic_evaluation as old_math
import two_write_evaluation as legacy_observer
import two_write_acquisition as acquisition
import two_write_memory as memory
from hopfield_diagnostics import hopfield_rhs, hopfield_jacobian, weight_matrix
from phase_memory import neutral_left_phase

METHODS=('tight-dop853','tight-rk45')
TANGENT_METHODS=('dop853','rk45')
GRIDS=('4096','8192')
WINDOWS=('immediate','k2','k4','k8')
ARMS=('identity','timing','transverse')
EPSILON=SIGMA=1e-4
STATE_LIMIT=1e-6
ACTIVATION_LIMIT=1e-7
MARGIN_LIMIT=.01
ACCURACY_LIMIT=.001
NORMALIZATION_LIMIT=1e-7
PHASE_SEED=2026091702
SLOT_IDS=('anchor-core','anchor-failed-near-edge','core-rho025','core-rho050',
          'failed-near-edge-rho025','failed-near-edge-rho050')
_array=old_math._array
_number=old_math._number
_equal=old_math._equal
_strict=old_math._strict
_tree=legacy_reference._tree
_method=legacy_reference._method
_solver=legacy_reference._solver
_ivp_complete=legacy_reference._ivp_complete
_forward=legacy_reference._forward
_certificate=legacy_reference._certificate
_metrics=legacy_reference._metrics
candidate_comparison=legacy_reference.candidate_comparison
array_agreement=old_math.array_agreement
normalization=old_math.normalization
validate_shoot_packet=legacy_reference.validate_shoot_packet
partial_array_census=legacy_reference.partial_array_census
_scalar=legacy_observer._scalar
_exact=legacy_observer._exact
_segments=legacy_observer._segments
_difference=legacy_observer._difference
_predict=legacy_observer._predict


def phase_census():
    fractions=np.random.Generator(np.random.PCG64(PHASE_SEED)).random(8)
    return [dict(phase_index=i,phase_fraction=float(f),split='fresh') for i,f in enumerate(fractions)]


def _phase_rows(rows):
    if not isinstance(rows,list) or len(rows)!=8:raise ValueError('Exactly eight phase rows required')
    for row,expected in zip(rows,phase_census(),strict=True):
        if not isinstance(row,Mapping) or set(row)!=set(expected):raise ValueError('Exact phase schema required')
        for key,value in expected.items():_exact(row[key],value,'exact phase '+key)
    return rows


def _context(context):
    if not isinstance(context,Mapping):raise ValueError('Context mapping required')
    cycle=context.get('cycle_id',context.get('setting'));i=context.get('phase_index')
    if type(cycle) is not str or not cycle or type(i) is not int or i not in range(8):raise ValueError('Declared cycle and phase required')
    _exact(context['setting'],cycle,'truthful cycle setting')
    _exact(context['id'],f'{cycle}--p{i:02d}','context id')
    for key,val in phase_census()[i].items():_exact(context[key],val,'context phase '+key)
    T=_scalar(context['period'],'period')
    if T<=0:raise ValueError('Positive period required')
    for key in ('initial','flow','phase_gradient','direction','nominal_event_phase_gradient'):_array(context[key],key,(3,))
    if abs(float(np.linalg.norm(context['direction']))-1)>1e-12:raise ValueError('Unit direction required')
    _exact(context['phase_epoch'],context['phase_fraction']*T,'phase epoch')
    _exact(context.get('new_normalizations'),0,'no per-phase normalization')
    _exact(context['nominal_event_phase_gradient'],context['phase_gradient'],'unchanged periodic event gradient')
    length=float(np.linalg.norm(context['phase_gradient']))
    if length==0 or not np.isfinite(length):raise ValueError('Nonzero finite phase gradient required')
    unit=context['phase_gradient']/length;projector=np.eye(3)-np.outer(unit,unit)
    coordinate=int(np.argmax(np.linalg.norm(projector,axis=0)))
    expected=projector[:,coordinate]/np.linalg.norm(projector[:,coordinate])
    _strict(context['direction'],expected,'reference-defined deterministic phase-null write')
    for key in ('alpha','gamma'):_scalar(context[key],key)
    return T


def contract():
    return dict(schema='matched-cycle-reference-evaluation-v1',cycles=6,phases=8,phase_seed=PHASE_SEED,
        phase_split='fresh',shoot_methods=list(METHODS),tangent_methods=list(TANGENT_METHODS),
        jobs=dict(shooting=12,references=6,tangents=96,total=114),
        ivp_slots=dict(shooting=300,crossverification=24,adjoints=12,tangent_segments=192,total=528),
        comparisons=dict(phase_contexts=48,candidate_bin_pairs=2,cycle_pairs=15),
        epsilon=EPSILON,sigma=SIGMA,windows=list(WINDOWS),
        thresholds=dict(normalization=1e-7,state=1e-6,activation=1e-7,standardized_margin=.01,
            probability=.001,rho_targets=[.25,.50],rho_bin_half_width=.025,
            period_relative=.02,immediate_rms_relative=.10,detectability_floor=1e-6),
        mode_resolution=dict(multiplier_order='minimum-sum-distance two-mode matching; fixed lexicographic tie',
            floor_factor=64.,machine_epsilon=float(np.finfo(float).eps),
            floor='64*eps*max(1, spectral matrix norm)*max(1, eigenvector condition)',
            radius='maximum paired-method displacement and either method arithmetic floor',
            rigorous_interval=False,positive_real_requires_exact_zero_imaginary_both_methods=True,
            unique_dominant_requires_separated_modulus_discs=True,
            unresolved_secondary_does_not_invalidate_qualified_reference=True,
            unresolved_log_rate=None),
        rho_bin=dict(rule='max(abs(primary-target),abs(alternate-target))+resolution_radius <= 1/40',
            arithmetic='exact rational comparison of stored binary64 values and exact 1/40',
            resolution_radius_is_heuristic=True,dominant_positive_resolved_required=True),
        modal_orientation=dict(source='primary phase-local one-period operators at each write epoch',
            basis='unit right eigenvectors; largest-magnitude component has positive real phase',
            order='closest-to-one neutral first, then lexicographic real/imaginary transverse values',
            write='V inverse times epsilon*d; complex coefficients retained',
            read='observable operator times each right eigenvector and its write coefficient',
            orthogonal_energy_interpretation=False,qualification_gate=False,
            caveat='Repeated or ill-conditioned modes have nonunique or sensitive orientations; no eigenvector matching or rigorous resolution claim'),
        duplicate=dict(period_relative=1e-8,state_scaled_rms=1e-8,grid=8192,coarse_grid=4096,
            alignment='FFT cyclic shift then exactly64 golden-section steps within plus/minus one grid sample',
            interpolation='periodic-query cubic Hermite using unchanged states and physical flow times period',
            state_scale='coordinatewise max(1,maxabs both loops)',reversal=False,time_warp=False,
            same_exact_parameters_duplicate_fails_distinctness=True,
            same_exact_parameters_unresolved_fails_distinctness=True,
            cross_parameter_near_duplicates_warning_only=True,rigorous_orbit_identity=False),
        arithmetic=dict(rtol=1e-12,atol=1e-14,reported_separately=True),
        finite_writes=0,geometry_effect_claim=False,
        complete='All six numerical reference rows and48 context computations available; matching may still fail',
        feasible='All planned references and contexts qualified, both fixed bin pairs match, no same-parameter duplicate or unresolved comparison')


def _complex(value):
    return dict(real=float(value.real),imag=float(value.imag),modulus=float(abs(value)))


def _complex_array(value):
    """Numeric-codec compatible representation, preserving both complex parts."""
    return dict(real=np.asarray(value.real,dtype=np.float64),imag=np.asarray(value.imag,dtype=np.float64))


def rho_bin_containment(modes,target):
    """Exact arithmetic around a descriptive, not interval-certified, radius."""
    if target not in (.25,.5):raise ValueError('Known rho target required')
    row=dict(target_rho=target,passed=False,half_width_numerator=1,half_width_denominator=40,
        distance_plus_radius_exact=None,scope='Heuristic radius containment; not a rigorous eigenvalue enclosure')
    d=modes['rows'][modes['dominant_index']];radius=d['resolution_radius']
    if not modes['dominant_positive_resolved'] or radius is None:return row
    values=[_number(d[k]['real'],k+' multiplier') for k in ('primary','alternate')]
    radius=_number(radius,'resolution radius')
    if radius<0:raise ValueError('Nonnegative radius required')
    bound=max(abs(Fraction(x)-Fraction(target)) for x in values)+Fraction(radius)
    row.update(passed=bound<=Fraction(1,40),
        distance_plus_radius_exact=dict(numerator=str(bound.numerator),denominator=str(bound.denominator)))
    return row


def _modal_basis(matrix,write):
    """Descriptive coordinates only: right modes need not be orthogonal.

    First and second writes use their own directly propagated one-period
    matrices. Neither eigenvectors nor the global adjoint are repaired here.
    Repeated eigenvalues can yield arbitrary orientations even when this
    algebraic decomposition is available; conditioning is always retained.
    """
    row=dict(available=False,qualification_gate=False)
    try:
        M=_array(matrix,'phase-local monodromy',(3,3));delta=_array(write,'write',(3,))
        values,V=np.linalg.eig(M);neutral=int(np.argmin(abs(values-1.)))
        order=[neutral]+sorted((i for i in range(3) if i!=neutral),key=lambda i:(float(values[i].real),float(values[i].imag),i))
        values=values[order];V=V[:,order].astype(np.complex128);pivots=[]
        for i in range(3):
            v=V[:,i];v/=np.linalg.norm(v);pivot=int(np.argmax(abs(v)));pivots.append(pivot)
            v*=np.conj(v[pivot])/abs(v[pivot])
        condition=float(np.linalg.cond(V));dual=np.linalg.solve(V,np.eye(3));coefficients=dual@delta
        if not all(np.isfinite(x).all() for x in (V,dual,coefficients)) or not np.isfinite(condition):raise ValueError('Nonfinite modal decomposition')
        reconstruct=V@coefficients
        row.update(available=True,eigenvalues=[_complex(v) for v in values],
            labels=['neutral','transverse-0','transverse-1'],pivot_coordinates=pivots,
            right_vectors=_complex_array(V),dual_rows=_complex_array(dual),
            write_coefficients=_complex_array(coefficients),eigenbasis_condition=condition,
            minimum_eigenvalue_separation=float(min(abs(values[i]-values[j]) for i,j in itertools.combinations(range(3),2))),
            eigen_residual_max=float(np.max(abs(M@V-V*values[None,:]))),
            write_reconstruction_max=float(np.max(abs(reconstruct-delta))),
            write_reconstruction_imaginary_max=float(np.max(abs(reconstruct.imag))),
            raw_monodromy=M.copy(),requested_write=delta.copy())
    except (ValueError,TypeError,np.linalg.LinAlgError,FloatingPointError,OverflowError) as exc:row['error']=f'{type(exc).__name__}: {exc}'
    return row


def _modal_readout(basis,operator,response):
    if not basis['available']:return dict(available=False,qualification_gate=False,error=basis.get('error'))
    try:
        V=basis['right_vectors']['real']+1j*basis['right_vectors']['imag']
        c=basis['write_coefficients']['real']+1j*basis['write_coefficients']['imag']
        with np.errstate(over='raise',invalid='raise'):
            read=operator@V;contributions=read*c[None,:];total=np.sum(contributions,axis=1)
            norms=np.linalg.norm(contributions,axis=0)
            error=float(np.max(abs(total-response)));imag=float(np.max(abs(total.imag)))
        if not all(np.isfinite(x).all() for x in (read,contributions,norms,total)) or not np.isfinite(error+imag):raise ValueError('Nonfinite modal readout')
        return dict(available=True,qualification_gate=False,read_vectors=_complex_array(read),
            signed_contributions=_complex_array(contributions),contribution_l2=norms,
            reconstruction_max=error,reconstruction_imaginary_max=imag,
            scope='Nonorthogonal modal decomposition; contribution norms are not additive energies or causal effects')
    except (ValueError,TypeError,np.linalg.LinAlgError,FloatingPointError,OverflowError) as exc:
        return dict(available=False,qualification_gate=False,error=f'{type(exc).__name__}: {exc}')


def mode_resolution(left_matrix,right_matrix,period):
    """Descriptive resolution floor; never a rigorous eigenvalue enclosure.

    The old neutral/stability qualification is a prerequisite, not replaced by
    this heuristic. An unresolved tiny transverse multiplier has no log rate.
    """
    T=_number(period,'period',True);records=[]
    for M in (left_matrix,right_matrix):
        M=_array(M,'monodromy',(3,3));values,vectors=np.linalg.eig(M)
        ni=int(np.argmin(abs(values-1.)));trans=np.delete(values,ni)
        trans=trans[np.lexsort((trans.imag,trans.real))]
        condition=float(np.linalg.cond(vectors));norm=float(np.linalg.norm(M,2))
        floor=64*np.finfo(float).eps*max(1.,norm)*max(1.,condition)
        if not np.isfinite(floor):floor=None
        records.append(dict(values=trans,neutral_index=ni,condition=condition if np.isfinite(condition) else None,
                            matrix_norm=norm,floor=floor))
    a,b=records;permutation=min(itertools.permutations(range(2)),key=lambda p:(sum(abs(a['values'][i]-b['values'][p[i]]) for i in range(2)),p))
    rows=[]
    for i,j in enumerate(permutation):
        x,y=a['values'][i],b['values'][j];delta=float(abs(x-y))
        radius=max(delta,a['floor'],b['floor']) if a['floor'] is not None and b['floor'] is not None else None
        resolved=radius is not None and min(abs(x),abs(y))>radius
        real=bool(x.imag==0 and y.imag==0)
        positive=bool(resolved and real and min(x.real,y.real)>radius)
        rows.append(dict(index=i,alternate_index=j,primary=_complex(x),alternate=_complex(y),
            method_difference=delta,resolution_radius=radius,nonzero_resolved=bool(resolved),
            real_in_both_methods=real,positive_resolved=positive,
            log_modulus_per_model_time=float(math.log(abs(x))/T) if resolved else None,
            decay_rate_per_model_time=float(-math.log(abs(x))/T) if resolved else None))
    dominant=max(range(2),key=lambda i:(rows[i]['primary']['modulus'],-i));other=1-dominant
    d,o=rows[dominant],rows[other]
    unique=bool(d['resolution_radius'] is not None and o['resolution_radius'] is not None and
        min(d['primary']['modulus'],d['alternate']['modulus'])-d['resolution_radius']>
        max(o['primary']['modulus'],o['alternate']['modulus'])+o['resolution_radius'])
    return dict(complete=True,rows=rows,dominant_index=dominant,dominant_unique=unique,
        dominant_positive_resolved=bool(unique and d['positive_resolved']),
        dominant_signed_value=d['primary']['real'] if unique and d['positive_resolved'] else None,
        method_diagnostics=[{k:v for k,v in r.items() if k!='values'} for r in records],
        resolution_semantics='heuristic paired-method/roundoff sensitivity diagnostic, not a rigorous interval',
        prior_scientific_stability_gates_unchanged=True)


def _aligned_loop(left,right,n):
    """Normalized-time circular alignment; never reverse or warp time."""
    la,lb=left['states'],right['states'];fa,fb=left['derivatives'],right['derivatives']
    for x,f in ((la,fa),(lb,fb)):
        _array(x,'loop states',(8193,3));_array(f,'loop derivatives',(8193,3))
    stride=8192//n;la,lb,fa,fb=(a[::stride] for a in (la,lb,fa,fb))
    scale=np.maximum(1.,np.maximum(np.max(abs(la),axis=0),np.max(abs(lb),axis=0)))
    aa,bb=la[:-1]/scale,lb[:-1]/scale
    correlation=np.fft.irfft(np.sum(np.conj(np.fft.rfft(aa,axis=0))*np.fft.rfft(bb,axis=0),axis=1),n=n)
    index=int(np.argmax(correlation));center=index/n;phase=np.arange(n,dtype=np.float64)/n
    interpolator=CubicHermiteSpline(np.arange(n+1,dtype=np.float64)/n,lb,right['period']*fb)
    def objective(shift):
        delta=(la[:-1]-interpolator(np.mod(phase+shift,1.)))/scale
        return float(np.mean(delta*delta))
    lo,hi=center-1/n,center+1/n;ratio=(math.sqrt(5)-1)/2
    c=hi-ratio*(hi-lo);d=lo+ratio*(hi-lo);fc,fd=objective(c),objective(d)
    for _ in range(64):
        if fc<=fd:hi,d,fd=d,c,fc;c=hi-ratio*(hi-lo);fc=objective(c)
        else:lo,c,fc=c,d,fd;d=lo+ratio*(hi-lo);fd=objective(d)
    shift,value=min([(center,objective(center)),(c,fc),(d,fd)],key=lambda item:(item[1],item[0]))
    delta=(la[:-1]-interpolator(np.mod(phase+shift,1.)))/scale
    if not np.isfinite(delta).all() or not np.isfinite(value):raise ValueError('Alignment arithmetic nonfinite')
    return dict(grid=n,fft_index=index,shift_fraction=float(shift%1),scaled_rms=float(np.sqrt(value)),
        scaled_max=float(np.max(abs(delta))),scale=scale,golden_steps=64,
        raw_closure_left=la[-1]-la[0],raw_closure_right=lb[-1]-lb[0])


def duplicate_comparison(left,right):
    row=dict(left_id=left.get('id'),right_id=right.get('id'),complete=False,
             duplicate=False,near_duplicate=False,distinctness_pass=False)
    try:
        same=all(_number(left[k],k)==_number(right[k],k) for k in ('alpha','gamma'))
        row['same_exact_parameters']=same
        if left.get('qualified') is not True or right.get('qualified') is not True:raise ValueError('Qualified complete cycle descriptors required')
        a,b=left['loop'],right['loop'];T0,T1=(_number(v['period'],'loop period',True) for v in (a,b))
        period=abs(T0-T1)/max(T0,T1);coarse=_aligned_loop(a,b,4096);fine=_aligned_loop(a,b,8192)
        near=period<=1e-8 and fine['scaled_rms']<=1e-8
        row.update(complete=True,period_relative=period,coarse=coarse,fine=fine,
            coarse_fine_rms_difference=abs(coarse['scaled_rms']-fine['scaled_rms']),
            duplicate=bool(same and near),near_duplicate=bool(not same and near),distinctness_pass=bool(not same or not near),
            scope='Numerical time-preserving loop similarity; no rigorous orbit or branch identity proof')
    except (ValueError,KeyError,TypeError,IndexError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:
        row['error']=f'{type(exc).__name__}: {exc}'
        row['distinctness_pass']=row.get('same_exact_parameters') is False
    return row


def _slot(slot):
    if not isinstance(slot,Mapping) or slot.get('id') not in SLOT_IDS:raise ValueError('Declared slot required')
    ident=slot['id'];anchor=ident.startswith('anchor-');source=ident.removeprefix('anchor-') if anchor else ident.rsplit('-rho',1)[0]
    for key,value in dict(cycle_id=ident,role='anchor' if anchor else 'candidate',source_setting=source).items():
        if key in slot:_equal(slot[key],value,'slot '+key)
        elif key!='cycle_id':raise ValueError('Missing slot '+key)
    target=None if anchor else (.25 if ident.endswith('025') else .5)
    _equal(slot.get('target_rho'),target,'slot target rho')
    if type(slot.get('available')) is not bool:raise ValueError('Explicit slot availability required')
    return slot


def _slot_packet(slot,packet):
    _slot(slot)
    if not isinstance(packet,Mapping):raise ValueError('Packet unavailable')
    for key,value in dict(setting=slot['id'],cycle_id=slot['id'],source_setting=slot['source_setting'],role=slot['role']).items():
        _equal(packet.get(key),value,'cycle packet '+key)
    _tree(packet['seed_provenance'],slot['seed_provenance'],'seed provenance')
    for key in ('alpha','gamma'):_equal(packet[key],slot[key],'declared parameter '+key)


def _cycle_nested(ident,shoots,reference):
    rows=[];reference=reference if isinstance(reference,Mapping) else {}
    for m in METHODS:
        p=shoots.get(m,shoots.get(ident+'--'+m,{}));trials=p.get('trials',[]) if isinstance(p,Mapping) else []
        by={r.get('index'):r for r in trials if isinstance(r,Mapping)}
        for i in range(25):
            r=by.get(i,{})
            rows.append(dict(id=f'{ident}--{m}--trial-{i:02d}',kind='shoot_trial',method=m,index=i,
                status=r.get('status','unavailable'),attempted=r.get('attempted',False),raw_available=bool(r),
                complete=r.get('ivp_completed',False),error=r.get('error')))
    for cm,m in itertools.product(METHODS,METHODS):
        r=reference.get('verifications',{}).get(cm+'--'+m,{})
        rows.append(dict(id=f'{ident}--verify--{cm}--{m}',kind='verification',method=m,candidate_method=cm,
            status=r.get('status','unavailable'),attempted=r.get('ivp_started',False),complete=r.get('ivp_completed',False),raw_available=bool(r),error=r.get('error')))
    for m in METHODS:
        r=reference.get('adjoints',{}).get(m,{})
        rows.append(dict(id=f'{ident}--adjoint--{m}',kind='adjoint',method=m,
            status=r.get('status','unavailable'),attempted=r.get('ivp_started',False),complete=r.get('ivp_completed',False),raw_available=bool(r),error=r.get('error')))
    return rows


def evaluate_cycle(slot,shoots_by_method,reference_bundle,seed_reference,phases=None,*,rhs=hopfield_rhs,jacobian=hopfield_jacobian):
    """Validate every supplied complete computation and retain failures."""
    slot=_slot(slot);phases=_phase_rows(phase_census() if phases is None else phases);ident=slot['id']
    shots=shoots_by_method if isinstance(shoots_by_method,Mapping) else {}
    row=dict(id=ident,slot=copy.deepcopy(slot),complete=False,qualified=False,status='unavailable',
        arithmetic_replay_pass=False,shoots=[],reference=None,descriptors=None,loop=None,
        alpha=slot.get('alpha'),gamma=slot.get('gamma'),nested_slots=_cycle_nested(ident,shots,reference_bundle))
    valid={}
    for method in METHODS:
        packet=shots.get(method,shots.get(ident+'--'+method));r=dict(id=ident+'--'+method,complete=False,qualified=False,
            converged=False,array_census=partial_array_census(packet))
        try:
            _slot_packet(slot,packet);_slot_packet(slot,seed_reference)
            validate_shoot_packet(packet,seed_reference,ident+'--'+method,rhs=rhs,jacobian=jacobian)
            r.update(complete=True,qualified=packet['qualified'],converged=packet['converged'],status=packet['status'],candidate=packet['candidate'])
            valid[method]=packet
        except (ValueError,KeyError,TypeError,IndexError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as exc:
            r['error']=f'{type(exc).__name__}: {exc}'
            if isinstance(packet,Mapping) and isinstance(seed_reference,Mapping):
                r['partial_replay']=validate_partial_packet(packet,seed_reference,phases,rhs=rhs,jacobian=jacobian)
        row['shoots'].append(r)
    try:
        if not slot['available']:raise ValueError('Historical slot unavailable; no replacement')
        _slot_packet(slot,reference_bundle);_slot_packet(slot,seed_reference)
        validate_reference_packet(reference_bundle,valid,seed_reference,phases,ident+'--reference-pair',rhs=rhs,jacobian=jacobian)
        p=reference_bundle
        row.update(arithmetic_replay_pass=True,complete=all(r['complete'] for r in row['shoots']) and p['complete'],status=p['status'],
            reference=dict(id=p['id'],complete=p['complete'],qualified=p['qualified'],gates=p['gates'],
                verifications={k:dict(complete=v['complete'],qualified=v.get('qualified',False),status=v['status'],certificate=v.get('certificate')) for k,v in p['verifications'].items()}))
        if p.get('complete') is True:
            primary,alternate=(p['verifications'][METHODS[0]+'--'+m]['forward']['trajectory'] for m in METHODS)
            T=_number(p['canonical']['period'],'period',True)
            modes=mode_resolution(primary['phi'][-1],alternate['phi'][-1],T)
            target=slot['target_rho'];bin_details=None if target is None else rho_bin_containment(modes,target)
            bin_pass=True if bin_details is None else bin_details['passed']
            row['descriptors']=dict(period=T,mode_resolution=modes,target_rho=target,rho_bin_pass=bin_pass,
                rho_bin_comparison=bin_details,
                rho_bin_semantics='both observed methods plus descriptive resolution radius inside frozen bin; no rigorous eigenvalue bound')
            row['loop']=dict(period=T,states=primary['states'].copy(),derivatives=primary['derivatives'].copy())
            row['qualified']=bool(row['complete'] and all(r['qualified'] for r in row['shoots']) and p['qualified'] and bin_pass)
    except (ValueError,KeyError,TypeError,IndexError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as exc:
        row['error']=f'{type(exc).__name__}: {exc}'
        if isinstance(reference_bundle,Mapping) and isinstance(seed_reference,Mapping):
            row['partial_replay']=validate_partial_packet(reference_bundle,seed_reference,phases,shoot_packets=shots,rhs=rhs,jacobian=jacobian)
    return row


def _tangent_physics(packet,context,method,*,rhs,jacobian):
    validate_tangent_packet(packet,context,method,rhs=rhs,jacobian=jacobian)


def validate_tangent_packet(packet,context,method,*,allow_partial=False,rhs=hopfield_rhs,jacobian=hopfield_jacobian):
    """Replay present complete segments; retain failed arrays without a solve claim.

    With allow_partial=True, an explicit incomplete prefix can pass integrity
    replay but never becomes a qualified reference or a completed parent job.
    """
    T=_context(context);checks=[]
    if not isinstance(packet,Mapping):raise ValueError('Tangent packet required')
    _exact(packet['id'],context['id']+'--reference--'+method,'reference job id')
    _exact(packet['kind'],'tangent','tangent kind')
    for key in ('setting','cycle_id','source_setting','role','alpha','gamma'):_exact(packet[key],context[key],'tangent '+key)
    for key in ('phase_index','phase_fraction','split','phase_epoch'):_exact(packet[key],context[key],'tangent phase '+key)
    _exact(packet['context_id'],context['id'],'tangent context');_exact(packet['method'],method,'tangent method')
    _exact(packet['period'],T,'tangent period');_exact(packet['event_time'],T,'tangent event')
    _tree(packet['seed_provenance'],context['seed_provenance'],'tangent seed provenance')
    declared=dict(acquisition.TANGENT_METHODS[method],id=method,max_step=T/acquisition.TANGENT_METHODS[method]['max_step_period_divisor'])
    _exact(packet['method_spec'],declared,'tangent method spec')
    if type(packet.get('complete')) is not bool:raise ValueError('Explicit tangent completeness required')
    if not allow_partial and packet['complete'] is not True:raise ValueError('Complete tangent required')
    if packet['complete']:_exact(packet['status'],'complete','complete tangent status')
    elif packet.get('status')=='complete':raise ValueError('Incomplete tangent labeled complete')
    if set(packet['ivp_slots'])!={'pre','post'}:raise ValueError('Both planned tangent segments required')
    grid=acquisition.build_grid(T)
    for name in ('pre','post'):
        record=packet['ivp_slots'][name]
        _exact(record['id'],packet['id']+'--'+name,'segment IVP identity')
        for key in ('ivp_started','ivp_returned','ivp_completed'):
            if type(record.get(key)) is not bool:raise ValueError('Boolean segment lifecycle flags required')
        if record['ivp_completed'] and not(record['ivp_started'] and record['ivp_returned']):raise ValueError('Contradictory completed IVP flags')
        if record['ivp_returned'] and not record['ivp_started']:raise ValueError('IVP returned without starting')
        if name not in packet:
            if packet['complete']:raise ValueError('Complete tangent missing segment')
            checks.append(dict(segment=name,available=False,physical_replay=False));continue
        g=packet[name];offset=0. if name=='pre' else T
        _exact(g['times'],grid[name],name+' exact grid');_exact(g['offset'],offset,name+' offset')
        N=len(grid[name]);_array(g['states'],name+' states',(N,3));_array(g['derivatives'],name+' derivatives',(N,3))
        _array(g['phi'],name+' phi',(N,3,3));_array(g['phi_derivatives'],name+' phi derivatives',(N,3,3))
        legacy_observer._solver(g,method,T,tangent=True)
        initial=context['initial'] if name=='pre' else packet['pre']['states'][-1]
        _exact(g['initial'],initial,name+' initial');_exact(g['states'][0],initial,name+' stored initial')
        _exact(g['initial_phi'],np.eye(3),name+' initial Phi');_exact(g['phi'][0],np.eye(3),name+' saved initial Phi')
        for field in ('propagated_directly','full_tangent_reinitialized_at_segment_start'):_exact(g[field],True,name+' '+field)
        g=packet[name];x=g['states'];a,b=context['alpha'],context['gamma']
        if rhs is hopfield_rhs and jacobian is hopfield_jacobian:
            act=np.tanh(x);W=weight_matrix(a,b)
            f=-x+act@W.T;J=-np.eye(3)[None,:,:]+W[None,:,:]*(1.-act*act)[:,None,:]
            dphi=J@g['phi']
        else:
            clock=context['phase_epoch']+g['offset']+g['times']
            f=np.asarray([rhs(float(t),v,a,b) for t,v in zip(clock,x,strict=True)],dtype=np.float64)
            dphi=np.asarray([jacobian(float(t),v,a,b)@P for t,v,P in zip(clock,x,g['phi'],strict=True)],dtype=np.float64)
        _strict(g['derivatives'],f,'physical RHS '+name)
        _strict(g['phi_derivatives'],dphi,'physical J@Phi '+name)
        for key in ('ivp_started','ivp_returned','ivp_completed'):_exact(record[key],True,'complete segment '+key)
        _exact(record['status'],'complete','segment status')
        checks.append(dict(segment=name,available=True,physical_replay=True))
    return dict(partial_replay_pass=True,complete=packet['complete'],qualified=False,checks=checks,
        array_census=partial_array_census(packet),scope='Present complete segment arithmetic only; partial solver buffers retained without a physical-solution claim')


def _descriptor_windows(context,tangent,sealed,bases):
    windows={}
    for window in WINDOWS:
        sample=acquisition.readout_samples(tangent,window);Phi=tangent[sample['segment']]['phi'][sample['indices']]
        first=Phi if window=='immediate' else Phi@tangent['pre']['phi'][-1]
        second=np.zeros_like(Phi) if window=='immediate' else Phi
        activation_factor=1.-sample['activations']**2
        OA=(activation_factor[:,:,None]*first).reshape(96,3)
        OB=(activation_factor[:,:,None]*second).reshape(96,3)
        p=sealed['windows'][window]['linear_predictions']['identity'];A,B=p['A'],p['B']
        gram=np.array([[A@A,A@B],[B@A,B@B]],dtype=np.float64)
        windows[window]=dict(A=A.copy(),B=B.copy(),gram=gram,noise_scaled_gram=gram/SIGMA**2,
            first_detectability=float(np.linalg.norm(A)/SIGMA),second_detectability=float(np.linalg.norm(B)/SIGMA),
            observable_first_operator=OA,observable_second_operator=OB,
            first_observable_gain=float(np.linalg.norm(OA,2)),second_observable_gain=float(np.linalg.norm(OB,2)),
            first_state_gain_max=float(np.max(np.linalg.svd(first,compute_uv=False)[:,0])),
            second_state_gain_max=float(np.max(np.linalg.svd(second,compute_uv=False)[:,0])),
            modal_first=_modal_readout(bases['first_write'],OA,A),
            modal_second=_modal_readout(bases['second_write'],OB,B),
            reference_linear_accuracy=copy.deepcopy(p['primary_accuracy']),
            scope='Unforced-reference prediction and tangent gain; no finite-write outcome')
    return windows


def seal_context(slot,context,tangent_primary,tangent_alternate,*,rhs=hopfield_rhs,jacobian=hopfield_jacobian):
    """Reproduce scientific gates with no finite pulse or mixture decoder."""
    slot=_slot(slot);ident=context.get('id') if isinstance(context,Mapping) else None
    result=dict(id=ident,context_id=ident,cycle_id=slot['id'],complete=False,qualified=False,status='unavailable',
        seal=None,descriptors=None,nested_slots=[],arithmetic_replay_pass=False)
    i=context.get('phase_index') if isinstance(context,Mapping) else None
    result.update(phase_index=i,phase_fraction=context.get('phase_fraction') if isinstance(context,Mapping) else None)
    for m,p in zip(TANGENT_METHODS,(tangent_primary,tangent_alternate),strict=True):
        p=p if isinstance(p,Mapping) else {};rows=p.get('ivp_slots',{})
        for segment in ('pre','post'):
            r=rows.get(segment,{})
            result['nested_slots'].append(dict(id=f'{ident}--reference--{m}--{segment}',kind='tangent_segment',method=m,segment=segment,
                status=r.get('status','unavailable'),attempted=r.get('ivp_started',False),complete=r.get('ivp_completed',False),raw_available=bool(r),error=r.get('error')))
    try:
        _slot_packet(slot,context);_context(context)
        _strict(context['flow'],np.asarray(rhs(context['phase_epoch'],context['initial'],context['alpha'],context['gamma']),dtype=np.float64),'physical initial flow')
        for m,p in zip(TANGENT_METHODS,(tangent_primary,tangent_alternate),strict=True):
            _slot_packet(slot,p);_tangent_physics(p,context,m,rhs=rhs,jacobian=jacobian)
        sealed=_seal_reference(context,tangent_primary,tangent_alternate);result['seal']=sealed
        result.update(complete=sealed['complete'],qualified=sealed['qualified'],status=sealed['status'],
                      arithmetic_replay_pass=bool(sealed.get('arithmetic_qualified',False)))
        if sealed['complete']:
            # The post segment uses step T/4096; index4096 is exactly one
            # period after the second write, with its own fresh tangent I.
            post_one_period=4096
            _exact(tangent_primary['post']['times'][post_one_period],context['period'],'second-write one-period epoch')
            bases=dict(first_write=_modal_basis(tangent_primary['pre']['phi'][-1],EPSILON*context['direction']),
                second_write=_modal_basis(tangent_primary['post']['phi'][post_one_period],EPSILON*context['direction']))
            result['descriptors']=dict(period=context['period'],epsilon=EPSILON,sigma=SIGMA,
                windows=_descriptor_windows(context,tangent_primary,sealed,bases),modal_bases=bases,
                immediate_detectability=float(np.linalg.norm(sealed['windows']['immediate']['linear_predictions']['identity']['A'])/SIGMA),
                source='primary directly propagated canonical-reference tangent; alternate gates retained in seal',finite_writes=0)
        if 'error' in sealed:result['error']=sealed['error']
    except (ValueError,KeyError,TypeError,IndexError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    return result


def _rms_detectability(cycle,contexts):
    values=[]
    for i in range(8):
        row=contexts[cycle+f'--p{i:02d}']
        if row.get('qualified') is not True:raise ValueError('All eight qualified phase descriptors required')
        value=_number(row['descriptors']['immediate_detectability'],'immediate detectability')
        if value<0:raise ValueError('Nonnegative detectability required')
        values.append(value)
    return dict(values=values,rms=math.hypot(*values)/math.sqrt(8))


def match_pair(target,left,right,contexts):
    row=dict(target_rho=target,left_id=left['id'],right_id=right['id'],complete=False,matched=False,gates={})
    try:
        if target not in (.25,.5):raise ValueError('Known multiplier bin required')
        if left.get('qualified') is not True or right.get('qualified') is not True:raise ValueError('Both planned cycles must qualify')
        a,b=(c['descriptors'] for c in (left,right))
        for d in (a,b):
            _equal(d['target_rho'],target,'matched rho target')
            if d['rho_bin_pass'] is not True or d['mode_resolution']['dominant_positive_resolved'] is not True:raise ValueError('Resolved positive dominant mode in original bin required')
        T0,T1=(_number(d['period'],'period',True) for d in (a,b));period=abs(T0-T1)/max(T0,T1)
        da,db=(_rms_detectability(c['id'],contexts) for c in (left,right));x,y=da['rms'],db['rms']
        signal=abs(x-y)/max(x,y,1e-12);sufficient=not(x<1e-6 and y<1e-6)
        gates=dict(period=period<=.02,immediate_detectability=signal<=.1,sufficient_detectability=sufficient,
                   signed_dominant_mode=True,original_bin=True)
        row.update(complete=True,matched=all(gates.values()),gates=gates,period_relative=period,
            immediate_rms_relative=signal,left_immediate=da,right_immediate=db,
            scope='Fixed feasibility calipers, not practical-utility thresholds or a geometry-effect estimate')
    except (ValueError,KeyError,TypeError,IndexError,OverflowError,FloatingPointError) as exc:row['error']=f'{type(exc).__name__}: {exc}'
    return row


def summarize(slots,cycle_rows,context_rows,*,phase_rows=None):
    """Keep all6/48/2/15 rows and all528 nested slots, including failures."""
    phases=_phase_rows(phase_census() if phase_rows is None else phase_rows)
    if not isinstance(slots,(list,tuple)) or [s['id'] for s in slots]!=list(SLOT_IDS):raise ValueError('Six fixed ordered slots required')
    for s in slots:_slot(s)
    cycles=cycle_rows if isinstance(cycle_rows,Mapping) else {r['id']:r for r in cycle_rows}
    contexts=context_rows if isinstance(context_rows,Mapping) else {r['context_id']:r for r in context_rows}
    if len(cycles)!=len(cycle_rows) or len(contexts)!=len(context_rows):raise ValueError('Duplicate result identity')
    wanted={s+f'--p{i:02d}' for s in SLOT_IDS for i in range(8)}
    errors=[]
    for label,supplied,allowed in [('cycles',cycles,set(SLOT_IDS)),('contexts',contexts,wanted)]:
        if set(supplied)-allowed:errors.append(dict(scope=label,unexpected_ids=sorted(set(supplied)-allowed)))
    retained_cycles=[];retained_contexts=[];nested=[]
    for slot in slots:
        s=slot['id'];c=cycles.get(s)
        if c is None:c=dict(id=s,slot=copy.deepcopy(slot),alpha=slot.get('alpha'),gamma=slot.get('gamma'),complete=False,qualified=False,status='unavailable',nested_slots=_cycle_nested(s,{},None))
        if c['id']!=s:raise ValueError('Misbound cycle map')
        _tree(c['slot'],slot,'planned retained slot')
        for key in ('alpha','gamma'):_exact(c[key],slot.get(key),'retained cycle parameter '+key)
        for key in ('complete','qualified'):
            if type(c.get(key)) is not bool:raise ValueError('Explicit cycle status required')
        if c['qualified'] and not c['complete']:raise ValueError('Cycle qualified without complete reference computation')
        retained_cycles.append(c)
        if len(c.get('nested_slots',[]))!=56:raise ValueError('Exactly56 nested cycle slots required')
        nested.extend(c['nested_slots'])
        for i,p in enumerate(phases):
            ident=s+f'--p{i:02d}';r=contexts.get(ident)
            if r is None:
                r=dict(id=ident,context_id=ident,cycle_id=s,phase_index=i,phase_fraction=p['phase_fraction'],complete=False,qualified=False,status='unavailable',descriptors=None,
                    nested_slots=[dict(id=f'{ident}--reference--{m}--{seg}',kind='tangent_segment',method=m,segment=seg,status='unavailable',attempted=False,complete=False,raw_available=False) for m in TANGENT_METHODS for seg in ('pre','post')])
            for key,value in dict(id=ident,context_id=ident,cycle_id=s,phase_index=i,phase_fraction=p['phase_fraction']).items():_equal(r[key],value,'retained context '+key)
            if len(r.get('nested_slots',[]))!=4:raise ValueError('Exactly four tangent segment slots required')
            for key in ('complete','qualified'):
                if type(r.get(key)) is not bool or type(c.get(key)) is not bool:raise ValueError('Explicit result status required')
            if r['qualified'] and not(c['qualified'] and r['complete']):raise ValueError('Context qualified without cycle prerequisites')
            retained_contexts.append(r);nested.extend(r['nested_slots'])
    if len(nested)!=528 or len({r['id'] for r in nested})!=528:raise ValueError('Complete unique528 IVP-slot census required')
    expected={}
    for s in SLOT_IDS:
        for r in _cycle_nested(s,{},None):expected[r['id']]=r['kind']
        for i in range(8):
            for m in TANGENT_METHODS:
                for seg in ('pre','post'):expected[f'{s}--p{i:02d}--reference--{m}--{seg}']='tangent_segment'
    for r in nested:
        if expected.get(r['id'])!=r['kind']:raise ValueError('Nested slot identity/kind differs from exact census')
        for field in ('attempted','complete','raw_available'):
            if type(r[field]) is not bool:raise ValueError('Nested slot status must be Boolean')
        if type(r['status']) is not str:raise ValueError('Nested slot needs explicit status')
    cm={c['id']:c for c in retained_cycles};pm={c['context_id']:c for c in retained_contexts}
    pairs=[match_pair(t,cm['core-rho'+suffix],cm['failed-near-edge-rho'+suffix],pm) for t,suffix in ((.25,'025'),(.5,'050'))]
    duplicates=[duplicate_comparison(a,b) for a,b in itertools.combinations(retained_cycles,2)]
    complete=bool(not errors and all(c['complete'] for c in retained_cycles+retained_contexts))
    numerical=bool(complete and all(c['qualified'] for c in retained_cycles+retained_contexts))
    distinct=all(d['distinctness_pass'] for d in duplicates)
    feasible=bool(numerical and distinct and all(p['matched'] for p in pairs))
    return dict(complete=complete,qualified=feasible,feasible=feasible,numerical_qualified=numerical,
        all_cycles_distinct=distinct,cycles=retained_cycles,contexts=retained_contexts,pairs=pairs,
        duplicate_comparisons=duplicates,nested_slots=nested,input_errors=errors,contract=contract(),
        counts=dict(cycles_planned=6,cycles_complete=sum(c['complete'] for c in retained_cycles),cycles_qualified=sum(c['qualified'] for c in retained_cycles),
            contexts_planned=48,contexts_complete=sum(c['complete'] for c in retained_contexts),contexts_qualified=sum(c['qualified'] for c in retained_contexts),
            matching_pairs_planned=2,matching_pairs_complete=sum(p['complete'] for p in pairs),matching_pairs_pass=sum(p['matched'] for p in pairs),
            duplicate_comparisons_planned=15,duplicate_comparisons_complete=sum(d['complete'] for d in duplicates),nested_slots_planned=528,nested_slots_retained=len(nested)),
        finite_write_outcomes=0,original_studies_requalified=False,
        scope='Selected unforced-cycle matching feasibility; reference predictions only, no finite memory outcome or geometry-specific advantage')


def _base(packet,canonical,forward,phases,*,rhs,jacobian):
    b=packet;T=canonical['period'];m=b['method'];s=b['setting'];z=canonical['terminal_adjoint'];x=canonical['state']
    if b.get('complete') is not True or b.get('status')!='complete':raise ValueError('Complete refined base required')
    _ivp_complete(b,'complete base');_equal(b['schema'],'refined-phase-base-v1','new base schema');_equal(b['kind'],'base','new base kind');_equal(b['id'],s+'--'+m,'new base id');_equal(b['period'],T,'canonical period')
    _equal(b['canonical_state'],x,'canonical state');_equal(b['global_terminal_adjoint'],z,'global terminal');_equal(b['new_normalizations'],0,'no base renormalization')
    _equal(b['gradient_origin'],'one global normalization of canonical DOP853 verification left-neutral vector','gradient provenance')
    _tree(b['method_spec'],_method(m,T),'base method');_tree(b['forward'],forward['solver'],'original verified forward')
    _solver(b['backward'],m,T,z,backward=True)
    if set(b['grids'])!=set(GRIDS):raise ValueError('Both refined grids required')
    for key in GRIDS:
        g=b['grids'][key];old_math._samples(g,T,int(key),adjoint=True,rhs=rhs,jacobian=jacobian,alpha=b['alpha'],gamma=b['gamma'])
        _equal(g['states'][0],x,'new section initial');_equal(g['z'][-1],z,'unchanged canonical terminal')
    fine=b['grids']['8192'];coarse=b['grids']['4096']
    for key in coarse:_equal(coarse[key],fine[key][::2],'nested grid '+key)
    for key in ('times','states','phi','derivatives','phi_derivatives'):_equal(fine[key],forward['trajectory'][key],'verification path reused '+key)
    for key,val in dict(initial_state=fine['states'][0],final_state=fine['states'][-1],initial_phi=fine['phi'][0],final_phi=fine['phi'][-1],initial_z=fine['z'][0],final_z=fine['z'][-1]).items():_equal(b['raw_endpoints'][key],val,'raw endpoint '+key)
    if len(b['phases'])!=8 or len(b['phase_rows'])!=8:raise ValueError('Complete8 phase rows required')
    exact=b['exact_phase_arrays']
    for key,shape in dict(times=(8,),states=(8,3),phi=(8,3,3),derivatives=(8,3),phi_derivatives=(8,3,3),z=(8,3),z_derivatives=(8,3),normalization_residuals=(8,)).items():_array(exact[key],'dense '+key,shape)
    for i,(p,w) in enumerate(zip(b['phases'],phases,strict=True)):
        metadata=dict(w,context_id=f'{s}--p{i:02d}',phase_epoch=w['phase_fraction']*T)
        for key,value in metadata.items():_equal(p[key],value,'new phase '+key)
        _tree(b['phase_rows'][i],metadata,'phase rows')
        for pk,ak in [('state','states'),('phi','phi'),('flow','derivatives'),('phi_derivative','phi_derivatives'),('z','z'),('z_derivative','z_derivatives'),('normalization_residual','normalization_residuals'),('phase_epoch','times')]:_equal(p[pk],exact[ak][i],'exact new phase '+pk)
        f=np.asarray(rhs(p['phase_epoch'],p['state'],b['alpha'],b['gamma']),dtype=np.float64);J=jacobian(p['phase_epoch'],p['state'],b['alpha'],b['gamma'])
        _strict(p['flow'],f,'new dense flow');_strict(p['phi_derivative'],J@p['phi'],'new dense Phi derivative');_strict(p['z_derivative'],-J.T@p['z'],'new dense z derivative')
        _tree(p['normalization_residual'],float(p['z']@f-1.),'new phase normalization')
    z0,zT=fine['z'][0],fine['z'][-1];periodicity=float(np.linalg.norm(z0-zT)/max(np.linalg.norm(z0),np.linalg.norm(zT)))
    diagnostic=dict(adjoint_relative_periodicity=periodicity,adjoint_periodic_difference=z0-zT,grid_normalization_max=normalization(fine['z'],fine['derivatives'])['max_abs'],exact_phase_normalization_max=normalization(exact['z'],exact['derivatives'])['max_abs'])
    gates=dict(adjoint_periodicity=periodicity<=1e-10,grid_normalization=diagnostic['grid_normalization_max']<=1e-7,exact_phase_normalization=diagnostic['exact_phase_normalization_max']<=1e-7)
    _tree(b['diagnostics'],diagnostic,'base diagnostics');_tree(b['gates'],gates,'base gates');_equal(b['qualified'],all(gates.values()),'base qualification')
    return b


def validate_reference_packet(packet,shoot_packets,old_reference,phase_rows,expected_id=None,*,rhs=hopfield_rhs,jacobian=hopfield_jacobian):
    """Validate retained subcalculations even when a bundle cannot qualify."""
    p=packet;old=old_reference;phases=_phase_rows(phase_rows);s=old['setting'];ident=s+'--reference-pair'
    if not isinstance(p,Mapping):raise ValueError('Reference bundle unavailable')
    _equal(p['id'],ident,'reference id');_equal(p['schema'],'periodic-reference-pair-v1','reference schema');_equal(p['kind'],'reference','reference kind')
    if expected_id is not None:_equal(ident,expected_id,'planned reference')
    for key in ('setting','alpha','gamma'):_equal(p[key],old[key],'reference original '+key)
    _equal(p['original_state'],np.asarray(old['certificate']['section_state'],dtype=np.float64),'reference original state');_equal(p['original_period'],float(old['period']),'reference original T');_equal(p['canonical_method'],METHODS[0],'fixed canonical method')
    shots={m:shoot_packets.get(s+'--'+m,shoot_packets.get(m)) for m in METHODS}
    for m,shot in shots.items():
        validate_shoot_packet(shot,old,s+'--'+m,rhs=rhs,jacobian=jacobian)
        if shot['converged'] is not True:raise ValueError('Reference dispatched from nonconverged shoot')
        _tree(p['candidates'][m],shot['candidate'],'unchanged candidate '+m)
    c=candidate_comparison(shots[METHODS[0]]['candidate'],shots[METHODS[1]]['candidate'],old)
    _tree(p['candidate_agreement'],dict(state_scaled_infinity=c['state_scaled_infinity'],period_relative=c['period_relative'],state_difference=c['state_difference'],period_difference=c['period_difference'],state_threshold=1e-10,period_threshold=1e-11,qualified=c['qualified']),'candidate agreement')
    keys={a+'--'+b for a in METHODS for b in METHODS}
    if set(p['verifications'])!=keys or set(p['adjoints'])!=set(METHODS):raise ValueError('Exactly4 verification and2 adjoint entries required')
    if set(p['bases'])-set(METHODS):raise ValueError('Unexpected reconstructed base')
    valid={};errors=[]
    for cm,m in itertools.product(METHODS,METHODS):
        key=cm+'--'+m;r=p['verifications'][key]
        try:
            _equal(r['id'],f'{s}--verify--{cm}--{m}','nested verification id');_equal(r['candidate_method'],cm,'verification candidate');_equal(r['method'],m,'verification method')
            if r.get('complete') is not True:continue
            _ivp_complete(r,'complete verification');_equal(r['status'],'complete','verification status')
            cand=shots[cm]['candidate'];_forward(r['forward'],cand['state'],cand['period'],m,old['alpha'],old['gamma'],rhs=rhs,jacobian=jacobian,events=True)
            cert=_certificate(r['forward'],old['alpha'],old['gamma'],rhs=rhs,jacobian=jacobian);_tree(r['certificate'],cert,'verification certificate '+key);_equal(r['qualified'],cert['qualified'],'verification qualification');valid[key]=r
        except (ValueError,KeyError,TypeError,IndexError,OverflowError,FloatingPointError) as exc:errors.append(key+': '+str(exc))
    if errors:raise ValueError('; '.join(errors))
    gates=dict(candidate_agreement=c['qualified'],all_cross_verifications=len(valid)==4 and all(v['qualified'] for v in valid.values()))
    if len(valid)==4:
        agreements={cm:array_agreement(valid[cm+'--'+METHODS[0]]['forward']['trajectory'],valid[cm+'--'+METHODS[1]]['forward']['trajectory']) for cm in METHODS}
        _tree(p['candidate_method_agreements'],agreements,'candidate method agreements')
        _tree(p['canonical_method_agreement'],agreements[METHODS[0]],'canonical agreement alias')
        gates['canonical_method_agreement']=agreements[METHODS[0]]['agreement_pass'];gates['both_candidate_method_agreements']=all(a['agreement_pass'] for a in agreements.values())
    else:gates.update(canonical_method_agreement=False,both_candidate_method_agreements=False)
    if not all(gates.values()):
        _tree(p['gates'],gates,'blocked crossverification gates');_equal(p['qualified'],False,'blocked reference qualified');_equal(p['complete'],False,'blocked reference complete');_equal(p['status'],'blocked_crossverification','blocked reference status')
        if 'canonical' in p or p['bases']:raise ValueError('Adjoint construction despite failed crosschecks')
        for r in p['adjoints'].values():
            if r.get('complete') is True or r.get('ivp_started') is not False or r.get('ivp_completed') is not False:raise ValueError('Adjoint dispatched after blocked prerequisite')
            _equal(r['status'],'blocked_prerequisite','blocked adjoint status')
        return p
    source=valid[METHODS[0]+'--'+METHODS[0]]['forward'];tr=source['trajectory'];canon=p['canonical'];candidate=shots[METHODS[0]]['candidate']
    expected_neutral=neutral_left_phase(tr['phi'][-1],tr['derivatives'][0],residual_atol=1e-10,unique_singular_rtol=1e-12,normalization_cosine_min=1e-12)
    _,singular,vh=np.linalg.svd(tr['phi'][-1].T-np.eye(3),full_matrices=False)
    _equal(canon['state'],candidate['state'],'canonical state');_equal(canon['period'],candidate['period'],'canonical period');_equal(canon['normalization_count'],1,'one new global normalization');_equal(canon['source_verification'],METHODS[0]+'--'+METHODS[0],'canonical source')
    _strict(canon['unnormalized_svd_vector'],vh[-1],'canonical raw SVD');_strict(canon['singular_values'],singular,'canonical singular values');_tree(canon['normalization_scalar'],float(vh[-1]@tr['derivatives'][0]),'canonical scale');_tree(canon['normalization_applied_scale'],1./canon['normalization_scalar'],'canonical applied scale');_tree(canon['neutral'],expected_neutral,'canonical neutral diagnostics');_strict(canon['terminal_adjoint'],expected_neutral['phase_gradient'],'canonical terminal')
    complete=True;qualified=[]
    for m,r in p['adjoints'].items():
        _equal(r['id'],s+'--adjoint--'+m,'adjoint id');_equal(r['method'],m,'adjoint method')
        if r.get('complete') is not True:
            complete=False;qualified.append(False);continue
        base=p['bases'][m];_ivp_complete(r,'complete adjoint');_equal(r['status'],'complete','adjoint status');_equal(base['method'],m,'base map method');_equal(base['id'],s+'--'+m,'base map id')
        for key in ('setting','alpha','gamma'):_equal(base[key],old[key],'base original '+key)
        _base(base,canon,valid[METHODS[0]+'--'+m]['forward'],phases,rhs=rhs,jacobian=jacobian)
        _equal(r['base_id'],base['id'],'adjoint base id');_equal(r['qualified'],base['qualified'],'adjoint qualification');qualified.append(base['qualified'])
    gates['both_reconstructed_bases']=all(qualified);_tree(p['gates'],gates,'reference gates');_equal(p['complete'],complete,'reference complete');_equal(p['qualified'],all(gates.values()),'reference qualified');_equal(p['status'],'complete' if complete else 'incomplete_adjoint','reference terminal status')
    return p


def validate_partial_packet(packet,old_reference,phase_rows,*,shoot_packets=None,reference_packet=None,rhs=hopfield_rhs,jacobian=hopfield_jacobian):
    """Audit completed nested calculations inside incomplete saved evidence.

    Failed solver arrays remain an explicit census, including nonfinite values;
    they are not asserted to solve the ODE. No incomplete outer packet is
    promoted to a qualified result. All completed nested checks are attempted.
    """
    p=packet;old=old_reference;phases=_phase_rows(phase_rows);checks=[];errors=[]
    if not isinstance(p,Mapping):return dict(partial_replay_pass=False,checks=[],errors=['Missing packet'],array_census=[])
    def attempt(name,fn):
        try:fn();checks.append(dict(path=name,passed=True))
        except (ValueError,KeyError,TypeError,IndexError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as exc:
            message=f'{type(exc).__name__}: {exc}';checks.append(dict(path=name,passed=False,error=message));errors.append(name+': '+message)
    for key in ('setting','alpha','gamma'):
        if key in p:attempt('binding.'+key,lambda key=key:_equal(p[key],old[key],'partial original '+key))
    kind=p.get('kind')
    if kind=='shoot':
        ident=old['setting']+'--'+p['method'];trials=p.get('trials',[])
        attempt('shoot_identity',lambda:_equal(p['id'],ident,'partial shoot id'))
        attempt('original_state',lambda:_equal(p['original_state'],np.asarray(old['certificate']['section_state'],dtype=np.float64),'partial original state'))
        attempt('original_period',lambda:_equal(p['original_period'],old['period'],'partial original T'))
        attempt('trial_census',lambda:_equal(len(trials),25,'partial25 trial slots'))
        for i,row in enumerate(trials):
            attempt(f'trial{i}.identity',lambda i=i,row=row:_tree(dict(index=row['index'],id=row['id']),dict(index=i,id=ident+f'--trial-{i:02d}'),'partial trial identity'))
            forward=row.get('forward')
            if isinstance(forward,Mapping) and forward.get('complete') is True:
                def replay(row=row,forward=forward):
                    tr=_forward(forward,row['state'],row['period'],p['method'],old['alpha'],old['gamma'],rhs=rhs,jacobian=jacobian)
                    if 'metrics' in row:_tree(row['metrics'],_metrics(row['state'],row['period'],tr,old,rhs=rhs,jacobian=jacobian),'partial completed trial metrics')
                attempt(f'trial{i}.completed_forward',replay)
    elif kind=='reference':
        s=old['setting'];attempt('reference_identity',lambda:_equal(p['id'],s+'--reference-pair','partial reference id'))
        attempt('original_state',lambda:_equal(p['original_state'],np.asarray(old['certificate']['section_state'],dtype=np.float64),'partial reference original state'))
        attempt('original_period',lambda:_equal(p['original_period'],old['period'],'partial reference original T'))
        expected={a+'--'+b for a in METHODS for b in METHODS}
        attempt('verification_census',lambda:_equal(set(p['verifications']),expected,'partial verification census'))
        attempt('adjoint_census',lambda:_equal(set(p['adjoints']),set(METHODS),'partial adjoint census'))
        for key,row in p.get('verifications',{}).items():
            cm=row.get('candidate_method');m=row.get('method');forward=row.get('forward')
            attempt(key+'.identity',lambda row=row,cm=cm,m=m:_equal(row['id'],f'{s}--verify--{cm}--{m}','partial verification identity'))
            if isinstance(forward,Mapping) and forward.get('complete') is True:
                def replay(cm=cm,m=m,row=row,forward=forward):
                    cand=p['candidates'][cm]
                    if shoot_packets is not None:
                        shot=shoot_packets.get(s+'--'+cm,shoot_packets.get(cm));_tree(cand,shot['candidate'],'partial candidate binding')
                    _forward(forward,cand['state'],cand['period'],m,old['alpha'],old['gamma'],rhs=rhs,jacobian=jacobian,events=True)
                    if 'certificate' in row:_tree(row['certificate'],_certificate(forward,old['alpha'],old['gamma'],rhs=rhs,jacobian=jacobian),'partial certificate')
                attempt(key+'.completed_forward',replay)
        if 'canonical' in p:
            def canonical():
                tr=p['verifications'][METHODS[0]+'--'+METHODS[0]]['forward']['trajectory'];z=neutral_left_phase(tr['phi'][-1],tr['derivatives'][0],residual_atol=1e-10,unique_singular_rtol=1e-12,normalization_cosine_min=1e-12)
                canon=p['canonical'];cand=p['candidates'][METHODS[0]];_,singular,vh=np.linalg.svd(tr['phi'][-1].T-np.eye(3),full_matrices=False)
                _equal(canon['state'],cand['state'],'partial canonical state');_equal(canon['period'],cand['period'],'partial canonical T');_equal(canon['source_verification'],METHODS[0]+'--'+METHODS[0],'partial canonical source')
                _strict(canon['unnormalized_svd_vector'],vh[-1],'partial raw SVD');_strict(canon['singular_values'],singular,'partial SVD spectrum')
                _tree(canon['normalization_scalar'],float(vh[-1]@tr['derivatives'][0]),'partial normalization denominator');_tree(canon['normalization_applied_scale'],1./canon['normalization_scalar'],'partial applied normalization scale')
                _tree(canon['neutral'],z,'partial canonical neutral');_strict(canon['terminal_adjoint'],z['phase_gradient'],'partial canonical z');_equal(canon['normalization_count'],1,'partial one normalization')
            attempt('canonical_gradient',canonical)
        for m,base in p.get('bases',{}).items():
            if base.get('complete') is True:
                def basecheck(m=m,base=base):
                    _equal(base['method'],m,'partial base map method');_equal(base['id'],s+'--'+m,'partial base map id')
                    return _base(base,p['canonical'],p['verifications'][METHODS[0]+'--'+m]['forward'],phases,rhs=rhs,jacobian=jacobian)
                attempt(m+'.completed_base',basecheck)
    else:errors.append('Unknown partial packet kind')
    return dict(partial_replay_pass=not errors,checks=checks,errors=errors,array_census=partial_array_census(p),qualified=False,scope='Completed saved subcalculations replayed; failed prefixes retained without ODE-solution claim')


def _tangent(packet, context, method):
    period = _context(context)
    _exact(packet.get('context_id'), context['id'], 'tangent context')
    _exact(packet.get('method'), method, 'tangent method')
    declared = dict(acquisition.TANGENT_METHODS[method], id=method, max_step=period/acquisition.TANGENT_METHODS[method]['max_step_period_divisor'])
    _exact(packet.get('method_spec'), declared, 'tangent method_spec')
    _segments(packet, period, method, tangent=True)
    _exact(packet['pre']['states'][0], context['initial'], 'reference initial')
    _exact(packet['post']['states'][0], packet['pre']['states'][-1], 'direct reference event')


def _seal_reference(context, tangent_primary, tangent_alternate):
    """Build all observers from saved references before any pulse is scored.

    The alternate uses the primary's sealed Q/P, so it tests the same physical
    intervention. Direct saved midpoint nodes are used; no new solve or
    interpolation is performed. Exceptions become explicit failed prerequisites.
    """
    result = dict(context_id=context.get('id'), context=copy.deepcopy(context), complete=False,
        qualified=False, status='failed', gates={}, windows={}, reference_comparison=None,
        epsilon=EPSILON, sigma=SIGMA)
    try:
        period = _context(context)
        _tangent(tangent_primary, context, 'dop853')
        _tangent(tangent_alternate, context, 'rk45')
        ctx = copy.deepcopy(context)
        base = tangent_primary['pre']['states'][-1].copy()
        flow = tangent_primary['pre']['derivatives'][-1].copy()
        gradient = ctx['nominal_event_phase_gradient']
        Q = np.outer(flow, gradient)
        for key, expected in dict(reference_at_event=base,event_flow=flow,event_phase_gradient=gradient,Q=Q,P=np.eye(3)-Q).items():
            if key in context:
                _exact(context[key], expected, 'supplied sealed '+key)
        ctx.update(reference_at_event=base, event_flow=flow,
            event_phase_gradient=gradient.copy(), Q=Q, P=np.eye(3)-Q,
            event_normalization_residual=float(gradient@flow-1.),
            event_projector_idempotence_residual=Q@Q-Q,
            event_reference_method='dop853',
            event_reference_semantics='direct unforced state at T; nominal periodic adjoint at phase_epoch+T')
        result['context'] = ctx
        result['local_amplification'] = memory.oblique_components(flow, gradient,
            tangent_primary['pre']['phi'][-1]@(EPSILON*ctx['direction']))
        normalization_max = max(abs(float(ctx['phase_gradient']@ctx['flow']-1.)),
            abs(float(gradient@flow-1.)),
            abs(float(gradient@tangent_alternate['pre']['derivatives'][-1]-1.)))
        gates = dict(original_reference_qualified=ctx.get('provenance', {}).get('reference_qualified') is True,
            normalization=normalization_max <= NORMALIZATION_LIMIT,
            primary_observers=False, reference_states=True, reference_activation=True,
            scaled_full_tangent=True, scaled_activation_tangent=True,
            composed_state_predictions=True, composed_activation_predictions=True,
            prediction_margins=True, prediction_accuracies=True)
        comparison = dict(normalization_max=normalization_max, segments={}, windows={})
        result.update(gates=gates, reference_comparison=comparison)
        for key in ('pre', 'post'):
            p, a = tangent_primary[key], tangent_alternate[key]
            state = _difference(p['states'], a['states'])
            activation = _difference(np.tanh(p['states']), np.tanh(a['states']))
            tangent = _difference(EPSILON*p['phi'], EPSILON*a['phi'])
            p_activation_phi = (1.-np.tanh(p['states'])**2)[:, :, None]*p['phi']
            a_activation_phi = (1.-np.tanh(a['states'])**2)[:, :, None]*a['phi']
            activation_tangent = _difference(EPSILON*p_activation_phi, EPSILON*a_activation_phi)
            comparison['segments'][key] = dict(state=state, activation=activation,
                raw_phi_all_nine=_difference(p['phi'], a['phi']),
                pulse_scaled_phi_all_nine=tangent, pulse_scaled_activation_phi_all_nine=activation_tangent)
            gates['reference_states'] &= state['max_abs'] <= STATE_LIMIT
            gates['reference_activation'] &= activation['max_abs'] <= ACTIVATION_LIMIT
            gates['scaled_full_tangent'] &= tangent['max_abs'] <= STATE_LIMIT
            gates['scaled_activation_tangent'] &= activation_tangent['max_abs'] <= ACTIVATION_LIMIT
        for window in WINDOWS:
            samples, predictions = _predict(ctx, tangent_primary, window)
            _, alternate = _predict(ctx, tangent_alternate, window)
            observer = memory.linear_observer(predictions['identity']['A'], predictions['identity']['B'], SIGMA)
            h = observer['h']
            packet = dict(times=samples['times'], segment=samples['segment'], indices=samples['indices'],
                baseline=samples['activations'], baseline_states=samples['states'], h=h,
                observer=observer, linear_predictions=predictions, arm_observers={},
                one_mode_prediction=None)
            velocity = tangent_primary[samples['segment']]['derivatives'][samples['indices']]
            G = ((1.-samples['activations']**2)*velocity).ravel()
            G_norm = float(np.linalg.norm(G))
            packet['derivative_observer'] = (dict(available=True,h=G/G_norm,noise_sd=SIGMA,
                scope='direct-reference physical activation velocity; fixed positive-flow orientation; not the old periodic-Hermite clock')
                if np.isfinite(G_norm) and G_norm > 0 else dict(available=False,h=None,reason='Zero/unrepresentable activation velocity'))
            for arm in ARMS:
                try:
                    secondary = memory.linear_observer(predictions[arm]['A'], predictions[arm]['B'], SIGMA)
                    packet['arm_observers'][arm] = dict(available=True, observer=secondary)
                except ValueError as error:
                    packet['arm_observers'][arm] = dict(available=False, error=str(error), observer=None)
                predictions[arm]['primary_accuracy'] = memory.balanced_accuracy(h,
                    memory.linear_means(predictions[arm]['A'], predictions[arm]['B']), SIGMA)
            if window == 'immediate':
                one = dict(available=True, A=predictions['identity']['A'].copy(), B=np.zeros(96),
                    scope='matched immediate calibration; not a one-mode temporal prediction')
            elif ctx.get('rho') is None:
                one = dict(available=False, reason='Signed dominant real transverse multiplier unavailable')
            else:
                one = dict(memory.periodic_single_mode(predictions['identity']['B'], _scalar(ctx['rho'], 'rho')), available=True)
            if one['available']:
                one['primary_accuracy'] = memory.balanced_accuracy(h, memory.linear_means(one['A'], one['B']), SIGMA)
            packet['one_mode_prediction'] = one
            wc = {}
            for arm in ARMS:
                first = abs(float(h@(predictions[arm]['A']-alternate[arm]['A'])))/observer['noise_sd']
                second = abs(float(h@(predictions[arm]['B']-alternate[arm]['B'])))/observer['noise_sd']
                pmeans = memory.linear_means(predictions[arm]['A'], predictions[arm]['B'])
                ameans = memory.linear_means(alternate[arm]['A'], alternate[arm]['B'])
                pa = memory.balanced_accuracy(h, pmeans, SIGMA)
                aa = memory.balanced_accuracy(h, ameans, SIGMA)
                margin = _difference(pa['signed_standardized_means'], aa['signed_standardized_means'])
                accuracy = _difference(pa['probability_correct'], aa['probability_correct'])
                wc[arm] = dict(first_projection_noise_difference=first, second_projection_noise_difference=second,
                    signed_margins=margin, probability_correct=accuracy,
                    first_activation=_difference(predictions[arm]['A'], alternate[arm]['A']),
                    second_activation=_difference(predictions[arm]['B'], alternate[arm]['B']))
                wc[arm]['first_state'] = _difference(predictions[arm]['first_state'], alternate[arm]['first_state'])
                wc[arm]['second_state'] = _difference(predictions[arm]['second_state'], alternate[arm]['second_state'])
                gates['composed_state_predictions'] &= max(wc[arm]['first_state']['max_abs'], wc[arm]['second_state']['max_abs']) <= STATE_LIMIT
                gates['composed_activation_predictions'] &= max(wc[arm]['first_activation']['max_abs'], wc[arm]['second_activation']['max_abs']) <= ACTIVATION_LIMIT
                gates['prediction_margins'] &= max(first, second, margin['max_abs']) <= MARGIN_LIMIT
                gates['prediction_accuracies'] &= accuracy['max_abs'] <= ACCURACY_LIMIT
            comparison['windows'][window] = wc
            result['windows'][window] = packet
        gates['primary_observers'] = True
        result.update(complete=True, status='complete', gates={k: bool(v) for k,v in gates.items()},
            reference_comparison=comparison, qualified=bool(all(gates.values())),
            arithmetic_qualified=all(p['composition_arithmetic_pass'] for w in result['windows'].values()
                for p in w['linear_predictions'].values()))
    except (ValueError, KeyError, TypeError, IndexError, FloatingPointError, OverflowError) as error:
        result.update(error=f'{type(error).__name__}: {error}', qualified=False)
    return result
