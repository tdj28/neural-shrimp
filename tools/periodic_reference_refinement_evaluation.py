"""Prospective global-reference refinement evaluation from supplied arrays only.

No integration, fitting, file IO, candidate replacement or phase renormalization.
The new canonical global adjoint is never described as the old inherited vector.
All numerical gates remain separate from computational completeness.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping
import numpy as np

from hopfield_diagnostics import hopfield_rhs, hopfield_jacobian, _floquet_diagnostics
from phase_memory import neutral_left_phase
import phase_reference_diagnostic_evaluation as old_math

SETTINGS=('core','failed-near-edge')
METHODS=('tight-dop853','tight-rk45')
GRIDS=('4096','8192')


def contract():
    return dict(settings=list(SETTINGS),methods=list(METHODS),shooting_jobs=4,reference_jobs=2,event_jobs=96,
        contexts=48,trial_slots_per_shoot=25,verification_paths=8,backward_paths=4,maximum_ivp_solves=208,
        thresholds=dict(closure_relative=1e-12,section_absolute=1e-12,state_neighborhood=1e-6,period_neighborhood=1e-6,
            candidate_state=1e-10,candidate_period=1e-11,bordered_condition=1e10,neutral_residual=1e-10,adjoint_periodicity=1e-10,
            unique_singular_rtol=1e-12,normalization_cosine_min=1e-12,normalization=1e-7,state=1e-6,activation=1e-7,epsilon=1e-4),
        canonical_method=METHODS[0],global_normalizations_per_setting=1,individual_phase_normalization=False,
        diagnostic_only=True,original_study_requalified=False,eight_period_memory_qualified=False,
        arithmetic_replay=dict(rtol=1e-12,atol=1e-14,left_unit_residual_order='norm((M.T-I)@unit)',normalization_order='pointwise z@f-1'),
        completion='All required saved computations are available and arithmetic-valid; science gates are separate',
        qualification='All planned candidate, reference, and48 context gates; unavailable rows remain in denominator')


_array=old_math._array
_number=old_math._number
_equal=old_math._equal
_strict=old_math._strict
difference=old_math.difference
normalization=old_math.normalization
array_agreement=old_math.array_agreement


def _tree(saved,expected,path):
    if isinstance(expected,Mapping):
        if not isinstance(saved,Mapping) or set(saved)!=set(expected):raise ValueError(path+': keys differ')
        for key,value in expected.items():_tree(saved[key],value,path+'.'+str(key))
    elif isinstance(expected,(list,tuple)):
        if not isinstance(saved,(list,tuple)) or len(saved)!=len(expected):raise ValueError(path+': list length differs')
        for i,(a,b) in enumerate(zip(saved,expected,strict=True)):_tree(a,b,path+f'[{i}]')
    elif isinstance(expected,np.ndarray):
        if expected.dtype.kind in 'biu':
            if type(saved) is not np.ndarray or saved.dtype!=expected.dtype or saved.shape!=expected.shape or not np.array_equal(saved,expected):raise ValueError(path+': exact discrete array differs')
        else:_strict(saved,expected,path)
    elif isinstance(expected,(float,np.floating)):
        a=_number(saved,path);b=_number(expected,path)
        if not np.isclose(a,b,rtol=1e-12,atol=1e-14):raise ValueError(path+': numeric arithmetic differs')
    else:_equal(saved,expected,path)


def _method(method,T):
    if method not in METHODS:raise ValueError('Unknown new tight method')
    return dict(id=method,kind='DOP853' if method==METHODS[0] else 'RK45',rtol=1e-13,atol=1e-15,max_step=T/16384,max_step_period_divisor=16384)


def _phase_rows(phases):
    if not isinstance(phases,list) or len(phases)!=24:raise ValueError('Exactly24 declared phases required')
    for i,p in enumerate(phases):
        _equal(p['phase_index'],i,'phase index');_equal(p['split'],'development' if i<8 else 'heldout','phase split')
        q=_number(p['phase_fraction'],'phase fraction')
        if not 0<=q<1:raise ValueError('phase fraction outside unit interval')
    return phases


def shooting_arithmetic(state,period,old_state,old_period,endpoint,monodromy,*,rhs,jacobian,alpha,gamma):
    """Rebuild exact scaled four-variable bordered Newton system without a solve."""
    x=_array(state,'state',(3,));seed=_array(old_state,'oldstate',(3,));end=_array(endpoint,'endpoint',(3,));M=_array(monodromy,'monodromy',(3,3))
    T=_number(period,'period',True);T0=_number(old_period,'oldperiod',True);S=np.maximum(1.,np.abs(seed))
    oldJ=_array(np.asarray(jacobian(0.,seed,alpha,gamma),dtype=np.float64),'oldJ',(3,3))
    section_scale=max(1.,float(np.linalg.norm(oldJ[2]*S)))
    f0=np.asarray(rhs(0.,x,alpha,gamma),dtype=np.float64);fT=np.asarray(rhs(T,end,alpha,gamma),dtype=np.float64)
    J0=np.asarray(jacobian(0.,x,alpha,gamma),dtype=np.float64)
    residual=np.r_[end-x,f0[2]];bordered=np.zeros((4,4));bordered[:3,:3]=M-np.eye(3);bordered[:3,3]=fT;bordered[3,:3]=J0[2]
    columns=np.r_[S,T0];rows=np.r_[S,section_scale];scaled=bordered*columns[None,:]/rows[:,None]
    return dict(state_scales=S,section_scale=section_scale,residual=residual,jacobian=bordered,scaled_residual=residual/rows,scaled_jacobian=scaled,
        condition=float(np.linalg.cond(scaled)),relative_closure=float(np.linalg.norm(end-x)/max(1.,np.linalg.norm(x))),
        section_residual=float(abs(f0[2])),state_neighborhood=float(np.max(abs(x-seed)/S)),period_neighborhood=float(abs(T-T0)/T0))


def candidate_comparison(left,right,old_reference):
    seed=_array(np.asarray(old_reference['certificate']['section_state'],dtype=np.float64),'old seed',(3,));T0=_number(old_reference['period'],'old period',True)
    a=_array(left['state'],'left candidate',(3,));b=_array(right['state'],'right candidate',(3,))
    delta=a-b;td=float(left['period']-right['period']);sd=float(np.max(abs(delta)/np.maximum(1.,abs(seed))));relative=abs(td)/T0
    return dict(state_difference=delta,period_difference=td,state_scaled_infinity=sd,period_relative=relative,
        gates=dict(state=sd<=1e-10,period=relative<=1e-11),qualified=sd<=1e-10 and relative<=1e-11,
        scope='same section and original seed scale; no branch-uniqueness proof')


def phase_locked_agreement(left,right):
    """Different candidate periods: compare equal fractions, not physical times."""
    if left['states'].shape!=right['states'].shape:raise ValueError('Common phase sample count required')
    n=len(left['times'])-1
    if n<1:raise ValueError('Nonempty normalized phase grid required')
    for g in (left,right):
        T=_number(g['times'][-1],'path terminal time',True);expected=np.arange(n+1,dtype=np.float64)*(T/n);expected[-1]=T
        _equal(g['times'],expected,'declared phase grid')
    phase=np.arange(n+1,dtype=np.float64)/n
    result=array_agreement(dict(left,times=phase),dict(right,times=phase))
    result.update(comparison_clock='equal normalized phase; candidate physical times can differ',periods=[float(left['times'][-1]),float(right['times'][-1])])
    return result


def cycle_diagnostics(state,period,trajectory,event_times,event_states,*,rhs,jacobian,alpha,gamma):
    """Unoptimized candidate checks, retaining all detected section events."""
    x=_array(state,'candidate state',(3,));T=_number(period,'period',True)
    states=_array(trajectory['states'],'states');phi=_array(trajectory['phi'],'phi')
    if states.ndim!=2 or states.shape[1]!=3 or phi.shape!=(len(states),3,3):raise ValueError('Complete state/Phi shapes required')
    times=np.asarray(event_times);points=np.asarray(event_states)
    if times.dtype!=np.float64 or points.dtype!=np.float64 or not np.isfinite(times).all() or not np.isfinite(points).all():raise ValueError('Finite float64 event samples required')
    if times.ndim!=1 or points.shape!=(len(times),3) or np.any(np.diff(times)<=0) or np.any(times<0) or np.any(times>T):raise ValueError('Ordered in-interval maximum events required')
    flow=np.asarray(rhs(0.,x,alpha,gamma),dtype=np.float64);J0=np.asarray(jacobian(0.,x,alpha,gamma),dtype=np.float64)
    endpoint=states[-1];M=phi[-1];closure=float(np.linalg.norm(endpoint-x)/max(1.,np.linalg.norm(x)));section=abs(float(flow[2]));curvature=float((J0@flow)[2]);flow_norm=float(np.linalg.norm(flow));amplitude=float(np.max(np.ptp(states,axis=0)))
    endpoint_tolerance=max(1e-7,1e-6*T);interior=(times>endpoint_tolerance)&(times<T-endpoint_tolerance)
    distances=np.linalg.norm(points[interior]-x,axis=1)/max(1.,np.linalg.norm(x));peak_count=1+int(interior.sum())
    event_flows=np.asarray([rhs(float(t),v,alpha,gamma) for t,v in zip(times,points,strict=True)],dtype=np.float64).reshape(-1,3)
    event_curvatures=np.asarray([(jacobian(float(t),v,alpha,gamma)@f)[2] for t,v,f in zip(times,points,event_flows,strict=True)],dtype=np.float64)
    floquet=_floquet_diagnostics(M,flow)
    _,singular,vh=np.linalg.svd(M.T-np.eye(3),full_matrices=False);unit=vh[-1]
    left_residual=float(np.linalg.norm((M.T-np.eye(3))@unit));right_residual=float(np.linalg.norm(M@flow-flow)/flow_norm) if flow_norm>0 else None
    gates=dict(closure=closure<=1e-12,section=section<=1e-12,section_is_maximum=curvature< -1e-8,nonstationary=flow_norm>1e-7 and amplitude>1e-6,
        expected_peak_count=peak_count==9,primitive=bool(np.all(distances>1e-7)),neutral_mode=floquet['neutral_verified'],transverse_stability=floquet['transversely_attracting'],
        event_maxima=bool(np.all(event_curvatures< -1e-8)),neutral_right=right_residual is not None and right_residual<=1e-10,neutral_left=left_residual<=1e-10,
        unique_neutral=bool(singular[-2]>1e-12*max(1.,singular[0])))
    return dict(relative_closure=closure,section_residual=section,section_curvature=curvature,flow_norm=flow_norm,state_amplitude=amplitude,peak_count=peak_count,
        endpoint_tolerance=endpoint_tolerance,interior_mask=interior,interior_return_distances=distances,event_section_residuals=np.abs(event_flows[:,2]),event_curvatures=event_curvatures,
        floquet=floquet,neutral_singular_values=singular,neutral_left_unit_residual=left_residual,neutral_flow_relative_residual=right_residual,gates=gates,qualified=all(gates.values()))


def _solver(record,method,T,initial,offset=0.,backward=False):
    _tree(record['method_spec'],_method(method,T),'solver method')
    if record['solver']['success'] is not True:raise ValueError('Successful solver required')
    for key in ('nfev','njev','nlu'):
        v=record['solver'][key]
        if type(v) is not int or v<0:raise ValueError('Nonnegative integer solver count required')
    _equal(record['initial'],initial,'solver initial')
    _equal(record['span'],np.array([T,0.] if backward else [0.,T]),'solver span')
    _equal(record['offset'],offset,'solver offset')


def _ivp_complete(record,path):
    for key in ('ivp_started','ivp_completed'):_equal(record.get(key),True,path+'.'+key)


def _forward(record,state,T,method,alpha,gamma,*,rhs,jacobian,offset=0.,events=False):
    if not isinstance(record,Mapping) or record.get('complete') is not True or record.get('status')!='complete':raise ValueError('Complete forward acquisition required')
    _ivp_complete(record,'complete forward');_equal(record['initial'],state,'forward initial');_equal(record['period'],T,'forward T');_equal(record['method'],method,'forward method');_equal(record['offset'],offset,'forward offset')
    _tree(record['method_spec'],_method(method,T),'forward method_spec')
    _solver(record['solver'],method,T,np.r_[state,np.eye(3).ravel()],offset)
    tr=record['trajectory'];old_math._samples(tr,T,8192,offset=offset,rhs=rhs,jacobian=jacobian,alpha=alpha,gamma=gamma)
    _equal(tr['offset'],offset,'trajectory offset');_equal(tr['states'][0],state,'trajectory initial')
    if events:
        maxima=record['maxima'];t=np.asarray(maxima['times']);x=np.asarray(maxima['states']);p=np.asarray(maxima['phi'])
        if any(v.dtype!=np.float64 or not np.isfinite(v).all() for v in (t,x,p)):raise ValueError('Finite float64 maximum arrays required')
        if t.ndim!=1 or x.shape!=(len(t),3) or p.shape!=(len(t),3,3) or np.any(np.diff(t)<=0) or np.any(t<0) or np.any(t>T):raise ValueError('Event shapes/order differ')
    return tr


def _metrics(state,T,tr,old,*,rhs,jacobian):
    seed=np.asarray(old['certificate']['section_state'],dtype=np.float64);T0=float(old['period'])
    a=shooting_arithmetic(state,T,seed,T0,tr['states'][-1],tr['phi'][-1],rhs=rhs,jacobian=jacobian,alpha=old['alpha'],gamma=old['gamma'])
    singular=np.linalg.svd(a['scaled_jacobian'],compute_uv=False);condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None
    f=rhs(0.,state,old['alpha'],old['gamma']);J=jacobian(0.,state,old['alpha'],old['gamma'])
    return dict(raw_residual=a['residual'],raw_bordered=a['jacobian'],scaled_residual=a['scaled_residual'],scaled_bordered=a['scaled_jacobian'],
        scaled_singular_values=singular,scaled_condition=condition,merit=float(np.linalg.norm(a['scaled_residual'])),relative_closure=a['relative_closure'],section_residual=a['section_residual'],
        section_curvature=float((J@f)[2]),seed_scaled_distance=a['state_neighborhood'],seed_relative_period_change=a['period_neighborhood'],
        state_spacing=np.spacing(np.abs(state)),endpoint_difference=tr['states'][-1]-state)


def _converged(metrics):
    return bool(metrics['relative_closure']<=1e-12 and metrics['section_residual']<=1e-12 and metrics['scaled_condition'] is not None and metrics['scaled_condition']<=1e10)


def validate_shoot_packet(packet,old_reference,expected_id=None,*,rhs=hopfield_rhs,jacobian=hopfield_jacobian):
    """Replay every declared Newton slot and decision; never run a trial IVP."""
    p=packet;old=old_reference
    if not isinstance(p,Mapping):raise ValueError('Shoot packet unavailable')
    s=old['setting'];m=p['method'];ident=s+'--'+m
    if m not in METHODS:raise ValueError('Tight method required')
    _equal(p['id'],ident,'shoot identity');_equal(p['kind'],'shoot','shoot kind');_equal(p['schema'],'periodic-reference-shoot-v1','shoot schema')
    if expected_id is not None:_equal(ident,expected_id,'planned shoot')
    for key in ('setting','alpha','gamma'):_equal(p[key],old[key],'shoot original '+key)
    seed=np.asarray(old['certificate']['section_state'],dtype=np.float64);T0=float(old['period']);S=np.maximum(1.,abs(seed));ss=max(1.,float(np.linalg.norm(jacobian(0.,seed,old['alpha'],old['gamma'])[2]*S)))
    _equal(p['original_state'],seed,'original state');_equal(p['original_period'],T0,'original period');_equal(p['state_scales'],S,'state scales');_equal(p['section_scale'],ss,'section scale')
    _tree(p['limits'],dict(maximum_updates=6,line_factors=[1.,.5,.25,.125],trial_slots=25,armijo=1e-4,state_radius=1e-6,period_radius=1e-6,condition_max=1e10,relative_closure=1e-12,section_residual=1e-12),'shoot limits')
    trials=p['trials']
    if not isinstance(trials,list) or len(trials)!=25:raise ValueError('Exactly25 trial slots required')
    errors=[];metrics={}
    for i,row in enumerate(trials):
        try:
            _equal(row['index'],i,'trial index');_equal(row['id'],ident+f'--trial-{i:02d}','trial id')
            if type(row['attempted']) is not bool:raise ValueError('Boolean attempted required')
            if not row['attempted']:
                expected=dict(index=i,id=ident+f'--trial-{i:02d}',status='not_needed',attempted=False,ivp_attempted=False,ivp_started=False,ivp_completed=False,reason=row.get('reason'))
                _equal(row,expected,'unused trial')
                if type(row.get('reason')) is not str:raise ValueError('Unused reason required')
                continue
            x=_array(row['state'],'trial state',(3,));T=_number(row['period'],'trial period')
            bounded=T>0 and np.max(abs(x-seed)/S)<=1e-6 and abs(T-T0)/T0<=1e-6
            if row['status']=='outside_neighborhood':
                if bounded or row.get('accepted') is not False or 'forward' in row or any(row.get(k) is not False for k in ('ivp_attempted','ivp_started','ivp_completed')):raise ValueError('Outside-neighborhood evidence differs')
                continue
            if row['status']!='complete':
                errors.append(f'trial{i}: incomplete retained acquisition ({row["status"]})');continue
            if not bounded:raise ValueError('Integrated trial outside declared neighborhood')
            if any(row.get(k) is not True for k in ('ivp_attempted','ivp_started','ivp_completed')):raise ValueError('Completed IVP flags differ')
            tr=_forward(row['forward'],x,T,m,old['alpha'],old['gamma'],rhs=rhs,jacobian=jacobian)
            actual=_metrics(x,T,tr,old,rhs=rhs,jacobian=jacobian);_tree(row['metrics'],actual,f'trial{i}.metrics');metrics[i]=actual
        except (ValueError,KeyError,TypeError,IndexError,OverflowError,FloatingPointError) as exc:errors.append(f'trial{i}: {exc}')
    if errors:raise ValueError('; '.join(errors))
    if p.get('status')!='complete' or p.get('complete') is not True:raise ValueError('Incomplete shooting routine retained')
    _equal(trials[0]['state'],seed,'first seed');_equal(trials[0]['period'],T0,'first period');_equal(trials[0]['factor'],None,'first factor');_equal(trials[0]['accepted'],True,'first accepted')
    current=0;used={0};stop=None;iterations=p['iterations']
    if not isinstance(iterations,list) or len(iterations)>6:raise ValueError('At most6 Newton updates')
    for i,iteration in enumerate(iterations):
        if _converged(metrics[current]):raise ValueError('Update after convergence')
        r=metrics[current]
        if r['scaled_condition'] is None or r['scaled_condition']>1e10:raise ValueError('Update from ill-conditioned solve')
        _equal(iteration['iteration'],i,'Newton index');_equal(iteration['from_trial'],current,'Newton start')
        delta=np.linalg.solve(r['scaled_bordered'],-r['scaled_residual']);_strict(iteration['scaled_correction'],delta,'Newton correction')
        ids=iteration['trial_indices']
        if not isinstance(ids,list) or not 1<=len(ids)<=4 or ids!=list(range(1+4*i,1+4*i+len(ids))):raise ValueError('Fixed damping sequence differs')
        start=current;moved=False
        for j,k in enumerate(ids):
            row=trials[k];factor=(1.,.5,.25,.125)[j];used.add(k)
            _equal(row['factor'],factor,'damping factor');_equal(row['state'],trials[start]['state']+factor*delta[:3]*S,'trial candidate state');_equal(row['period'],float(trials[start]['period']+factor*delta[3]*T0),'trial candidate period')
            accepted=k in metrics and metrics[k]['merit']<=(1.-1e-4*factor)*r['merit']
            _equal(row['accepted'],bool(accepted),'Armijo acceptance')
            if accepted:
                if k!=ids[-1]:raise ValueError('Trial after accepted step')
                moved=True;current=k
        if not moved:
            if i!=len(iterations)-1 or len(ids)!=4:raise ValueError('Incomplete line-search/stopping sequence')
            stop='line_search_no_decrease'
    _equal({i for i,r in enumerate(trials) if r['attempted']},used,'all attempted trials bound to updates')
    candidate=p['candidate'];_equal(candidate['source_trial_index'],current,'candidate source');_equal(candidate['state'],trials[current]['state'],'candidate state');_equal(candidate['period'],trials[current]['period'],'candidate T');_tree(candidate['metrics'],metrics[current],'candidate metrics')
    converged=_converged(metrics[current]);_equal(p['converged'],converged,'converged');_equal(p['qualified'],converged,'shoot qualification')
    if converged:stop='residual_targets'
    elif metrics[current]['scaled_condition'] is None or metrics[current]['scaled_condition']>1e10:stop='singular_or_ill_conditioned_bordered_system'
    elif stop is None:stop='maximum_updates'
    if stop=='maximum_updates' and len(iterations)!=6:raise ValueError('Premature update-budget stop')
    _equal(p['stop_reason'],stop,'stop reason')
    skipped=set()
    for i,it in enumerate(iterations):
        last=it['trial_indices'][-1]
        if trials[last].get('accepted') is True:skipped.update(range(last+1,5+4*i))
    for i,r in enumerate(trials):
        if not r['attempted']:_equal(r['reason'],'line_search_accepted' if i in skipped else 'unreached_after_'+stop,'unused reason')
    _equal(p['attempted_trials'],len(used),'attempted count');_equal(p['ivp_evaluations'],sum(r['ivp_attempted'] for r in trials),'IVP attempt count');_equal(p['ivp_evaluations'],len(metrics),'IVP count')
    return p


def _certificate(forward,alpha,gamma,*,rhs,jacobian):
    tr=forward['trajectory'];x=tr['states'][0];T=forward['period'];maximum=forward['maxima']
    a=cycle_diagnostics(x,T,tr,maximum['times'],maximum['states'],rhs=rhs,jacobian=jacobian,alpha=alpha,gamma=gamma)
    M=tr['phi'][-1];_,singular,vh=np.linalg.svd(M.T-np.eye(3),full_matrices=False);unit=vh[-1];fl=a['floquet']
    gates=dict(closure=a['relative_closure']<=1e-12,section=a['section_residual']<=1e-12,section_is_maximum=a['section_curvature']< -1e-8,
        all_recorded_events_are_maxima=bool(np.all(a['event_curvatures']< -1e-8)),nonstationary=a['flow_norm']>1e-7 and a['state_amplitude']>1e-6,
        expected_peak_count=a['peak_count']==9,primitive=bool(np.all(a['interior_return_distances']>1e-7)),
        neutral_flow=fl['neutral_flow_residual'] is not None and fl['neutral_flow_residual']<=1e-10,neutral_eigenvalue=fl['neutral_multiplier_error']<=1e-10,
        neutral_alignment=fl['neutral_flow_alignment']>=.99,transverse_stability=fl['transversely_attracting'],
        left_neutral=a['neutral_left_unit_residual']<=1e-10,unique_left_neutral=bool(singular[-2]>1e-12*max(1.,singular[0])))
    inside=a['interior_mask']
    return dict(qualified=all(gates.values()),gates=gates,relative_closure=a['relative_closure'],section_residual=a['section_residual'],section_curvature=a['section_curvature'],
        state_amplitude=a['state_amplitude'],expected_peaks=9,peak_count=a['peak_count'],endpoint_exclusion=a['endpoint_tolerance'],interior_mask=inside,
        interior_return_distances=a['interior_return_distances'],event_curvatures=a['event_curvatures'],excluded_event_times=maximum['times'][~inside].copy(),excluded_event_states=maximum['states'][~inside].copy(),
        left_neutral_unit_vector=unit,left_neutral_singular_values=singular,left_neutral_relative_residual=a['neutral_left_unit_residual'],unique_left_neutral=gates['unique_left_neutral'],**fl)


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
    if len(b['phases'])!=24 or len(b['phase_rows'])!=24:raise ValueError('Complete24 phase rows required')
    exact=b['exact_phase_arrays']
    for key,shape in dict(times=(24,),states=(24,3),phi=(24,3,3),derivatives=(24,3),phi_derivatives=(24,3,3),z=(24,3),z_derivatives=(24,3),normalization_residuals=(24,)).items():_array(exact[key],'dense '+key,shape)
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


def _event_diagnostics(packet,base):
    p=base['phases'][packet['phase_index']];g=packet['trajectory'];x=g['states'][-1];f0,f=g['derivatives'][0],g['derivatives'][-1];z=p['z'];Q=np.outer(f,z);P=np.eye(3)-Q
    return dict(state=x,flow=f,phase_gradient=z,initial_flow=f0,Q=Q,P=P,initial_normalization_residual=float(z@f0-1.),normalization_residual=float(z@f-1.),event_minus_initial_normalization=float(z@(f-f0)),return_difference=x-p['state'],
        Q_idempotence_residual=Q@Q-Q,P_idempotence_residual=P@P-P,QP_residual=Q@P,phase_leakage=z@P,Q_operator_norm=float(np.linalg.norm(Q,2)),P_operator_norm=float(np.linalg.norm(P,2)))


def validate_event_packet(packet,base,expected_id=None,*,rhs=hopfield_rhs,jacobian=hopfield_jacobian):
    e=packet;b=base
    if not isinstance(e,Mapping) or e.get('complete') is not True or e.get('status')!='complete':raise ValueError('Complete event required')
    if b.get('qualified') is not True:raise ValueError('Qualified base prerequisite required')
    _ivp_complete(e,'complete event')
    s=b['setting'];m=b['method'];i=e['phase_index'];T=b['period']
    if type(i) is not int or i not in range(24):raise ValueError('Declared phase index required')
    p=b['phases'][i];ident=f'{s}--p{i:02d}--{m}'
    _equal(e['id'],ident,'event identity');_equal(e['schema'],'refined-phase-event-v1','event schema');_equal(e['kind'],'event','event kind');_equal(e['base_id'],b['id'],'event base')
    if expected_id is not None:_equal(ident,expected_id,'planned event')
    for key in ('setting','method','period','alpha','gamma'):_equal(e[key],b[key],'event/base '+key)
    for key in ('context_id','phase_index','phase_fraction','phase_epoch','split'):_equal(e[key],p[key],'event phase '+key)
    _equal(e['event_time'],T,'elapsed event T');_equal(e['physical_event_time'],p['phase_epoch']+T,'physical event');_equal(e['initial'],p['state'],'new dense event initial');_equal(e['phase_gradient'],p['z'],'unchanged nominal phase gradient');_equal(e['new_normalizations'],0,'no event normalization')
    _tree(e['method_spec'],_method(m,T),'event method');_solver(e['forward'],m,T,np.r_[p['state'],np.eye(3).ravel()],p['phase_epoch'])
    g=e['trajectory'];old_math._samples(g,T,8192,offset=p['phase_epoch'],rhs=rhs,jacobian=jacobian,alpha=b['alpha'],gamma=b['gamma']);_equal(g['offset'],p['phase_epoch'],'event grid offset');_equal(g['states'][0],p['state'],'event saved initial')
    _tree(e['event'],_event_diagnostics(e,b),'event derived diagnostics')
    return e


def partial_array_census(packet):
    """Retain all nested arrays, including nonfinite solver-failure evidence.

    This is an evidence census, not a claim that failed prefixes satisfy the
    dynamics. Completed subcalculations are physically replayed by validators.
    """
    rows=[]
    def visit(value,path):
        if isinstance(value,np.ndarray):rows.append(dict(path=path,shape=list(value.shape),dtype=value.dtype.str,size=int(value.size),finite=bool(np.isfinite(value).all()) if value.dtype.kind in 'fciu' else None))
        elif isinstance(value,Mapping):
            for k,v in value.items():visit(v,path+'.'+str(k))
        elif isinstance(value,(tuple,list)):
            for i,v in enumerate(value):visit(v,path+f'[{i}]')
    visit(packet,'packet');return rows


def _nested_census(shoots,references,events):
    rows=[]
    for s in SETTINGS:
        for m in METHODS:
            ident=s+'--'+m;p=shoots.get(ident);trials=p.get('trials',[]) if isinstance(p,Mapping) else []
            by={r.get('index'):r for r in trials if isinstance(r,Mapping)}
            for i in range(25):
                r=by.get(i);rows.append(dict(id=ident+f'--trial-{i:02d}',kind='shoot_trial',setting=s,method=m,index=i,status=r.get('status','unavailable') if r else 'unavailable',attempted=r.get('attempted',False) if r else False,
                    raw_available=isinstance(r,Mapping),forward_available=isinstance(r,Mapping) and 'forward' in r,partial_failure_available=isinstance(r,Mapping) and 'partial_failure' in r,error=r.get('error') if r else None))
        bundle=references.get(s+'--reference-pair',references.get(s));bundle=bundle if isinstance(bundle,Mapping) else {}
        for cm,m in itertools.product(METHODS,METHODS):
            r=bundle.get('verifications',{}).get(cm+'--'+m,{});rows.append(dict(id=f'{s}--verify--{cm}--{m}',kind='verification',setting=s,candidate_method=cm,method=m,status=r.get('status','unavailable'),complete=r.get('complete',False),qualified=r.get('qualified',False),raw_available=bool(r),partial_failure_available='partial_failure' in r,error=r.get('error')))
        for m in METHODS:
            r=bundle.get('adjoints',{}).get(m,{});rows.append(dict(id=f'{s}--adjoint--{m}',kind='adjoint',setting=s,method=m,status=r.get('status','unavailable'),complete=r.get('complete',False),qualified=r.get('qualified',False),raw_available=bool(r),partial_failure_available='partial_failure' in r,error=r.get('error')))
        for i in range(24):
            for m in METHODS:
                ident=f'{s}--p{i:02d}--{m}';r=events.get(ident,{});rows.append(dict(id=ident,kind='event',setting=s,method=m,phase_index=i,status=r.get('status','unavailable'),complete=r.get('complete',False),raw_available=bool(r),partial_failure_available='partial_failure' in r,error=r.get('error')))
    return rows


def summarize(shoots,references,events,old_references,phase_rows,*,rhs=hopfield_rhs,jacobian=hopfield_jacobian):
    """Full4/2/96/48 census; missing subcalculations can never vanish."""
    phases=_phase_rows(phase_rows);input_errors=[]
    for name,values,expected in [('shoots',shoots,{s+'--'+m for s in SETTINGS for m in METHODS}),
        ('events',events,{f'{s}--p{i:02d}--{m}' for s in SETTINGS for i in range(24) for m in METHODS}),('old_references',old_references,set(SETTINGS))]:
        if not isinstance(values,Mapping):raise ValueError(name+' must be mapping')
        if set(values)-expected:input_errors.append(dict(scope=name,error='Unexpected identity',ids=sorted(set(values)-expected)))
    if not isinstance(references,Mapping):raise ValueError('references must be mapping')
    expected_ref={s+'--reference-pair' for s in SETTINGS}|set(SETTINGS)
    if set(references)-expected_ref:input_errors.append(dict(scope='references',error='Unexpected identity',ids=sorted(set(references)-expected_ref)))
    for s in SETTINGS:
        if s in references and s+'--reference-pair' in references:input_errors.append(dict(scope='references',error='Duplicate setting aliases',ids=[s,s+'--reference-pair']))
    shot_rows=[];reference_rows=[];event_rows=[];context_rows=[];valid_shots={};valid_refs={};valid_events={}
    for s in SETTINGS:
        for m in METHODS:
            ident=s+'--'+m;p=shoots.get(ident);row=dict(id=ident,setting=s,method=m,complete=False,converged=False,qualified=False,arithmetic_replay_pass=False,
                acquired_complete=isinstance(p,Mapping) and p.get('complete') is True,status='unavailable',array_census=partial_array_census(p))
            try:
                validate_shoot_packet(p,old_references[s],ident,rhs=rhs,jacobian=jacobian)
                row.update(complete=True,converged=p['converged'],qualified=p['qualified'],arithmetic_replay_pass=True,status=p['status'],stop_reason=p['stop_reason'],attempted_trials=p['attempted_trials'],ivp_evaluations=p['ivp_evaluations'],candidate=p['candidate'])
                valid_shots[ident]=p
            except (ValueError,KeyError,TypeError,IndexError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
                if isinstance(p,Mapping):row['partial_replay']=validate_partial_packet(p,old_references[s],phases,rhs=rhs,jacobian=jacobian)
            shot_rows.append(row)
        ident=s+'--reference-pair';p=references.get(ident,references.get(s));row=dict(id=ident,setting=s,complete=False,qualified=False,arithmetic_replay_pass=False,
            acquired_complete=isinstance(p,Mapping) and p.get('complete') is True,status='unavailable',array_census=partial_array_census(p))
        try:
            validate_reference_packet(p,valid_shots,old_references[s],phases,ident,rhs=rhs,jacobian=jacobian)
            row.update(complete=p['complete'],qualified=p['qualified'],arithmetic_replay_pass=True,status=p['status'],gates=p['gates'],candidate_agreement=p['candidate_agreement'],candidate_comparison_details=candidate_comparison(p['candidates'][METHODS[0]],p['candidates'][METHODS[1]],old_references[s]),
                verifications={k:dict(complete=v.get('complete',False),qualified=v.get('qualified',False),status=v['status'],certificate=v.get('certificate'),error=v.get('error')) for k,v in p['verifications'].items()},
                adjoints=p['adjoints'],canonical=p.get('canonical'),bases={m:dict(complete=b['complete'],qualified=b['qualified'],gates=b['gates'],diagnostics=b['diagnostics']) for m,b in p['bases'].items()},
                historical_threshold_diagnostics=dict(scope='Old criteria reported only; not used to waive tighter new gates',
                    closure_threshold=1e-8,adjoint_periodicity_threshold=1e-7,
                    verification_closure={k:dict(value=v['certificate']['relative_closure'],passes_old=v['certificate']['relative_closure']<=1e-8) for k,v in p['verifications'].items() if v.get('complete') is True},
                    adjoint_periodicity={m:dict(value=b['diagnostics']['adjoint_relative_periodicity'],passes_old=b['diagnostics']['adjoint_relative_periodicity']<=1e-7) for m,b in p['bases'].items()}))
            if all(r.get('complete') is True for r in p['verifications'].values()):
                row['candidate_method_agreements']=p['candidate_method_agreements']
                row['cross_candidate_phase_comparisons']={m:phase_locked_agreement(p['verifications'][METHODS[0]+'--'+m]['forward']['trajectory'],p['verifications'][METHODS[1]+'--'+m]['forward']['trajectory']) for m in METHODS}
            if p['complete'] and p['qualified']:valid_refs[s]=p
        except (ValueError,KeyError,TypeError,IndexError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as exc:
            row['error']=f'{type(exc).__name__}: {exc}'
            if isinstance(p,Mapping):row['partial_replay']=validate_partial_packet(p,old_references[s],phases,shoot_packets=shoots,rhs=rhs,jacobian=jacobian)
        reference_rows.append(row)
        for i in range(24):
            cid=f'{s}--p{i:02d}';ctx=dict(context_id=cid,setting=s,phase_index=i,phase_fraction=phases[i]['phase_fraction'],split=phases[i]['split'],complete=False,qualified=False,events={},method_comparison=None,dense_phase_comparison=None)
            for m in METHODS:
                ident=cid+'--'+m;e=events.get(ident);r=dict(id=ident,context_id=cid,setting=s,phase_index=i,method=m,complete=False,qualified=False,arithmetic_replay_pass=False,
                    acquired_complete=isinstance(e,Mapping) and e.get('complete') is True,status='unavailable',array_census=partial_array_census(e))
                try:
                    if s not in valid_refs:raise ValueError('Qualified full canonical reference unavailable')
                    b=valid_refs[s]['bases'][m];validate_event_packet(e,b,ident,rhs=rhs,jacobian=jacobian);p=b['phases'][i];d=_event_diagnostics(e,b)
                    initial=normalization(p['z'],p['flow']);event=normalization(p['z'],d['flow']);qualified=initial['pass_bound'] and event['pass_bound']
                    r.update(complete=True,qualified=qualified,arithmetic_replay_pass=True,status='complete',initial_normalization=initial,event_normalization=event,event_diagnostics=d,
                        after_one_period_state_difference=difference(d['state'],p['state']),after_one_period_activation_difference=difference(np.tanh(d['state']),np.tanh(p['state'])))
                    valid_events[ident]=e
                except (ValueError,KeyError,TypeError,IndexError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    r['error']=f'{type(exc).__name__}: {exc}'
                    if isinstance(e,Mapping):r['partial_replay']=validate_partial_packet(e,old_references[s],phases,reference_packet=valid_refs.get(s),rhs=rhs,jacobian=jacobian)
                event_rows.append(r);ctx['events'][m]=dict(id=ident,complete=r['complete'],qualified=r['qualified'],initial_normalization=r.get('initial_normalization'),event_normalization=r.get('event_normalization'),event_diagnostics=r.get('event_diagnostics'),error=r.get('error'))
            try:
                a,b=(valid_events[cid+'--'+m] for m in METHODS);ctx['method_comparison']=array_agreement(a['trajectory'],b['trajectory'])
                pa,pb=(valid_refs[s]['bases'][m]['phases'][i] for m in METHODS)
                ctx['dense_phase_comparison']=old_math._phase_comparison(pa,pb)
                ctx['complete']=all(v['complete'] for v in ctx['events'].values());ctx['qualified']=bool(ctx['complete'] and all(v['qualified'] for v in ctx['events'].values()) and ctx['method_comparison']['agreement_pass'] and ctx['dense_phase_comparison']['agreement_pass'])
            except (ValueError,KeyError,TypeError,IndexError,OverflowError,FloatingPointError) as exc:ctx['error']=f'{type(exc).__name__}: {exc}'
            context_rows.append(ctx)
    complete=not input_errors and all(r['complete'] for rows in (shot_rows,reference_rows,event_rows,context_rows) for r in rows)
    qualified=bool(complete and all(r['qualified'] for rows in (shot_rows,reference_rows,event_rows,context_rows) for r in rows))
    nested=_nested_census(shoots,references,events)
    counts=dict(shoots_planned=4,shoots_complete=sum(r['complete'] for r in shot_rows),shoots_converged=sum(r['converged'] for r in shot_rows),references_planned=2,references_complete=sum(r['complete'] for r in reference_rows),references_qualified=sum(r['qualified'] for r in reference_rows),events_planned=96,events_complete=sum(r['complete'] for r in event_rows),events_qualified=sum(r['qualified'] for r in event_rows),contexts_planned=48,contexts_complete=sum(r['complete'] for r in context_rows),contexts_qualified=sum(r['qualified'] for r in context_rows),nested_slots_planned=208,nested_slots_retained=len(nested))
    return dict(complete=complete,diagnostic_complete=complete,qualified=qualified,all_contexts_qualified=qualified,diagnostic_only=True,original_study_requalified=False,eight_period_memory_qualified=False,
        contract=contract(),input_errors=input_errors,counts=counts,shoots=shot_rows,references=reference_rows,events=event_rows,contexts=context_rows,nested_slots=nested,
        scope='Globally refined reference numerical qualification at the same24 old phases; no finite-input outcomes, fresh heldout phases, all-phase proof, or8T memory qualification')


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
    elif kind=='event':
        s=old['setting'];m=p['method'];i=p['phase_index'];attempt('event_identity',lambda:_equal(p['id'],f'{s}--p{i:02d}--{m}','partial event identity'))
        if reference_packet is not None:
            base=reference_packet['bases'][m];phase=base['phases'][i]
            for key,val in [('initial',phase['state']),('phase_gradient',phase['z']),('period',base['period']),('phase_epoch',phase['phase_epoch'])]:attempt('event.'+key,lambda key=key,val=val:_equal(p[key],val,'partial event '+key))
            if 'trajectory' in p:
                def trcheck():
                    tr=p['trajectory'];old_math._samples(tr,base['period'],8192,offset=phase['phase_epoch'],rhs=rhs,jacobian=jacobian,alpha=base['alpha'],gamma=base['gamma'])
                    if 'event' in p:_tree(p['event'],_event_diagnostics(p,base),'partial completed event metrics')
                attempt('completed_event_trajectory',trcheck)
    else:errors.append('Unknown partial packet kind')
    return dict(partial_replay_pass=not errors,checks=checks,errors=errors,array_census=partial_array_census(p),qualified=False,scope='Completed subcalculations replayed; failed solver prefixes retained without ODE-solution claim')
