"""Bounded fixed-parameter cycle refinement; no IO, scheduling or import solves.

The caller freezes the plan and preserves all trial/subsolve payloads. Old
references are inputs, never modified. New phase gradients have one global
normalization per setting, not an event-dependent repair.
"""
from __future__ import annotations
import copy
import numpy as np
from scipy.integrate import solve_ivp
from hopfield_diagnostics import hopfield_rhs, hopfield_jacobian, _floquet_diagnostics
from phase_memory import neutral_left_phase
from phase_memory_acquisition import AcquisitionFailure, reference_templates
import phase_reference_diagnostic as prior

METHODS = {
    'tight-dop853': dict(kind='DOP853',rtol=1e-13,atol=1e-15,max_step_period_divisor=16384),
    'tight-rk45': dict(kind='RK45',rtol=1e-13,atol=1e-15,max_step_period_divisor=16384),
}
PRIMARY = 'tight-dop853'
EXPECTED_PEAKS = 9
MAX_UPDATES = 6
FACTORS = (1.,.5,.25,.125)
TRIALS = 25
CLOSURE = 1e-12
SECTION = 1e-12
NEUTRAL = 1e-10
ADJOINT_PERIODICITY = 1e-10
STATE_RADIUS = 1e-6
PERIOD_RADIUS = 1e-6
CONDITION_MAX = 1e10


def method_spec(method,period):
    period=prior._scalar(period,'period',positive=True)
    if method not in METHODS:raise ValueError('Unknown fixed refinement method')
    return dict(METHODS[method],id=method,max_step=period/16384)


def _seed(reference):
    if reference.get('status')!='complete' or reference.get('qualified') is not True:
        raise ValueError('Qualified unchanged historical reference required')
    x=prior._array(reference['certificate']['section_state'],'original section state',(3,)).copy()
    T=prior._scalar(reference['period'],'original period',positive=True)
    if reference['certificate']['period']!=T or not np.array_equal(x,reference['orbit']['states'][0]):
        raise ValueError('Original seed/period binding differs')
    return x,T


def _failure(packet,stage,error):
    packet.update(status='failed',complete=False,qualified=False,failed_stage=stage,error=repr(error))
    if hasattr(error,'payload'):packet['partial_failure']=error.payload
    return AcquisitionFailure('Global reference acquisition failed',packet)


def _forward(state,period,alpha,gamma,method,*,rhs,jacobian,solver,events=False,offset=0.):
    spec=method_spec(method,period);times=prior.build_grids(period)['8192']
    packet=dict(status='started',complete=False,method=method,method_spec=spec,
        initial=state.copy(),period=period,offset=float(offset),ivp_started=False,ivp_completed=False);captured=[]
    def maxima(t,packed):return float(rhs(offset+t,packed[:3],alpha,gamma)[2])
    maxima.direction=-1;maxima.terminal=False
    def wrapped(fun,span,initial,**kw):
        if events:kw['events']=maxima
        packet['ivp_started']=True
        solution=solver(fun,span,initial,**kw);captured.append(solution);return solution
    try:
        raw,dense=prior._solve(prior._augmented(alpha,gamma,rhs,jacobian),(0.,period),
            np.r_[state,np.eye(3).ravel()],times,spec,solver=wrapped,stage='forward',offset=offset)
        packet['raw_solver']=raw
        packet['ivp_completed']=True
        packet['trajectory']=prior._trajectory(times,raw['values'])
        packet['trajectory']['offset']=float(offset)
        prior._derivatives(packet['trajectory'],alpha,gamma,rhs,jacobian,offset=offset)
        if events:
            solution=captured[0]
            te=prior._array(np.asarray(solution.t_events[0]),'maximum event times')
            ye=prior._array(np.asarray(solution.y_events[0]),'maximum event packed')
            if len(te)==0 and ye.size==0:ye=ye.reshape(0,12)
            if te.ndim!=1 or ye.shape!=(len(te),12) or np.any(te<0) or np.any(te>period) or np.any(np.diff(te)<=0):
                raise ValueError('Event times/states invalid')
            packet['maxima']=dict(times=te.copy(),states=ye[:,:3].copy(),phi=ye[:,3:].reshape(-1,3,3).copy())
        packet['solver']=prior._compact_solver(raw);del packet['raw_solver']
        packet.update(status='complete',complete=True)
        return packet,dense
    except Exception as error:raise _failure(packet,'forward',error) from error


def shooting_metrics(state,period,trajectory,seed,seed_period,scales,section_scale,alpha,gamma,*,rhs,jacobian):
    end=trajectory['states'][-1];M=trajectory['phi'][-1]
    f0=prior._array(rhs(0.,state,alpha,gamma),'section flow',(3,))
    fT=prior._array(rhs(period,end,alpha,gamma),'endpoint flow',(3,))
    J0=prior._array(jacobian(0.,state,alpha,gamma),'section Jacobian',(3,3))
    residual=np.r_[end-state,f0[2]]
    bordered=np.zeros((4,4));bordered[:3,:3]=M-np.eye(3);bordered[:3,3]=fT;bordered[3,:3]=J0[2]
    column=np.r_[scales,seed_period];rows=np.r_[scales,section_scale]
    scaled=bordered*column[None,:]/rows[:,None];scaled_r=residual/rows
    singular=np.linalg.svd(scaled,compute_uv=False)
    condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None
    return dict(raw_residual=residual,raw_bordered=bordered,scaled_residual=scaled_r,
        scaled_bordered=scaled,scaled_singular_values=singular,scaled_condition=condition,
        merit=float(np.linalg.norm(scaled_r)),relative_closure=float(np.linalg.norm(end-state)/max(1.,np.linalg.norm(state))),
        section_residual=float(abs(f0[2])),section_curvature=float((J0@f0)[2]),
        seed_scaled_distance=float(np.max(np.abs(state-seed)/scales)),seed_relative_period_change=float(abs(period-seed_period)/seed_period),
        state_spacing=np.spacing(np.abs(state)),endpoint_difference=end-state)


def _converged(metrics):
    return bool(metrics['relative_closure']<=CLOSURE and metrics['section_residual']<=SECTION
        and metrics['scaled_condition'] is not None and metrics['scaled_condition']<=CONDITION_MAX)


def refine_cycle(old_reference,method,*,rhs=hopfield_rhs,jacobian=hopfield_jacobian,solver=solve_ivp):
    seed,T0=_seed(old_reference);method_spec(method,T0)
    alpha,gamma=old_reference['alpha'],old_reference['gamma'];setting=old_reference['setting']
    scales=np.maximum(1.,np.abs(seed));J0=prior._array(jacobian(0.,seed,alpha,gamma),'original Jacobian',(3,3))
    section_scale=max(1.,float(np.linalg.norm(J0[2]*scales)))
    packet=dict(schema='periodic-reference-shoot-v1',kind='shoot',id=f'{setting}--{method}',
        setting=setting,alpha=alpha,gamma=gamma,method=method,original_state=seed.copy(),original_period=T0,
        state_scales=scales,section_scale=section_scale,complete=False,status='started',converged=False,qualified=False,
        limits=dict(maximum_updates=MAX_UPDATES,line_factors=list(FACTORS),trial_slots=TRIALS,
            armijo=1e-4,state_radius=STATE_RADIUS,period_radius=PERIOD_RADIUS,condition_max=CONDITION_MAX,
            relative_closure=CLOSURE,section_residual=SECTION),
        trials=[dict(index=i,id=f'{setting}--{method}--trial-{i:02d}',status='not_needed',attempted=False,
            ivp_attempted=False,ivp_started=False,ivp_completed=False,reason='not_reached') for i in range(TRIALS)],iterations=[],ivp_evaluations=0)
    def evaluate(index,state,period,factor=None):
        row=packet['trials'][index];row.update(attempted=True,status='started',state=state.copy(),period=float(period),factor=factor)
        if (not np.isfinite(state).all() or not np.isfinite(period) or period<=0
                or np.max(np.abs(state-seed)/scales)>STATE_RADIUS or abs(period-T0)/T0>PERIOD_RADIUS):
            row.update(status='outside_neighborhood',complete=True,accepted=False);return None
        try:
            row['ivp_attempted']=True;packet['ivp_evaluations']+=1
            def tracked_solver(*args,**kwargs):
                row['ivp_started']=True
                return solver(*args,**kwargs)
            forward,_=_forward(state,period,alpha,gamma,method,rhs=rhs,jacobian=jacobian,solver=tracked_solver)
            row['ivp_completed']=forward['ivp_completed']
            row['forward']=forward
            row['metrics']=shooting_metrics(state,period,forward['trajectory'],seed,T0,scales,section_scale,
                alpha,gamma,rhs=rhs,jacobian=jacobian)
            row.update(status='complete',complete=True,accepted=False);return row
        except Exception as error:
            row.update(status='failed',complete=False,error=repr(error))
            if hasattr(error,'payload'):
                row['partial_failure']=error.payload
                row['ivp_completed']=bool(error.payload.get('ivp_completed',False))
            raise
    stage='initial_trial'
    try:
        current=evaluate(0,seed,T0);current['accepted']=True
        reason='maximum_updates'
        for iteration in range(MAX_UPDATES+1):
            metrics=current['metrics']
            if _converged(metrics):packet['converged']=True;reason='residual_targets';break
            if iteration==MAX_UPDATES:break
            if metrics['scaled_condition'] is None or metrics['scaled_condition']>CONDITION_MAX:
                reason='singular_or_ill_conditioned_bordered_system';break
            stage=f'newton_update_{iteration}'
            delta=np.linalg.solve(metrics['scaled_bordered'],-metrics['scaled_residual'])
            prior._array(delta,'scaled Newton correction',(4,))
            record=dict(iteration=iteration,from_trial=current['index'],scaled_correction=delta.copy(),trial_indices=[])
            packet['iterations'].append(record);moved=False
            for half,factor in enumerate(FACTORS):
                index=1+iteration*len(FACTORS)+half;record['trial_indices'].append(index)
                state=current['state']+factor*delta[:3]*scales
                period=float(current['period']+factor*delta[3]*T0)
                candidate=evaluate(index,state,period,factor)
                if candidate is not None and candidate['metrics']['merit']<=(1.-1e-4*factor)*metrics['merit']:
                    candidate['accepted']=True;current=candidate;moved=True
                    for skipped in range(index+1,1+(iteration+1)*len(FACTORS)):
                        packet['trials'][skipped]['reason']='line_search_accepted'
                    break
            if not moved:reason='line_search_no_decrease';break
        packet['candidate']=dict(state=current['state'].copy(),period=current['period'],source_trial_index=current['index'],metrics=copy.deepcopy(current['metrics']))
        for row in packet['trials']:
            if not row['attempted'] and row['reason']=='not_reached':row['reason']='unreached_after_'+reason
        packet.update(status='complete',complete=True,qualified=packet['converged'],stop_reason=reason,
            attempted_trials=sum(r['attempted'] for r in packet['trials']))
        return packet
    except Exception as error:raise _failure(packet,stage,error) from error


def _certificate(forward,alpha,gamma,*,rhs,jacobian):
    tr=forward['trajectory'];x0,xT=tr['states'][0],tr['states'][-1];T=forward['period'];M=tr['phi'][-1]
    f0=tr['derivatives'][0];J0=jacobian(0.,x0,alpha,gamma)
    te=forward['maxima']['times'];xe=forward['maxima']['states'];tol=max(1e-7,1e-6*T)
    inside=(te>tol)&(te<T-tol);interior=xe[inside]
    distances=np.linalg.norm(interior-x0,axis=1)/max(1.,np.linalg.norm(x0))
    curvature=np.asarray([(jacobian(t,x,alpha,gamma)@rhs(t,x,alpha,gamma))[2] for t,x in zip(te,xe,strict=True)],dtype=np.float64)
    floquet=_floquet_diagnostics(M,f0)
    _,left_singular,left_vh=np.linalg.svd(M.T-np.eye(3),full_matrices=False)
    left_unit=left_vh[-1].copy()
    left_residual=float(np.linalg.norm((M.T-np.eye(3))@left_unit))
    left_unique=bool(left_singular[-2]>1e-12*max(1.,left_singular[0]))
    closure=float(np.linalg.norm(xT-x0)/max(1.,np.linalg.norm(x0)))
    section=float(abs(f0[2]));section_curvature=float((J0@f0)[2]);amplitude=float(np.max(np.ptp(tr['states'],axis=0)))
    gates=dict(closure=closure<=CLOSURE,section=section<=SECTION,section_is_maximum=section_curvature < -1e-8,
        all_recorded_events_are_maxima=bool(np.all(curvature < -1e-8)),nonstationary=np.linalg.norm(f0)>1e-7 and amplitude>1e-6,
        expected_peak_count=1+int(inside.sum())==EXPECTED_PEAKS,primitive=bool(np.all(distances>1e-7)),
        neutral_flow=floquet['neutral_flow_residual'] is not None and floquet['neutral_flow_residual']<=NEUTRAL,
        neutral_eigenvalue=floquet['neutral_multiplier_error']<=NEUTRAL,neutral_alignment=floquet['neutral_flow_alignment']>=.99,
        left_neutral=left_residual<=NEUTRAL,unique_left_neutral=left_unique,
        transverse_stability=floquet['transversely_attracting'])
    return dict(qualified=bool(all(gates.values())),gates={k:bool(v) for k,v in gates.items()},relative_closure=closure,
        section_residual=section,section_curvature=section_curvature,state_amplitude=amplitude,expected_peaks=EXPECTED_PEAKS,
        peak_count=1+int(inside.sum()),endpoint_exclusion=tol,interior_mask=inside,interior_return_distances=distances,
        event_curvatures=curvature,excluded_event_times=te[~inside].copy(),excluded_event_states=xe[~inside].copy(),
        left_neutral_unit_vector=left_unit,left_neutral_singular_values=left_singular,
        left_neutral_relative_residual=left_residual,unique_left_neutral=left_unique,**floquet)


def _refined_base(reference,forward,dense,terminal,phase_rows,method,*,rhs,jacobian,solver):
    setting=reference['setting'];alpha,gamma=reference['alpha'],reference['gamma'];T=forward['period']
    phases=prior._phases(phase_rows,T,setting);tr=forward['trajectory'];times=tr['times'];spec=method_spec(method,T)
    packet=dict(schema='refined-phase-base-v1',kind='base',id=f'{setting}--{method}',setting=setting,
        alpha=alpha,gamma=gamma,period=T,method=method,method_spec=spec,status='started',complete=False,
        canonical_state=tr['states'][0].copy(),global_terminal_adjoint=terminal.copy(),
        gradient_origin='one global normalization of canonical DOP853 verification left-neutral vector',
        new_normalizations=0,phase_rows=copy.deepcopy(phases),phases=[],grids={},forward=copy.deepcopy(forward['solver']),
        ivp_started=False,ivp_completed=False)
    stage='backward'
    try:
        def adjoint(t,z):
            x=prior._array(dense(t),'verification dense packed',(12,))[:3]
            return -jacobian(t,x,alpha,gamma).T@z
        def tracked_solver(*args,**kwargs):
            packet['ivp_started']=True
            return solver(*args,**kwargs)
        raw,dense_z=prior._solve(adjoint,(T,0.),terminal,times[::-1].copy(),spec,solver=tracked_solver,stage='backward')
        packet['ivp_completed']=True
        packet['backward']=raw
        fine={k:copy.deepcopy(v) for k,v in tr.items() if k!='offset'}
        fine['z']=raw['values'][::-1].copy();packet['grids']['8192']=fine
        stage='grid_derivatives';prior._derivatives(fine,alpha,gamma,rhs,jacobian)
        packet['grids']['4096']={k:v[::2].copy() for k,v in fine.items()}
        packet['raw_endpoints']=dict(initial_state=fine['states'][0].copy(),final_state=fine['states'][-1].copy(),
            initial_phi=fine['phi'][0].copy(),final_phi=fine['phi'][-1].copy(),initial_z=fine['z'][0].copy(),final_z=fine['z'][-1].copy())
        if not np.array_equal(fine['z'][-1],terminal):raise ValueError('Global terminal vector changed')
        stage='exact_phases';epochs=np.asarray([p['phase_epoch'] for p in phases],dtype=np.float64)
        data=prior._trajectory(epochs,prior._array(dense(epochs),'exact dense state/Phi',(12,24)).T)
        packet['exact_phase_arrays']=data;data['z']=prior._array(dense_z(epochs),'exact dense adjoint',(3,24)).T.copy()
        prior._derivatives(data,alpha,gamma,rhs,jacobian)
        for i,p in enumerate(phases):
            packet['phases'].append(dict(p,state=data['states'][i].copy(),phi=data['phi'][i].copy(),
                flow=data['derivatives'][i].copy(),phi_derivative=data['phi_derivatives'][i].copy(),
                z=data['z'][i].copy(),z_derivative=data['z_derivatives'][i].copy(),
                normalization_residual=float(data['normalization_residuals'][i])))
        z0,zT=fine['z'][0],fine['z'][-1];scale=max(np.linalg.norm(z0),np.linalg.norm(zT))
        if scale==0 or not np.isfinite(scale):raise ValueError('Finite nonzero adjoint scale required')
        periodicity=float(np.linalg.norm(z0-zT)/scale)
        packet['diagnostics']=dict(adjoint_relative_periodicity=periodicity,adjoint_periodic_difference=z0-zT,
            grid_normalization_max=float(np.max(np.abs(fine['normalization_residuals']))),
            exact_phase_normalization_max=float(np.max(np.abs(data['normalization_residuals']))))
        packet['gates']=dict(adjoint_periodicity=periodicity<=ADJOINT_PERIODICITY,
            grid_normalization=packet['diagnostics']['grid_normalization_max']<=1e-7,
            exact_phase_normalization=packet['diagnostics']['exact_phase_normalization_max']<=1e-7)
        packet['backward']=prior._compact_solver(raw)
        packet.update(status='complete',complete=True,qualified=bool(all(packet['gates'].values())))
        return packet
    except Exception as error:raise _failure(packet,stage,error) from error


def crossverify_reference(old_reference,candidates,phase_rows,*,rhs=hopfield_rhs,jacobian=hopfield_jacobian,solver=solve_ivp):
    """Four unoptimized candidates/algorithms before one canonical global Z."""
    seed,T0=_seed(old_reference);setting=old_reference['setting'];alpha,gamma=old_reference['alpha'],old_reference['gamma']
    if set(candidates)!=set(METHODS):raise ValueError('Both fixed independent shooting candidates required')
    for method,shoot in candidates.items():
        if (shoot.get('complete') is not True or shoot.get('converged') is not True or shoot.get('method')!=method
                or shoot.get('id')!=f'{setting}--{method}' or shoot.get('original_period')!=T0
                or not np.array_equal(shoot.get('original_state'),seed)):
            raise ValueError('Matching completed converged original-seed shooting candidates required')
    primary=candidates[PRIMARY]['candidate'];alternate=candidates['tight-rk45']['candidate'];scales=np.maximum(1.,np.abs(seed))
    agreement=dict(state_scaled_infinity=float(np.max(np.abs(primary['state']-alternate['state'])/scales)),
        period_relative=float(abs(primary['period']-alternate['period'])/T0),
        state_difference=primary['state']-alternate['state'],period_difference=float(primary['period']-alternate['period']),
        state_threshold=1e-10,period_threshold=1e-11)
    agreement['qualified']=agreement['state_scaled_infinity']<=1e-10 and agreement['period_relative']<=1e-11
    packet=dict(schema='periodic-reference-pair-v1',kind='reference',id=f'{setting}--reference-pair',setting=setting,
        alpha=alpha,gamma=gamma,original_state=seed.copy(),original_period=T0,
        candidates={m:copy.deepcopy(s['candidate']) for m,s in candidates.items()},candidate_agreement=agreement,
        canonical_method=PRIMARY,complete=False,status='started',qualified=False,gates={},bases={},
        verifications={f'{c}--{m}':dict(id=f'{setting}--verify--{c}--{m}',candidate_method=c,method=m,status='not_started',complete=False,ivp_started=False,ivp_completed=False)
            for c in METHODS for m in METHODS},
        adjoints={m:dict(id=f'{setting}--adjoint--{m}',method=m,status='blocked_prerequisite',complete=False,ivp_started=False,ivp_completed=False) for m in METHODS})
    dense_paths={};stage='crossverification'
    try:
        # All independent forward jobs run, even if a scientific gate fails.
        for candidate_method in METHODS:
            candidate=candidates[candidate_method]['candidate']
            for method in METHODS:
                key=f'{candidate_method}--{method}';row=packet['verifications'][key]
                try:
                    def tracked_solver(*args,**kwargs):
                        row['ivp_started']=True
                        return solver(*args,**kwargs)
                    forward,dense=_forward(candidate['state'],candidate['period'],alpha,gamma,method,
                        rhs=rhs,jacobian=jacobian,solver=tracked_solver,events=True)
                    row['ivp_completed']=forward['ivp_completed']
                    row['forward']=forward;row['certificate']=_certificate(forward,alpha,gamma,rhs=rhs,jacobian=jacobian)
                    row.update(status='complete',complete=True,qualified=row['certificate']['qualified'])
                    if candidate_method==PRIMARY:dense_paths[method]=dense
                except Exception as error:
                    row.update(status='failed',complete=False,qualified=False,error=repr(error))
                    if hasattr(error,'payload'):
                        row['partial_failure']=error.payload;row['ivp_completed']=bool(error.payload.get('ivp_completed',False))
        packet['gates']=dict(candidate_agreement=agreement['qualified'],
            all_cross_verifications=all(r.get('qualified') is True for r in packet['verifications'].values()))
        # Same candidate/T, so clocks are exactly shared. Other-candidate
        # normalized-phase comparisons are owned by the saved-array evaluator.
        if all(r['complete'] for r in packet['verifications'].values()):
            from phase_reference_diagnostic_evaluation import array_agreement
            packet['candidate_method_agreements']={c:array_agreement(
                packet['verifications'][f'{c}--{PRIMARY}']['forward']['trajectory'],
                packet['verifications'][f'{c}--tight-rk45']['forward']['trajectory']) for c in METHODS}
            packet['canonical_method_agreement']=packet['candidate_method_agreements'][PRIMARY]
            packet['gates']['canonical_method_agreement']=packet['canonical_method_agreement']['agreement_pass']
            packet['gates']['both_candidate_method_agreements']=all(x['agreement_pass'] for x in packet['candidate_method_agreements'].values())
        else:packet['gates'].update(canonical_method_agreement=False,both_candidate_method_agreements=False)
        if not all(packet['gates'].values()):
            packet.update(status='blocked_crossverification',complete=False,qualified=False);return packet
        stage='canonical_global_gradient'
        source=packet['verifications'][f'{PRIMARY}--{PRIMARY}']['forward'];tr=source['trajectory']
        neutral=neutral_left_phase(tr['phi'][-1],tr['derivatives'][0],residual_atol=NEUTRAL,
            unique_singular_rtol=1e-12,normalization_cosine_min=1e-12)
        terminal=neutral['phase_gradient'].copy()
        # Preserve the actual SVD sign/normalization evidence; no alternate
        # solver's vector can substitute for this canonical boundary.
        _,singular,vh=np.linalg.svd(tr['phi'][-1].T-np.eye(3),full_matrices=False)
        packet['canonical']=dict(state=primary['state'].copy(),period=primary['period'],terminal_adjoint=terminal,
            neutral=neutral,unnormalized_svd_vector=vh[-1].copy(),singular_values=singular,
            normalization_scalar=float(vh[-1]@tr['derivatives'][0]),normalization_count=1,
            normalization_applied_scale=1./float(vh[-1]@tr['derivatives'][0]),
            source_verification=f'{PRIMARY}--{PRIMARY}')
        stage='backward_reconstruction'
        for method in METHODS:
            row=packet['adjoints'][method];forward=packet['verifications'][f'{PRIMARY}--{method}']['forward']
            try:
                def tracked_solver(*args,**kwargs):
                    row['ivp_started']=True
                    return solver(*args,**kwargs)
                base=_refined_base(old_reference,forward,dense_paths[method],terminal,phase_rows,method,
                    rhs=rhs,jacobian=jacobian,solver=tracked_solver)
                row['ivp_completed']=base['ivp_completed']
                packet['bases'][method]=base;row.update(status='complete',complete=True,qualified=base['qualified'],base_id=base['id'])
            except Exception as error:
                row.update(status='failed',complete=False,qualified=False,error=repr(error))
                if hasattr(error,'payload'):
                    row['partial_failure']=error.payload;row['ivp_completed']=bool(error.payload.get('ivp_completed',False))
        packet['gates']['both_reconstructed_bases']=all(r.get('qualified') is True for r in packet['adjoints'].values())
        packet.update(status='complete' if all(r['complete'] for r in packet['adjoints'].values()) else 'incomplete_adjoint',
            complete=all(r['complete'] for r in packet['adjoints'].values()),qualified=bool(all(packet['gates'].values())))
        return packet
    except Exception as error:
        if hasattr(error,'diagnostics'):packet['gradient_failure_diagnostics']=error.diagnostics
        raise _failure(packet,stage,error) from error


def acquire_refined_event(base,phase_index,*,rhs=hopfield_rhs,jacobian=hopfield_jacobian,solver=solve_ivp):
    if base.get('complete') is not True or base.get('qualified') is not True or base.get('schema')!='refined-phase-base-v1':
        raise ValueError('Qualified new global reference base required')
    if type(phase_index) is not int or phase_index not in range(24):raise ValueError('Original phase index0..23 required')
    method=base['method'];T=base['period'];spec=method_spec(method,T);setting=base['setting']
    if base['id']!=f'{setting}--{method}' or base['method_spec']!=spec:raise ValueError('New base method identity differs')
    phase=base['phases'][phase_index]
    expected=prior._phases(base['phase_rows'],T,setting)[phase_index]
    if any(phase[k]!=v for k,v in expected.items()):raise ValueError('Original phase metadata differs')
    x=prior._array(phase['state'],'new exact phase state',(3,)).copy();z=prior._array(phase['z'],'new nominal phase adjoint',(3,)).copy()
    for key,name in [('state','states'),('z','z'),('flow','derivatives')]:
        if not np.array_equal(phase[key],base['exact_phase_arrays'][name][phase_index]):raise ValueError('New dense phase binding differs')
    packet=dict(schema='refined-phase-event-v1',kind='event',id=f'{setting}--p{phase_index:02d}--{method}',
        context_id=f'{setting}--p{phase_index:02d}',base_id=base['id'],setting=setting,method=method,method_spec=spec,
        alpha=base['alpha'],gamma=base['gamma'],period=T,phase_index=phase_index,
        phase_fraction=phase['phase_fraction'],phase_epoch=phase['phase_epoch'],split=phase['split'],
        event_time=T,physical_event_time=phase['phase_epoch']+T,initial=x,phase_gradient=z,
        new_normalizations=0,status='started',complete=False,ivp_started=False,ivp_completed=False)
    try:
        def tracked_solver(*args,**kwargs):
            packet['ivp_started']=True
            return solver(*args,**kwargs)
        forward,_=_forward(x,T,base['alpha'],base['gamma'],method,rhs=rhs,jacobian=jacobian,
            solver=tracked_solver,offset=phase['phase_epoch'])
        packet['ivp_completed']=forward['ivp_completed']
        packet['forward']=forward['solver'];tr=forward['trajectory'];packet['trajectory']=tr
        f0,f=tr['derivatives'][0],tr['derivatives'][-1];end=tr['states'][-1];Q=np.outer(f,z);P=np.eye(3)-Q
        packet['event']=dict(state=end.copy(),flow=f.copy(),phase_gradient=z.copy(),initial_flow=f0.copy(),Q=Q,P=P,
            initial_normalization_residual=float(z@f0-1.),normalization_residual=float(z@f-1.),
            event_minus_initial_normalization=float(z@(f-f0)),return_difference=end-x,
            Q_idempotence_residual=Q@Q-Q,P_idempotence_residual=P@P-P,QP_residual=Q@P,phase_leakage=z@P,
            Q_operator_norm=float(np.linalg.norm(Q,ord=2)),P_operator_norm=float(np.linalg.norm(P,ord=2)))
        for key,value in packet['event'].items():
            if isinstance(value,np.ndarray):prior._array(value,'event '+key)
            else:prior._scalar(value,'event '+key)
        packet.update(status='complete',complete=True);return packet
    except Exception as error:
        if hasattr(error,'payload'):packet['ivp_completed']=bool(error.payload.get('ivp_completed',False))
        raise _failure(packet,'event_continuation',error) from error
