"""Six-cycle/eight-phase unforced-reference acquisition, with no IO or solves at import.

The generic shooting kernel and scalar numerical helpers are frozen EXP018
implementations. The reference-pair and context functions are scoped copies
with explicit eight-phase metadata, truthful cycle IDs, and seed admission
separate from new-reference qualification. No historical module is patched.
"""
from __future__ import annotations
from collections.abc import Mapping
import copy
import re
import numpy as np
from scipy.integrate import solve_ivp
from hopfield_diagnostics import hopfield_rhs, hopfield_jacobian
from phase_memory import neutral_left_phase
from phase_memory_acquisition import AcquisitionFailure
import periodic_reference_refinement as refinement
import phase_reference_diagnostic as prior
import refined_two_write_context as old_context
import two_write_acquisition as tangent

PHASE_COUNT=8
PHASE_SEED=2026091702
METHODS=copy.deepcopy(refinement.METHODS)
TANGENT_METHODS=copy.deepcopy(tangent.TANGENT_METHODS)
PRIMARY=refinement.PRIMARY
CANONICAL_VERIFICATION=PRIMARY+'--'+PRIMARY
NEUTRAL=refinement.NEUTRAL
ADJOINT_PERIODICITY=refinement.ADJOINT_PERIODICITY
SOURCE_FIELDS=('experiment_id','source_commit','plan_sha256','reference_id','packet_sha256','packet_semantic_sha256')
CYCLE_IDS=('anchor-core','anchor-failed-near-edge','core-rho025','core-rho050',
    'failed-near-edge-rho025','failed-near-edge-rho050')
method_spec=refinement.method_spec
_forward=refinement._forward
_certificate=refinement._certificate
_failure=refinement._failure


def contract():
    return dict(cycle_count=6,phase_count=PHASE_COUNT,phase_seed=PHASE_SEED,
        phase_split='fresh',shooting_methods=copy.deepcopy(METHODS),
        tangent_methods=copy.deepcopy(TANGENT_METHODS),shooting_jobs=12,
        reference_jobs=6,tangent_jobs=96,top_level_jobs=114,
        shoot_trial_slots=300,cross_verification_slots=24,adjoint_slots=12,
        tangent_segment_slots=192,total_ivp_slots=528,
        global_normalizations_per_new_cycle=1,per_phase_normalizations=0,
        anchors_freshly_reshot=True,seed_admission_is_new_reference_qualification=False,
        tangent_initial_identity_per_segment=True,finite_write_jobs=0)


def phase_rows():
    values=np.random.Generator(np.random.PCG64(PHASE_SEED)).random(PHASE_COUNT)
    return [dict(phase_index=i,phase_fraction=float(v),split='fresh') for i,v in enumerate(values)]


def _identifier(value,name):
    if type(value) is not str or re.fullmatch('[a-z][a-z0-9_-]*',value) is None:
        raise ValueError(name+' must be a safe truthful cycle identifier')
    return value


def phase_metadata(rows,period,cycle_id):
    period=prior._scalar(period,'period',positive=True);_identifier(cycle_id,'cycle_id')
    if not isinstance(rows,list) or len(rows)!=PHASE_COUNT:raise ValueError('Exactly8 ordered phase rows required')
    result=[]
    for i,row in enumerate(rows):
        if not isinstance(row,Mapping) or type(row.get('phase_index')) is not int or row['phase_index']!=i:
            raise ValueError('Exact ordered phase indices0..7 required')
        if set(row) not in ({'phase_index','phase_fraction','split'},{'context_id','phase_index','phase_fraction','split','phase_epoch'}):
            raise ValueError('Exact phase metadata fields required')
        fraction=prior._scalar(row['phase_fraction'],'phase fraction')
        if not 0<=fraction<1 or row['split']!='fresh':raise ValueError('Fresh fraction in[0,1) required')
        expected=dict(context_id=f'{cycle_id}--p{i:02d}',phase_index=i,phase_fraction=fraction,split='fresh',phase_epoch=fraction*period)
        for key in ('context_id','phase_epoch'):
            if key in row:old_context._same(row[key],expected[key],'phase '+key)
        result.append(expected)
    if len({p['phase_fraction'] for p in result})!=PHASE_COUNT:raise ValueError('Eight distinct phase fractions required')
    return result


def _seed(reference):
    if not isinstance(reference,Mapping) or reference.get('schema')!='matched-cycle-seed-v1':raise ValueError('Explicit admitted cycle seed required')
    if reference.get('status')!='admitted_historical_seed' or reference.get('admitted_historical_seed') is not True or reference.get('qualified') is not False or reference.get('new_reference_qualified') is not False:
        raise ValueError('Historical seed admission must remain separate from new qualification')
    cycle_id=_identifier(reference['cycle_id'],'cycle_id')
    if cycle_id not in CYCLE_IDS:raise ValueError('One of the six declared cycle slots required')
    if reference.get('source_setting') not in ('core','failed-near-edge') or reference.get('role') not in ('anchor','candidate'):
        raise ValueError('Historical source setting and anchor/candidate role required')
    old_context._same(reference['setting'],cycle_id,'truthful setting alias')
    for key in ('alpha','gamma'):prior._scalar(reference[key],key)
    state=prior._array(reference['certificate']['section_state'],'seed section',(3,)).copy()
    period=prior._scalar(reference['period'],'seed period',positive=True)
    old_context._same(reference['certificate']['period'],period,'seed certificate period')
    old_context._same(reference['orbit']['states'],state[None,:],'seed-only orbit row')
    if not isinstance(reference.get('seed_provenance'),Mapping) or not reference['seed_provenance']:
        raise ValueError('Immutable historical seed provenance required')
    selected=reference.get('selection')
    if not isinstance(selected,Mapping) or selected.get('available') is not True or selected.get('status')!='selected':
        raise ValueError('Original selected slot must remain attached to its admitted seed')
    for key,expected in [('id',cycle_id),('role',reference['role']),('source_setting',reference['source_setting']),
                         ('alpha',reference['alpha']),('gamma',reference['gamma']),('period',period)]:
        old_context._same(selected.get(key),expected,'unchanged selected seed '+key)
    old_context._same(np.asarray(selected['first_maximum_state']),state,'unchanged selected maximum')
    if selected.get('seed_provenance')!=reference['seed_provenance']:
        raise ValueError('Original selected seed provenance changed')
    return state,period


def prepare_seed(slot):
    """Admit selected historical metadata, without certifying a new cycle.

    The selector/controller binds historical packet and branch bytes before
    calling this helper. The one-row `orbit` and small `certificate` below are
    seed interfaces only, not a newly acquired trajectory or certificate.
    """
    if not isinstance(slot,Mapping) or slot.get('available') is not True or slot.get('status')!='selected':
        raise ValueError('Available selected historical slot required; never replace a missing slot')
    cycle_id=_identifier(slot['id'],'slot id')
    if cycle_id not in CYCLE_IDS:raise ValueError('Exact declared cycle slot required')
    if 'cycle_id' in slot:old_context._same(slot['cycle_id'],cycle_id,'slot cycle_id')
    role=slot['role'];source=slot['source_setting']
    if role not in ('anchor','candidate') or source not in ('core','failed-near-edge'):
        raise ValueError('Historical role/source required')
    if role=='anchor':
        if cycle_id!='anchor-'+source or slot.get('target_rho') is not None:
            raise ValueError('Anchor identity/target differs')
    else:
        target=prior._scalar(slot['target_rho'],'target rho')
        if target not in (.25,.5) or cycle_id!=source+('-rho025' if target==.25 else '-rho050'):
            raise ValueError('Candidate identity/target differs')
        if type(slot.get('node_index')) is not int or slot['node_index']<0 or not isinstance(slot.get('branch_id'),str) or not isinstance(slot.get('node_id'),str):
            raise ValueError('Exact selected historical node identity required')
    x=prior._array(np.asarray(slot['first_maximum_state']),'historical maximum seed',(3,)).copy()
    T=prior._scalar(slot['period'],'historical seed period',positive=True)
    a,g=(prior._scalar(slot[k],k) for k in ('alpha','gamma'))
    if not isinstance(slot.get('seed_provenance'),Mapping) or not slot['seed_provenance']:
        raise ValueError('Bound historical seed provenance required')
    result=dict(schema='matched-cycle-seed-v1',id=cycle_id,cycle_id=cycle_id,setting=cycle_id,
        source_setting=source,role=role,alpha=a,gamma=g,period=T,
        status='admitted_historical_seed',admitted_historical_seed=True,qualified=False,new_reference_qualified=False,
        certificate=dict(section_state=x.copy(),period=T,seed_interface_only=True),
        orbit=dict(states=x[None,:].copy(),seed_interface_only=True),
        seed_provenance=copy.deepcopy(dict(slot['seed_provenance'])),selection=copy.deepcopy(dict(slot)))
    _seed(result);return result


def _metadata(reference):
    return dict(cycle_id=reference['cycle_id'],source_setting=reference['source_setting'],
        role=reference['role'],seed_provenance=copy.deepcopy(reference['seed_provenance']),
        admitted_historical_seed=True,seed_was_not_a_qualified_new_reference=True)


def _internal_seed(reference):
    """Private legacy guard shim, never returned as a qualified reference."""
    _seed(reference);internal=copy.deepcopy(dict(reference))
    internal.update(status='complete',qualified=True,
        internal_legacy_guard_only='historical seed admitted; not new-reference qualification')
    return internal


def refine_cycle(seed_reference,method,*,rhs=hopfield_rhs,jacobian=hopfield_jacobian,solver=solve_ivp):
    _seed(seed_reference)
    try:
        result=refinement.refine_cycle(_internal_seed(seed_reference),method,rhs=rhs,jacobian=jacobian,solver=solver)
        result.update(_metadata(seed_reference));return result
    except AcquisitionFailure as error:
        error.payload.update(_metadata(seed_reference));raise


def _refined_base(reference,forward,dense,terminal,phase_rows,method,*,rhs,jacobian,solver):
    setting=reference['setting'];alpha,gamma=reference['alpha'],reference['gamma'];T=forward['period']
    phases=phase_metadata(phase_rows,T,setting);tr=forward['trajectory'];times=tr['times'];spec=method_spec(method,T)
    packet=dict(schema='refined-phase-base-v1',kind='base',id=f'{setting}--{method}',setting=setting,
        alpha=alpha,gamma=gamma,period=T,method=method,method_spec=spec,status='started',complete=False,
        canonical_state=tr['states'][0].copy(),global_terminal_adjoint=terminal.copy(),
        gradient_origin='one global normalization of canonical DOP853 verification left-neutral vector',
        new_normalizations=0,phase_rows=copy.deepcopy(phases),phases=[],grids={},forward=copy.deepcopy(forward['solver']),
        ivp_started=False,ivp_completed=False)
    packet.update(_metadata(reference))
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
        data=prior._trajectory(epochs,prior._array(dense(epochs),'exact dense state/Phi',(12,PHASE_COUNT)).T)
        packet['exact_phase_arrays']=data;data['z']=prior._array(dense_z(epochs),'exact dense adjoint',(3,PHASE_COUNT)).T.copy()
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
    phase_metadata(phase_rows,T0,setting)
    if set(candidates)!=set(METHODS):raise ValueError('Both fixed independent shooting candidates required')
    for method,shoot in candidates.items():
        for key in ('setting','alpha','gamma','cycle_id','source_setting','role'):
            old_context._same(shoot.get(key),old_reference[key],'shoot seed '+key)
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
    packet.update(_metadata(old_reference))
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


def _source(source_binding,setting):
    if not isinstance(source_binding,Mapping) or set(source_binding)!=set(SOURCE_FIELDS):
        raise ValueError('Exact current-run reference source binding required')
    if type(source_binding['experiment_id']) is not str or re.fullmatch('EXP-[0-9]{3}',source_binding['experiment_id']) is None:
        raise ValueError('Current study experiment identity required')
    old_context._same(source_binding['reference_id'],setting+'--reference-pair','source reference')
    for key,length in [('source_commit',40),('plan_sha256',64),('packet_sha256',64),('packet_semantic_sha256',64)]:
        if type(source_binding[key]) is not str or re.fullmatch('[0-9a-f]{'+str(length)+'}',source_binding[key]) is None:
            raise ValueError('Valid current source '+key+' required')
    return copy.deepcopy(dict(source_binding))


def prepare_context(reference_pair, phase_row, *, source_binding, rhs=hopfield_rhs):
    """Use one canonical exact dense state/Z for both future 8T algorithms.

    ``source_binding`` contains the six fields in SOURCE_FIELDS. Its hashes
    identify caller-verified immutable artifacts; they are not recomputed here.
    ``phase_row`` contains phase_index, phase_fraction and the fresh split.
    Eight new common fractions are used on historical cycle slots; this is
    not a fresh set of networks or independent parameter-space branches.
    """
    pair = reference_pair
    if not isinstance(pair, Mapping) or pair.get('complete') is not True or pair.get('qualified') is not True:
        raise ValueError('Complete qualified global reference pair required')
    old_context._same(pair.get('status'), 'complete', 'reference status')
    old_context._same(pair.get('schema'), 'periodic-reference-pair-v1', 'reference schema')
    old_context._same(pair.get('kind'), 'reference', 'reference kind')
    setting = pair.get('setting')
    if setting not in CYCLE_IDS:
        raise ValueError('One of the six declared cycle IDs required')
    old_context._same(pair.get('cycle_id'),setting,'truthful cycle identity')
    old_context._same(pair.get('id'), setting+'--reference-pair', 'reference identity')
    old_context._same(pair.get('canonical_method'), PRIMARY, 'canonical method')
    provenance_binding = _source(source_binding, setting)
    if not isinstance(pair.get('gates'), Mapping) or not pair['gates'] or any(v is not True for v in pair['gates'].values()):
        raise ValueError('Every reference-pair prerequisite must qualify')
    canonical = pair['canonical']
    old_context._same(canonical.get('source_verification'), CANONICAL_VERIFICATION, 'canonical verification')
    old_context._same(canonical.get('normalization_count'), 1, 'one original global normalization')
    period = tangent._positive(canonical['period'], 'canonical period')
    section = tangent._array(canonical['state'], 'canonical section', (3,))
    terminal = tangent._array(canonical['terminal_adjoint'], 'canonical terminal gradient', (3,))
    old_context._same(canonical['neutral']['phase_gradient'], terminal, 'canonical gradient')
    if set(pair['bases']) != {'tight-dop853', 'tight-rk45'}:
        raise ValueError('Both canonical reference reconstructions required')
    for method, candidate_base in pair['bases'].items():
        if candidate_base.get('complete') is not True or candidate_base.get('qualified') is not True:
            raise ValueError('Both canonical reference reconstructions must qualify')
        old_context._same(candidate_base['method'], method, 'base method')
        old_context._same(candidate_base['id'], setting+'--'+method, 'base identity')
        old_context._same(candidate_base['period'], period, 'base canonical period')
        old_context._same(candidate_base['canonical_state'], section, 'base canonical section')
        old_context._same(candidate_base['global_terminal_adjoint'], terminal, 'shared terminal gradient')
    base = pair['bases'][PRIMARY]
    old_context._same(base.get('schema'), 'refined-phase-base-v1', 'canonical base schema')
    old_context._same(base.get('status'), 'complete', 'canonical base status')
    old_context._same(base.get('new_normalizations'), 0, 'no base gradient repair')
    old_context._same(base['raw_endpoints']['final_z'], terminal, 'unchanged backward terminal')
    alpha, gamma = (old_context._number(pair[name], name) for name in ('alpha', 'gamma'))
    for name, value in [('alpha', alpha), ('gamma', gamma)]:
        old_context._same(base[name], value, 'base '+name)
    if not isinstance(phase_row, Mapping) or set(phase_row) != {'phase_index', 'phase_fraction', 'split'}:
        raise ValueError('Exact declared phase row required')
    index = phase_row['phase_index']
    if type(index) is not int or index not in range(PHASE_COUNT):
        raise ValueError('Fresh phase index0..7 required')
    if len(base['phases']) != PHASE_COUNT or len(base['phase_rows']) != PHASE_COUNT:
        raise ValueError('All8 fresh phases required')
    exact = base['exact_phase_arrays']
    for key, shape in [('times', (PHASE_COUNT,)), ('states', (PHASE_COUNT, 3)), ('z', (PHASE_COUNT, 3)), ('derivatives', (PHASE_COUNT, 3))]:
        tangent._array(exact[key], 'exact dense '+key, shape)
    for i, phase in enumerate(base['phases']):
        fraction = old_context._number(phase['phase_fraction'], 'phase fraction')
        if not 0 <= fraction < 1:
            raise ValueError('Fresh phase fraction must be in [0,1)')
        metadata = dict(context_id=f'{setting}--p{i:02d}', phase_index=i,
            phase_fraction=fraction, split='fresh', phase_epoch=fraction*period)
        if set(base['phase_rows'][i]) != set(metadata):
            raise ValueError('Fresh phase metadata fields differ')
        for key, value in metadata.items():
            old_context._same(phase[key], value, 'phase '+key)
            old_context._same(base['phase_rows'][i][key], value, 'phase row '+key)
        for pk, ak in [('state', 'states'), ('z', 'z'), ('flow', 'derivatives'), ('phase_epoch', 'times')]:
            value = exact[ak][i]
            if pk == 'phase_epoch':
                value = float(value)
            old_context._same(phase[pk], value, 'exact dense '+pk)
    phase = base['phases'][index]
    for key, value in phase_row.items():
        old_context._same(phase[key], value, 'declared '+key)
    state = phase['state'].copy()
    z = phase['z'].copy()
    flow = tangent._array(rhs(phase['phase_epoch'], state, alpha, gamma), 'physical phase flow', (3,)).copy()
    if not np.allclose(flow, phase['flow'], rtol=1e-12, atol=1e-14):
        raise ValueError('Saved dense phase flow differs from physical RHS')
    znorm = float(np.linalg.norm(z))
    if not np.isfinite(znorm) or znorm == 0:
        raise ValueError('Nonzero finite phase gradient required')
    unit = z/znorm
    projector = np.eye(3)-np.outer(unit, unit)
    coordinate = int(np.argmax(np.linalg.norm(projector, axis=0)))
    direction = projector[:, coordinate]/np.linalg.norm(projector[:, coordinate])
    verification = pair['verifications'][CANONICAL_VERIFICATION]
    if verification.get('complete') is not True or verification.get('qualified') is not True or verification['certificate'].get('qualified') is not True:
        raise ValueError('Qualified canonical verification certificate required')
    old_context._same(verification['candidate_method'], PRIMARY, 'certificate candidate')
    old_context._same(verification['method'], PRIMARY, 'certificate verifier')
    multipliers, rho = old_context._multipliers(verification['certificate'])
    context = dict(id=phase['context_id'], setting=setting, phase_index=index,
        split=phase['split'], alpha=alpha, gamma=gamma, period=period,
        phase_fraction=phase['phase_fraction'], phase_epoch=phase['phase_epoch'],
        initial=state, phase_gradient=z, flow=flow, direction=direction,
        nominal_event_phase_gradient=z.copy(), null_coordinate_index=coordinate,
        normalization_residual=float(z@flow-1.), direction_phase_leakage=float(z@direction),
        transverse_multipliers=multipliers, rho=rho,
        rho_provenance='signed dominant real transverse multiplier of the new canonical DOP853 candidate/DOP853 verification certificate; null if unavailable/nonreal',
        context_schema='matched-cycle-two-write-context-v1', new_normalizations=0,
        phase_cohort='eight new common PCG64 fractions; historical cycle slots, not fresh networks',
        provenance=dict(initial='new canonical DOP853 exact solver-dense phase state',
            adjoint='new canonical DOP853 exact dense adjoint, unchanged after its one global construction normalization',
            event_adjoint='same unchanged phase adjoint at nominal phase_epoch+T; approximate periodic reference',
            projection_flow='physical RHS of independently integrated fresh8T primary reference event state',
            reference_qualified=True, reference_status='complete', source_binding=provenance_binding,
            canonical_source_verification=CANONICAL_VERIFICATION, canonical_base_id=base['id'],
            source_global_normalization_count=1, new_normalizations=0,
            both_fresh_reference_methods_use_identical_initial_state=True,
            eight_period_reference_qualified=False))
    context.update(cycle_id=setting,source_setting=pair['source_setting'],role=pair['role'],seed_provenance=copy.deepcopy(pair['seed_provenance']))
    tangent._context(context)
    return context


def acquire_tangent(context,method='dop853',*,rhs=hopfield_rhs,jacobian=hopfield_jacobian,solver=solve_ivp):
    """Two direct12D segments with an explicit fixed two-IVP ledger.

    The autonomous physical model uses elapsed clocks0..T andT..8T. The
    phase_epoch separately identifies the saved reference start. A fresh
    identity tangent starts at each segment; no repeated-period composition.
    """
    period=tangent._context(context);cycle_id=context.get('cycle_id')
    if cycle_id not in CYCLE_IDS or context.get('setting')!=cycle_id:
        raise ValueError('Truthful six-cycle context identity required')
    i=context.get('phase_index')
    if type(i) is not int or i not in range(PHASE_COUNT) or context.get('id')!=f'{cycle_id}--p{i:02d}' or context.get('split')!='fresh':
        raise ValueError('Exact fresh context identity required')
    if method not in TANGENT_METHODS:raise ValueError('Declared tangent algorithm required')
    fraction=prior._scalar(context['phase_fraction'],'phase fraction')
    if not 0<=fraction<1 or context['phase_epoch']!=fraction*period:raise ValueError('Exact phase epoch required')
    if context.get('new_normalizations')!=0:raise ValueError('Per-phase gradient normalization prohibited')
    for key in ('alpha','gamma'):prior._scalar(context[key],key)
    grid=tangent.build_grid(period);declared=TANGENT_METHODS[method]
    spec=dict(declared,id=method,max_step=period/declared['max_step_period_divisor'])
    ident=context['id']+'--reference--'+method
    packet=dict(id=ident,kind='tangent',cycle_id=cycle_id,setting=cycle_id,
        source_setting=context['source_setting'],role=context['role'],
        seed_provenance=copy.deepcopy(context['seed_provenance']),
        context_id=context['id'],phase_index=i,phase_fraction=fraction,split='fresh',phase_epoch=context['phase_epoch'],
        method=method,method_spec=spec,status='started',complete=False,period=period,event_time=period,
        alpha=context['alpha'],gamma=context['gamma'],
        provenance='Direct unforced state and full tangent; independent identity atT; no finite writes',
        ivp_slots={stage:dict(id=ident+'--'+stage,stage=stage,status='blocked_prerequisite',
            ivp_started=False,ivp_returned=False,ivp_completed=False) for stage in ('pre','post')})
    stage='pre'
    try:
        for stage,initial,offset in (('pre',context['initial'],0.),('post',None,period)):
            row=packet['ivp_slots'][stage];row['status']='started'
            if stage=='post':initial=packet['pre']['states'][-1]
            def tracked_solver(fun,span,start,**kwargs):
                row['ivp_started']=True
                result=solver(fun,span,start,**kwargs);row['ivp_returned']=True
                y=np.asarray(result.y);t=np.asarray(result.t)
                row['ivp_completed']=bool(result.success and np.array_equal(t,kwargs['t_eval'])
                    and y.shape==(len(start),len(t)) and np.isfinite(y).all() and np.array_equal(y[:,0],start))
                return result
            packet[stage]=tangent._tangent_segment(context,initial,grid[stage],offset,spec,rhs,jacobian,tracked_solver)
            row.update(status='complete',complete=True)
        packet.update(status='complete',complete=True)
        return packet
    except Exception as error:
        packet.update(status='failed',complete=False,failed_stage=stage,error=repr(error))
        packet['ivp_slots'][stage].update(status='failed',complete=False,error=repr(error))
        if hasattr(error,'payload'):packet['partial_failure']=error.payload
        raise AcquisitionFailure('Matched-cycle tangent acquisition failed',packet) from error


def seal_context(context,primary_tangent,*,rhs=hopfield_rhs):
    """Use the frozen physical event projector without phase-gradient repair."""
    return tangent.seal_context(context,primary_tangent,rhs=rhs)


def job_census():
    jobs=[]
    for cycle_id in CYCLE_IDS:
        for method in METHODS:jobs.append(dict(id=cycle_id+'--'+method,kind='shoot',cycle_id=cycle_id,method=method))
    for cycle_id in CYCLE_IDS:jobs.append(dict(id=cycle_id+'--reference-pair',kind='reference',cycle_id=cycle_id))
    for cycle_id in CYCLE_IDS:
        for i in range(PHASE_COUNT):
            for method in TANGENT_METHODS:
                jobs.append(dict(id=f'{cycle_id}--p{i:02d}--reference--{method}',kind='tangent',cycle_id=cycle_id,phase_index=i,method=method))
    return jobs


def ivp_census():
    rows=[]
    for cycle_id in CYCLE_IDS:
        for method in METHODS:
            rows.extend(dict(id=f'{cycle_id}--{method}--trial-{i:02d}',kind='shoot_trial',cycle_id=cycle_id,method=method,index=i) for i in range(25))
        for candidate in METHODS:
            for method in METHODS:rows.append(dict(id=f'{cycle_id}--verify--{candidate}--{method}',kind='verification',cycle_id=cycle_id,candidate_method=candidate,method=method))
        for method in METHODS:rows.append(dict(id=f'{cycle_id}--adjoint--{method}',kind='adjoint',cycle_id=cycle_id,method=method))
        for i in range(PHASE_COUNT):
            for method in TANGENT_METHODS:
                for stage in ('pre','post'):rows.append(dict(id=f'{cycle_id}--p{i:02d}--reference--{method}--{stage}',kind='tangent_segment',cycle_id=cycle_id,phase_index=i,method=method,stage=stage))
    return rows
