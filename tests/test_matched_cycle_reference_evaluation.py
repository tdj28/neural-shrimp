"""Synthetic saved-array matching, failure census and phase-reference checks."""
from pathlib import Path
import copy
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import matched_cycle_reference_evaluation as e
import matched_cycle_reference_acquisition as a
import run_phase_memory_recovery as codec


def slots():
    result=[]
    for i,ident in enumerate(e.SLOT_IDS):
        anchor=ident.startswith('anchor-');source=ident.removeprefix('anchor-') if anchor else ident.rsplit('-rho',1)[0]
        result.append(dict(id=ident,cycle_id=ident,source_setting=source,role='anchor' if anchor else 'candidate',
            target_rho=None if anchor else .25 if ident.endswith('025') else .5,
            branch_id=None if anchor else source+'-direction-+1',node_index=None if anchor else i,
            node_id=None if anchor else 'synthetic-'+str(i),available=True,status='selected',reasons=[],
            alpha=float(i),gamma=0.,period=2*np.pi,first_maximum_state=np.array([1.,0.,1.]),
            seed_provenance={'synthetic':True,'slot':ident}))
    return result


def loop(shift=0.,period=2*np.pi):
    t=np.linspace(0.,2*np.pi,8193)+shift
    return dict(period=period,states=np.column_stack([np.cos(t),np.sin(t),np.cos(9*t)]).astype(np.float64),
        derivatives=np.column_stack([-np.sin(t),np.cos(t),-9*np.sin(9*t)]).astype(np.float64)*(2*np.pi/period))


def cycle(slot):
    target=slot['target_rho'];rho=.25 if target is None else target
    return dict(id=slot['id'],slot=copy.deepcopy(slot),alpha=slot['alpha'],gamma=slot['gamma'],
        complete=True,qualified=True,loop=loop(),
        nested_slots=e._cycle_nested(slot['id'],{},None),
        descriptors=dict(period=2*np.pi,target_rho=target,rho_bin_pass=True,
            mode_resolution=e.mode_resolution(np.diag([1.,rho,.01]),np.diag([1.,rho,.01]),2*np.pi)))


def context_rows():
    result={}
    for s in slots():
        for p in e.phase_census():
            ident=s['id']+f"--p{p['phase_index']:02d}"
            result[ident]=dict(id=ident,context_id=ident,cycle_id=s['id'],**p,complete=True,qualified=True,
                descriptors=dict(immediate_detectability=1.+p['phase_index']/10),
                nested_slots=[dict(id=f'{ident}--reference--{m}--{seg}',kind='tangent_segment',method=m,segment=seg,status='complete',attempted=True,complete=True,raw_available=True)
                    for m in e.TANGENT_METHODS for seg in ('pre','post')])
    return result


def test_exact_prospective_contract():
    c=e.contract();assert c['jobs']['total']==114 and c['ivp_slots']['total']==528
    assert c['finite_writes']==0 and not c['mode_resolution']['rigorous_interval']
    rows=e.phase_census();assert len(rows)==8
    np.testing.assert_array_equal([p['phase_fraction'] for p in rows],np.random.Generator(np.random.PCG64(2026091702)).random(8))
    assert all(p['split']=='fresh' for p in rows)
    assert c['rho_bin']['arithmetic']=='exact rational comparison of stored binary64 values and exact 1/40'
    assert c['modal_orientation']['qualification_gate'] is False


def test_tiny_secondary_mode_has_no_invented_log_rate():
    r=e.mode_resolution(np.diag([1.,.25,1e-30]),np.diag([1.,.25,0.]),10.)
    assert r['dominant_positive_resolved'] and r['dominant_signed_value']==.25
    tiny=min(r['rows'],key=lambda x:x['primary']['modulus'])
    assert not tiny['nonzero_resolved'] and tiny['log_modulus_per_model_time'] is None


@pytest.mark.parametrize('matrix',[
    np.diag([1.,-.25,.01]),np.diag([1.,.25,.25]),
    np.array([[1.,0.,0.],[0.,.4,-.02],[0.,.02,.4]])])
def test_negative_complex_or_unseparated_dominant_not_positive_resolved(matrix):
    r=e.mode_resolution(matrix,matrix,3.)
    assert not r['dominant_positive_resolved'] and r['dominant_signed_value'] is None


def test_method_matching_and_rate_are_descriptive():
    r=e.mode_resolution(np.diag([1.,.5,.1]),np.diag([.1,1.,.5000001]),4.)
    assert r['dominant_positive_resolved']
    d=r['rows'][r['dominant_index']]
    assert d['method_difference']==pytest.approx(1e-7)
    assert d['decay_rate_per_model_time']==pytest.approx(-np.log(.5)/4)


def test_exact_rational_bin_boundary_does_not_round_caliper():
    modes=e.mode_resolution(np.diag([1.,.25,.1]),np.diag([1.,.25,.1]),1.)
    d=modes['rows'][modes['dominant_index']];d['resolution_radius']=0.
    # Binary64 .275 lies above the exact rational upper edge 11/40.
    for key in ('primary','alternate'):d[key]['real']=.275
    assert not e.rho_bin_containment(modes,.25)['passed']
    for key in ('primary','alternate'):d[key]['real']=float(np.nextafter(.275,0.))
    assert e.rho_bin_containment(modes,.25)['passed']
    d['resolution_radius']=1e-8
    assert not e.rho_bin_containment(modes,.25)['passed']


@pytest.mark.parametrize('M',[
    np.array([[1.,.2,.3],[0.,.5,.2],[0.,0.,.1]]),
    np.array([[1.,0.,0.],[0.,.4,-.2],[0.,.2,.4]]),np.eye(3)])
def test_descriptive_modal_coordinates_reconstruct_nonnormal_complex_and_repeated_modes(M):
    delta=np.array([1e-4,-2e-4,3e-4]);basis=e._modal_basis(M,delta)
    assert basis['available'] and not basis['qualification_gate']
    V=basis['right_vectors']['real']+1j*basis['right_vectors']['imag']
    np.testing.assert_allclose(np.linalg.norm(V,axis=0),1.,rtol=1e-14,atol=1e-14)
    for j,p in enumerate(basis['pivot_coordinates']):
        assert V[p,j].real>0 and abs(V[p,j].imag)<1e-15
    operator=np.arange(288,dtype=np.float64).reshape(96,3)/288
    row=e._modal_readout(basis,operator,operator@delta)
    assert row['available'] and row['reconstruction_max']<1e-17
    assert row['reconstruction_imaginary_max']<1e-17
    assert basis['eigen_residual_max']<1e-14


def test_modal_failure_is_diagnostic_unavailable_not_a_new_gate():
    basis=e._modal_basis(np.full((3,3),np.nan),np.ones(3))
    assert not basis['available'] and not basis['qualification_gate']
    row=e._modal_readout(basis,np.zeros((96,3)),np.zeros(96))
    assert not row['available'] and not row['qualification_gate']


def test_modal_overflow_retains_diagnostic_unavailability():
    basis=e._modal_basis(np.eye(3),np.ones(3))
    basis['write_coefficients']['real']*=1e200
    row=e._modal_readout(basis,np.ones((96,3))*1e200,np.zeros(96))
    assert not row['available'] and not row['qualification_gate'] and 'error' in row


def test_phase_shifted_same_parameter_loop_is_duplicate():
    left=cycle(slots()[0]);right=copy.deepcopy(left);right['id']='other';right['loop']=loop(.13749013)
    r=e.duplicate_comparison(left,right)
    assert r['complete'] and r['duplicate'] and not r['distinctness_pass']
    assert r['fine']['scaled_rms']<1e-8 and r['fine']['golden_steps']==64
    assert abs(((r['fine']['shift_fraction']+.13749013/(2*np.pi)+.5)%1)-.5)<1e-8


def test_cross_parameter_near_loop_is_warning_only():
    left=cycle(slots()[0]);right=cycle(slots()[1]);right['loop']=loop(2*np.pi/9)
    r=e.duplicate_comparison(left,right)
    assert r['complete'] and r['near_duplicate'] and not r['duplicate'] and r['distinctness_pass']


def test_alignment_does_not_reverse_time():
    left=cycle(slots()[0]);right=copy.deepcopy(left);right['id']='reverse'
    right['loop']['states']=right['loop']['states'][::-1].copy()
    right['loop']['derivatives']=-right['loop']['derivatives'][::-1].copy()
    r=e.duplicate_comparison(left,right)
    assert r['complete'] and not r['duplicate'] and r['fine']['scaled_rms']>.1


@pytest.mark.parametrize('same',[True,False])
def test_unresolved_duplicate_is_not_silently_distinct_at_same_parameters(same):
    left=cycle(slots()[0]);right=cycle(slots()[1]);right['loop']['states']=np.zeros((8,3))
    if same:right['alpha']=left['alpha']
    r=e.duplicate_comparison(left,right)
    assert not r['complete'] and r['distinctness_pass'] is (not same)


def test_wholly_missing_outputs_keep_every_denominator():
    r=e.summarize(slots(),{},{});c=r['counts']
    assert not r['complete'] and not r['feasible']
    assert len(r['cycles'])==6 and len(r['contexts'])==48 and len(r['pairs'])==2 and len(r['duplicate_comparisons'])==15
    assert c['nested_slots_retained']==528 and len({x['id'] for x in r['nested_slots']})==528


def test_full_summary_all_contexts_and_fixed_matching():
    s=slots();r=e.summarize(s,{x['id']:cycle(x) for x in s},context_rows())
    assert r['complete'] and r['numerical_qualified'] and r['feasible']
    assert r['counts']['matching_pairs_pass']==2 and r['counts']['contexts_qualified']==48


def test_missing_phase_never_becomes_complete_case_matching():
    s=slots();p=context_rows();del p['core-rho025--p07']
    r=e.summarize(s,{x['id']:cycle(x) for x in s},p)
    assert not r['complete'] and not r['feasible'] and len(r['contexts'])==48
    assert not r['pairs'][0]['complete'] and r['pairs'][1]['matched']


@pytest.mark.parametrize('period_ratio,signal,passed',[(1.,1.,True),(1.03,1.,False),(1.,1.2,False),(1.,0.,False)])
def test_calipers_and_zero_detectability(period_ratio,signal,passed):
    s=slots();left=cycle(s[2]);right=cycle(s[4]);right['descriptors']['period']*=period_ratio;p=context_rows()
    for k,v in p.items():
        if k.startswith(right['id']):v['descriptors']['immediate_detectability']*=signal
        if signal==0 and k.startswith(left['id']):v['descriptors']['immediate_detectability']=0.
    r=e.match_pair(.25,left,right,p)
    assert r['complete'] and r['matched'] is passed


def flow(t,x,a,g):return np.array([1.,0.,0.])
def jac(t,x,a,g):return np.zeros((3,3))
def analytic(fun,span,initial,*,t_eval,**kwargs):
    values=np.repeat(initial[:,None],len(t_eval),axis=1);values[0]+=t_eval
    return SimpleNamespace(t=t_eval.copy(),y=values,success=True,message='synthetic constant drift',nfev=0,njev=0,nlu=0)


@pytest.fixture(scope='module')
def reference():
    slot=slots()[0];p=e.phase_census()[0];T=1.;ident=slot['id']+'--p00'
    ctx=dict(id=ident,cycle_id=slot['id'],setting=slot['id'],source_setting=slot['source_setting'],role=slot['role'],seed_provenance=slot['seed_provenance'],
        **p,alpha=slot['alpha'],gamma=slot['gamma'],period=T,phase_epoch=p['phase_fraction']*T,
        initial=np.zeros(3),flow=np.array([1.,0.,0.]),phase_gradient=np.array([1.,0.,0.]),
        nominal_event_phase_gradient=np.array([1.,0.,0.]),direction=np.array([0.,1.,0.]),rho=.25,new_normalizations=0,
        provenance=dict(reference_qualified=True,reference_status='complete'))
    primary=a.acquire_tangent(ctx,'dop853',rhs=flow,jacobian=jac,solver=analytic)
    alternate=a.acquire_tangent(ctx,'rk45',rhs=flow,jacobian=jac,solver=analytic)
    return slot,ctx,primary,alternate


def test_actual_new_tangent_api_and_saved_codec(reference,tmp_path):
    slot,ctx,p,q=reference
    receipt=codec.save_packet(tmp_path,'synthetic-reference',dict(context=ctx,primary=p,alternate=q))
    loaded=codec.load_packet(receipt['path'])
    r=e.seal_context(slot,loaded['context'],loaded['primary'],loaded['alternate'],rhs=flow,jacobian=jac)
    assert r['complete'] and r['qualified'],r.get('error')
    assert r['descriptors']['immediate_detectability']==pytest.approx(np.sqrt(32))
    w=r['descriptors']['windows']['k4']
    np.testing.assert_allclose(w['A'],w['B'],rtol=0,atol=0)
    assert w['gram'].shape==(2,2) and w['observable_first_operator'].shape==(96,3)
    assert r['descriptors']['modal_bases']['first_write']['available']
    assert w['modal_first']['reconstruction_max']<1e-17 and w['modal_second']['reconstruction_max']<1e-17
    assert r['descriptors']['windows']['immediate']['modal_second']['reconstruction_max']==0.
    assert len(r['nested_slots'])==4 and all(x['complete'] for x in r['nested_slots'])
    saved=codec.save_packet(tmp_path,'synthetic-seal',r);reloaded=codec.load_packet(saved['path'])
    np.testing.assert_array_equal(reloaded['descriptors']['modal_bases']['first_write']['right_vectors']['real'],
        r['descriptors']['modal_bases']['first_write']['right_vectors']['real'])


@pytest.mark.parametrize('edit',[
    lambda c,p,q:q['pre']['derivatives'].__setitem__((5,1),1.),
    lambda c,p,q:q['post']['phi_derivatives'].__setitem__((8,0,0),1.),
    lambda c,p,q:q.update(id='wrong'),
    lambda c,p,q:q['ivp_slots']['post'].update(ivp_completed=False),
    lambda c,p,q:c.update(split='heldout'),
    lambda c,p,q:c.update(phase_index=True),
    lambda c,p,q:q['pre']['solver'].update(nfev=True),
    lambda c,p,q:q.update(phase_index=True),
    lambda c,p,q:q.update(phase_fraction=float(np.nextafter(q['phase_fraction'],1.))),
    lambda c,p,q:q.update(phase_epoch=q['phase_epoch']+1.),
    lambda c,p,q:q.update(split='heldout'),
    lambda c,p,q:q.update(kind='trace'),
])
def test_tangent_tampering_retained_as_failed_reference(reference,edit):
    slot,c,p,q=copy.deepcopy(reference);edit(c,p,q)
    r=e.seal_context(slot,c,p,q,rhs=flow,jacobian=jac)
    assert not r['complete'] and not r['qualified'] and 'error' in r and len(r['nested_slots'])==4


def test_missing_cycle_seed_retains56_ivp_slots():
    r=e.evaluate_cycle(slots()[0],{},None,None)
    assert not r['qualified'] and len(r['shoots'])==2 and len(r['nested_slots'])==56


def test_failed_tangent_prefix_replays_completed_pre_without_qualification(reference):
    slot,c,p,q=copy.deepcopy(reference);del p['post'];p.update(complete=False,status='failed')
    p['ivp_slots']['post'].update(status='failed',ivp_started=True,ivp_returned=False,ivp_completed=False)
    r=e.validate_tangent_packet(p,c,'dop853',allow_partial=True,rhs=flow,jacobian=jac)
    assert r['partial_replay_pass'] and not r['complete'] and not r['qualified']
    assert r['checks'][0]['physical_replay'] and not r['checks'][1]['physical_replay']
    with pytest.raises(ValueError):e.validate_tangent_packet(p,c,'dop853',rhs=flow,jacobian=jac)
    p['pre']['phi_derivatives'][3,0,1]=1.
    with pytest.raises(ValueError):e.validate_tangent_packet(p,c,'dop853',allow_partial=True,rhs=flow,jacobian=jac)


@pytest.mark.parametrize('mutation',[
    lambda p:p[1].update(phase_index=True),lambda p:p[0].update(split='development'),
    lambda p:p[0].update(phase_fraction=np.nextafter(p[0]['phase_fraction'],1.))])
def test_exact_phase_metadata_never_uses_arithmetic_tolerance(mutation):
    p=e.phase_census();mutation(p)
    with pytest.raises(ValueError):e._phase_rows(p)


def test_nested_slot_renaming_cannot_preserve_false_complete_census():
    s=slots();c={x['id']:cycle(x) for x in s};c[s[0]['id']]['nested_slots'][0]['id']='invented-but-unique'
    with pytest.raises(ValueError):e.summarize(s,c,context_rows())


def test_outer_parameter_tampering_cannot_bypass_same_parameter_duplicate():
    s=slots()
    for slot in s:slot['alpha']=0.
    c={x['id']:cycle(x) for x in s}
    assert not e.summarize(s,c,context_rows())['feasible']
    for i,row in enumerate(c.values()):row['alpha']=float(i)
    with pytest.raises(ValueError,match='parameter alpha'):e.summarize(s,c,context_rows())


def test_qualified_cycle_requires_complete_computation():
    s=slots();c={x['id']:cycle(x) for x in s};c[s[0]['id']]['complete']=False
    with pytest.raises(ValueError,match='qualified without complete'):e.summarize(s,c,context_rows())
