"""Pure context adaptation tested with a genuine synthetic nine-peak cycle."""
from pathlib import Path
import copy
import hashlib
import importlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import refined_two_write_context as adapter
import periodic_reference_refinement as refine
import two_write_acquisition as acquire
import two_write_evaluation as evaluate
import test_periodic_reference_refinement_integration as oscillator


def binding(setting='core'):
    return dict(experiment_id='EXP-018', source_commit='a'*40, plan_sha256='b'*64,
        receipt_sha256='c'*64, reference_id=setting+'--reference-pair',
        packet_sha256='d'*64, packet_semantic_sha256='e'*64)


@pytest.fixture(scope='module')
def data():
    old=oscillator.reference()
    rows=oscillator.phase_rows()
    fractions=[r['phase_fraction'] for r in rows][::-1]
    rows=[dict(r,phase_fraction=f) for r,f in zip(rows,fractions,strict=True)]
    shots={m:refine.refine_cycle(old,m,rhs=oscillator.rhs,jacobian=oscillator.jacobian,
        solver=oscillator.exact_solver) for m in refine.METHODS}
    pair=refine.crossverify_reference(old,shots,rows,rhs=oscillator.rhs,
        jacobian=oscillator.jacobian,solver=oscillator.exact_solver)
    assert pair['complete'] and pair['qualified']
    return pair,rows


def context(data,index=3):
    pair,rows=data
    return adapter.prepare_context(pair,rows[index],source_binding=binding(),rhs=oscillator.rhs)


def test_exact_canonical_dense_context_has_no_historical_template(data,monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('Hermite reference unexpectedly used')
    monkeypatch.setattr(acquire,'reference_templates',forbidden)
    pair,rows=data;result=context(data,17);phase=pair['bases']['tight-dop853']['phases'][17]
    assert result['id']=='core--p17' and result['split']=='heldout'
    assert result['phase_cohort'].endswith('no fresh holdout')
    assert result['phase_fraction']==rows[17]['phase_fraction']
    assert result['phase_epoch']==rows[17]['phase_fraction']*pair['canonical']['period']
    for key,source in [('initial','state'),('phase_gradient','z'),('nominal_event_phase_gradient','z'),('flow','flow')]:
        np.testing.assert_array_equal(result[key],phase[source])
    assert result['new_normalizations']==0 and 'template_velocity' not in result
    assert result['provenance']['reference_qualified']
    assert result['provenance']['eight_period_reference_qualified'] is False
    assert result['provenance']['source_binding']==binding()


def test_adapter_preserves_unsorted_phase_order(data):
    pair,rows=data
    assert rows[0]['phase_fraction']>rows[-1]['phase_fraction']
    all_contexts=[context(data,i) for i in range(24)]
    assert [c['phase_index'] for c in all_contexts]==list(range(24))
    assert [c['phase_fraction'] for c in all_contexts]==[r['phase_fraction'] for r in rows]
    for c in all_contexts:
        np.testing.assert_array_equal(c['initial'],pair['bases']['tight-dop853']['exact_phase_arrays']['states'][c['phase_index']])


def test_context_copies_input_arrays_and_provenance(data):
    pair,rows=data;source=binding()
    before=pair['bases']['tight-dop853']['phases'][2]['z'].copy()
    result=adapter.prepare_context(pair,rows[2],source_binding=source,rhs=oscillator.rhs)
    result['phase_gradient'][:]=123.
    result['nominal_event_phase_gradient'][:]=456.
    result['initial'][:]=789.
    result['provenance']['source_binding']['packet_sha256']='f'*64
    np.testing.assert_array_equal(pair['bases']['tight-dop853']['phases'][2]['z'],before)
    assert source==binding()


def test_phase_gradient_is_never_normalized_per_context(data):
    pair,rows=data;changed=copy.deepcopy(pair);base=changed['bases']['tight-dop853'];i=4
    base['phases'][i]['z']*=1.01;base['exact_phase_arrays']['z'][i]*=1.01
    result=adapter.prepare_context(changed,rows[i],source_binding=binding(),rhs=oscillator.rhs)
    np.testing.assert_array_equal(result['phase_gradient'],base['phases'][i]['z'])
    assert result['normalization_residual']==pytest.approx(.01,abs=2e-12)
    assert result['new_normalizations']==0


def test_direction_first_coordinate_tie_rule_and_phase_null(data):
    pair,rows=data;changed=copy.deepcopy(pair);base=changed['bases']['tight-dop853'];i=0
    z=np.array([0.,0.,2.]);base['phases'][i]['z']=z.copy();base['exact_phase_arrays']['z'][i]=z
    result=adapter.prepare_context(changed,rows[i],source_binding=binding(),rhs=oscillator.rhs)
    assert result['null_coordinate_index']==0
    np.testing.assert_array_equal(result['direction'],np.array([1.,0.,0.]))
    assert result['direction_phase_leakage']==0.
    np.testing.assert_array_equal(result['phase_gradient'],z)


def test_rho_comes_only_from_canonical_dop_candidate_dop_verifier(data):
    pair,rows=data;changed=copy.deepcopy(pair)
    changed['verifications'][adapter.CANONICAL_VERIFICATION]['certificate']['transverse_multipliers']=[
        dict(real=-.8,imag=0.,modulus=.8),dict(real=.2,imag=0.,modulus=.2)]
    changed['verifications']['tight-rk45--tight-rk45']['certificate']['transverse_multipliers']=[
        dict(real=.9,imag=0.,modulus=.9),dict(real=.1,imag=0.,modulus=.1)]
    result=adapter.prepare_context(changed,rows[0],source_binding=binding(),rhs=oscillator.rhs)
    assert result['rho']==-.8
    assert 'new canonical' in result['rho_provenance']


@pytest.mark.parametrize('multipliers',[[],[dict(real=.3,imag=.4,modulus=.5),dict(real=.3,imag=-.4,modulus=.5)]])
def test_missing_or_complex_dominant_mode_stays_unavailable(data,multipliers):
    pair,rows=data;changed=copy.deepcopy(pair)
    changed['verifications'][adapter.CANONICAL_VERIFICATION]['certificate']['transverse_multipliers']=multipliers
    result=adapter.prepare_context(changed,rows[0],source_binding=binding(),rhs=oscillator.rhs)
    assert result['rho'] is None and result['transverse_multipliers']==multipliers


@pytest.mark.parametrize('field,value', [('complete',False),('qualified',False),('status','failed'),
    ('schema','old-reference'),('canonical_method','tight-rk45')])
def test_unqualified_or_wrong_reference_recipe_rejected(data,field,value):
    pair,rows=data;changed=copy.deepcopy(pair);changed[field]=value
    with pytest.raises(ValueError):
        adapter.prepare_context(changed,rows[0],source_binding=binding(),rhs=oscillator.rhs)


@pytest.mark.parametrize('change',['state','z','flow','phase_fraction','phase_epoch','split','index'])
def test_exact_phase_identity_corruption_rejected(data,change):
    pair,rows=data;changed=copy.deepcopy(pair);phase=changed['bases']['tight-dop853']['phases'][5]
    if change in ('state','z','flow'):phase[change][0]+=1e-8
    elif change=='phase_fraction':phase[change]+=1e-8
    elif change=='phase_epoch':phase[change]+=1e-8
    elif change=='split':phase[change]='heldout'
    else:phase['phase_index']=6
    with pytest.raises(ValueError):
        adapter.prepare_context(changed,rows[5],source_binding=binding(),rhs=oscillator.rhs)


def test_saved_flow_is_recomputed_from_physical_rhs(data):
    pair,rows=data
    def wrong(t,x,a,g):return oscillator.rhs(t,x,a,g)+1e-6
    with pytest.raises(ValueError,match='physical RHS'):
        adapter.prepare_context(pair,rows[0],source_binding=binding(),rhs=wrong)


@pytest.mark.parametrize('change',['receipt_sha256','source_commit','experiment_id','reference_id','extra','missing'])
def test_source_binding_schema_and_identity_are_required(data,change):
    pair,rows=data;source=binding()
    if change=='extra':source['extra']=True
    elif change=='missing':source.pop('packet_semantic_sha256')
    else:source[change]='wrong'
    with pytest.raises(ValueError):
        adapter.prepare_context(pair,rows[0],source_binding=source,rhs=oscillator.rhs)


def test_both_reconstructions_share_the_same_canonical_terminal(data):
    pair,rows=data;changed=copy.deepcopy(pair)
    changed['bases']['tight-rk45']['global_terminal_adjoint'][0]+=1e-8
    with pytest.raises(ValueError,match='shared terminal'):
        adapter.prepare_context(changed,rows[0],source_binding=binding(),rhs=oscillator.rhs)


def test_real_frozen_eight_period_reference_api_uses_common_new_initial_state(data):
    result=context(data,9);first_states=[]
    def logged(fun,span,initial,**kwargs):
        if span[0]==0. and kwargs['t_eval'][-1]==result['period']:
            first_states.append(initial[:3].copy())
        return oscillator.exact_solver(fun,span,initial,**kwargs)
    references={m:acquire.acquire_tangent(result,m,rhs=oscillator.rhs,
        jacobian=oscillator.jacobian,solver=logged) for m in acquire.TANGENT_METHODS}
    assert len(first_states)==2
    for state in first_states:np.testing.assert_array_equal(state,result['initial'])
    sealed_context=acquire.seal_context(result,references['dop853'],rhs=oscillator.rhs)
    sealed=evaluate.seal_reference(sealed_context,references['dop853'],references['rk45'])
    assert sealed['complete'] and sealed['qualified']
    assert set(sealed['windows'])=={'immediate','k2','k4','k8'}
    np.testing.assert_array_equal(sealed_context['event_phase_gradient'],result['phase_gradient'])
    assert all(sealed['windows'][w]['derivative_observer']['available'] for w in sealed['windows'])


def test_import_performs_no_solve_or_historical_template_sampling(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('Unexpected import work')
    monkeypatch.setattr(acquire,'prepare_context',forbidden)
    monkeypatch.setattr(acquire,'acquire_tangent',forbidden)
    monkeypatch.setattr(acquire,'acquire_trace',forbidden)
    monkeypatch.setattr(acquire,'reference_templates',forbidden)
    importlib.reload(adapter)
