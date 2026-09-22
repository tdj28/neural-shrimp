"""Independent closed-flow end-to-end reference checks for EXP022.

No historical target inputs are read. A known nine-maximum oscillator crosses
all real packet boundaries. Direct analytic state, adjoint and observable
responses are separate from the production tangent composition.
"""
from pathlib import Path
import copy
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import matched_cycle_reference_acquisition as acq
import matched_cycle_reference_evaluation as evaluate
import run_phase_memory_recovery as codec
import test_periodic_reference_refinement_integration as oscillator


def slot():
    return dict(id='anchor-core',cycle_id='anchor-core',role='anchor',source_setting='core',
        target_rho=None,branch_id=None,node_index=None,node_id=None,available=True,status='selected',
        reasons=[],alpha=0.,gamma=0.,period=float(oscillator.PERIOD),
        first_maximum_state=oscillator.SEED.tolist(),
        seed_provenance=dict(experiment='synthetic-analytic-oscillator',source_sha256='d'*64))


def roundtrip(path,name,value):
    record=codec.save_packet(path,name,value)
    restored=codec.load_packet(record['path'])
    assert codec.packet_digest(restored)==codec.packet_digest(value)
    return record,restored


@pytest.fixture(scope='module')
def pipeline(tmp_path_factory):
    out=tmp_path_factory.mktemp('matched-cycle-reference-analytic')
    selected=slot();phases=acq.phase_rows();seed=acq.prepare_seed(selected)
    _,seed=roundtrip(out,'seed',seed)
    shots={m:acq.refine_cycle(seed,m,rhs=oscillator.rhs,jacobian=oscillator.jacobian,
                             solver=oscillator.exact_solver) for m in acq.METHODS}
    for m in shots:_,shots[m]=roundtrip(out,'shoot-'+m,shots[m])
    pair=acq.crossverify_reference(seed,shots,phases,rhs=oscillator.rhs,
                                  jacobian=oscillator.jacobian,solver=oscillator.exact_solver)
    record,pair=roundtrip(out,'pair',pair)
    cycle=evaluate.evaluate_cycle(selected,shots,pair,seed,phases,rhs=oscillator.rhs,jacobian=oscillator.jacobian)
    source=dict(experiment_id='EXP-022',source_commit='a'*40,plan_sha256='b'*64,
        reference_id=pair['id'],packet_sha256=record['sha256'],packet_semantic_sha256=record['semantic_sha256'])
    contexts={};calls=[]
    for i in [0,7]:
        context=acq.prepare_context(pair,phases[i],source_binding=source,rhs=oscillator.rhs)
        _,context=roundtrip(out,'context-'+str(i),context)
        def solver(fun,span,initial,**kw):
            calls.append(dict(phase_index=i,span=tuple(span),method=kw['method'],initial=initial.copy(),samples=len(kw['t_eval'])))
            return oscillator.exact_solver(fun,span,initial,**kw)
        refs={m:acq.acquire_tangent(context,m,rhs=oscillator.rhs,jacobian=oscillator.jacobian,solver=solver) for m in acq.TANGENT_METHODS}
        for m in refs:_,refs[m]=roundtrip(out,'transport-'+str(i)+'-'+m,refs[m])
        physical=acq.seal_context(context,refs['dop853'],rhs=oscillator.rhs)
        evaluated=evaluate.seal_context(selected,physical,refs['dop853'],refs['rk45'],rhs=oscillator.rhs,jacobian=oscillator.jacobian)
        _,evaluated=roundtrip(out,'evaluated-'+str(i),evaluated)
        contexts[i]=dict(context=context,physical=physical,references=refs,evaluated=evaluated)
    return dict(directory=out,slot=selected,seed=seed,phases=phases,shots=shots,pair=pair,cycle=cycle,
                source=source,contexts=contexts,calls=calls)


def test_synthetic_reference_remains_admitted_until_new_qualification(pipeline):
    p=pipeline
    assert p['seed']['admitted_historical_seed'] is True
    assert p['seed']['qualified'] is False and p['seed']['new_reference_qualified'] is False
    assert p['pair']['complete'] and p['pair']['qualified']
    assert p['cycle']['complete'] and p['cycle']['qualified'],p['cycle']
    assert p['pair']['canonical']['normalization_count']==1
    for base in p['pair']['bases'].values():
        assert len(base['phases'])==len(base['phase_rows'])==8
        assert base['new_normalizations']==0
        assert all(x['split']=='fresh' for x in base['phases'])


@pytest.mark.parametrize('index',[0,7])
def test_analytic_phase_state_and_gradient_cross_real_packet_interfaces(pipeline,index):
    context=pipeline['contexts'][index]['context'];theta=context['phase_epoch']
    expected,_=oscillator.exact_flow(oscillator.SEED,theta)
    np.testing.assert_allclose(context['initial'],expected,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(context['phase_gradient'],[-np.sin(theta),np.cos(theta),0.],rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(context['direction'],[0.,0.,1.],rtol=0.,atol=2e-14)
    assert context['id']==f'anchor-core--p{index:02d}' and context['setting']=='anchor-core'
    assert context['source_setting']=='core'
    assert context['new_normalizations']==0


@pytest.mark.parametrize('index',[0,7])
def test_first_and_second_observable_responses_match_direct_scalar_solution(pipeline,index):
    item=pipeline['contexts'][index];result=item['evaluated']
    assert result['complete'] and result['qualified'],result.get('error',result)
    seal=result['seal'];context=item['context'];T=context['period'];epoch=context['phase_epoch']
    for window,row in seal['windows'].items():
        # Saved sample times are elapsed from the first write, not phase epoch.
        t=np.asarray(row['times']);phase=epoch+t
        activation_slope=1-np.tanh(np.cos(9*phase))**2
        A=np.zeros((32,3));A[:,2]=1e-4*activation_slope*np.exp(-oscillator.LAMBDA*t)
        B=np.zeros((32,3))
        if window!='immediate':B[:,2]=1e-4*activation_slope*np.exp(-oscillator.LAMBDA*(t-T))
        predicted=row['linear_predictions']['identity']
        np.testing.assert_allclose(predicted['A'],A.ravel(),rtol=2e-11,atol=2e-17)
        np.testing.assert_allclose(predicted['B'],B.ravel(),rtol=2e-11,atol=2e-17)
        if window!='immediate':
            np.testing.assert_allclose(predicted['A'],np.exp(-oscillator.LAMBDA*T)*predicted['B'],rtol=2e-11,atol=2e-17)
        descriptors=result['descriptors']['windows'][window]
        np.testing.assert_allclose(descriptors['A'],A.ravel(),rtol=2e-11,atol=2e-17)
        np.testing.assert_allclose(descriptors['B'],B.ravel(),rtol=2e-11,atol=2e-17)


def test_two_segment_transport_retains_identity_resets_and_exact_ownership_census(pipeline):
    assert len(pipeline['calls'])==8
    for index,item in pipeline['contexts'].items():
        for method,reference in item['references'].items():
            assert reference['context_id']==item['context']['id']
            for segment,count in [('pre',4097),('post',28673)]:
                assert reference[segment]['states'].shape==(count,3)
                np.testing.assert_array_equal(reference[segment]['phi'][0],np.eye(3))
                assert reference['ivp_slots'][segment]['ivp_completed'] is True
            np.testing.assert_array_equal(reference['pre']['states'][-1],reference['post']['states'][0])
    jobs=acq.job_census();nested=acq.ivp_census()
    assert len(jobs)==len({j['id'] for j in jobs})==114
    assert len(nested)==len({j['id'] for j in nested})==528


def test_changed_physical_source_is_rejected_after_roundtrip(pipeline):
    p=pipeline;changed=copy.deepcopy(p['pair']);changed['canonical']['normalization_count']=2
    with pytest.raises(ValueError):
        acq.prepare_context(changed,p['phases'][0],source_binding=p['source'],rhs=oscillator.rhs)
    changed=copy.deepcopy(p['contexts'][7]['physical']);changed['phase_fraction']=p['phases'][0]['phase_fraction']
    refs=p['contexts'][7]['references']
    result=evaluate.seal_context(p['slot'],changed,refs['dop853'],refs['rk45'],rhs=oscillator.rhs,jacobian=oscillator.jacobian)
    assert not result['qualified']
