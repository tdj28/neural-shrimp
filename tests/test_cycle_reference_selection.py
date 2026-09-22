"""Synthetic metadata only: no continuation artifacts or target arrays read."""
from copy import deepcopy
from pathlib import Path
import builtins
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import cycle_reference_selection as select


def node(index, rho, *, other=0.01):
    modes = [dict(real=v, imag=0., modulus=abs(v)) for v in (other, rho)]
    radius = max(m['modulus'] for m in modes)
    return dict(index=index, alpha=1., gamma=10.+0.41*index/100,
        parameter=index/100, period=12.+index/10, state=[-1.,-2.,-3.],
        first_maximum_state=[1.+index,2.,3.],
        transverse_multipliers=modes, transverse_spectral_radius=radius,
        attracting=radius < 1-1e-5, unstable=radius > 1+1e-5,
        gates={g: True for g in select.OLD_GATES}, verified=True,
        bracket_flags=[], checkpoint_certificate={'verified':False})


def fixture(rhos=(0.1,0.25,0.5)):
    branches = {b:dict(alpha_seed=1.,gamma_seed=10., direction=1 if s=='core' else -1,
        parameterization='alpha=alpha_seed; gamma=gamma_seed+0.41*s',
        nodes=[node(i,r) for i,r in enumerate(rhos)]) for s,b in select.BRANCHES.items()}
    anchors = {s:dict(id=s+'--reference-pair',setting=s,canonical_method='tight-dop853',
        complete=True,qualified=True,status='complete',gates={'all_original':True},alpha=1.,gamma=10.,
        canonical=dict(source_verification='tight-dop853--tight-dop853',state=[3.,2.,1.],period=12.))
        for s in select.SETTINGS}
    bindings = {key:dict(experiment_id='EXP-018' if key.startswith('anchor-') else 'EXP-006',
        source_commit='a'*40,receipt_sha256='b'*64,
        artifacts=[dict(path=key+'.json',sha256='c'*64,bytes=10)])
        for key in [*select.BRANCHES.values(),*('anchor-'+s for s in select.SETTINGS)]}
    return branches,anchors,bindings


def run(data):
    return select.select_cycles(data[0],data[1],bindings=data[2])


def core(data,index):
    return data[0]['core-direction-+1']['nodes'][index]


def test_full_fixed_six_slot_census_and_seed_convention():
    result=run(fixture())
    assert [s['id'] for s in result['slots']]==['anchor-core','anchor-failed-near-edge',
        'core-rho025','core-rho050','failed-near-edge-rho025','failed-near-edge-rho050']
    assert result['counts']==dict(planned=6,available=6,unavailable=0)
    assert result['all_slots_available'] and not result['scientific_qualification']
    assert result['new_neural_computations']==0
    assert result['slots'][2]['first_maximum_state']==[2.,2.,3.]
    assert result['slots'][2]['first_maximum_state']!=[-1.,-2.,-3.]
    assert result['slots'][0]['seed_provenance']['state_field']=='canonical.state'
    assert len(result['branch_audits'])==2 and len(result['candidate_decisions'])==4
    ids=[s['node_id'] for s in result['slots'] if s['node_id']]
    assert len(ids)==len(set(ids))==4


def test_old_checkpoint_failure_is_not_repurposed_as_selector_gate():
    data=fixture()
    assert core(data,1)['checkpoint_certificate']['verified'] is False
    assert run(data)['slots'][2]['available']


@pytest.mark.parametrize('crossing',select.CROSSINGS)
def test_crossing_ends_prefix_before_node_and_never_resumes(crossing):
    data=fixture((.1,.25,.5,.25))
    core(data,1)['bracket_flags']=[crossing]
    result=run(data)
    assert [s['status'] for s in result['slots'][2:4]]==['no_historical_match']*2
    audit=result['branch_audits'][0]
    assert audit['prefix_stop']==dict(index=1,reasons=[crossing])
    assert [n['in_initial_attracting_prefix'] for n in audit['nodes']]==[True,False,False,False]
    assert all('outside_initial_attracting_prefix' in n['reasons'] for n in audit['nodes'][1:])


@pytest.mark.parametrize('rho',[1.,1.00002])
def test_undecided_and_unstable_both_stop_initial_prefix(rho):
    result=run(fixture((.1,rho,.25,.5)))
    assert result['branch_audits'][0]['prefix_stop']['index']==1
    assert result['counts']['unavailable']==4


def test_gate_failure_excludes_only_that_node_and_retains_every_decision():
    data=fixture((.1,.25,.26,.5))
    core(data,1)['gates']['closure']=False;core(data,1)['verified']=False
    result=run(data)
    assert result['slots'][2]['node_index']==2
    assert result['branch_audits'][0]['prefix_stop'] is None
    assert 'old_gate_failed:closure' in result['candidate_decisions'][0]['nodes'][1]['reasons']
    assert sum(n['selected'] for n in result['candidate_decisions'][0]['nodes'])==1


@pytest.mark.parametrize('kind',['negative','complex','tied'])
def test_positive_modulus_is_not_sufficient_and_ambiguous_modes_are_retained(kind):
    data=fixture((.1,.25,.5))
    n=core(data,1)
    if kind=='negative': n['transverse_multipliers'][1]['real']=-.25
    elif kind=='complex':
        n['transverse_multipliers'][1].update(real=.25,imag=1e-20)
    else: n['transverse_multipliers'][0]=dict(real=-.25,imag=0.,modulus=.25)
    result=run(data)
    assert not result['slots'][2]['available']
    reason='nonunique_dominant_modulus' if kind=='tied' else 'dominant_not_exact_real_positive'
    assert reason in result['branch_audits'][0]['nodes'][1]['reasons']


def test_nearest_then_exact_tie_uses_lowest_original_index():
    data=fixture((.1,.234375,.265625,.49,.5))
    result=run(data)
    assert result['slots'][2]['node_index']==1
    assert result['slots'][3]['node_index']==4
    decisions=result['candidate_decisions'][0]['nodes']
    assert decisions[1]['exact_distance']==decisions[2]['exact_distance']=='1/64'


def test_exact_rational_caliper_has_no_unrecorded_epsilon():
    # float(.275) is just above the exact .25+.025 boundary.
    result=run(fixture((.1,.275,.5)))
    assert not result['slots'][2]['available']
    result=run(fixture((.1,math.nextafter(.275,-math.inf),.5)))
    assert result['slots'][2]['available']


def test_absent_maximum_evidence_retained_without_moving_state_fallback():
    data=fixture();core(data,1)['first_maximum_state']=None
    result=run(data)
    assert not result['slots'][2]['available']
    assert result['slots'][2]['first_maximum_state'] is None
    assert result['slots'][2]['status']=='missing_seed_evidence'
    assert 'absent_first_maximum_state' in result['candidate_decisions'][0]['nodes'][1]['seed_reasons']


def test_absent_chosen_seed_does_not_search_next_ranked_node():
    data=fixture((.1,.25,.251,.5));core(data,1)['first_maximum_state']=None
    result=run(data);row=result['slots'][2]
    assert row['status']=='missing_seed_evidence' and row['node_index']==1
    assert row['node_id']=='core-direction-+1--node-1'
    assert row['first_maximum_state'] is None and row['dominant_rho']==.25
    assert result['candidate_decisions'][0]['nodes'][1]['selected']
    assert not result['candidate_decisions'][0]['nodes'][2]['selected']


def test_valid_failed_anchor_does_not_remove_slot_or_expand_candidates():
    data=fixture();data[1]['core'].update(qualified=False,complete=False,status='failed')
    result=run(data)
    assert result['slots'][0]['status']=='invalid_anchor'
    assert len(result['slots'])==6 and result['counts']['available']==5


@pytest.mark.parametrize('mutation',[
    lambda d:core(d,1).update(index=0),
    lambda d:core(d,1).update(index=True),
    lambda d:core(d,1).update(alpha=2.),
    lambda d:core(d,1).update(gamma=11.),
    lambda d:core(d,1).update(verified=False),
    lambda d:core(d,1).update(attracting=False),
    lambda d:core(d,1).update(unstable=True),
    lambda d:core(d,1).update(period=float('nan')),
    lambda d:core(d,1).update(period=True),
    lambda d:core(d,1).update(first_maximum_state=[1.,2.,float('inf')]),
    lambda d:core(d,1).update(first_maximum_state=[[1.],[2.],[3.]]),
    lambda d:core(d,1)['gates'].update(closure=1),
    lambda d:core(d,1)['gates'].pop('primitive_nine_peaks'),
    lambda d:core(d,1).update(bracket_flags=['unknown']),
    lambda d:core(d,1)['transverse_multipliers'][1].update(modulus=.2),
    lambda d:d[0]['core-direction-+1'].update(direction=-1),
    lambda d:d[1]['core'].update(setting='failed-near-edge'),
    lambda d:d[1]['core'].update(canonical_method='tight-rk45'),
    lambda d:d[1]['core']['gates'].update(all_original=False),
    lambda d:d[2]['anchor-core'].update(receipt_sha256='wrong'),
    lambda d:d[2]['anchor-core']['artifacts'].append(deepcopy(d[2]['anchor-core']['artifacts'][0])),
])
def test_malformed_metadata_is_fatal_not_silently_skipped(mutation):
    data=fixture();mutation(data)
    with pytest.raises(ValueError):run(data)


def test_malformed_post_prefix_nodes_are_still_validated():
    data=fixture((1.1,.25,.5));core(data,2)['period']=float('nan')
    with pytest.raises(ValueError,match='finite'):run(data)


def test_missing_extra_or_reordered_branch_census_rejects():
    data=fixture();data[0]['extra']=deepcopy(data[0]['core-direction-+1'])
    with pytest.raises(ValueError,match='exact two'):run(data)
    data=fixture();data[0]['core-direction-+1']['nodes'].reverse()
    with pytest.raises(ValueError,match='indices'):run(data)


def test_outputs_do_not_alias_inputs_and_need_no_file_access(monkeypatch):
    data=fixture();snapshot=deepcopy(data)
    def forbidden(*args,**kwargs):raise AssertionError('unexpected file access')
    monkeypatch.setattr(builtins,'open',forbidden)
    result=run(data)
    assert data==snapshot
    result['slots'][2]['first_maximum_state'][0]=999.
    result['slots'][0]['seed_provenance']['source_binding']['artifacts'][0]['sha256']='changed'
    result['branch_audits'][0]['nodes'][1]['recorded_modes'][0]['real']=999.
    assert data==snapshot
    assert run(data)==run(data)


def test_contract_and_census_are_defensive_copies():
    a=select.slot_census();a[0]['source_setting']='changed'
    b=select.contract();b['branches']['core']='changed'
    assert select.slot_census()[0]['source_setting']=='core'
    assert select.contract()['branches']['core']=='core-direction-+1'
