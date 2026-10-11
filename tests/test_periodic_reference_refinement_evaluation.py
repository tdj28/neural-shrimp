"""Synthetic only: no Hopfield integrations or old target packet loading."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
import pytest
from scipy.linalg import expm
import periodic_reference_refinement_evaluation as ev

A=np.array([[0.,0.,-1.],[0.,-.4,0.],[1.,0.,0.]])
def rhs(t,x,a,g):return A@x
def jac(t,x,a,g):return A


def test_bordered_derivative_including_time_and_section():
    x=np.array([.07,.02,1.]);T=6.1;seed=np.array([0.,0.,1.]);T0=2*np.pi
    M=expm(A*T);end=M@x
    result=ev.shooting_arithmetic(x,T,seed,T0,end,M,rhs=rhs,jacobian=jac,alpha=0.,gamma=0.)
    def residual(v):return np.r_[expm(A*v[3])@v[:3]-v[:3],(A@v[:3])[2]]
    u=np.r_[x,T];d=1e-5;finite=np.column_stack([(residual(u+d*np.eye(4)[i])-residual(u-d*np.eye(4)[i]))/(2*d) for i in range(4)])
    np.testing.assert_allclose(result['jacobian'],finite,rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(result['scaled_jacobian'],finite*np.r_[result['state_scales'],T0][None,:]/np.r_[result['state_scales'],result['section_scale']][:,None],rtol=1e-9,atol=1e-9)
    assert result['section_scale']==1.


def test_section_scale_uses_original_not_trial():
    def j(t,x,a,g):return np.diag([1.,1.,3.*x[2]])
    r=ev.shooting_arithmetic(np.array([.2,.3,4.]),2.,np.array([0.,0.,2.]),2.,np.ones(3),np.eye(3),rhs=rhs,jacobian=j,alpha=0.,gamma=0.)
    assert r['section_scale']==12.


def test_candidate_comparison_old_denominators_and_inclusive_bounds():
    old={'certificate':{'section_state':[0.,2.,0.]},'period':2.}
    a={'state':np.zeros(3),'period':2.};b={'state':np.array([1e-10,2e-10,0.]),'period':2.}
    r=ev.candidate_comparison(a,b,old);assert r['qualified'];assert r['state_scaled_infinity']==1e-10
    b['state'][0]=np.nextafter(1e-10,np.inf);assert not ev.candidate_comparison(a,b,old)['qualified']


def test_candidate_period_uses_old_period():
    old={'certificate':{'section_state':[0.,0.,0.]},'period':2.}
    a={'state':np.zeros(3),'period':100.};b={'state':np.zeros(3),'period':100.+1e-9}
    r=ev.candidate_comparison(a,b,old);assert not r['gates']['period']
    assert r['period_relative']==abs(a['period']-b['period'])/2.


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_tree_nonfinite_never_equal(bad):
    with pytest.raises(ValueError):ev._tree({'x':bad},{'x':bad},'test')


def test_strict_method_ids_and_limits():
    assert ev._method('tight-rk45',16.384)['max_step']==.001
    with pytest.raises(ValueError):ev._method('fine-rk45',2.)
    assert ev.contract()['maximum_ivp_solves']==208


def test_phase_rows_no_silent_missing_or_wrong_identity():
    rows=[dict(phase_index=i,phase_fraction=i/25,split='development' if i<8 else 'heldout') for i in range(24)]
    ev._phase_rows(rows)
    with pytest.raises(ValueError):ev._phase_rows(rows[:-1])
    rows[12]['phase_index']=True
    with pytest.raises(ValueError):ev._phase_rows(rows)


def test_missing_everything_retains_full_denominator():
    rows=[dict(phase_index=i,phase_fraction=i/25,split='development' if i<8 else 'heldout') for i in range(24)]
    summary=ev.summarize({},{},{},{},rows)
    assert not summary['complete'] and not summary['qualified']
    assert (len(summary['shoots']),len(summary['references']),len(summary['events']),len(summary['contexts']),len(summary['nested_slots']))==(4,2,96,48,208)
    assert len({r['id'] for r in summary['nested_slots']})==208
    assert summary['counts']['contexts_qualified']==0
    assert sum(r['kind']=='shoot_trial' for r in summary['nested_slots'])==100


def test_discrete_masks_replay_exactly_without_float_cast():
    ev._tree(np.array([True,False]),np.array([True,False]),'mask')
    with pytest.raises(ValueError):ev._tree(np.array([1.,0.]),np.array([True,False]),'mask')


def test_partial_array_census_keeps_nonfinite_evidence():
    r=ev.partial_array_census({'solver':{'values':np.array([[np.nan,np.inf]])}})
    assert len(r)==1 and r[0]['finite'] is False and r[0]['size']==2


# Real producer API fixtures are synthetic analytic nine-maximum oscillators.
from test_periodic_reference_refinement_integration import synthetic_packets, rhs as circle_rhs, jacobian as circle_jac, phase_rows as circle_phases
import copy


@pytest.mark.parametrize('change', ['unused_reason','candidate_source','newton_budget','initial_state','metric','ivp_count'])
def test_shoot_tamper_rejected(synthetic_packets,change):
    old,shots,pair=synthetic_packets;p=copy.deepcopy(shots[ev.METHODS[0]])
    if change=='unused_reason':p['trials'][1]['reason']='selected_after_outcome'
    elif change=='candidate_source':p['candidate']['source_trial_index']=1
    elif change=='newton_budget':p['limits']['maximum_updates']=7
    elif change=='initial_state':p['trials'][0]['forward']['trajectory']['states'][0,0]+=.01
    elif change=='metric':p['trials'][0]['metrics']['raw_residual'][0]+=.01
    elif change=='ivp_count':p['ivp_evaluations']+=1
    with pytest.raises(ValueError):ev.validate_shoot_packet(p,old,rhs=circle_rhs,jacobian=circle_jac)


@pytest.mark.parametrize('change',['candidate','global_z','terminal_z','phase_id','dense_count','canonical_choice'])
def test_reference_tamper_rejected(synthetic_packets,change):
    old,shots,pair=synthetic_packets;p=copy.deepcopy(pair);m=ev.METHODS[0]
    if change=='candidate':p['candidates'][m]['state'][0]+=.01
    elif change=='global_z':p['canonical']['terminal_adjoint'][0]+=.01
    elif change=='terminal_z':p['bases'][m]['grids']['8192']['z'][-1,0]+=.01
    elif change=='phase_id':p['bases'][m]['phases'][7]['phase_index']=8
    elif change=='dense_count':p['bases'][m]['exact_phase_arrays']['states']=p['bases'][m]['exact_phase_arrays']['states'][:-1]
    elif change=='canonical_choice':p['canonical_method']=ev.METHODS[1]
    with pytest.raises(ValueError):ev.validate_reference_packet(p,shots,old,circle_phases(),rhs=circle_rhs,jacobian=circle_jac)


def test_failed_outer_shoot_replays_completed_trial_and_keeps_25(synthetic_packets):
    old,shots,pair=synthetic_packets;p=copy.deepcopy(shots[ev.METHODS[0]])
    p.update(status='failed',complete=False,qualified=False,error='synthetic interruption after completed trial')
    r=ev.validate_partial_packet(p,old,circle_phases(),rhs=circle_rhs,jacobian=circle_jac)
    assert r['partial_replay_pass'] and not r['qualified']
    assert any(v['path']=='trial0.completed_forward' for v in r['checks'])
    p['trials'][0]['metrics']['merit']+=.01
    bad=ev.validate_partial_packet(p,old,circle_phases(),rhs=circle_rhs,jacobian=circle_jac)
    assert not bad['partial_replay_pass'] and len(bad['errors'])==1


def test_failed_outer_reference_replays_all_completed_paths(synthetic_packets):
    old,shots,pair=synthetic_packets;p=copy.deepcopy(pair);p.update(complete=False,status='failed',qualified=False,error='synthetic interrupted seal')
    p['verifications'][ev.METHODS[0]+'--'+ev.METHODS[1]]['certificate']['section_residual']+=.01
    r=ev.validate_partial_packet(p,old,circle_phases(),shoot_packets=shots,rhs=circle_rhs,jacobian=circle_jac)
    assert not r['partial_replay_pass']
    assert sum(v['path'].endswith('.completed_forward') for v in r['checks'])==4
    assert sum(v['path'].endswith('.completed_base') for v in r['checks'])==2


@pytest.mark.parametrize('change',['applied_scale','verification_started','verification_completed','adjoint_started','base_started','bundle_status','base_map_method'])
def test_reference_completion_metadata_tamper(synthetic_packets,change):
    old,shots,pair=synthetic_packets;p=copy.deepcopy(pair);m=ev.METHODS[0]
    if change=='applied_scale':p['canonical']['normalization_applied_scale']+=.1
    elif change=='verification_started':p['verifications'][m+'--'+m]['ivp_started']=False
    elif change=='verification_completed':p['verifications'][m+'--'+m]['ivp_completed']=False
    elif change=='adjoint_started':p['adjoints'][m]['ivp_started']=False
    elif change=='base_started':p['bases'][m]['ivp_started']=False
    elif change=='bundle_status':p['status']='blocked_crossverification'
    elif change=='base_map_method':p['bases'][m]['method']=ev.METHODS[1]
    with pytest.raises(ValueError):ev.validate_reference_packet(p,shots,old,circle_phases(),rhs=circle_rhs,jacobian=circle_jac)


def test_completed_path_flags_require_true():
    ev._ivp_complete({'ivp_started':True,'ivp_completed':True},'synthetic')
    for bad in (False,1,None):
        with pytest.raises(ValueError):ev._ivp_complete({'ivp_started':bad,'ivp_completed':True},'synthetic')


def test_partial_reference_applied_scale_is_still_bound(synthetic_packets):
    old,shots,pair=synthetic_packets;p=copy.deepcopy(pair);p.update(complete=False,status='failed',qualified=False)
    p['canonical']['normalization_applied_scale']+=.1
    r=ev.validate_partial_packet(p,old,circle_phases(),shoot_packets=shots,rhs=circle_rhs,jacobian=circle_jac)
    assert not r['partial_replay_pass']
    assert any('applied normalization scale' in error for error in r['errors'])
