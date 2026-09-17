"""Only algebraic circles/injected solvers; no historical inputs or neural solves."""
from pathlib import Path
import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import phase_reference_diagnostic as acquisition
import phase_reference_diagnostic_evaluation as evaluation
import run_phase_memory_recovery as codec

J=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]])

def rhs(t,x,a,g):return J@x

def jacobian(t,x,a,g):return J


def rotations(times):
    t=np.asarray(times);R=np.zeros(t.shape+(3,3));R[...,0,0]=np.cos(t);R[...,0,1]=-np.sin(t)
    R[...,1,0]=np.sin(t);R[...,1,1]=np.cos(t);R[...,2,2]=1.;return R


def solver(fun,span,initial,*,t_eval,**kwargs):
    def dense(times):
        t=np.asarray(times);R=rotations(t-span[0]);x=R@initial[:3]
        if len(initial)==12:
            phi=R@initial[3:].reshape(3,3)
            value=np.concatenate([x,phi.reshape(t.shape+(9,))],axis=-1)
        else:value=x
        return np.moveaxis(value,-1,0)
    y=dense(t_eval);y[:,0]=initial
    return SimpleNamespace(t=t_eval.copy(),y=y,sol=dense,success=True,message='algebraic rotation, no solve',nfev=0,njev=0,nlu=0)


def phases():return [dict(phase_index=i,phase_fraction=(i+.37)/24,split='development' if i<8 else 'heldout') for i in range(24)]


def reference(setting='core',scale=1.):
    T=2*np.pi;t=acquisition.build_grids(T)['4096'];x=np.column_stack([np.cos(t),np.sin(t),np.zeros_like(t)])
    f=x@J.T;z=scale*f;terminal=np.array([0.,scale,0.]);z[-1]=terminal
    return dict(status='complete',qualified=True,setting=setting,alpha=0.,gamma=0.,period=T,
        certificate=dict(section_state=np.array([1.,0.,0.]),period=T),orbit=dict(times=t,states=x,derivatives=f),
        adjoint=dict(times=t,values=z,derivatives=z@J.T,terminal_normalized_once=terminal,normalization_count=1))


def acquire_base(setting='core',method='fine-dop853',scale=1.):
    return acquisition.acquire_base(reference(setting,scale),phases(),method,rhs=rhs,jacobian=jacobian,solver=solver)


def old_seal(base,index):
    phase=base['phases'][index];old=phase['old_hermite'];xT=rotations(np.array(base['period']))@old['state'];fT=J@xT
    context=dict(id=phase['context_id'],setting=base['setting'],period=base['period'],alpha=0.,gamma=0.,
        phase_index=index,phase_fraction=phase['phase_fraction'],split=phase['split'],phase_epoch=phase['phase_epoch'],
        initial=old['state'],flow=old['flow'],phase_gradient=old['z'],nominal_event_phase_gradient=old['z'],
        reference_at_event=xT,event_flow=fT,event_phase_gradient=old['z'])
    return dict(context_id=phase['context_id'],status='complete',qualified=False,acquisition=context,
        evaluation=dict(gates=dict(normalization=False))) # Deliberately failed old qualification is immutable.


@pytest.fixture(scope='module')
def base():return acquire_base()


@pytest.fixture(scope='module')
def event(base):return acquisition.acquire_event(base,7,rhs=rhs,jacobian=jacobian,solver=solver)


def test_actual_acquisition_packets_validate(base,event):
    assert evaluation.validate_base_packet(base,rhs=rhs,jacobian=jacobian) is base
    assert evaluation.validate_event_packet(event,base,rhs=rhs,jacobian=jacobian) is event
    assert len(event['trajectory']['times'])==8193
    assert event['forward']['method_spec']['max_step']==base['period']/8192


def test_interpolation_ladder_keeps_both_grids_and_raw_endpoints(base):
    for grid in evaluation.GRIDS:
        out=evaluation.interpolation_diagnostics(base,base['phases'][7],grid,rhs=rhs)
        assert out['periodic_wrap_uses_initial_sample'] and out['raw_endpoint_retained']
        assert out['state_error']['max_abs']<1e-12
        assert out['adjoint_error']['max_abs']<1e-12
        for key in ['dense_normalization','interpolated_z_dense_flow','interpolated_z_interpolated_state_physical_flow','interpolated_z_template_velocity']:
            assert out[key]['pass_bound']
    assert not np.array_equal(base['raw_endpoints']['initial_state'],base['raw_endpoints']['final_state'])


@pytest.mark.parametrize('mutate',[
    lambda b:b.update(method='unknown'),lambda b:b['forward']['method_spec'].update(max_step=.2),
    lambda b:b['backward']['solver'].update(success=False),lambda b:b['forward']['solver'].update(nfev=True),
    lambda b:b['inherited'].update(new_normalizations=1),lambda b:b['inherited']['terminal_adjoint'].__setitem__(0,1.),
    lambda b:b['grids']['8192'].update(phi=b['grids']['8192']['phi'].astype(np.float32)),
    lambda b:b['grids']['4096']['z'].__setitem__((5,0),np.nan),
    lambda b:b['raw_endpoints']['final_state'].__setitem__(1,1.),
    lambda b:b['exact_phase_arrays']['states'].__setitem__((0,0),1.),
    lambda b:b['phases'][0].update(phase_index=True),
    lambda b:b['grids']['8192']['derivatives'].__setitem__((100,1),1.),
])
def test_base_identity_physics_and_no_repair_guards(base,mutate):
    b=copy.deepcopy(base);mutate(b)
    with pytest.raises(ValueError):evaluation.validate_base_packet(b,rhs=rhs,jacobian=jacobian)


@pytest.mark.parametrize('mutate',[
    lambda e:e.update(base_id='core--fine-rk45'),lambda e:e['initial'].__setitem__(0,1.),
    lambda e:e['phase_gradient'].__setitem__(0,1.),lambda e:e.update(physical_event_time=0.),
    lambda e:e['trajectory']['phi'].__setitem__((0,0,0),2.),
    lambda e:e['event']['Q'].__setitem__((0,0),1.),
    lambda e:e['forward']['method_spec'].update(rtol=1e-8),
    lambda e:e['trajectory']['phi_derivatives'].__setitem__((100,0,0),1.),
])
def test_event_identity_physics_and_nominal_z_guards(base,event,mutate):
    e=copy.deepcopy(event);mutate(e)
    with pytest.raises(ValueError):evaluation.validate_event_packet(e,base,rhs=rhs,jacobian=jacobian)


def test_no_normalization_of_failed_reference(base):
    b=acquire_base(scale=1.03);before=b['inherited']['terminal_adjoint'].copy()
    evaluation.validate_base_packet(b,rhs=rhs,jacobian=jacobian)
    row=evaluation._base_diagnostics(b)
    assert not row['all_grid_diagnostics_pass']
    assert row['grids']['8192']['normalization']['max_abs']==pytest.approx(.03)
    np.testing.assert_array_equal(before,b['inherited']['terminal_adjoint'])
    event=acquisition.acquire_event(b,7,rhs=rhs,jacobian=jacobian,solver=solver)
    evaluation.validate_event_packet(event,b,rhs=rhs,jacobian=jacobian)
    assert abs(event['event']['normalization_residual'])>1e-2


def test_full_tangent_gate_retains_raw_operator_and_location(base):
    a=base['grids']['4096'];b=copy.deepcopy(a);b['phi'][100,2,2]+=.02
    value=evaluation.array_agreement(a,b,adjoint=True)
    assert not value['agreement_pass'] and not value['gates']['pulse_scaled_tangent']
    assert value['gates']['state'] and value['gates']['activation']
    assert value['raw_phi']['location']==[100,2,2]
    assert value['raw_phi']['max_abs']==pytest.approx(.02)
    assert value['pulse_scaled_phi']['max_abs']==pytest.approx(2e-6)


def test_missing_everything_retains_complete_planned_denominators():
    out=evaluation.summarize({}, {}, {},rhs=rhs,jacobian=jacobian)
    assert not out['diagnostic_complete'] and not out['all_dense_fine_pairs_pass']
    assert len(out['bases'])==6 and len(out['events'])==96 and len(out['contexts'])==48 and len(out['base_pairs'])==6
    assert out['counts']['events_complete']==0 and not out['original_study_requalified']


def test_unexpected_input_identity_retained():
    out=evaluation.summarize({'extra':None},{},{},rhs=rhs,jacobian=jacobian)
    assert out['input_errors']==[dict(scope='bases',error='Unexpected identities',ids=['extra'])]
    assert not out['diagnostic_complete'] and len(out['contexts'])==48


def test_packet_codec_roundtrip_is_lossless_and_replayable(base,event,tmp_path):
    a=codec.save_packet(tmp_path,'base',base);b=codec.save_packet(tmp_path,'event',event)
    restored_base=codec.load_packet(a['path']);restored_event=codec.load_packet(b['path'])
    evaluation.validate_base_packet(restored_base,rhs=rhs,jacobian=jacobian)
    evaluation.validate_event_packet(restored_event,restored_base,rhs=rhs,jacobian=jacobian)
    assert codec.packet_digest(restored_base)==codec.packet_digest(base)


def test_actual_full_synthetic_census_and_one_missing_dependency():
    bases={};events={};old={}
    for setting in evaluation.SETTINGS:
        for method in evaluation.METHODS:
            b=acquire_base(setting,method);bases[b['id']]=b
            if method in evaluation.FINE_METHODS:
                for index in range(24):
                    e=acquisition.acquire_event(b,index,rhs=rhs,jacobian=jacobian,solver=solver);events[e['id']]=e
            if method=='baseline-dop853':
                for index in range(24):old[f'{setting}--p{index:02d}']=old_seal(b,index)
    out=evaluation.summarize(bases,events,old,rhs=rhs,jacobian=jacobian)
    assert out['diagnostic_complete'],[(r['id'],r.get('error')) for r in out['bases']+out['events'] if not r['complete']]
    assert out['all_dense_fine_pairs_pass'] and not out['original_study_requalified']
    assert out['counts']==dict(bases_planned=6,bases_complete=6,events_planned=96,events_complete=96,
        base_pairs_planned=6,base_pairs_complete=6,contexts_planned=48,contexts_complete=48,dense_fine_pair_diagnostic_pass=48)
    assert all(r['old']['original_qualified'] is False for r in out['contexts'])
    # Structural failure retention does not require a second million-node physics
    # pass; retain previously validated packets and check aggregation only.
    missing='core--p07--fine-dop853';del events[missing]
    from unittest.mock import patch
    with patch.object(evaluation,'validate_base_packet',side_effect=lambda b,*a,**k:b),patch.object(evaluation,'validate_event_packet',side_effect=lambda e,*a,**k:e if e is not None else (_ for _ in ()).throw(ValueError('missing'))):
        failed=evaluation.summarize(bases,events,old,rhs=rhs,jacobian=jacobian)
    assert not failed['diagnostic_complete'] and failed['counts']['events_complete']==95
    assert failed['counts']['contexts_complete']==47 and len(failed['events'])==96 and len(failed['contexts'])==48


@pytest.mark.parametrize('field',['neutral_flow_relative_residual','transported_flow_relative_residual','inherited_terminal_left_relative_residual'])
def test_saved_neutral_diagnostics_are_replayed(base,field):
    b=copy.deepcopy(base);b['diagnostics'][field]+=.01
    with pytest.raises(ValueError,match='base diagnostics'):
        evaluation.validate_base_packet(b,rhs=rhs,jacobian=jacobian)


@pytest.mark.parametrize('field',['initial_normalization_residual','event_minus_initial_normalization','old_z_event_normalization','Q_operator_norm'])
def test_saved_event_scalar_diagnostics_are_replayed(base,event,field):
    e=copy.deepcopy(event);e['event'][field]+=.01
    with pytest.raises(ValueError,match='event diagnostics'):
        evaluation.validate_event_packet(e,base,rhs=rhs,jacobian=jacobian)


def test_event_old_template_duplicate_is_bound(base,event):
    e=copy.deepcopy(event);e['old_hermite']['z'][0]+=.01
    with pytest.raises(ValueError,match='old Hermite'):
        evaluation.validate_event_packet(e,base,rhs=rhs,jacobian=jacobian)


def test_exact_phase_normalization_array_is_bound(base):
    b=copy.deepcopy(base);b['exact_phase_arrays']['normalization_residuals'][0]+=.01
    with pytest.raises(ValueError,match='exact phase normalization'):
        evaluation.validate_base_packet(b,rhs=rhs,jacobian=jacobian)


def test_comparison_rejects_cross_phase_metadata(base):
    left,right=base['phases'][0],copy.deepcopy(base['phases'][0]);right['phase_index']=1
    with pytest.raises(ValueError,match='compared phase'):
        evaluation._phase_comparison(left,right)


def test_pointwise_dot_normalization_matches_acquisition_under_cancellation():
    z=np.array([[1e16,1.,-1e16],[1.,2.,3.]]);f=np.ones_like(z)
    expected=np.array([float(a@b-1.) for a,b in zip(z,f)])
    np.testing.assert_array_equal(evaluation.normalization(z,f)['residuals'],expected)
    for i in range(2):assert evaluation.normalization(z[i],f[i])['residuals']==expected[i]
