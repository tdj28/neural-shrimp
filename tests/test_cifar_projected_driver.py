import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
import explore_cifar_projected as W
import launch_cifar_projected as L


def test_complete_fixed_negative_and_guarded_trial():
    counts=dict(action=0,j2vp=0,hvp=12,direct_gradient=14)
    row=dict(status='projected_pilot_complete',comparison_qualified=True,
        accepted_orbit=False,spectrum_requested=False,operation_caps=W.CAPS,
        operation_attempted=counts,operation_returned=counts,qualification=dict(passed=True),
        linear=dict(dimension=4,action_linearity_pass=True,forcing_pass=False,usable_descent=True),
        trials=[],accepted_index=None,last_accepted_state='initial-theta.npy')
    assert L.scientific_complete(row)
    bad=copy.deepcopy(row);bad['linear']['action_linearity_pass']=False;assert not L.scientific_complete(bad)
    bad=copy.deepcopy(row);bad.update(trials=[dict(accepted_local_step=True)],accepted_index=0,last_accepted_state='trial-0-theta.npy')
    bad['operation_attempted']['direct_gradient']=16;bad['operation_returned']['direct_gradient']=16
    assert not L.scientific_complete(bad)
    bad['linear']['forcing_pass']=True;assert L.scientific_complete(bad)
    bad['operation_attempted']['hvp']=14;assert not L.scientific_complete(bad)


def test_fixed_parent_rejects_changed_state_or_operations():
    row=dict(source_commit=W.CORR_EXEC,status='phase_scaled_pilot_complete',qualification=dict(passed=True),
        accepted_index=None,trials=[],linear=dict(forcing_pass=False),
        operation_attempted=dict(action=0,j2vp=0,hvp=818,direct_gradient=14),
        operation_returned=dict(action=0,j2vp=0,hvp=818,direct_gradient=14))
    W.check_parent(row)
    bad=copy.deepcopy(row);bad['accepted_index']=0
    with pytest.raises(ValueError):W.check_parent(bad)
    bad=copy.deepcopy(row);bad['operation_returned']['hvp']=817
    with pytest.raises(ValueError):W.check_parent(bad)


def test_bounded_profile_counts_the_complete_projected_unit():
    events=[dict(kind='operation_returned',operation=k,seconds=v)for k,v in [('hvp',.2),('direct_gradient',.1)]]
    assert W.measured_gate(events)['forecast_seconds']==pytest.approx(2*(12*.2+40*.1)+20)
    events[1]['seconds']=2
    with pytest.raises(ValueError):W.measured_gate(events)
