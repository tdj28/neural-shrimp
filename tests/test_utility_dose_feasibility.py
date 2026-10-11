"""Synthetic gradients/data only; no target gradient or state update."""
from dataclasses import dataclass
import importlib.util,json,os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from neural_shrimp.utility_dose import dose_prediction,training_data,plan_array_hash
from neural_shrimp.gdmaps import TwoLayerNet


def module(name):
    p=Path(__file__).parents[1]/'tools'/(name+'.py')
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


@pytest.fixture(autouse=True)
def priority(monkeypatch):monkeypatch.setattr(os,'getpriority',lambda *_:10)


def test_disjoint_blocks_have_equal_treatment_dose_but_unequal_total_steps():
    result=dose_prediction(np.array([3.,4.]),np.ones(2),1)
    assert result['status']=='feasible' and result['radius']==.25
    assert result['fractions']['hidden']==.25/3 and result['fractions']['output']==.25/4
    for name,arm in result['arm_predictions'].items():
        assert arm['dose_equality_pass']
        np.testing.assert_allclose(arm['dose'],0 if name=='sham' else .25,rtol=1e-15)
    norms=[v['predicted_total_step_length'] for v in result['arm_predictions'].values()]
    assert len(set(norms))==4
    np.testing.assert_array_equal(result['arm_predictions']['hidden']['treatment_minus_sham_increment'],[.25,0.])
    np.testing.assert_array_equal(result['arm_predictions']['output']['treatment_minus_sham_increment'],[0.,.25])


def test_exact_fraction_cap_and_next_float_beyond_are_not_widened():
    a=dose_prediction(np.array([1.,np.sqrt(24.)]),np.ones(2),1)
    assert a['fractions']['hidden']==.25 and a['status']=='feasible'
    b=dose_prediction(np.array([1.,np.nextafter(np.sqrt(24.),np.inf)]),np.ones(2),1)
    assert b['fractions']['hidden']>.25 and b['status']=='infeasible' and b['reason']=='fraction_cap'


@pytest.mark.parametrize('gradient,reason',[(np.array([0.,1.]),'zero_or_nonfinite_block_norm'),
 (np.array([1.,0.]),'zero_or_nonfinite_block_norm'),(np.array([np.nan,1.]),'nonfinite_gradient'),
 (np.array([np.inf,1.]),'nonfinite_gradient'),(np.array([1e308,1e308]),'nonfinite_scaled_gradient'),
 (np.array([1e-320,1.]),'nonfinite_or_zero_dose')])
def test_bad_gradient_blocks_abstain_without_hiding_raw_gradient(gradient,reason):
    rates=np.array([2.,2.]) if reason=='nonfinite_scaled_gradient' else np.ones(2)
    result=dose_prediction(gradient,rates,1)
    assert result['status']=='infeasible' and result['reason']==reason
    # Scalar metadata is JSON-safe; raw arrays are retained by the NPZ writer.
    json.dumps({k:v for k,v in result.items() if not isinstance(v,np.ndarray)},allow_nan=False)


def test_finite_components_with_overflowing_vector_norm_fail_closed():
    result=dose_prediction(np.array([1.7e308,1.7e308,1.]),np.ones(2),2)
    assert result['reason']=='zero_or_nonfinite_block_norm' and result['hidden_norm'] is None


def test_training_only_reconstruction_has_exact_frozen_train_hashes():
    x,y=training_data()
    assert x.shape==(24,2) and y.shape==(24,)
    assert plan_array_hash(x)=='306db8f39a2d813dcabfdaa8cf975954c40f766fd7271c37a8a6dec1730f8f0e'
    assert plan_array_hash(y)=='6ddb5c9fb9e4facd87ca74bcf8fb4917a9708064b1d4fe056503f74016d36f38'


def test_synthetic_gradient_matches_loss_difference_including_all_biases():
    net=TwoLayerNet(2,3,bias=True,ridge=.001,loss_scale=.5)
    state=np.linspace(-.3,.4,13);x=np.array([[.2,-.1],[.8,.3],[-.4,.7]]);y=np.array([.1,-.2,.3])
    gradient=net.gradient(state[None,:],x,y)[0]
    oracle=[]
    for i in range(13):
        direction=np.zeros(13);direction[i]=1e-6
        oracle.append((net.loss((state+direction)[None,:],x,y)[0]-net.loss((state-direction)[None,:],x,y)[0])/2e-6)
    np.testing.assert_allclose(gradient,oracle,rtol=2e-7,atol=2e-10)
    assert net.hidden_parameter_count==9
    np.testing.assert_array_equal(net.rate_vector([2.],[3.])[0],np.r_[np.full(9,2.),np.full(4,3.)])


def synthetic_inputs():
    @dataclass
    class Model:
        n_parameters:int=2
        hidden_parameter_count:int=1
        calls:int=0
        def gradient(self,state,x,y):
            self.calls+=1
            assert state.shape==(1,2)
            return np.array([[3.,4.]]) if self.calls!=2 else np.array([[0.,1.]])
    forks=[dict(point=i//2,origin=str(i%2),native_state=np.array([i,1.]),native_rates=np.ones(2),state={'sha256':'synthetic'}) for i in range(4)]
    return Model(),np.ones((1,2)),np.zeros(1),forks,dict(projected_seconds=1.)


def test_driver_retains_all_four_forks_even_when_second_is_infeasible(tmp_path,monkeypatch):
    runner=module('explore_utility_dose_feasibility');inputs=synthetic_inputs()
    monkeypatch.setattr(runner,'sources',lambda:{'synthetic':{}})
    monkeypatch.setattr(runner,'bound_inputs',lambda:inputs)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runner.ROOT,text=True).strip()
    result=runner.run(tmp_path/'out',head)
    assert result['gradient_attempts']==result['gradient_returns']==4 and inputs[0].calls==4
    assert len(result['forks'])==4 and result['status']=='fixed_candidate_infeasible' and result['candidate_feasible'] is False
    events=[json.loads(v)['kind'] for v in (tmp_path/'out'/'journal.jsonl').read_text().splitlines()]
    assert events.count('gradient_return')==4 and events.count('dose_prediction')==4


def test_driver_deadline_after_gradient_retains_return_without_outcome(tmp_path,monkeypatch):
    runner=module('explore_utility_dose_feasibility');inputs=synthetic_inputs();clock=[0.]
    original=inputs[0].gradient
    def gradient(*args):
        value=original(*args);clock[0]=6.;return value
    inputs[0].gradient=gradient
    monkeypatch.setattr(runner,'sources',lambda:{'synthetic':{}})
    monkeypatch.setattr(runner,'bound_inputs',lambda:inputs)
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0])
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runner.ROOT,text=True).strip()
    result=runner.run(tmp_path/'out',head)
    assert result['status']=='resource_stop' and result['gradient_returns']==1 and result['candidate_feasible'] is None
    events=[json.loads(v)['kind'] for v in (tmp_path/'out'/'journal.jsonl').read_text().splitlines()]
    assert 'gradient_return' in events and 'dose_prediction' not in events


def test_state_and_rate_corruption_are_rejected():
    from neural_shrimp.period2_branch_io import identity
    runner=module('explore_utility_dose_feasibility');state=np.array([0.,-0.,1.])
    record={**identity(state),'binary64_hex':[format(int(v),'016x') for v in state.view('<u8')]}
    np.testing.assert_array_equal(runner.array_from_record(state.tolist(),record),state)
    with pytest.raises(ValueError,match='identity'):runner.array_from_record([0.,0.,1.],record)
    with pytest.raises(ValueError,match='finite'):runner.array_from_record([0.,0.,np.nan],record)


def test_fixed_launcher_has_only_ten_plus_five_second_gate():
    launcher=module('launch_utility_dose_feasibility')
    cmd=launcher.command(Path('/synthetic'),'a'*40)
    assert cmd[:7]==['timeout','--signal=TERM','--kill-after=5s','10s','nice','-n','10']
    assert launcher.SUPERVISOR_SECONDS==15 and 'explore_utility_dose_feasibility.py' in cmd[8]
    assert launcher.LAUNCHER_BYTES+launcher.STORE_BYTES==1024**2


@pytest.mark.parametrize('rates', [[0.,1.],[-1.,1.],[np.nan,1.],[1.,np.inf]])
def test_invalid_recovery_rates_cannot_create_predictions(rates):
    with pytest.raises(ValueError,match='rates'):dose_prediction(np.ones(2),rates,1)


def test_driver_priority_guard_remains_active(tmp_path,monkeypatch):
    runner=module('explore_utility_dose_feasibility')
    monkeypatch.setattr(runner,'sources',lambda:{'synthetic':{}})
    monkeypatch.setattr(os,'getpriority',lambda *_:0)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runner.ROOT,text=True).strip()
    with pytest.raises(ValueError,match='nice10'):runner.run(tmp_path/'out',head)
    assert not (tmp_path/'out').exists()
