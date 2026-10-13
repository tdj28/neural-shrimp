"""Fold-only preparation tests use analytic synthetic charts, never checkpoints."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'codex/hostile-fold-crosscheck'


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


W = load('task7_fold_worker_test', TASK/'fold_worker.py')
C = load('task7_existing_synthetic_fold_chart', ROOT/'tests/test_codex_fresh_fold.py')


def run(model, seed=(3.,0.), **kwargs):
    stages=[]
    result=W.qualify_fold(model,np.asarray(seed,dtype=np.float64),
        save=lambda name,payload:stages.append((name,payload)),check=lambda:None,**kwargs)
    return result,stages


@pytest.mark.parametrize('b', [3., -3.])
def test_actual_corrector_nested_ad_full_gates_gauge_units_and_clock(b):
    target=W.target()
    result,stages=run(C.ChartCycle(target,b=b),seed=(3.000005,1e-7))
    assert result['passed'], result
    assert result['solver_history_complete'] is True
    expected_a=2*np.log(10)*10**target.initial_parameter
    assert result['parameter_c']==pytest.approx(target.initial_parameter,abs=2e-12)
    assert result['raw_a']==pytest.approx(expected_a,rel=1e-9)
    assert result['raw_b']==pytest.approx(b,rel=1e-9)
    assert result['a']==pytest.approx(np.sign(b)*expected_a,rel=1e-9)
    assert result['b']==pytest.approx(abs(b),rel=1e-9)
    assert result['ab']==pytest.approx(expected_a*b,rel=1e-9)
    assert result['absolute_ab']==pytest.approx(abs(expected_a*b),rel=1e-9)
    assert result['asymptotic_inverse_passage_coefficient']==pytest.approx(np.sqrt(abs(expected_a*b))/np.pi)
    assert result['ghost_parameter_direction']==np.sign(b)
    assert result['r']@result['l']==pytest.approx(1.)
    assert np.linalg.norm(result['r'])==pytest.approx(1.)
    assert result['clock']['forward_steps_per_return']==2
    assert result['active_parameter_units']=='log10T'
    assert result['matrix'].shape==(2,2) and result['orbitpoints'].shape==(3,2)
    assert result['spectrum']['left_eigenvectors'].shape==(2,2)
    assert result['spectrum']['eigenvectors'].shape==(2,2)
    assert result['spectrum']['critical_modulus']==pytest.approx(1.)
    assert result['spectrum']['next_modulus']==pytest.approx(.16)
    assert result['orbit_diagnostics']['numerically_attracting'] is None
    assert list(result['orbit_diagnostics']['proper_divisor_returns'])==[1]
    assert len(result['phase_returns'])==2
    assert result['phase_return_points'].shape==(2,2)
    np.testing.assert_allclose(result['phase_returns'],[np.linalg.norm(y-x)/np.linalg.norm(x)
        for x,y in zip(result['orbitpoints'][:-1],result['phase_return_points'])],rtol=0,atol=0)
    assert all(result['gates'].values())
    assert len(result['composed_precision'])==2
    for row in result['composed_precision']:
        assert len(row['finite_difference'])==3 and row['jvp'].shape==(2,)
    assert len(result['parameter_fd'])==len(result['state_fd'])==3
    assert result['operation_counts']['critical_gradient_ad']>0
    assert result['certificate_passed'] is False
    assert result['passage_trajectories_evaluated'] is False
    assert stages[-1][0].startswith('fold-result-')
    assert sum(name.startswith('monodromy-') and 'matrix' in payload for name,payload in stages)>=2


class Nonnormal(C.ChartCycle):
    # Chart coordinates y=(x0-2*x1,x1); physical right=(1,0), left=(1,-2).
    def forward(self,x,alpha,temperature,cache=None):
        y=torch.stack((x[0]-2*x[1],x[1]))
        z=super().forward(y,alpha,temperature,cache=cache)
        return torch.stack((z[0]+2*z[1],z[1]))


def test_nonnormal_left_gauge_and_all_full_arrays_persist(tmp_path):
    result,_=run(Nonnormal(W.target(),b=-3.))
    assert result['passed']
    np.testing.assert_allclose(result['raw_l'],[1.,-2.],atol=1e-12)
    setup,_=W.dependencies();setup.persist(tmp_path,'fold',result)
    doc=json.loads((tmp_path/'fold.json').read_text())
    assert doc['array_file_sha256']==setup.sha256(tmp_path/'fold.npz')
    with np.load(tmp_path/'fold.npz',allow_pickle=False) as archive:
        for key in ('matrix','orbitpoints','phase_return_points','r','l','raw_r','raw_l','parameter_derivative','directional_second',
                    'spectrum/eigenvalues','spectrum/eigenvectors','spectrum/left_eigenvectors',
                    'composed_precision/0/finite_difference/0/derivative'):
            assert key in archive
        np.testing.assert_array_equal(archive['l'],result['l'])


@pytest.mark.parametrize('kind', ['lower_period','complement','nonconverged'])
def test_failed_prerequisite_stops_before_any_normal_form(kind):
    options={}
    seed=(3.,0.)
    if kind=='lower_period':options['actual_period']=1
    if kind=='complement':options['transverse']=.99
    if kind=='nonconverged':seed=(3.01,0.)
    result,_=run(C.ChartCycle(W.target(),**options),seed=seed,max_newton=0)
    assert not result['passed'] and 'raw_a' not in result
    assert 'nested_state_ad' not in result['operation_counts']
    assert result['stage']=={'lower_period':'orbit-gates-failed',
        'complement':'crossing-classification-failed','nonconverged':'fold-corrector-failed'}[kind]


class WrongComposedJVP(C.ChartCycle):
    def jvp(self,x,v,alpha,temperature,period=1,cache=None):
        answer=super().jvp(x,v,alpha,temperature,period,cache)
        return answer+.01*v if period==2 else answer


def test_composed_precision_failure_prevents_coefficient_calculation():
    result,_=run(WrongComposedJVP(W.target()))
    assert result['stage']=='composed-precision-failed'
    assert not result['gates']['composed_precision']
    assert not result['gates']['critical_multiplier_jvp']
    assert 'raw_a' not in result and 'nested_state_ad' not in result['operation_counts']
    assert len(result['composed_precision'])==2


class OpposingParameterFD(C.ChartCycle):
    def period_map(self,x,alpha,temperature,period,cache=None):
        result=super().period_map(x,alpha,temperature,period,cache)
        if not isinstance(temperature,torch.Tensor):
            delta=float(temperature)-self.c
            for h,relative_error in ((2e-6,.003),(7e-7,-.003)):
                if abs(abs(delta)-h)<1e-12:
                    a=2*np.log(10)*10**self.c
                    result=result+torch.tensor([delta*a*relative_error,0.],dtype=torch.float64)
        return result


def test_each_parameter_fd_row_must_agree_even_if_average_cancels():
    result,_=run(OpposingParameterFD(W.target()))
    assert result['stage']=='coefficient-gates-failed'
    assert not result['gates']['parameter_ad_fd']
    values=[row['a'] for row in result['parameter_fd'][-2:]]
    assert np.mean(values)==pytest.approx(result['raw_a'],rel=1e-8)
    assert all(abs(v-result['raw_a'])>1e-3*abs(result['raw_a']) for v in values)


def test_early_headroom_failure_touches_no_model():
    class NoModel:
        def __getattr__(self,key):raise AssertionError('model accessed')
    def stop():raise RuntimeError('headroom')
    with pytest.raises(RuntimeError,match='headroom'):
        W.qualify_fold(NoModel(),np.zeros(2),save=lambda *args:None,check=stop)


def test_preparation_skeleton_cannot_become_a_runtime_plan():
    skeleton=json.loads((TASK/'fold-plan-skeleton.json').read_text())
    assert W.validate_skeleton(skeleton)['execution_status']=='not_admitted_pending_measured_profile'
    for key,value in (('profile_receipt',{'sha256':'a'*64}),('profile_review','approved'),
                      ('execution_status','approved')):
        bad=dict(skeleton);bad[key]=value
        with pytest.raises(ValueError):W.validate_skeleton(bad)
    bad=dict(skeleton,resources=dict(skeleton['resources'],wall_seconds=1800))
    with pytest.raises(ValueError,match='wall cap'):W.validate_skeleton(bad)


def test_solver_trial_error_retained_without_invented_history():
    class BadPeriod(C.ChartCycle):
        def period_map(self,x,alpha,temperature,period,cache=None):
            return torch.full_like(x,float('nan'))
    stages=[]
    with pytest.raises(ValueError,match='nonfinite'):
        W.qualify_fold(BadPeriod(W.target()),np.array([3.,0.]),
            save=lambda n,v:stages.append((n,v)),check=lambda:None)
    error=next(v for n,v in stages if n.startswith('corrector-trial-error-'))
    np.testing.assert_array_equal(error['state'],[3.,0.])
    assert error['solver_history_complete'] is False
    assert any(n.startswith('corrector-trial-start-')for n,v in stages)
    assert not any(n.startswith('corrector-001')for n,v in stages)
