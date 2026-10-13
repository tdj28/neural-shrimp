"""Synthetic Task5 setup gates; no model checkpoint or producer tool is used."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / 'codex/confirm-fold/fold_setup.py'
SPEC = importlib.util.spec_from_file_location('codex_fresh_fold_setup', PATH)
setup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = setup
SPEC.loader.exec_module(setup)


class ChartCycle:
    """Primitive n-cycle, one quadratic fold leg; all remaining legs affine."""
    def __init__(self, target, *, b=3., actual_period=None, transverse=.4):
        self.target = target
        self.n = target.period if actual_period is None else actual_period
        self.b, self.transverse = b, transverse
        self.c = target.initial_parameter

    def prefix_cache(self, alpha):
        return 2*alpha

    def forward(self, x, alpha, temperature, cache=None):
        cache = self.prefix_cache(alpha) if cache is None else cache
        index = int(round(float(x[0].detach()-3)/10)) % self.n
        q = x[0] - (3+10*index)
        if index == 0:
            if self.target.coordinate == 'alpha':
                # Both current-token alpha and alpha-dependent prefix contribute.
                delta = alpha + cache - 3*self.c
            else:
                delta = 2*(10**temperature - 10**self.c)
            q = q + delta + self.b*q*q + .7*delta*q
        return torch.stack([3+10*((index+1)%self.n)+q, self.transverse*x[1]])

    def period_map(self,x,alpha,temperature,period,cache=None):
        for _ in range(period):
            x = self.forward(x,alpha,temperature,cache)
        return x

    def jvp(self,x,v,alpha,temperature,period=1,cache=None):
        return torch.func.jvp(lambda z:self.period_map(z,alpha,temperature,period,cache),(x,),(v,))[1]

    def jacobian(self,x,alpha,temperature,period=1,chunk_size=16,cache=None):
        return torch.func.jacfwd(lambda z:self.period_map(z,alpha,temperature,period,cache))(x)


@pytest.mark.parametrize('label', ['A','B'])
@pytest.mark.parametrize('b', [3.,-3.])
def test_actual_fold_setup_nested_ad_units_signs_and_complete_arrays(label,b):
    target=setup.TARGETS[label]
    model=ChartCycle(target,b=b)
    stages=[]
    result=setup.prepare_fold(model,np.array([3.000005,1e-7]),target,
                              parameter=target.initial_parameter+1e-6,
                              save=lambda name,payload:stages.append((name,payload)))
    assert result['passed'], result
    expected_a=3. if label=='A' else 2*np.log(10)*10**target.initial_parameter
    assert result['parameter_c']==pytest.approx(target.initial_parameter,abs=2e-12)
    assert result['raw_a']==pytest.approx(expected_a,rel=1e-9)
    assert result['raw_b']==pytest.approx(b,rel=1e-9)
    assert result['a_positive']==pytest.approx(expected_a,rel=1e-9)
    assert result['b_positive']==abs(b)
    assert result['ghost_parameter_direction']==np.sign(b)
    assert result['r'] @ result['l']==pytest.approx(1.)
    assert result['r'][0]==np.sign(b)
    assert result['matrix'].shape==(2,2)
    assert result['orbitpoints'].shape==(target.period+1,2)
    assert result['spectrum']['next_modulus']==pytest.approx(.4**target.period)
    assert len(result['orbit_diagnostics']['proper_divisor_returns'])==(1 if label=='A' else 3)
    assert result['spectrum']['critical_modulus']==pytest.approx(1.)
    assert result['spectrum']['critical_argument_turns']==0.
    assert result['operation_counts']['critical_gradient_ad']>0
    assert stages[-1][0].startswith('setup-result-')
    assert result['orbit_diagnostics']['numerically_attracting'] is None
    # Every equation in the saved quadratic model uses return-map iterates.
    assert result['target']['period']==target.period


def test_log_temperature_nested_mixed_derivative_and_prefix_derivative():
    for label in ('A','B'):
        target=setup.TARGETS[label]
        ops=setup.Operations(ChartCycle(target),target)
        p=target.initial_parameter
        x=np.array([3.,0.]); r=np.array([1.,0.]); l=r.copy()
        expected=3. if label=='A' else 2*np.log(10)*10**p
        assert ops.parameter_derivative(x,p)[0]==pytest.approx(expected)
        gx,gp=ops.critical_grad(x,p,l,r)
        assert gx[0]==pytest.approx(6.)
        assert gp==pytest.approx(.7*expected)
        assert ops.second(x,r,p)[0]==pytest.approx(6.)


@pytest.mark.parametrize('matrix,message', [
    (np.diag([-1.,.2]),'real \\+1'),
    (np.array([[0.,-1.],[1.,0.]]),'real \\+1'),
    (np.diag([1.,.97]),'complementary'),
    (np.diag([1.,1.]),'complementary'),
    (np.diag([1.,-1.]),'complementary'),
    (np.diag([1.,1.1]),'complementary'),
])
def test_wrong_or_ambiguous_crossings_refused(matrix,message):
    with pytest.raises(ValueError,match=message):
        setup.classify_fold(matrix)


def test_lower_period_alias_refused_before_normal_form():
    target=setup.TARGETS['B']
    result=setup.prepare_fold(ChartCycle(target,actual_period=2),np.array([3.,0.]),target)
    assert not result['passed']
    assert result['stage']=='orbit-gates-failed'
    assert not result['gates']['primitive']
    assert 'raw_a' not in result
    assert 'nested_state_ad' not in result['operation_counts']


def test_complement_gate_precedes_coefficient_calculation():
    target=setup.TARGETS['A']
    result=setup.prepare_fold(ChartCycle(target,transverse=.99),np.array([3.,0.]),target)
    assert not result['passed']
    assert result['stage']=='crossing-classification-failed'
    assert 'nested_state_ad' not in result['operation_counts']


def test_bounds_stop_wrong_branch_and_nonfinite_state():
    ops=setup.Operations(ChartCycle(setup.TARGETS['A']),setup.TARGETS['A'])
    with pytest.raises(ValueError,match='bounds'):
        ops.period_map(np.array([3.,0.]),.0419)
    with pytest.raises(ValueError,match='finite'):
        setup.prepare_fold(ChartCycle(setup.TARGETS['A']),np.array([np.nan,0.]),setup.TARGETS['A'])


def test_saved_setup_schema_roundtrip_and_seed_selection(tmp_path):
    result=setup.prepare_fold(ChartCycle(setup.TARGETS['A']),np.array([3.,0.]),setup.TARGETS['A'])
    setup.persist(tmp_path,'setup',result)
    doc=json.loads((tmp_path/'setup.json').read_text())
    assert doc['passed']
    assert doc['x_c']['key']=='x_c'
    assert doc['x_c']['file']=='setup.npz'
    assert doc['array_file_sha256']==setup.sha256(tmp_path/'setup.npz')
    arrays=np.load(tmp_path/'setup.npz',allow_pickle=False)
    for key in ('x_c','r','l','raw_r','raw_l','matrix','orbitpoints','parameter_derivative','directional_second'):
        assert key in arrays
    assert np.array_equal(setup.load_seed(tmp_path/'setup.npz',key='x_c'),result['x_c'])
    with pytest.raises(ValueError,match='explicit'):
        setup.load_seed(tmp_path/'setup.npz')
    np.save(tmp_path/'seeds.npy',np.array([[3.,0.],[13.,0.]]))
    with pytest.raises(ValueError,match='one-dimensional'):
        setup.load_seed(tmp_path/'seeds.npy')
    assert np.array_equal(setup.load_seed(tmp_path/'seeds.npy',index=1),[13.,0.])


def test_fd_gate_cannot_pass_through_averaging_opposite_errors():
    table=[dict(a=1.0015),dict(a=.9985)]
    assert np.mean([row['a'] for row in table])==1.
    assert not setup.fd_agrees(table,'a',1.,1e-3,1e-10)
    assert setup.fd_agrees([dict(a=1.0001),dict(a=.9999)],'a',1.,1e-3,1e-10)
    assert not setup.fd_agrees([dict(a=-1e-12),dict(a=1e-12)],'a',1e-12,1.,1e-10)


def test_source_snapshot_commit_comes_only_from_validated_launcher_environment(monkeypatch):
    monkeypatch.setenv('TASK5_SOURCE_COMMIT','A'*40)
    assert setup.source_commit_from_environment()=='a'*40
    for invalid in ('','a'*39,'g'*40,'a'*40+'\n'):
        monkeypatch.setenv('TASK5_SOURCE_COMMIT',invalid)
        with pytest.raises(ValueError,match='40-hex'):
            setup.source_commit_from_environment()
    monkeypatch.delenv('TASK5_SOURCE_COMMIT')
    with pytest.raises(ValueError,match='40-hex'):
        setup.source_commit_from_environment()
