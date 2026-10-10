import importlib.util
from pathlib import Path
import pytest
import numpy as np
from flint import arb,arb_mat,ctx
from neural_shrimp.correlated_cocycle_budget import decompose
from neural_shrimp.rigorous_correlated_cocycle import cocycle
from neural_shrimp.rigorous_connection import restore_ball


def test_saved_decomposition_needs_no_floating_decomposition(monkeypatch):
    def forbidden(*a,**k):pytest.fail('no new decomposition allowed')
    for name in ('solve','inv','eig','eigh','svd','norm','cond','cholesky'):monkeypatch.setattr(np.linalg,name,forbidden)
    with ctx.workprec(256):
        cs=[arb_mat([['1/2']]),arb_mat([['1/100']]),arb_mat([[0]])];e=arb_mat([[arb(0,'1/1000')]])
        result=decompose(cs,e,arb_mat([[1]]),arb_mat([[1]]),[{'phase':0,'coefficients':cs,'remainder':e,'center_J':cs[0],'scaled_Jprime':cs[1]}])
        assert result['whitened_full_check']['passed']
        assert restore_ball(result['zero_phase_error_counterfactual_truncation_inf'])==0
        assert restore_ball(result['phases'][0]['center_J_uncertainty_inf'])==0


def test_remainder_failure_separates_from_polynomial_variation():
    with ctx.workprec(256):
        cs=[arb_mat([['1/2']]),arb_mat([[0]]),arb_mat([[0]])];e=arb_mat([[arb(0,2)]])
        r=decompose(cs,e,arb_mat([[1]]),arb_mat([[1]]),[{'phase':0,'coefficients':cs,'remainder':e,'center_J':cs[0],'scaled_Jprime':cs[1]}])
        assert r['whitened_polynomial_only_check']['passed'] and not r['whitened_full_check']['passed']


def test_native_truncation_counterfactual_keeps_high_terms():
    with ctx.workprec(256):
        phase=[arb_mat([['1/2']]),arb_mat([['1/2']]),arb_mat([[0]])]
        rows=[{'phase':j,'coefficients':phase,'remainder':arb_mat([[0]]),'center_J':phase[0],'scaled_Jprime':phase[1]} for j in range(4)]
        coeff,error=cocycle([(phase,arb_mat([[0]]))]*4)
        r=decompose(coeff,error,arb_mat([[1]]),arb_mat([[1]]),rows)
        assert restore_ball(r['zero_phase_error_counterfactual_truncation_inf'])>0


def test_cap_and_interrupt():
    with pytest.raises(ValueError):decompose([],arb_mat([[0]]),arb_mat([[1]]),arb_mat([[1]]),[])
    cs=[arb_mat([['1/2']]),arb_mat([[0]]),arb_mat([[0]])]
    def stop():raise TimeoutError('synthetic')
    with pytest.raises(TimeoutError):decompose(cs,arb_mat([[0]]),arb_mat([[1]]),arb_mat([[1]]),[{}],stop)


def test_launcher_names_and_fixed_budget_are_current():
    source=(Path(__file__).parents[1]/'tools/launch_s2_period18_cocycle_budget.py').read_text()
    assert 'HEAD:tools/launch_s2_period18_cocycle_budget.py' in source
    assert "'20s'" in source and "'external_seconds':20" in source and 'timeout=27' in source
    assert 'launch_s2_period18_slice_budget.py' not in source
