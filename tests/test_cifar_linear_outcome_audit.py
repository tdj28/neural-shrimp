"""Synthetic semantic-tamper controls for independent saved-array arithmetic."""
import importlib.util
from pathlib import Path
import math
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('cifar_outcome_audit',Path(__file__).parents[1]/'tools/audit_cifar_linear_outcome.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def fixture():
    r=np.array([-1.,-1.]);p=np.array([.6,.6]);ap=np.array([.6,1.2]);e=ap+r
    record=dict(rhs_norm=math.sqrt(2),true_linear_residual_norm=math.sqrt(.2),
      true_forcing_ratio=math.sqrt(.1),merit_directional_derivative=-1.8,
      forcing_test_pass=True,descent_direction=True,requested_tolerance_met=False)
    return r,p,ap,e,record


def test_independent_arithmetic_and_forced_linear_metadata_are_distinguished():
    r,p,ap,e,record=fixture()
    assert a.validate_linear(r,p,ap,e,record)['forcing_test_pass']
    for field,value in [('true_forcing_ratio',.01),('forcing_test_pass',False),('descent_direction',False)]:
        bad={**record,field:value}
        with pytest.raises(ValueError):a.validate_linear(r,p,ap,e,bad)
    with pytest.raises(ValueError,match='residual array'):a.validate_linear(r,p,ap,e+1,record)


def test_forged_trial_decision_and_chained_initialization_are_rejected():
    r,p,ap,e,record=fixture();linear=a.validate_linear(r,p,ap,e,record)
    theta=np.zeros(2);trial=p.copy();one=trial+.1;two=trial+e
    row=dict(alpha=1.,relative_step_norm=math.sqrt(.72),armijo_upper=.99982,
      residual_norm=math.sqrt(.2),relative_residual=math.sqrt(.2),merit=.1,merit_ratio=.1,
      relative_separation=math.sqrt(.02),closure_screen=False,primitive_screen=True,
      merit_decreased=True,armijo_pass=True,finite=True,inexact_step_candidate=True,
      accepted_orbit=False,applied_to_continuation=False)
    assert a.validate_trial(theta,r,p,ap,1.,trial,one,two,e,row,linear)['inexact_step_candidate']
    with pytest.raises(ValueError,match='accepted_orbit'):
        a.validate_trial(theta,r,p,ap,1.,trial,one,two,e,{**row,'accepted_orbit':True},linear)
    with pytest.raises(ValueError,match='base-state'):
        a.validate_trial(theta,r,p,ap,1.,trial+p,one,two,e,row,linear)
