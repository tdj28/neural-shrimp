import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).parents[1]/'tools/diagnose_cifar_smaller_alpha.py';s=importlib.util.spec_from_file_location('smaller',p);M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def test_fixed_nine_and_no_large_alpha():
 assert M.ALPHAS==tuple(2.**(-k)for k in range(4,13))
 with pytest.raises(ValueError):M.metrics(np.ones(2),np.ones(2),np.ones(2),.125,{'initial_merit':1.,'slope':-1.,'true_forcing_ratio':.2})
def test_forcing_rejection():
 with pytest.raises(ValueError):M.reductions(np.ones(2),np.zeros(2))
def test_armijo_uses_absolute_merit_not_state_denominator():
 red=M.reductions(np.array([1.,0.]),np.array([-1.,0.]));t=np.array([1000.,0.]);r=M.metrics(t,t+1,t+np.array([1.1,0.]),2**-4,red)
 assert r['relative_residual']<.002 and not r['armijo_pass'] and not r['inexact_step_candidate']
def test_decrease_does_not_accept_orbit():
 red=M.reductions(np.array([1.,0.]),np.array([-1.,0.]));r=M.metrics(np.zeros(2),np.ones(2),np.array([.5,0.]),2**-4,red)
 assert r['inexact_step_candidate'] and not r['accepted_orbit'] and not r['applied_to_continuation']
@pytest.mark.parametrize('kind',['hvp','j2vp','action'])
def test_no_derivative_or_solver_calls(kind):
 b=M.SmallBudget(lambda _:None)
 with pytest.raises(ValueError):b.call(kind,lambda:0)
def test_twenty_gradient_attempt_limit():
 b=M.SmallBudget(lambda _:None)
 for _ in range(20):assert b.call('direct_gradient',lambda:1)==1
 with pytest.raises(ValueError):b.call('direct_gradient',lambda:1)
def test_measured_gate_invalid_and_slow():
 for value in(float('nan'),float('inf'),2.):
  with pytest.raises(ValueError):M.measured_gate([dict(kind='operation_returned',operation='direct_gradient',seconds=value)]*11)
