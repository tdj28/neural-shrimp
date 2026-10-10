import importlib.util
from pathlib import Path
import pytest
import numpy as np

def module():
 p=Path(__file__).parents[1]/'tools/diagnose_cifar_large_space.py';s=importlib.util.spec_from_file_location('large_space',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_prefix_all200_then_continue400():
 m=module();events=[];p=m.Prefix400(np.ones(200),events.append)
 for i in range(400):p({'kind':'gmres_inner','iteration':i+1,'estimated_relative_residual':1 if i<200 else .5})
 p.complete();assert len(events)==400
 with pytest.raises(ValueError):p({'kind':'gmres_inner','iteration':401,'estimated_relative_residual':.4})

@pytest.mark.parametrize('index',[0,49,50,199])
def test_first200_mismatch_retained(index):
 m=module();events=[];p=m.Prefix400(np.ones(200),events.append)
 for i in range(index):p({'kind':'gmres_inner','iteration':i+1,'estimated_relative_residual':1})
 with pytest.raises(ValueError,match='prefix mismatch'):p({'kind':'gmres_inner','iteration':index+1,'estimated_relative_residual':.1})
 assert len(events)==index+1

def test_incomplete_prefix_rejected():
 m=module();p=m.Prefix400(np.ones(200),lambda row:None)
 for i in range(199):p({'kind':'gmres_inner','iteration':i+1,'estimated_relative_residual':1})
 with pytest.raises(ValueError):p.complete()

def test_attempted_calls_consume_cap_and_log():
 m=module();events=[];b=m.Budget400(events.append);b.attempted['action']=401
 def fail():raise ArithmeticError('failure')
 with pytest.raises(ArithmeticError):b.call('action',fail)
 assert b.attempted['action']==402 and b.returned['action']==0
 with pytest.raises(ValueError):b.call('action',lambda:0)

def test_settings_keep_scientific_gates():
 m=module();assert m.SETTINGS['forcing_limit']==.5 and m.SETTINGS['rtol']==1e-10 and m.SETTINGS['armijo']==1e-4 and m.SETTINGS['alphas']==[1.,.5,.25,.125]
 assert m.CAPS==dict(action=402,j2vp=403,hvp=807,direct_gradient=11)

@pytest.mark.parametrize('bad',['none','nan','short','oversized'])
def test_cost_gate_is_finite_measured_and_bounded(bad):
 m=module();r=dict(source_commit=m.EXECUTION,status='diagnostic_complete',comparison_qualified=True,accepted_orbit=False,settings=dict(restart=200,cycles=1),linear=dict(inner_iterations=200,matvec_calls=202,seconds=12.),elapsed_seconds=14.)
 events=[dict(kind='operation_returned',operation='action',seconds=.04)for _ in range(202)]
 if bad=='nan':events[0]['seconds']=float('nan')
 if bad=='short':events.pop()
 if bad=='oversized':r['linear']['seconds']=r['elapsed_seconds']=40.
 if bad=='none':assert m.measured_gate(r,events)['safety_projected_seconds']<90
 else:
  with pytest.raises(ValueError):m.measured_gate(r,events)
