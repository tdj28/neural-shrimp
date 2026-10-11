import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).parents[1]/'tools/complete_clark_rnn_n6_p7.py';s=importlib.util.spec_from_file_location('completion',p);M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def test_original_budget_only():
 t={'stop':'qualification_or_arithmetic: ValueError: primitive/simple boundary qualification','nodes':[{'qualified_primitive_simple_boundary':True}]*34}
 assert M.remaining_edges(t)==95
 t['stop']='node_cap'
 with pytest.raises(ValueError):M.remaining_edges(t)
def test_failed_node_not_admitted():
 t={'stop':'qualification_or_arithmetic: ValueError: primitive/simple boundary qualification','nodes':[{'qualified_primitive_simple_boundary':False}]}
 with pytest.raises(ValueError):M.remaining_edges(t)
def test_stricter_corrector_takes_another_step(tmp_path):
 w=M.Work(tmp_path);returns=[5e-11,0.];events=[];w.emit=lambda k,v:events.append((k,v))
 def full(y,*args):
  v=returns.pop(0);return np.r_[v,np.zeros(12)],np.eye(14)[:13],{'endpoint':np.zeros(6),'theta':np.zeros(6)},0.
 w.full=full;y=np.zeros(14);y[6]=7/5;y[7]=.6/.1;t=np.eye(14)[13]
 result,status=w.correct(y,y,t,np.eye(6),1,'synthetic');assert status=='closed' and not returns and len(events)==1
 assert events[0][0]=='pal_solve'
def test_domain_and_target_slots_unchanged():
 assert M.SEEDS==[('lower-fold',-1,1),('lower-fold',1,1),('upper-fold',1,1),('lower-flip',-1,-1)]
 y=np.zeros(14);y[6]=5.5/5;y[7]=.6/.1;assert not M.in_domain(y)
