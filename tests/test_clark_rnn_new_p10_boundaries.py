import importlib.util
from pathlib import Path
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('p10boundary',Path(__file__).parents[1]/'tools/explore_clark_rnn_new_p10_boundaries.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def test_composite_period_checks_all_proper_divisors():
    cycle=np.tile([[1.,2.,3.,4.],[-1.,-2.,-3.,-4.]],(5,1))
    v=M.primitive_separations(cycle)
    assert set(v)=={'1','2','5'} and v['1']>1 and v['2']==0 and v['5']>1
def test_only_declared_component_cells_seed_the_slice():
    c={'id':80,'period':10,'count':34,'grid_edge_margin':3,'cells':[42*51+9,42*51+10,44*51+12]}
    assert M.component_seed_indices(c,42).tolist()==[9,10]
    c['id']=0
    with pytest.raises(ValueError):M.component_seed_indices(c,42)
