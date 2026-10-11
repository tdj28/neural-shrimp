import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).resolve().parents[1]/'tools/explore_saved_lm_inference.py'
s=importlib.util.spec_from_file_location('saved_lm',p);M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def test_margin_safe():
    r=M.gap_metrics([[2.,1.,0.]],[[1.8,1.2,0.]])
    assert r['changed']==0 and r['measured_gap_sufficient']==1
def test_near_tie_changes():
    r=M.gap_metrics([[1.,.99,0.]],[[.98,1.,0.]])
    assert r['changed']==1 and r['measured_gap_sufficient']==0
def test_ties_are_not_certified():
    assert M.gap_metrics([[1.,1.]],[[1.,1.]])['measured_gap_sufficient']==0
@pytest.mark.parametrize('v',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite(v):
    with pytest.raises(ValueError):M.gap_metrics([[1.,v]],[[1.,0.]])
def test_first_divergence():
    assert M.first_difference(np.array([[1,2],[3,4],[5,6]]),np.array([[1,3],[4,4],[5,7]]))==[1,0]
def test_no_divergence():
    x=np.ones((32,8),int);assert M.first_difference(x,x)==[None]*8
