import importlib.util
from pathlib import Path
import numpy as np
import pytest

def module():
 p=Path(__file__).parents[1]/'tools/continue_clark_rnn_compact_p12.py';s=importlib.util.spec_from_file_location('p12pal',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_native_full_twelve_and_all_divisors(tmp_path):
 m=module();w=m.Work(tmp_path);w.emit=lambda *a:None;y=np.r_[np.arange(1,5)/10,3.5,.8,[1.,0,0,0]]/m.COL
 h,b,a,_=w.full(y,np.eye(4)*.2,1,'synthetic');assert a['cycle'].shape==(12,4)and set(m.proper_separations(a['cycle']))=={'1','2','3','4','6'}

def test_native_association_collapse_not_fold(tmp_path):
 m=module();w=m.Work(tmp_path);w.emit=lambda *a:None;c=np.arange(48.).reshape(12,4);c[6]=c[0];w.it=lambda *a:(c[0],np.eye(4)*.2,c,np.ones((12,4,4)),0.)
 q,s=m.native_correct(w,c[0],np.eye(4),3.5,.8,'toy');assert q is None and s=='collapsed'

def test_domain_and_eigenmode_finite_guard(tmp_path):
 m=module();y=np.r_[np.ones(4),3.5,.8,np.ones(4)]/m.COL;assert m.in_domain(y);y[4]=5.61/5;assert not m.in_domain(y)
 w=m.Work(tmp_path);w.emit=lambda *a:None
 with pytest.raises(ValueError):w.tangent(np.full((9,10),np.nan),None,'toy')

def test_parameter_total_crossing_contains_branch_motion():
 m=module();second=np.zeros((4,6,6));second[0,0,0]=2.;second[0,0,4]=3.;q=np.array([1.,0,0,0]);xs=np.zeros((4,2));xs[0]=[5.,7.]
 np.testing.assert_array_equal(m.total_crossing(second,q,q,xs),[13.,14.])
