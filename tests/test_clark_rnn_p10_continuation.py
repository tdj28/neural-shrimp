import importlib.util
from pathlib import Path
import numpy as np
import pytest
P=Path(__file__).resolve().parents[1]/'tools/continue_clark_rnn_new_p10.py'
s=importlib.util.spec_from_file_location('p10pal',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_total_crossing_keeps_moving_branch_term():
    second=np.zeros((4,6,6));second[0,0,0]=3.;second[0,0,4]=2.;second[0,0,5]=-4.
    xs=np.zeros((4,2));xs[0]=[5.,-2.];q=np.array([1.,0,0,0]);p=q.copy()
    np.testing.assert_array_equal(m.total_crossing(second,q,p,xs),[17.,-10.])

def test_composite_period_divisors_reject_two_and_five():
    a=np.arange(20).reshape(5,4).astype(float)
    assert m.B.B.primitive_separations(np.tile(a,(2,1)))['5']==0
    b=np.arange(8).reshape(2,4).astype(float)
    assert m.B.B.primitive_separations(np.tile(b,(5,1)))['2']==0

def test_domain_is_fixed_open_rectangle():
    y=np.zeros(10);y[4]=1.2;y[5]=4.5;assert m.in_domain(y)
    for g,d in [(8,.45),(20,.45),(12,.4),(12,.52)]:
        y[4]=g/10;y[5]=d/.1;assert not m.in_domain(y)

def test_failed_corrector_return_is_retained(tmp_path):
    w=m.Work(tmp_path);seen=[]
    def full(y,j,sign,tag):
        seen.append(y.copy());return np.ones(9),np.zeros((9,10)),{'endpoint':np.zeros(4),'theta':np.zeros(4)},0.
    w.full=full
    y=np.zeros(10);y[4]=1.2;y[5]=4.5
    point,status=w.correct(y,y,np.ones(10),np.eye(4),1,'synthetic')
    assert point is None and status=='linear_failure' and len(seen)==1
