import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).resolve().parents[1]/'tools/continue_clark_rnn_compact_p5.py'
s=importlib.util.spec_from_file_location('compactpal',p);M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def test_native_five_return_retained(tmp_path):
    w=M.Work(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v));y=np.r_[.1,-.2,.3,-.4,5.,.6,1.,0,0,0]/M.COL
    h,b,a,t=w.full(y,np.eye(4)*.2,1,'toy');assert a['cycle'].shape==(5,4) and b.shape==(9,10)
    assert events[0][0]=='family' and events[0][1]['cycle'].shape==(5,4)
def test_prime_period_guard_and_shape():
    assert M.proper_separations(np.zeros((5,4)))['1']==0
    with pytest.raises(ValueError):M.proper_separations(np.zeros((10,4)))
def test_moving_branch_total_flip_derivative():
    a=np.zeros((4,6,6));a[0,0,0]=3;a[0,0,4]=2;a[0,0,5]=-4;xs=np.zeros((4,2));xs[0]=[5,-2];q=np.array([1.,0,0,0])
    np.testing.assert_array_equal(M.total_crossing(a,q,q,xs),[17,-10])
def test_fixed_domain():
    y=np.r_[np.zeros(4),5.,.6,np.zeros(4)]/M.COL;assert M.in_domain(y)
    for g,d in [(3.8,.6),(8.,.6),(5.,.42),(5.,.85)]:
        y[4]=g/5;y[5]=d/.1;assert not M.in_domain(y)

def test_nonfinite_tangent_is_retained_and_rejected(tmp_path,monkeypatch):
    w=M.Work(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    monkeypatch.setattr(M,'oriented_null',lambda *a:(np.full(10,np.nan),{'s':np.ones(9),'null_residual':float('nan')}))
    with pytest.raises(ValueError,match='tangent'):w.tangent(np.zeros((9,10)),None,'nan')
    assert events[0][0]=='null_svd'

def test_nonfinite_corrector_return_rejected(tmp_path,monkeypatch):
    w=M.Work(tmp_path);monkeypatch.setattr(w,'full',lambda *a:(np.full(9,np.nan),np.zeros((9,10)),{'endpoint':np.zeros(4)},0.))
    with pytest.raises(ValueError,match='nonfinite'):w.correct(np.zeros(10),np.zeros(10),np.zeros(10),np.eye(4),1,'nan')

def test_nonfinite_eigenvectors_retained_and_rejected(tmp_path,monkeypatch):
    w=M.Work(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    monkeypatch.setattr(M.np.linalg,'eig',lambda x:(np.ones(4),np.full((4,4),np.nan)))
    with pytest.raises(ValueError,match='nonfinite'):w.node(np.zeros(10),np.zeros((9,10)),{'monodromy':np.eye(4)},np.zeros(10),1,'nan')
    assert [e[0] for e in events]==['node_eigen','node_left_eigen']
