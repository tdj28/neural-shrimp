import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).resolve().parents[1]/'tools/continue_clark_rnn_n6_p7.py'
s=importlib.util.spec_from_file_location('compactpal',p);M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def test_native_seven_return_retained(tmp_path):
    w=M.Work(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v));y=np.r_[.1,-.2,.3,-.4,.5,-.6,7.,.6,1.,0,0,0,0,0]/M.COL
    h,b,a,t=w.full(y,np.eye(6)*.2,1,'toy');assert a['cycle'].shape==(7,6) and b.shape==(13,14)
    assert events[0][0]=='family' and events[0][1]['cycle'].shape==(7,6)
def test_prime_period_guard_and_shape():
    assert M.proper_separations(np.zeros((7,6)))['1']==0
    with pytest.raises(ValueError):M.proper_separations(np.zeros((14,6)))
def test_moving_branch_total_flip_derivative():
    a=np.zeros((6,8,8));a[0,0,0]=3;a[0,0,6]=2;a[0,0,7]=-4;xs=np.zeros((6,2));xs[0]=[5,-2];q=np.array([1.,0,0,0,0,0])
    np.testing.assert_array_equal(M.total_crossing(a,q,q,xs),[17,-10])
def test_fixed_domain():
    y=np.r_[np.zeros(6),7.,.6,np.zeros(6)]/M.COL;assert M.in_domain(y)
    for g,d in [(5.5,.6),(12.,.6),(7.,.48),(7.,.67)]:
        y[6]=g/5;y[7]=d/.1;assert not M.in_domain(y)

def test_nonfinite_tangent_is_retained_and_rejected(tmp_path,monkeypatch):
    w=M.Work(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    monkeypatch.setattr(M,'oriented_null',lambda *a:(np.full(14,np.nan),{'s':np.ones(13),'null_residual':float('nan')}))
    with pytest.raises(ValueError,match='tangent'):w.tangent(np.zeros((13,14)),None,'nan')
    assert events[0][0]=='null_svd'

def test_nonfinite_corrector_return_rejected(tmp_path,monkeypatch):
    w=M.Work(tmp_path);monkeypatch.setattr(w,'full',lambda *a:(np.full(13,np.nan),np.zeros((13,14)),{'endpoint':np.zeros(6)},0.))
    with pytest.raises(ValueError,match='nonfinite'):w.correct(np.zeros(14),np.zeros(14),np.zeros(14),np.eye(6),1,'nan')

def test_nonfinite_eigenvectors_retained_and_rejected(tmp_path,monkeypatch):
    w=M.Work(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    monkeypatch.setattr(M.np.linalg,'eig',lambda x:(np.ones(6),np.full((6,6),np.nan)))
    with pytest.raises(ValueError,match='nonfinite'):w.node(np.zeros(14),np.zeros((13,14)),{'monodromy':np.eye(6)},np.zeros(14),1,'nan')
    assert [e[0] for e in events]==['node_eigen','node_left_eigen']


def test_original_failure_requires_separate_passed_association():
    meta={'native_identity':{'passed':True},'status':'boundary_identity_unresolved','reverse_status':'step_guard'}
    assert not M.seed_qualified(meta,None)
    assoc={'status':'numerically_associated_fold','parent_check':{'passed':True},'attempts':[{'association':{'passed':True}}]}
    assert M.seed_qualified(meta,assoc)
    meta['native_identity']['passed']=False
    assert not M.seed_qualified(meta,assoc)

def test_observer_failure_not_swallowed_as_numerical(tmp_path,monkeypatch):
    w=M.Work(tmp_path)
    monkeypatch.setattr(M.B.Worker,'emit',lambda *a:(_ for _ in ()).throw(ValueError('storage')))
    with pytest.raises(M.DurabilityError):w.emit('test',{})
