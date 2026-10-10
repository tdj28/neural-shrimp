import importlib.util
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp.clark_rnn import iterate

def module():
    p=Path(__file__).resolve().parents[1]/'tools/explore_clark_rnn_compact_p5_boundaries.py'
    s=importlib.util.spec_from_file_location('p7_control',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_native_worker_uses_five_and_retains_return(tmp_path):
    m=module();w=m.Worker(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    x=np.array([.1,-.2,.3,-.4]);j=np.eye(4)*.2
    a=w.it(x,j,2.,.7,'synthetic');b=iterate(x,j,2.,.7,5)
    np.testing.assert_array_equal(a[0],b[0]);assert a[2].shape==(5,4)
    assert events[0][0]=='native' and events[0][1]['cycle'].shape==(5,4)

def test_native_phase_and_odd_identity_rules():
    m=module();a=np.arange(20.).reshape(5,4)
    assert m.phase_match(a,a+1e-5)['passed']
    assert not m.phase_match(a,-a)['passed']
    assert not m.phase_match(a,np.roll(a,1,axis=0))['passed']

def test_interrupt_before_any_native_call(tmp_path):
    m=module();w=m.Worker(tmp_path);w.interrupted=True
    with pytest.raises(InterruptedError):w.it(np.zeros(4),np.eye(4),1.,.7,'synthetic')
    assert w.calls==0


def test_leading_complex_is_not_replaced_by_nearby_real_mode():
    m=module()
    assert m.leading_boundary_sign([1.01+.1j,1.01-.1j,.999,0.]) is None
    assert m.leading_boundary_sign([-1.02,.999,.2,.1])==-1
    assert m.leading_boundary_sign([1.02,-.999,.2,.1])==1
    assert m.leading_boundary_sign([np.nan,1.]) is None

def test_local_root_does_not_override_failed_reverse_association(tmp_path,monkeypatch):
    m=module();w=m.Worker(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    cycle=np.arange(20.).reshape(5,4);point={'theta':cycle[0].copy(),'cycle':cycle,'delta':.6,'eigenvalues':np.array([1.,.2,.3,.4]),'eigenvectors':np.eye(4)}
    w.aug=lambda *a:(np.zeros(9),np.eye(9),{},0.)
    monkeypatch.setattr(m,'qualify_boundary',lambda *a:({'theta':point['theta'],'cycle':cycle},{'status':'numerical_boundary_candidate'}))
    w.correct=lambda *a:({'theta':point['theta']+.01},'closed')
    result=w.localize(point,np.eye(4),5.,1,'toy')
    assert result['status']=='boundary_identity_unresolved' and result['reverse_error']>1e-7
    assert events[-1][0]=='boundary_result' and any(k=='boundary_reverse' for k,v in events)
