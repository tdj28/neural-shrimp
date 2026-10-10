import importlib.util
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp.clark_rnn import iterate

def module():
    p=Path(__file__).resolve().parents[1]/'tools/explore_clark_rnn_n6_p7_boundaries.py'
    s=importlib.util.spec_from_file_location('p7_control',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_native_worker_uses_seven_in_six_dimensions_and_retains_return(tmp_path):
    m=module();w=m.Worker(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    x=np.array([.1,-.2,.3,-.4,.5,-.6]);j=np.eye(6)*.2
    a=w.it(x,j,2.,.7,'synthetic');b=iterate(x,j,2.,.7,7)
    np.testing.assert_array_equal(a[0],b[0]);assert a[2].shape==(7,6)
    assert events[0][0]=='native' and events[0][1]['cycle'].shape==(7,6)

def test_native_phase_and_odd_identity_rules():
    m=module();a=np.arange(42.).reshape(7,6)
    assert m.phase_match(a,a+1e-5)['passed']
    assert not m.phase_match(a,-a)['passed']
    assert not m.phase_match(a,np.roll(a,1,axis=0))['passed']

def test_interrupt_before_any_native_call(tmp_path):
    m=module();w=m.Worker(tmp_path);w.interrupted=True
    with pytest.raises(InterruptedError):w.it(np.zeros(6),np.eye(6),1.,.7,'synthetic')
    assert w.calls==0


def test_leading_complex_is_not_replaced_by_nearby_real_mode():
    m=module()
    assert m.leading_boundary_sign([1.01+.1j,1.01-.1j,.999,0.]) is None
    assert m.leading_boundary_sign([-1.02,.999,.2,.1])==-1
    assert m.leading_boundary_sign([1.02,-.999,.2,.1])==1
    assert m.leading_boundary_sign([np.nan,1.]) is None

def test_local_root_does_not_override_failed_reverse_association(tmp_path,monkeypatch):
    m=module();w=m.Worker(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    cycle=np.arange(42.).reshape(7,6);point={'theta':cycle[0].copy(),'cycle':cycle,'delta':.6,'eigenvalues':np.array([1.,.2,.3,.4,.5,.6]),'eigenvectors':np.eye(6)}
    w.aug=lambda *a:(np.zeros(13),np.eye(13),{},0.)
    monkeypatch.setattr(m,'qualify_boundary',lambda *a:({'theta':point['theta'],'cycle':cycle},{'status':'numerical_boundary_candidate'}))
    w.correct=lambda *a:({'theta':point['theta']+.01},'closed')
    result=w.localize(point,np.eye(6),5.,1,'toy')
    assert result['status']=='boundary_identity_unresolved' and result['reverse_error']>1e-7
    assert events[-1][0]=='boundary_result' and any(k=='boundary_reverse' for k,v in events)


def test_start_must_be_attracting_before_first_loss_claim():
    m=module();assert m.initial_attracting({'radius':.9})
    for rho in [1.,1.1,np.nan,np.inf,1-1e-7]:assert not m.initial_attracting({'radius':rho})


def test_proper_divisor_one_collapse_is_rejected(tmp_path):
    m=module();w=m.Worker(tmp_path);w.emit=lambda *a:None
    c=np.arange(42.).reshape(7,6);c[1]=c[0]
    w.it=lambda *a:(c[0],np.eye(6)*.2,c,np.tile(np.eye(6),(7,1,1)),0.)
    p,status=w.correct(c[0],np.eye(6),3.,.8,'synthetic');assert p is None and status=='collapsed'

def test_nonfinite_return_fails_closed(tmp_path):
    m=module();w=m.Worker(tmp_path);w.emit=lambda *a:None
    w.it=lambda *a:(np.full(6,np.nan),np.eye(6),np.ones((7,6)),np.ones((7,6,6)),0.)
    p,status=w.correct(np.ones(6),np.eye(6),3.,.8,'synthetic');assert p is None and status=='nonfinite_native'


def test_localization_uses_six_state_coordinates_and_delta_slot(tmp_path,monkeypatch):
    m=module();w=m.Worker(tmp_path);w.emit=lambda *a:None
    cycle=np.arange(42.).reshape(7,6);point={'theta':cycle[0].copy(),'cycle':cycle,'delta':.6,'eigenvalues':np.arange(6)/10+.2,'eigenvectors':np.eye(6)};point['eigenvalues'][0]=1.
    w.aug=lambda z,*a:(np.zeros(13),np.eye(13),{},0.)
    monkeypatch.setattr(m,'qualify_boundary',lambda z,*a:({'theta':z[:6],'cycle':cycle},{'status':'numerical_boundary_candidate'}))
    seen=[]
    def correct(x,j,g,d,tag):seen.append((x.copy(),d));return {'theta':x.copy()},'closed'
    w.correct=correct;r=w.localize(point,np.eye(6),7.,1,'synthetic')
    assert seen[0][0].shape==(6,) and seen[0][1]==.6 and r['delta']==.6 and r['status']=='numerical_boundary_candidate'

def test_nonfinite_augmented_fails_closed(tmp_path):
    m=module();w=m.Worker(tmp_path);w.emit=lambda *a:None
    p={'theta':np.ones(6),'cycle':np.ones((7,6)),'delta':.6,'eigenvalues':np.array([1.,.2,.3,.4,.5,.6]),'eigenvectors':np.eye(6)}
    w.aug=lambda *a:(np.full(13,np.nan),np.eye(13),{},0.)
    assert w.localize(p,np.eye(6),7.,1,'synthetic')['status']=='nonfinite_augmented'
