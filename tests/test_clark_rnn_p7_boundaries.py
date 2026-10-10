import importlib.util
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp.clark_rnn import iterate

def module():
    p=Path(__file__).resolve().parents[1]/'tools/explore_clark_rnn_p7_boundaries.py'
    s=importlib.util.spec_from_file_location('p7_control',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_native_worker_uses_seven_and_retains_return(tmp_path):
    m=module();w=m.Worker(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    x=np.array([.1,-.2,.3,-.4]);j=np.eye(4)*.2
    a=w.it(x,j,2.,.7,'synthetic');b=iterate(x,j,2.,.7,7)
    np.testing.assert_array_equal(a[0],b[0]);assert a[2].shape==(7,4)
    assert events[0][0]=='native' and events[0][1]['cycle'].shape==(7,4)

def test_native_phase_and_odd_identity_rules():
    m=module();a=np.arange(28.).reshape(7,4)
    assert m.phase_match(a,a+1e-5)['passed']
    assert not m.phase_match(a,-a)['passed']
    assert not m.phase_match(a,np.roll(a,1,axis=0))['passed']

def test_interrupt_before_any_native_call(tmp_path):
    m=module();w=m.Worker(tmp_path);w.interrupted=True
    with pytest.raises(InterruptedError):w.it(np.zeros(4),np.eye(4),1.,.7,'synthetic')
    assert w.calls==0

def test_p7_pal_native_period_and_seed_schema(tmp_path):
    p=Path(__file__).resolve().parents[1]/'tools/continue_clark_rnn_p7.py'
    s=importlib.util.spec_from_file_location('p7_pal_control',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    w=m.Work(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v))
    y=np.r_[.1,-.2,.3,-.4,2.,.7,1.,0.,0.,0.]/m.COL
    h,b,a,t=w.full(y,np.eye(4)*.2,1,'synthetic')
    assert a['cycle'].shape==(7,4) and b.shape==(9,10)
    assert events[0][0]=='family' and events[0][1]['cycle'].shape==(7,4)
