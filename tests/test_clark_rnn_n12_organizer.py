import importlib.util,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('organizer',ROOT/'tools/continue_clark_rnn_n12_organizer.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)

def test_ordered_literal_phase_transport_uses_no_state_sign():
 rng=np.random.default_rng(472);cycle=rng.normal(size=(6,12));jac=np.eye(12)[None]+.02*rng.normal(size=(6,12,12));q=rng.normal(size=12);q/=np.linalg.norm(q)
 for k in range(6):
  x,v,p,n=C.phase_transport(cycle,jac,q,k)
  expected=np.eye(12)if k==0 else np.linalg.multi_dot([*reversed(jac[:k]),np.eye(12)])
  np.testing.assert_array_equal(x,cycle[k]);np.testing.assert_allclose(p,expected,atol=1e-14);np.testing.assert_allclose(v,expected@q/np.linalg.norm(expected@q),atol=1e-14)
 assert not np.allclose(jac[0]@jac[1],jac[1]@jac[0])

@pytest.mark.parametrize('fault',['shape','nonfinite','zero_mode','phase'])
def test_invalid_transport_cannot_supply_a_bridge(fault):
 cycle=np.arange(72.).reshape(6,12);jac=np.repeat(np.eye(12)[None],6,axis=0);q=np.eye(12)[0];k=1
 if fault=='shape':cycle=cycle[:5]
 elif fault=='nonfinite':jac[3,2,1]=np.nan
 elif fault=='zero_mode':jac[0]=0
 else:k=6
 with pytest.raises(ValueError):C.phase_transport(cycle,jac,q,k)

def test_mode_orientation_is_separate_from_native_state_identity():
 q=np.eye(12)[0];assert C.mode_join(q,-q)['passed']and C.mode_join(q,-q)['orientation']==-1
 assert not C.mode_join(q,np.eye(12)[1])['passed']
 cycle=np.arange(72.).reshape(6,12)+1
 assert not C.B.native_join(cycle,-cycle,.8,.8)['passed']
 assert not C.B.native_join(cycle,np.roll(cycle,1,axis=0),.8,.8)['passed']

def test_full_native_cycle_reverse_mode_in_linear_halfwave_map():
 # A real 60-degree rotation has F^3=-I and F^6=I: all six literal phases differ.
 angle=np.pi/3;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);a=np.kron(np.eye(6),r);x=np.arange(12.)+.2
 cycle=np.array([np.linalg.matrix_power(a,k)@x for k in range(6)]);jac=np.repeat(a[None],6,axis=0);q=np.eye(12)[0]
 for k in range(6):
  _,v,_,_=C.phase_transport(cycle,jac,q,k);back=np.linalg.matrix_power(a,6-k)@v;back/=np.linalg.norm(back)
  assert C.mode_join(q,back)['passed']
  assert sum(C.B.native_join(cycle,np.roll(cycle,-i,axis=0),.8,.8)['passed']for i in range(6))==1

def test_real_launcher_entrypoint_and_unchanged_caps():
 p=ROOT/'tools/launch_clark_rnn_n12_organizer.py';r=subprocess.run([sys.executable,str(p),'--help'],capture_output=True,text=True,timeout=10);assert r.returncode==0,r.stderr
 assert 'continue_clark_rnn_n12_organizer.py'in p.read_text();assert C.STEPS==16 and C.END_GAIN==4.0756 and C.J.SOLVE_CAP==1024
