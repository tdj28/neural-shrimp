import importlib.util
from pathlib import Path
import numpy as np

def module():
 p=Path(__file__).parents[1]/'tools/continue_clark_rnn_n12_daughter_leak.py';s=importlib.util.spec_from_file_location('daughter_leak',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_fixed_leak_tangent_and_profile_solve():
 m=module();j=np.eye(12)*.2+np.ones((12,12))*.025;x=np.arange(1,13)/20;q=np.eye(12)[0];z=np.r_[x,-x,.8];h,b,a=m.D.pair_fixed_amplitude(z,j,2.,.02,q)
 v=m.delta_tangent(b);assert max(abs(b[:24,:24]@v+b[:24,-1]))<1e-10
 correction=np.linalg.solve(b[:24,:24],-h[:24]);assert max(abs(b[:24,:24]@correction+h[:24]))<1e-10
 eps=1e-6
 fd=(m.D.pair_fixed_amplitude(z+eps*np.r_[v,1.],j,2.,.02,q)[0][:24]-m.D.pair_fixed_amplitude(z-eps*np.r_[v,1.],j,2.,.02,q)[0][:24])/(2*eps)
 assert max(abs(fd))<1e-7

def test_actual_delta_is_fixed_and_amplitude_is_a_diagnostic(tmp_path,monkeypatch):
 m=module();w=m.Work(tmp_path);q=np.eye(12)[0];target=np.r_[np.ones(12),-np.ones(12)];seen=[]
 def pair(z,j,g,amp,q,tag):
  seen.append((z[-1],amp));b=np.eye(25);h=np.r_[z[:24]-target,0.];return h,b,{},0.
 monkeypatch.setattr(w,'pair',pair);monkeypatch.setattr(w,'emit',lambda *a:None);anchor=np.r_[target+.001,.8];z,h,b,a,amp=w.correct_delta(anchor,anchor,np.eye(12),2.,.7998,q,'synthetic')
 assert all(v[0]==.7998 for v in seen)and z[-1]==.7998 and amp==1. and max(abs(h))==0
 assert m.STEP==-.0002 and m.STEPS==32
