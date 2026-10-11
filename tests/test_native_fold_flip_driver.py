import sys,shutil,subprocess,os,json
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import localize_native_fold_flip as D

def test_complete_actual_metadata_preflight_without_target_arrays(monkeypatch):
 class W:
  def check(self):pass
  def save(self,*a):pass
 old=D.read
 def read(p):
  assert p.name not in ('target-inputs.json','target-certificate.json.gz','independent512.json','independent470-anchor.json')
  return old(p)
 monkeypatch.setattr(D,'read',read);card=dict(old(ROOT/D.CARD),status='reviewed');assert D.binding(W(),card,metadata_only=True)==D.D.OUT.parent

def test_isolated_archive_closure_and_launcher_same_path(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code="import sys;from pathlib import Path;sys.path.insert(0,'tools');import localize_native_fold_flip as d;import launch_native_fold_flip as l;r=Path.cwd();assert d.OUT.parent==l.OUT;a={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert a<=set(d.FILES),a-set(d.FILES)"
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)

def test_full_unrelated_profile_and_inclusive_count(tmp_path,monkeypatch):
 monkeypatch.setattr(D.C,'OUT',tmp_path);monkeypatch.setattr(D.C,'SETTINGS',D.SETTINGS);w=D.C.Work();D.profile(w);r=D.read(tmp_path/'profile.json')
 assert r['passed']and r['target_arrays_unopened']and w.calls==432
 assert D.SETTINGS['native_calls']==432+65*36+18*18
 assert len(list((tmp_path/'events').glob('*native_input*')))==432

def test_complete_localization_retains_adjacent_and_both_full_reverse(monkeypatch):
 seed=np.r_[np.zeros(13),70.,.8,1.,np.zeros(12),0.,1.,np.zeros(11)]
 orbit=np.zeros((19,13));orbit[1:18,2]=np.arange(1,18)*.01;qs=np.tile(seed[15:28],(19,1));ws=np.tile(seed[28:],(19,1));M=np.diag([1.,-1.]+[.5]*11);data=dict(orbit=orbit,q_phases=qs,w_phases=ws,monodromy=M)
 def aug(m,y,p,tick,observe):return np.zeros(41),np.eye(41),data
 def sig(m,z,h,p,tick,observe):return np.zeros(27),np.eye(27),np.zeros(27),dict(orbit=orbit,modes=ws,monodromy=M)
 monkeypatch.setattr(D,'augmented',aug);monkeypatch.setattr(D,'evaluate',sig)
 z=np.r_[seed[:13],seed[14],seed[28:]];flat=D.flat(z,70.,dict(orbit=orbit,modes=ws,monodromy=M));targets=[dict(index=i,hidden=70.,z=z.tolist(),native_flat=flat.tolist())for i in(15,16)]
 class W:
  def __init__(self):self.events=[]
  def tick(self):pass
  def observe(self,k,v):self.events.append((k,v))
 w=W();r=D.localize(type('M',(),{'n':13})(),seed,targets,w);assert r['passed']and r['adjacent']['passed']and len(r['reverse'])==2 and all(v['coordinates']==470 for v in r['reverse']);assert len([1 for k,v in w.events if k=='adjacent_return'])==1
 targets[1]['native_flat'][31]+=.01
 with pytest.raises(ValueError,match='full470 reverse'):D.localize(type('M',(),{'n':13})(),seed,targets,W())
