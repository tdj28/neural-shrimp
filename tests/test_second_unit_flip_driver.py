import sys,json,subprocess,shutil
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import localize_native_second_unit_flip as D

def test_actual_metadata_without_arrays(monkeypatch):
 class W:
  def check(self):pass
  def save(self,*a):pass
 old=D.read
 def guarded(p):
  assert p.name not in ('survey.json','target-inputs.json','target-certificate.json.gz')and p.suffix!='.gz'
  return old(p)
 monkeypatch.setattr(D,'read',guarded);assert D.binding(W(),dict(old(ROOT/D.CARD),status='reviewed'),metadata_only=True)==D.S.OUT.parent

def test_complete_real_native_profile_counts_and_retention(tmp_path,monkeypatch):
 monkeypatch.setattr(D.C,'OUT',tmp_path/'run');D.C.OUT.mkdir();monkeypatch.setattr(D.C,'SETTINGS',D.SETTINGS)
 w=D.C.Work();D.profile(w);assert w.calls==540
 p=D.read(D.C.OUT/'profile.json');assert p['passed']and p['counts']==dict.fromkeys(D.CAPS,6)
 assert len(list((D.C.OUT/'events').glob('*-native_input.json.gz')))==len(list((D.C.OUT/'events').glob('*-native_return.json.gz')))==540
 counts=dict.fromkeys(D.CAPS,0);counts['augmented']=12
 with pytest.raises(RuntimeError,match='cap'):D.call(None,w,counts,'augmented',None)

def test_completed_nonfinite_native_return_is_retained():
 from neural_shrimp.unit_pair_scout import Jet
 class M:
  n=1
  def cusp_step(self,x,h,o):return[x[0]*np.nan]
 class W:
  rows=[]
  def observe(self,k,v):self.rows.append((k,v))
 w=W();m=D.Retained(M(),w);v=Jet.variable(1.,3,0);m.cusp_step([v],v,v);assert [k for k,v in w.rows]==['native_input','native_return']

def test_isolated_source_archive_and_literal_paths(tmp_path):
 for n in D.FILES:
  q=tmp_path/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,q)
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import localize_native_second_unit_flip as d;import launch_native_second_unit_flip as l;r=Path.cwd();assert d.OUT.parent==l.OUT;a={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert a<=set(d.FILES),a-set(d.FILES)"
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
