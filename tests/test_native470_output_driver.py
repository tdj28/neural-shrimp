import sys,subprocess,shutil,json
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import scout_native470_output as D

def test_section_change_preserves_literal_output_and_all470():
 old={'native_flat':[0.]*468+[70.,.8],'z':[0.]*13+[.8]+[0.]*13};x=np.array(old['native_flat'])
 assert D.terminal_association(x,old)['passed']
 for k in (0,13,26,467,468):
  y=x.copy();y[k]+=.001;assert not D.terminal_association(y,old)['passed']
 y=x.copy();y[-1]=np.nextafter(y[-1],1.);assert not D.terminal_association(y,old)['passed']

def test_actual_preflight_without_target_array_decode(monkeypatch):
 class W:
  def check(self):pass
  def save(self,*a):pass
 original=D.read
 def guarded(p):
  assert p.name not in ('survey.json','target-inputs.json','target-certificate.json.gz','independent512.json')and p.suffix!='.gz'
  return original(p)
 for m in(D,D.F,D.F.D,D.C):monkeypatch.setattr(m,'read',guarded)
 card=dict(original(ROOT/D.CARD),status='reviewed');assert D.binding(W(),card,metadata_only=True)==D.F.D.OUT.parent
 with pytest.raises(ValueError,match='reviewed'):D.binding(W(),dict(card,status='prepared'),metadata_only=True)

def test_full_unrelated_output_profile_and_inclusive_cap(tmp_path,monkeypatch):
 monkeypatch.setattr(D.C,'OUT',tmp_path);monkeypatch.setattr(D.C,'SETTINGS',D.SETTINGS);w=D.C.Work();D.profile(w);v=D.read(tmp_path/'profile.json')
 assert v['passed']and v['target_arrays_unopened']and w.calls==216 and len(v['seconds'])==12
 assert len(list((tmp_path/'events').glob('*native_return*')))==216
 w.calls=D.SETTINGS['native_calls']
 with pytest.raises(RuntimeError,match='phase cap'):w.tick()
 assert w.calls==4536

def test_isolated_archive_and_real_entrypoint_settings(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code="import sys;from pathlib import Path;sys.path.insert(0,'tools');import scout_native470_output as d;d.install();r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES);assert d.C.SETTINGS['native_calls']==216+240*18;assert d.C.survey is d.survey and d.C.profile is d.profile;assert d.C.OUT==d.OUT and str(d.OUT).endswith('signed-flip-output-scout001/attempt001/run')"
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
