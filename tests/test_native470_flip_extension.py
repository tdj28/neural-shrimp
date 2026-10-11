import sys,copy,subprocess,json,shutil
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import scout_native470_flip_extension as D

def test_terminal_association_retains_all470_and_literal_hidden():
 x=np.zeros(470);x[-2]=72.;x[-1]=.8;old={'native_flat':x.tolist(),'hidden':72.}
 assert D.terminal_association(x,old)['passed']
 for i in (0,13,26,467,469):
  y=x.copy();y[i]+=.001;assert not D.terminal_association(y,old)['passed']
 y=x.copy();y[-2]=np.nextafter(y[-2],73.);assert not D.terminal_association(y,old)['passed']

def test_default_preserving_settings_and_actual_metadata_preflight(monkeypatch):
 assert D.C.SETTINGS['max_points']==48 and D.SETTINGS['max_points']==24 and D.SETTINGS['native_calls']==216+160*18
 class W:
  def check(self):pass
  def save(self,*a):pass
 card=dict(D.read(ROOT/D.CARD),status='reviewed');monkeypatch.setattr(D.C,'SETTINGS',D.SETTINGS)
 old=D.read
 def guard(p):
  assert p.name not in ('survey.json','target-inputs.json','target-certificate.json.gz')
  return old(p)
 monkeypatch.setattr(D,'read',guard);monkeypatch.setattr(D.C,'read',guard)
 assert D.binding(W(),card,metadata_only=True).name=='attempt001'

def test_extension_archive_closure_and_settings_entrypoints(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code="import sys;from pathlib import Path;sys.path.insert(0,'tools');import scout_native470_flip_extension as d;d.install();r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES);assert d.C.SETTINGS['max_points']==24 and d.C.SETTINGS['native_calls']==3096;assert str(d.C.OUT).endswith('attempt002/run')"
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
