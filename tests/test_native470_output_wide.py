import sys,subprocess,shutil
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import scout_native470_output_wide as D

def test_actual_metadata_preflight_and_literal_output_packing(monkeypatch):
 class W:
  def check(self):pass
  def save(self,*a):pass
 original=D.read
 def guarded(p):
  assert p.name not in ('survey.json','target-inputs.json','target-certificate.json.gz','independent512.json')and p.suffix!='.gz'
  return original(p)
 for m in(D,D.E,D.D,D.D.F,D.D.F.D,D.C):monkeypatch.setattr(m,'read',guarded)
 assert D.binding(W(),dict(original(ROOT/D.CARD),status='reviewed'),metadata_only=True)==D.E.OUT.parent
 flat=np.zeros(470);flat[-2]=70.;flat[-1]=.84;old={'native_flat':flat.tolist(),'output':.84,'z':[0.]*13+[70.]+[0.]*13}
 assert D.terminal_association(flat,old)['passed'];flat[-1]=np.nextafter(.84,1.);assert not D.terminal_association(flat,old)['passed']

def test_actual_archive_closure_defaults_and_new_event_argument(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code="import sys,inspect;from pathlib import Path;sys.path.insert(0,'tools');import scout_native470_output_wide as d;assert inspect.signature(d.D.survey).parameters['near_plus_threshold'].default==.01;d.install();r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES);assert d.C.survey.keywords=={'near_plus_threshold':.0001};assert d.C.SETTINGS['step']==.00025 and d.C.SETTINGS['native_calls']==4536 and str(d.C.OUT).endswith('attempt003/run')"
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
