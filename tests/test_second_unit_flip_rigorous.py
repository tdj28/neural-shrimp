import os,subprocess,sys,time
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import certify_second_unit_flip as D
import launch_second_unit_flip_rigorous as L

def test_actual_metadata_before_new_arrays(monkeypatch):
 class W:
  start=time.monotonic()
  def check(self):pass
  def save(self,*a):pass
 card=dict(D.read(ROOT/D.CARD),status='reviewed');old=D.read
 def guarded(p):
  assert p.name not in ('candidate.json','input-binding.json')or p.parent!=D.N.OUT
  return old(p)
 monkeypatch.setattr(D,'read',guarded)
 assert D.binding(W(),card,True)==D.N.OUT.parent

def test_full_profile_and_retained_inputs(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path/'run');D.OUT.mkdir();w=D.Work()
 with ctx.workprec(384):D.profile(w)
 assert w.calls==w.native==108 and D.read(D.OUT/'profile.json')['passed']
 assert len(list((D.OUT/'events').glob('*native_input*')))==len(list((D.OUT/'events').glob('*native_return*')))==108

def test_whole_physical_bracket_rejects_boundary_and_broad_box():
 b=[arb(0)]*55;b[13]=arb(2);b[14]=arb(4);t=[dict(hidden=1.,output=3.),dict(hidden=3.,output=5.)];D.bracket(b,t)
 b[13]=arb(1)
 with pytest.raises(ValueError):D.bracket(b,t)
 b[13]=arb(2,2)
 with pytest.raises(ValueError):D.bracket(b,t)

def test_settings_counts_and_archive_closure(tmp_path):
 assert D.OUT.parent==L.OUT and D.LIMITS['native_calls']==(D.SETTINGS['newton_steps']+4)*54==432
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import certify_second_unit_flip as d;import launch_second_unit_flip_rigorous as l;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES);assert l.OUT==d.OUT.parent"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
