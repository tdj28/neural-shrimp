import sys,subprocess
from pathlib import Path
from flint import ctx
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native470_junction_saved as D
import launch_native470_junction_saved as L

def test_literal_saved_lift_and_resource_profile_paths():
 assert D.OUT.parent==L.OUT and D.PARENT.name=='attempt003'
 assert D.settings()['native_calls']==0 and D.settings()['profile_phase_blocks']+D.settings()['target_phase_blocks']==54
 for root,pin in zip((D.PARENT,D.PARENT/'run'),D.PINS):assert D.entry(root/'manifest.json')['sha256']==pin
 r=D.read(D.PARENT/'run/outcome.json');assert r['passed']and r['native_calls']==108 and r['events']==228
 assert D.read(D.PARENT/'run/runtime.json')['source_commit']==D.SOURCE

def test_full13_dimension18_block_unrelated_synthetic(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path)
 with ctx.workprec(512):
  w=D.Work();D.profile(w);p=D.read(tmp_path/'profile.json');assert p['passed']and p['target_arrays_unopened']and w.blocks==36
  assert D.read(tmp_path/'profile0-result.json')['shooting_dimension']==470

def test_complete_source_closure(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 subprocess.run([sys.executable,'-I','-B','-c',"import sys,pathlib;sys.path.insert(0,'tools');import probe_native470_junction_saved as d;r=pathlib.Path.cwd();loaded={str(pathlib.Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and pathlib.Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES)"],cwd=tmp_path,check=True,capture_output=True)
