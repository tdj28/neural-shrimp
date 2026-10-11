import importlib.util,json,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_junction_lift as D
import launch_native705_junction_lift as L


def test_exact_paths_limits_and_measurements():
 assert D.OUT.parent==L.OUT and D.settings()['profile_calls']+D.settings()['target_calls']==108
 assert D.forecast([1.,1.5],[1024,2048],100000)['admitted']
 assert not D.forecast([15.,15.],[1024,2048],100000)['admitted']
 assert not D.forecast([1.,1.],[20*1024**2]*2,100000)['admitted']
 with pytest.raises(ValueError):D.forecast([0.,1.],[1024,2048],100000)


def test_actual_metadata_shapes_without_target_reduction():
 p=D.PARENT;rt=D.read(p/'run/runtime.json');la=D.read(p/'launcher-result.json');owned=D.read(p/'owned-process.json');b=D.read(p/'run/binding.json')
 assert D.entry(p/'manifest.json')['sha256']==D.PINS[0]and D.entry(p/'run/manifest.json')['sha256']==D.PINS[1]
 assert rt['commit']==la['commit']==D.SOURCE and set(owned)=={'pid'}and 'owned_group_pid'not in la and rt['precision']==480
 assert b['source']==D.ORIGINAL_SOURCE
 old=Path(b['root']);owner=D.read(old/'owned-process.json');result=D.read(old/'launcher-result.json');assert owner['group_pid']==result['owned_group_pid']
 for n in D.NATIVE:assert D.entry(ROOT/n)==rt['sources'][n]


def test_full_archive_import_closure(tmp_path):
 for n in D.FILES:
  q=tmp_path/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/n).read_bytes())
 script="import sys,pathlib;sys.path.insert(0,'tools');import probe_native705_junction_lift as d;r=pathlib.Path.cwd();loaded={str(pathlib.Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and pathlib.Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES)"
 subprocess.run([sys.executable,'-I','-B','-c',script],cwd=tmp_path,check=True,capture_output=True)


def test_call_and_storage_limits_preserve_failure(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.calls=108
 with pytest.raises(ValueError,match='native ceiling'):w.tick()
 w.used=78*1024**2
 with pytest.raises(ValueError,match='closing reserve'):w.save('x.json',{})
 w.save('outcome.json',{'passed':False},True);assert D.read(tmp_path/'outcome.json')['passed']is False


def test_actual_complete_historical_derivative_association():
 from flint import ctx
 p=D.PARENT;b=D.read(p/'run/binding.json');root=D.read(p/'run/fresh-root.json');old=D.read(Path(b['root'])/'run/root.json');identity=D.read(p/'run/root-identity.json');out=D.read(p/'run/outcome.json');ce=D.read(p/'run/fresh-center_evaluation.json');wh=D.read(p/'run/fresh-box_evaluation.json')
 assert set(wh)=={'box','jacobian','monodromy'}
 with ctx.workprec(512):D.qualify(root,out,identity,old,ce,wh)
 from copy import deepcopy
 corrupted=deepcopy(wh);corrupted['jacobian'][0][0]=root['center'][0]
 with ctx.workprec(512),pytest.raises(ValueError,match='acquired derivative association'):D.qualify(root,out,identity,old,ce,corrupted)
 assert D.OUT.parent.name=='attempt003'and D.FAILED_SCHEMA.name=='attempt002'
 assert D.entry(D.FAILED_SCHEMA/'manifest.json')['sha256']==D.FAILED_SCHEMA_OUTER
