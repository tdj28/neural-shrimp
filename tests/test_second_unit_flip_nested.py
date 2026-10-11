import os,sys,subprocess,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tests')]
import audit_second_unit_flip_nested as D
import launch_second_unit_flip_nested as L

def test_exact_independent_reducer_provenance():
 card=D.read(ROOT/D.CARD)
 for n,e in card['independent_reducer']['files'].items():assert D.entry(ROOT/n)==e
 assert card['settings']==D.SETTINGS and card['limits']==D.LIMITS
 assert D.SETTINGS['newton_steps']==0 and D.LIMITS['native_calls']==162

def test_actual_metadata_before_native_or_dense_arrays(monkeypatch):
 class W:
  def check(self):pass
  def save(self,*a,**kw):pass
 card=dict(D.read(ROOT/D.CARD),status='reviewed');old=D.read
 def read(p):
  assert p.name not in('root.json','same-root-complement.json','same-root-crossing.json','candidate.json')
  assert '/events/'not in str(p)
  return old(p)
 monkeypatch.setattr(D,'read',read);monkeypatch.setattr(D,'augmented',lambda*a,**kw:pytest.fail('native before profile'));assert D.binding(W(),card,True)==D.C.OUT.parent
 with pytest.raises(ValueError):D.binding(W(),dict(card,status='prepared'),True)

def test_full55_profile_actual162_ceiling_and_retention(tmp_path):
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import audit_second_unit_flip_nested as d;d.OUT=Path(sys.argv[1]);d.OUT.mkdir();w=d.Work();d.ctx.prec=480;d.profile(w);assert w.calls==w.native==54;assert len(list((d.OUT/'events').glob('*-native_input.json.gz')))==54;assert len(list((d.OUT/'events').glob('*-native_return.json.gz')))==54;w.calls=162\ntry:w.tick()\nexcept ValueError:pass\nelse:raise AssertionError('native ceiling')"
 p=subprocess.run([sys.executable,'-I','-B','-c',code,str(tmp_path/'profile')],cwd=ROOT,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr

def test_isolated_archive_and_inherited_soft_as_startup(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import resource,sys;from pathlib import Path;resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,4*1024**3));resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);sys.path[:0]=['tools','src'];import audit_second_unit_flip_nested as d;import launch_second_unit_flip_nested as l;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES);assert d.OUT.parent==l.OUT"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
 source=(ROOT/'tools/audit_second_unit_flip_nested.py').read_text();assert source.index("if __name__=='__main__':resource.setrlimit")<source.index('import flint')


def test_real_exclusive_nested_binder_namespaces(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work()
 D.Prefix(w,'old-root-').save('input-binding.json',dict(kind='old'))
 D.Prefix(w,'numerical-').save('input-binding.json',dict(kind='numerical'))
 w.save('input-binding.json',dict(kind='final'))
 assert len(list(tmp_path.glob('*input-binding.json')))==3
 with pytest.raises(FileExistsError):D.Prefix(w,'old-root-').save('input-binding.json',{})


def test_retained_complete_cold_runtime_cannot_be_hidden():
 prior=dict(outcome=dict(seconds=21),profile=dict(seconds=2))
 assert D.measured_time(3,1,prior)==51
 assert D.measured_time(3,3,prior)==57
 with pytest.raises(ValueError):D.measured_time(3,1,dict(outcome=dict(seconds=-1),profile=dict(seconds=2)))
