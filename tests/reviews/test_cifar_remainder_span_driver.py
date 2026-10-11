import copy,json,os,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import diagnose_cifar_remainder_span as W
import launch_cifar_remainder_span as L


def test_actual_metadata_only_binding(monkeypatch):
 c=W.M.read(ROOT/W.CARD);c['status']='reviewed'
 class Meta:
  def tick(self):pass
  def save(self,*a,**kw):pass
 def forbidden(*a,**kw):raise AssertionError('metadata must not read scientific vectors')
 monkeypatch.setattr(W.M,'vector',forbidden);r,calls,trials,audit=W.bind(Meta(),c);assert len(trials)==7 and audit['numerical_action_rank']==512
 assert W.OUT.parent==L.OUT
 bad=copy.deepcopy(c);bad['status']='prepared'
 with pytest.raises(ValueError):W.bind(Meta(),bad)


def test_small_complete_profile_and_reserve(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();W.profile(w,40,5);assert W.M.read(tmp_path/'profile.json')['passed'];w.used=124*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',{})
 w.save('outcome.json',{'passed':False},True)


def test_isolated_archive_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools/reviews','tools','src'];import diagnose_cifar_remainder_span as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
