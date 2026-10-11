import os,subprocess,sys,time
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tests')]
import refine_second_unit_flip_crossing as D
import launch_second_unit_flip_crossing as L
from test_rigorous_unit_pair_flip import fixture,augmented

def test_preconditioned_same_root_known_crossing_and_zero_rejection():
 with ctx.workprec(384):
  model,y=fixture();_,J,data=augmented(model,y,1);v=D.refined_transverse_crossing(J,data['monodromy'],y[15:],4);assert v['passed']and v['multiplier_hidden_rate_derivative']['mid']==[-1,0]
  for i in range(14,18):
   for j in range(6):J[i,j]=0
  assert not D.refined_transverse_crossing(J,data['monodromy'],y[15:],4)['passed']

def test_full_nonnormal_validated_inverse_and_singular_rejection():
 with ctx.workprec(384):
  A=arb_mat([[1,2**30],[0,1]]);X=D.validated_inverse(A);assert X[0,1]==-2**30
  with pytest.raises((ValueError,ZeroDivisionError)):D.validated_inverse(arb_mat([[1,1],[1,1]]))

def test_actual_metadata_and_entire_box_event_only_after_profile(monkeypatch):
 class W:
  def check(self):pass
  def save(self,*a):pass
 card=dict(D.read(ROOT/D.CARD),status='reviewed');old=D.read
 def read(p):
  assert p.name not in('root.json','same-root-complement.json','same-root-crossing.json')
  return old(p)
 monkeypatch.setattr(D,'read',read);assert D.bind(W(),card,True)==D.C.OUT.parent

def test_full_synthetic_profile_and_archive(tmp_path,monkeypatch):
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import refine_second_unit_flip_crossing as d;d.OUT=Path(sys.argv[1]);d.OUT.mkdir();w=d.Work();d.ctx.prec=384;d.profile(w);assert d.read(d.OUT/'profile.json')['passed']"
 sub=subprocess.run([sys.executable,'-I','-B','-c',code,str(tmp_path/'run')],cwd=ROOT,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert sub.returncode==0,sub.stderr
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import refine_second_unit_flip_crossing as d;import launch_second_unit_flip_crossing as l;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES);assert d.OUT.parent==l.OUT"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
