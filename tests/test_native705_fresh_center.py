import sys,subprocess,os
from pathlib import Path
from fractions import Fraction as F
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native705_fresh_center as W
from neural_shrimp.rigorous_orbits import vector_json,matrix_json

def test_exact_original_domain_rejects_serialization_expansion():
 with ctx.workprec(256):
  p=[arb(1)]*705;old=vector_json([arb(0,arb(1)/8)]*705);good=[arb(1)]*705;W.original_contains(old,p,good)
  bad=good.copy();bad[-1]=arb(2)
  with pytest.raises(ValueError):W.original_contains(old,p,bad)
  badp=p.copy();badp[0]=arb(1,arb(1)/1024)
  with pytest.raises(ValueError):W.original_contains(old,badp,good)

def test_fixed_subinterval_and_call_counts():
 assert W.HALF==W.P.HALF/16 and W.ENDS==[F(-34,2048),F(-32,2048)]
 assert W.P.LEFT<W.ENDS[0]<W.ENDS[1]<W.P.RIGHT
 assert W.OUT.parent.name=='fresh-center001'


def test_retained_full_derivative_actual_domain_is_mandatory():
 with ctx.workprec(192):
  whole=[arb(1)]*704;v=[arb(0)]*704
  a=dict(kind='whole_tangent',phase=0,state=vector_json(whole[:13]),rates=vector_json(whole[-2:]),state_tangent=vector_json(v[:13]),rate_tangent=vector_json(v[-2:]));r=dict(kind='whole_tangent',phase=0,A=matrix_json(arb_mat(13,13)),Aprime=matrix_json(arb_mat(13,13)))
  W.retained_whole(a,r,0,whole,v)
  altered=whole.copy();altered[0]+=arb(1)/1024
  with pytest.raises(ValueError):W.retained_whole(a,r,0,altered,v)
  v[-1]=arb(1)
  with pytest.raises(ValueError):W.retained_whole(a,r,0,whole,v)

def test_full705_residual_inputs_before_call_and_complete_count(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path/'run');W.OUT.mkdir();w=W.Work();w.calls=108
 class Linear:
  n=13
  def cusp_step(self,x,a,b):return [(1-a*arb((1,-7)))*z+b*sum(x)*arb((1,-12)) for z in x]
 with ctx.workprec(128):
  c=[arb(1)/8]*702+[arb(1),arb(1),arb(1)/64];normal=[arb(0)]*704+[arb(1)];origin=[arb(0)]*705
  out=w.residual(Linear(),c,normal,origin,arb(1)/64,'control')
  assert out.nrows()==705 and w.calls==144
  assert len(list((W.OUT/'native').glob('*-input.json')))==len(list((W.OUT/'native').glob('*-return.json')))==36
  assert W.read(W.OUT/'native/109-input.json')['kind']=='residual'

def test_archive_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import probe_native705_fresh_center as w;loaded={str(Path(m.__file__).resolve().relative_to(Path.cwd()))for m in sys.modules.values()if getattr(m,'__file__',None)and Path.cwd()in Path(m.__file__).resolve().parents};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 r=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr
