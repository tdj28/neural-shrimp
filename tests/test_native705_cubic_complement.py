import copy,os,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import probe_native705_cubic_complement as W
import launch_native705_cubic_complement as L


def admitted():
 c=W.read(ROOT/W.CARD);c['status']='reviewed';return c


def test_fixed_scope_and_cap_card():
 c=admitted();W.card_check(c);assert W.OUT.parent==L.OUT
 for key,value in [('status','prepared'),('ordinal',33),('parameter_endpoints',['0','1']),('maximum_depth',4),('maximum_centers',31)]:
  d=copy.deepcopy(c);d[key]=value
  with pytest.raises(ValueError):W.card_check(d)
 d=copy.deepcopy(c);d['caps']['maximum_total_calls']+=1
 with pytest.raises(ValueError):W.card_check(d)
 d=copy.deepcopy(c);d['domains'][1]['endpoints'][0]='0'
 with pytest.raises(ValueError):W.card_check(d)


def test_actual_metadata_original_family_no_array_decoding(monkeypatch):
 def no(*_a,**_k):raise AssertionError('scientific target decoder called')
 monkeypatch.setattr(W.P,'small_matrix',no);monkeypatch.setattr(W.G,'exact_inverse',no);monkeypatch.setattr(W.B,'TwoRateCuspMap',no)
 d=W.G.metadata(admitted(),lambda:None)
 assert d['index']['ordinal']==32 and d['prior']['chart']==31 and d['proof']['gates']['uniform_root']
 lo,hi=map(W.F,d['anchor_projection']);assert W.F(39,1024)<=lo<=hi<=W.F(55,1024)


def test_exact_c0_shape_and_nonexact_rejection():
 with ctx.workprec(128):
  c=W.G.exact_inverse([[[1,0],[0,0]],[[0,0],[1,0]]],2);assert c[0,0]==1
  for x in [[[[1.,0],[0,0]],[[0,0],[1,0]]],[[[1,0]],[[0,0]]]]:
   with pytest.raises(ValueError):W.G.exact_inverse(x,2)


def test_full_residual_retention_cap_and_reserve(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.prefix='trial'
 class Linear:
  n=13
  def cusp_step(self,x,a,b):return[z*arb('0.5')for z in x]
 with ctx.workprec(128):
  x=[arb(0)]*702+[arb(1),arb(1),arb(1)/64];n=[arb(0)]*704+[arb(1)];o=[arb(0)]*705
  g=w.residual(Linear(),x,n,o,arb(1)/64,'point');assert g.nrows()==705 and w.calls==36
  assert len(list((tmp_path/'trial/native').glob('*-input.json')))==36
  w.calls=4104
  with pytest.raises(ValueError):w.jac(Linear(),x[:13],x[-3:-1],[arb(0)]*13,[arb(0)]*2,'center',0)
 w.used=6*1024**3-8*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',{})
 w.save('outcome.json',{'passed':False},True)
 assert W.read(tmp_path/'trial/outcome.json')['passed']is False


def test_archive_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','tools/reviews','src'];import probe_native705_cubic_complement as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
