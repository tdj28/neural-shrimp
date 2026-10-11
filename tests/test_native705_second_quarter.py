import copy,json,os,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native705_second_quarter as W
import launch_native705_second_quarter as L


def admitted():
 c=W.read(ROOT/W.CARD);c['status']='reviewed';return c


def test_literal_card_source_and_resource_boundaries():
 c=admitted();W.card_check(c);assert W.OUT.parent==L.OUT
 for k,v in [('status','prepared'),('maximum_depth',4),('maximum_centers',31),('parameter_endpoints',['0','1'])]:
  bad=copy.deepcopy(c);bad[k]=v
  with pytest.raises(ValueError):W.card_check(bad)
 for k in ['independent_first_quarter','width_diagnosis']:
  bad=copy.deepcopy(c);bad[k]['source']='0'*40
  with pytest.raises(ValueError):W.card_check(bad)
 bad=copy.deepcopy(c);bad['domains'][1]['endpoints'][0]='0'
 with pytest.raises(ValueError):W.card_check(bad)
 bad=copy.deepcopy(c);bad['caps']['maximum_total_calls']+=1
 with pytest.raises(ValueError):W.card_check(bad)


def test_full_residual_path_records_and_hard_cap(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.prefix='test'
 class Linear:
  n=13
  def cusp_step(self,x,a,b):return[z*arb('0.5')for z in x]
 with ctx.workprec(128):
  x=[arb(0)]*702+[arb(1),arb(1),arb(1)/64];normal=[arb(0)]*704+[arb(1)];origin=[arb(0)]*705
  G=w.residual(Linear(),x,normal,origin,arb(1)/64,'point');assert G.nrows()==705 and w.calls==36
  cp=w.paths(Linear(),x,[arb(0)]*705,normal,origin,arb(1)/64,'path');assert cp.nrows()==705 and w.calls==72
  assert len(list((tmp_path/'test/native').glob('*-input.json')))==72
  assert W.read(tmp_path/'test/native/0037-input.json')['direction']=='q'
  w.calls=4104
  with pytest.raises(ValueError):w.jac(Linear(),x[:13],x[-3:-1],[arb(0)]*13,[arb(0)]*2,'center',0)


def test_actual_reused_derivative_domain_rejects_expansion():
 with ctx.workprec(128):
  old=[W.ball_json(arb(0,arb(1)))];W.P.exact_coverage(old,[arb(0,arb('0.5'))],'inside')
  with pytest.raises(ValueError):W.P.exact_coverage(old,[arb(0,arb(2))],'outside')


def test_finalization_reserve(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.used=6*1024**3-8*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',{'x':1})
 w.save('outcome.json',{'passed':False},True);assert W.read(tmp_path/'outcome.json')['passed']is False


def test_archive_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','tools/reviews','src'];import probe_native705_second_quarter as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr


def test_actual_completed_prior_metadata_and_width_schema():
 class Saved:
  def __init__(self):self.rows={}
  def save(self,name,value):self.rows[name]=value
 w=Saved();overlap=W.bind_meta(w,admitted());assert overlap['second_quarter_contains_previous_endpoint']
 actual=w.rows['authority-binding.json']['width_outcome'];assert 'results'in actual and 'rows'not in actual
 assert [r['passed']for r in actual['results']]==[True,True,False,False,False]
