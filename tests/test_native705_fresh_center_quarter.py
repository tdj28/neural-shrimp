import copy,json,os,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native705_fresh_center_quarter as W
import launch_native705_fresh_center_quarter as L

def admitted():
 c=W.read(ROOT/W.CARD);c['status']='reviewed';c['independent_pilot']=dict(root='/data2/neural-shrimp-data/audits/control',source='a'*40,outer='b'*64,worker='c'*64);return c

def test_prepared_card_rejects_and_every_fixed_domain_is_bound():
 c=admitted();W.card_check(c)
 for key,value in [('status','prepared'),('independent_pilot',None),('original_quarter',['0','1'])]:
  bad=copy.deepcopy(c);bad[key]=value
  with pytest.raises((ValueError,TypeError)):W.card_check(bad)
 bad=copy.deepcopy(c);bad['cells'][0]['center']='0'
 with pytest.raises(ValueError):W.card_check(bad)
 bad=copy.deepcopy(c);bad['caps']['total_calls']+=1
 with pytest.raises(ValueError):W.card_check(bad)
 assert W.OUT.parent==L.OUT

def test_fresh_cell_full_residual_retention_and_stage_cap(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.calls=144;w.cell_start=144;w.prefix='cells/001'
 class Linear:
  n=13
  def cusp_step(self,x,a,b):return[z*arb('0.5')for z in x]
 with ctx.workprec(128):
  x=[arb(0)]*702+[arb(1),arb(1),arb(1)/64];normal=[arb(0)]*704+[arb(1)];origin=[arb(0)]*705
  out=w.residual(Linear(),x,normal,origin,arb(1)/64,'target0')
  assert out.nrows()==705 and w.calls==180
  assert len(list((tmp_path/'cells/001/native').glob('*-input.json')))==36
  assert W.read(tmp_path/'cells/001/native/0145-input.json')['kind']=='residual'
  w.calls=253
  with pytest.raises(ValueError):w.residual(Linear(),x,normal,origin,arb(1)/64,'invalid')
  w.calls=270
  with pytest.raises(ValueError):w.jac(Linear(),x[:13],x[-3:-1],[arb(0)]*13,[arb(0)]*2,'center',0)

def test_reserved_finalization_survives_failed_regular_write(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.used=3*1024**3-8*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',{'x':1})
 w.prefix='';w.save('outcome.json',dict(passed=False),closing=True)
 assert W.read(tmp_path/'outcome.json')==dict(passed=False)

def test_archive_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import probe_native705_fresh_center_quarter as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
