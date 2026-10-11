import copy,os,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import native705_metric_diagnosis as H
import diagnose_native705_cubic34_terminal as D
import launch_native705_cubic34_terminal_diagnosis as L

def test_error_floor_and_hypothetical_scope():
 with ctx.workprec(192):
  q=H.eye(11)/2;qp=H.eye(11)*arb(0,arb(1)/64);q[0,0]+=arb(0,arb(1)/4096);records=[]
  result=H.analyze([(q,qp)for _ in range(18)],arb(1)/2048,observe=lambda k,v:records.append((k,v)))
  assert len(result['scenarios'])==7 and len(records)==8
  assert result['scenarios'][1]['name']=='center_error_floor'and not result['scenarios'][1]['hypothetical']
  assert all(r['hypothetical']for r in result['scenarios'][2:5])and all(not r['hypothetical']for r in result['scenarios'][5:])
  c,e0,e1=H.phase_parts(q,qp,arb(0));assert not e0[0,0].is_zero()and e1[0,0].is_zero()

def test_numerical_neutral_midpoint_is_diagnostic_not_gate():
 with ctx.workprec(192):
  q=H.eye(11)*arb('0.999');result=H.analyze([(q,arb_mat(11,11))for _ in range(18)],arb(1)/2048)
  assert abs(result['numerical_midpoint']['spectral_radius']-.999**18)<1e-13
  assert result['scenarios'][0]['metric_passed']

def test_literal_card_and_reserve(tmp_path,monkeypatch):
 c=D.read(ROOT/D.CARD);c['status']='reviewed';D.card_check(c);assert D.OUT.parent==L.OUT
 for k,v in [('status','prepared'),('ordinal',75),('cell',dict(c['cell'],halfwidth='1/2048'))]:
  bad=copy.deepcopy(c);bad[k]=v
  with pytest.raises(ValueError):D.card_check(bad)
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.used=62*1024**2-1
 with pytest.raises(ValueError):w.save('blocked.json',{'x':1})
 w.save('outcome.json',{'passed':False},True)

def test_archive_import_closure(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','tools/reviews','src'];import diagnose_native705_cubic34_terminal as d;import launch_native705_cubic34_terminal_diagnosis as l;r=Path.cwd();m={str(Path(x.__file__).resolve().relative_to(r))for x in list(sys.modules.values())if getattr(x,'__file__',None)and Path(x.__file__).resolve().is_relative_to(r)};assert m<=set(d.FILES),m-set(d.FILES);assert d.OUT.parent==l.OUT"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True);assert p.returncode==0,p.stderr
