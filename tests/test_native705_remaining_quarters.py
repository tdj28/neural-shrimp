import copy,os,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native705_remaining_quarters as W
import launch_native705_remaining_quarters as L


def card():
 c=W.read(ROOT/W.CARD);c['status']='reviewed';c['second_quarter_independent']=dict(root='/data2/neural-shrimp-data/audits/control',source='a'*40,outer='b'*64,worker='c'*64);return c


def test_card_and_prior_gate():
 c=card();W.card_check(c);assert W.OUT.parent==L.OUT
 for k,v in [('status','prepared'),('quarters',[4,3]),('second_quarter_independent',None)]:
  bad=copy.deepcopy(c);bad[k]=v
  with pytest.raises((ValueError,TypeError)):W.card_check(bad)
 bad=copy.deepcopy(c);bad['domains']['3'][0]['endpoints'][0]='0'
 with pytest.raises(ValueError):W.card_check(bad)


def test_default_geometry_unchanged_and_two_remaining_unions():
 assert W.H.domains()==W.H.domains(2)
 for q in(2,3,4):
  rows=[dict(cell=W.H.cell(d,3,i,q),passed=True,completed=True,native_calls=126,dimension=705,complement_dimension=11)for d in(1,2)for i in range(8)]
  u=W.H.complete(rows,q);assert u['accepted_cells']==16 and u['endpoints']==[W.H.domains(q)[0]['endpoints'][0],W.H.domains(q)[1]['endpoints'][1]]
 assert W.H.domains(2)[1]['endpoints'][1]==W.H.domains(3)[0]['endpoints'][0]
 assert W.H.domains(3)[1]['endpoints'][1]==W.H.domains(4)[0]['endpoints'][0]
 with pytest.raises(ValueError):W.H.domains(5)


def test_complete_forecast_and_output_caps(tmp_path,monkeypatch):
 good=W.forecast(16,20*1024**2,37,67*1024**2,20,1800000,101,450*1024**2);assert good['passed']and good['maximum_total_calls']==8064
 bad=W.forecast(16,20*1024**2,37,67*1024**2,20,1800000,1000,450*1024**2);assert not bad['passed']
 monkeypatch.setattr(W.Work,'output_dir',tmp_path);w=W.Work();assert w.maximum_calls==8064 and w.raw_cap==12*1024**3
 w.used=w.raw_cap-8*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',{})
 w.save('outcome.json',{'passed':False},True);assert W.read(tmp_path/'outcome.json')['passed']is False


def test_archive_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import probe_native705_remaining_quarters as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
