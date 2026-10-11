import copy,os,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import probe_native705_cubic_panel as W
import launch_native705_cubic_panel as L


def card():
 c=W.read(ROOT/W.CARD);c['status']='reviewed';c['initial_independent']={k:'non-admission-test'for k in c['initial_independent']};return c


def test_card_and_incomplete_independent_prerequisite():
 c=card();W.card_check(c);assert W.OUT.parent==L.OUT
 bad=copy.deepcopy(c);bad['initial_independent']['outer']=None
 with pytest.raises(ValueError):W.card_check(bad)
 for key,value in [('maximum_centers',97),('maximum_depth',4),('maximum_domains',17)]:
  bad=copy.deepcopy(c);bad[key]=value
  with pytest.raises(ValueError):W.card_check(bad)


def test_launcher_uses_own_prepared_card_before_output(monkeypatch,tmp_path):
 assert L.CARD==W.CARD
 c=card();L.card_check(c)
 for key in ('root','source','outer','worker'):
  bad=copy.deepcopy(c);bad['initial_independent'][key]=None
  with pytest.raises(AssertionError):L.card_check(bad)
 bad=copy.deepcopy(c);bad['charts'][0]['ordinal']=32
 with pytest.raises(AssertionError):L.card_check(bad)
 bad=copy.deepcopy(c);bad['caps']['maximum_total_calls']+=1
 with pytest.raises(AssertionError):L.card_check(bad)
 monkeypatch.setattr(L.subprocess,'check_output',lambda *a,**k:b'')
 monkeypatch.setattr(L,'OUT',tmp_path/'unopened')
 with pytest.raises(AssertionError,match='reviewed cubic33'):L.main()
 assert not L.OUT.exists()


def test_two_chart_scheduler_uses_one_context_and_closed_union(monkeypatch,tmp_path):
 c=card();contexts=[];monkeypatch.setattr(W,'initial',lambda *a:{'first':True})
 def bind(w,s,p):contexts.append((s['ordinal'],p));return s['ordinal']
 def domain(w,row,context,domains):w.calls+=90;return {},{},[]
 def cell(w,row,context,*rest):
  w.calls+=126;return dict(cell=row,passed=True,completed=True,native_calls=126,dimension=705,complement_dimension=11)
 monkeypatch.setattr(W.G,'bind',bind);monkeypatch.setattr(W.C,'derivative_domain',domain);monkeypatch.setattr(W.C,'fresh_cell',cell)
 w=W.Work();w.output_dir=tmp_path;w.calls=144;result=W.target(w,c)
 assert result['passed']and w.calls==144+16*90+16*126 and len(w.completed_charts)==2
 assert contexts[1][1]['index']==c['charts'][0]['root_index']and contexts[1][1]['complete_complement_endpoints']==c['charts'][0]['parameter_endpoints']


def test_nonmetric_failure_never_advances_or_splits(monkeypatch,tmp_path):
 c=card();monkeypatch.setattr(W,'initial',lambda *a:{});monkeypatch.setattr(W.G,'bind',lambda *a:None)
 def domain(w,*a):w.calls+=90;return {},{},[]
 def fail(*a):raise ValueError('actual acquired domain missed')
 monkeypatch.setattr(W.C,'derivative_domain',domain);monkeypatch.setattr(W.C,'fresh_cell',fail)
 w=W.Work();w.output_dir=tmp_path;w.calls=144
 with pytest.raises(ValueError,match='domain missed'):W.target(w,c)
 assert not w.completed_charts and not w.attempts and len(w.pending)==1


def test_actual_metadata_and_full705_same_gauge_predecessor(monkeypatch):
 def no(*a,**k):raise AssertionError('scientific decoder/model forbidden')
 monkeypatch.setattr(W.G.P,'small_matrix',no);monkeypatch.setattr(W.G,'exact_inverse',no);monkeypatch.setattr(W.G.B,'TwoRateCuspMap',no)
 c=card();index=W.read(ROOT/W.G.M.CARD)
 for s in c['charts']:
  prior=W.G.M.select(index,s['ordinal']-1);p=dict(index=prior,complete_complement_endpoints=prior['parameter_endpoints'])
  data=W.G.metadata(s,p,lambda:None);assert data['proof']['anchor']['coordinates']==705 and data['index']['ordinal']==s['ordinal']
  with pytest.raises(ValueError):W.G.metadata(s,dict(p,complete_complement_endpoints=['0','1']),lambda:None)


def test_measured_complete_forecast_and_closing_reserve(tmp_path):
 out=dict(elapsed_seconds=334.237);old=dict(forecast_seconds=7187.96,forecast_bytes=12510871977)
 f=W.forecast(10,10**6,10,2000000,out,old,10**9);assert f['passed']and f['maximum_total_calls']==13680
 assert not W.forecast(10000,10**6,10,2000000,out,old,10**9)['passed']
 w=W.Work();w.output_dir=tmp_path;w.used=w.raw_cap-8*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',{})
 w.save('outcome.json',{'passed':False},True);assert W.read(tmp_path/'outcome.json')['passed']is False


def test_isolated_archive_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','tools/reviews','src'];import probe_native705_cubic_panel as w;import launch_native705_cubic_panel as l;assert w.OUT.parent==l.OUT;r=Path.cwd();a={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert a<=set(w.FILES),a-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True);assert p.returncode==0,p.stderr
