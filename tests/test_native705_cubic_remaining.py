import copy,os,subprocess,sys,time
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import probe_native705_cubic_remaining as W
import launch_native705_cubic_remaining as L


def card():
 c=W.read(ROOT/W.CARD);c['status']='reviewed';c['first_independent']={k:'not-an-admission'for k in c['first_independent']};return c


def test_exact_six_domain_card_and_missing_independent_block():
 c=card();W.card_check(c);assert W.OUT.parent==L.OUT
 actual=W.read(ROOT/W.CARD);actual['status']='reviewed';actual['first_independent']['worker']=None
 with pytest.raises(ValueError):W.card_check(actual)
 for key,value in [('parameter_endpoints',['39/1024','103/1024']),('maximum_centers',91),('maximum_depth',4),('ordinal',33)]:
  d=copy.deepcopy(c);d[key]=value
  with pytest.raises(ValueError):W.card_check(d)
 for n in range(6):assert W.F(c['domains'][n]['endpoints'][1])-W.F(c['domains'][n]['endpoints'][0])==W.F(1,128)


def test_full_remaining_union_and_no_earlier_reacquisition(monkeypatch,tmp_path):
 c=card();prior={'parameter_endpoints':['39/1024','55/1024']};context=object();seen=[]
 monkeypatch.setattr(W,'metadata',lambda w,cc:(cc,prior));monkeypatch.setattr(W.G,'bind',lambda w,cc:context)
 def domain(w,row,x,dd):
  assert x is context;w.calls+=90;return {},{},[]
 def cell(w,row,x,*rest):
  assert x is context;seen.append(row);w.calls+=126;return dict(cell=row,passed=True,completed=True,native_calls=126,dimension=705,complement_dimension=11)
 monkeypatch.setattr(W.C,'derivative_domain',domain);monkeypatch.setattr(W.C,'fresh_cell',cell)
 w=W.Work();w.output_dir=tmp_path;w.calls=144;out=W.target(w,c)
 assert out['passed']and w.calls==144+540+6*126 and len(seen)==6
 assert min(W.F(x['endpoints'][0])for x in seen)==W.F(55,1024)
 assert out['union']['whole_chart_endpoints']==['39/1024','103/1024']


def test_nonmetric_failure_stops_not_subdivides(monkeypatch,tmp_path):
 c=card();monkeypatch.setattr(W,'metadata',lambda *a:(c,{'parameter_endpoints':['39/1024','55/1024']}));monkeypatch.setattr(W.G,'bind',lambda *a:None)
 monkeypatch.setattr(W.C,'derivative_domain',lambda *a:({}, {}, []));calls=[]
 def fail(*a):calls.append(1);raise ValueError('native domain coverage failed')
 monkeypatch.setattr(W.C,'fresh_cell',fail);w=W.Work();w.output_dir=tmp_path
 with pytest.raises(ValueError,match='domain coverage'):W.target(w,c)
 assert len(calls)==1 and not w.attempts


def test_complete_cost_scaling_and_hard_cap():
 r=W.forecast(10,10**6,10,10**6,12,100000,100,10**7)
 assert r['passed']and r['maximum_total_calls']==12024 and r['maximum_centers']==90
 for i,val in [(0,1000),(6,10000),(7,10**10)]:
  a=[10,10**6,10,10**6,12,100000,100,10**7];a[i]=val;assert not W.forecast(*a)['passed']


def test_actual_original_cubic_metadata_without_scientific_decoders(monkeypatch):
 def no(*a,**k):raise AssertionError('scientific decoder used')
 monkeypatch.setattr(W.G.P,'small_matrix',no);monkeypatch.setattr(W.G,'exact_inverse',no);monkeypatch.setattr(W.B,'TwoRateCuspMap',no)
 data=W.G.metadata(W.read(ROOT/W.C.CARD),lambda:None)
 assert data['index']['parameter_endpoints']==['39/1024','103/1024']and data['prior']['chart']==31


def test_owned_cap_and_closing_reserve(tmp_path):
 w=W.Work();w.output_dir=tmp_path;assert w.maximum_calls==12024 and w.raw_cap==18*1024**3
 w.used=w.raw_cap-8*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',{})
 w.save('outcome.json',{'passed':False},True)
 assert W.read(tmp_path/'outcome.json')['passed']is False


def test_archive_import_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','tools/reviews','src'];import probe_native705_cubic_remaining as w;import launch_native705_cubic_remaining as l;assert w.OUT.parent==l.OUT;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr


def test_actual_complete_new_authority_before_arrays(monkeypatch):
 def no(*a,**k):raise AssertionError('scientific decoder used')
 monkeypatch.setattr(W.G.P,'small_matrix',no);monkeypatch.setattr(W.G,'exact_inverse',no);monkeypatch.setattr(W.B,'TwoRateCuspMap',no)
 class Meta:
  def tick(self):pass
  def save(self,*a):pass
 c=W.read(ROOT/W.CARD);c['status']='reviewed';W.card_check(c);original,prior=W.metadata(Meta(),c)
 assert prior['total_cells']==4 and original['ordinal']==32
 assert prior['parameter_endpoints'][1]==c['parameter_endpoints'][0]
