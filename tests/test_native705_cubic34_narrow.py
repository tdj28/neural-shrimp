import copy,os,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import probe_native705_cubic34_narrow as W
import launch_native705_cubic34_narrow as L

def card():
 c=W.read(ROOT/W.CARD);c['status']='reviewed';return c

def test_exact_partition_depth_and_full_cost():
 rows=W.H.partition();assert len(rows)==64 and rows[0]['endpoints'][0]=='83/512'and rows[-1]['endpoints'][1]=='115/512'
 assert all(a['endpoints'][1]==b['endpoints'][0]for a,b in zip(rows,rows[1:]))and W.H.CALLS==30096
 row=W.C.H.cell(rows,1);out=dict(cell=row,passed=False,completed=True,native_calls=126,dimension=705,complement_dimension=11)
 children=W.H.children(rows,row,out);assert children[0]['halfwidth']=='1/4096'
 with pytest.raises(ValueError):W.H.children(rows,children[0],dict(out,cell=children[0]))
 with pytest.raises(ValueError):W.H.children(rows,row,dict(out,completed=False))

def test_prepared_card_blocks_launcher_before_output(monkeypatch,tmp_path):
 c=card();W.card_check(c);L.card_check(c);assert W.OUT.parent==L.OUT and L.CARD==W.CARD
 bad=copy.deepcopy(c);bad['independent_previous']['outer']=None
 with pytest.raises(ValueError):W.card_check(bad)
 with pytest.raises(AssertionError):L.card_check(bad)
 monkeypatch.setattr(L.subprocess,'check_output',lambda *a,**k:b'');monkeypatch.setattr(L,'OUT',tmp_path/'unopened');monkeypatch.setattr(L.json,'loads',lambda *_:dict(c,status='prepared'))
 with pytest.raises(AssertionError):L.main()
 assert not L.OUT.exists()

def test_complete_worstcase_scheduler_and_no_extra_calls(monkeypatch,tmp_path):
 c=card();monkeypatch.setattr(W,'bind',lambda *a:{});monkeypatch.setattr(W.G,'bind',lambda *a:None)
 def domain(w,*a):w.calls+=90;return {},{},[]
 def cell(w,row,*a):w.calls+=126;return dict(cell=row,passed=row['depth']==1,completed=True,native_calls=126,dimension=705,complement_dimension=11)
 monkeypatch.setattr(W.C,'derivative_domain',domain);monkeypatch.setattr(W.C,'fresh_cell',cell);monkeypatch.setattr(W.gc,'collect',lambda:None)
 w=W.Work();w.output_dir=tmp_path;w.calls=144;r=W.target(w,c)
 assert r['passed']and w.calls==30096 and len(w.attempts)==192 and len(w.accepted)==128 and w.domains_done==64
 assert r['union']['endpoints']==W.H.ENDPOINTS and not w.pending
 with pytest.raises(ValueError):W.H.reserve(w.calls,64,192,'cell')

def test_nonmetric_failure_and_depth_limit_remain_failed(monkeypatch,tmp_path):
 c=card();monkeypatch.setattr(W,'bind',lambda *a:{});monkeypatch.setattr(W.G,'bind',lambda *a:None)
 def domain(w,*a):w.calls+=90;return {},{},[]
 def cell(w,row,*a):w.calls+=126;return dict(cell=row,passed=False,completed=True,native_calls=126,dimension=705,complement_dimension=11)
 monkeypatch.setattr(W.C,'derivative_domain',domain);monkeypatch.setattr(W.C,'fresh_cell',cell)
 w=W.Work();w.output_dir=tmp_path;w.calls=144
 with pytest.raises(ValueError,match='depth1'):W.target(w,c)
 assert len(w.attempts)==2 and w.pending[0]==w.attempts[-1]['cell']and len(w.pending)==2 and not w.accepted

def test_actual_metadata_and_source_without_scientific_reducers(monkeypatch):
 def no(*a,**k):raise AssertionError('target arithmetic/model prohibited')
 monkeypatch.setattr(W.G.P,'small_matrix',no);monkeypatch.setattr(W.G,'exact_inverse',no);monkeypatch.setattr(W.G.B,'TwoRateCuspMap',no)
 class Work:
  prefix=''
  def tick(self):pass
  def save(self,*a,**kw):pass
 p=W.bind(Work(),card());assert p['index']['ordinal']==33 and p['independent']==W.INDEPENDENT

def test_full_worstcase_cost_and_final_reserve(tmp_path):
 cost=dict(seconds=728.1414386,domains=9,centers=74,max_domain_bytes=70000000,max_cell_bytes=50000000)
 a=W.forecast(8,10000000,30,2000000,cost);assert a['passed']and a['maximum_total_calls']==30096 and a['forecast_seconds']>10000 and a['forecast_bytes']>28*10**9
 assert not W.forecast(40,10000000,30,2000000,cost)['passed']
 w=W.Work();w.output_dir=tmp_path;w.used=w.raw_cap-8*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',{})
 w.save('outcome.json',dict(passed=False),True)

def test_isolated_archive_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','tools/reviews','src'];import probe_native705_cubic34_narrow as w;import launch_native705_cubic34_narrow as l;r=Path.cwd();a={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert a<=set(w.FILES),a-set(w.FILES);assert w.OUT.parent==l.OUT"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True);assert p.returncode==0,p.stderr
