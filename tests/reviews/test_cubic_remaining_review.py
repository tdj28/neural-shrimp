import copy,json,os,subprocess,sys,hashlib
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
PRODUCER=ROOT;PACKET=ROOT/'docs/plans'
sys.path[:0]=[str(PACKET),str(ROOT/'tools/reviews'),str(PRODUCER/'tools'),str(PRODUCER/'tools/reviews'),str(PRODUCER/'src')]
import neural_shrimp
import review_cubic_remaining as W

def test_exact_remaining_union():
 rows=[dict(passed=True,cell=W.cell(d,1,i),complement_dimension=11)for d in range(1,7)for i in(0,1)]
 r=W.union(rows,['79/2048','79/2048']);assert r['total_cells']==12 and r['whole_chart_endpoints']==['39/1024','103/1024']
 for bad in[rows[:-1],rows+[rows[0]]]:
  with pytest.raises(ValueError):W.union(bad,['79/2048','79/2048'])

def test_uncompleted_parent_blocks_before_read():
 class No:
  def tick(self):raise AssertionError('premature reader')
 card=W.read(PACKET/'native705-cubic32-remaining-independent-inputs.json');card['status']='reviewed';card['producer']['outer']=None
 with pytest.raises(ValueError):W.metadata(No(),card)

def test_resource_closing_reserve(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.used=252*1024**2
 with pytest.raises(ValueError):w.save('bad.json',{})
 w.save('outcome.json',{'passed':False},True)
 assert W.read(tmp_path/'outcome.json')['passed']is False


def test_isolated_archive_closure(tmp_path):
 for n in W.FILES:
  dst=tmp_path/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools/reviews','tools','src'];import review_cubic_remaining as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True);assert p.returncode==0,p.stderr


def test_actual_complete_history_without_scientific_decoders(monkeypatch):
 def forbidden(*a,**k):raise AssertionError('scientific arrays forbidden')
 monkeypatch.setattr(W.G.P,'small_matrix',forbidden);monkeypatch.setattr(W.G,'exact_inverse',forbidden);monkeypatch.setattr(W.G.B,'TwoRateCuspMap',forbidden);monkeypatch.setattr(W,'ROOT',PRODUCER)
 class Metadata:
  def tick(self):pass
  def save(self,*a):pass
 card=W.read(PACKET/'native705-cubic32-remaining-independent-inputs.json');card['status']='reviewed'
 run,domains,selected,projection,data,original=W.metadata(Metadata(),card)
 assert len(domains)==6 and len(selected)==18 and original['ordinal']==32
 assert W.read(run/'outcome.json')['native_calls']==4464
 with pytest.raises(ValueError):W.union([dict(passed=True,complement_dimension=11,cell=z['cell'])for z in selected[:-1]],projection)
 assert W.union([dict(passed=True,complement_dimension=11,cell=z['cell'])for z in selected],projection)['total_cells']==18
