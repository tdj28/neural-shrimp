import copy,json,os,subprocess,sys
from pathlib import Path
from fractions import Fraction as F
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import review_cubic_first as W


def test_exact_cubic_union_and_prior_projection():
 rows=[dict(passed=True,cell=W.cell(d,1,i),complement_dimension=11)for d in(1,2)for i in(0,1)]
 assert W.union(rows,['79/2048','79/2048'])['total_cells']==4
 for bad in [rows[:-1],rows+[rows[0]]]:
  with pytest.raises(ValueError):W.union(bad,['79/2048','79/2048'])
 with pytest.raises(ValueError):W.union(rows,['0','0'])


def test_actual_complete_tree_and_negative_counter():
 p=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/cubic32-first-quarter001/run');out=W.read(p/'outcome.json');dd,cells=W.tree_history(out,p)
 assert [v['ordinal']for v in cells]==[2,3,5,6]and len(dd)==2 and cells[-1]['start']+126==1080
 bad=copy.deepcopy(out);bad['completed_attempts'][0]['completed']=False
 with pytest.raises(ValueError):W.tree_history(bad,p)


def test_missing_pins_reject_before_read():
 class Nope:
  def tick(self):raise AssertionError('must not open input')
 with pytest.raises(ValueError):W.metadata(Nope(),dict(status='prepared',producer={}))


def test_actual_metadata_no_scientific_decode(monkeypatch):
 def no(*_a,**_kw):raise AssertionError('scientific array or model decoder invoked')
 monkeypatch.setattr(W.G.P,'small_matrix',no);monkeypatch.setattr(W.G,'exact_inverse',no);monkeypatch.setattr(W.G.B,'TwoRateCuspMap',no)
 class Meta:
  def tick(self):pass
  def save(self,*_):pass
 card=W.read(ROOT/W.CARD);card['status']='reviewed';run,dd,cells,projection,data,pc=W.metadata(Meta(),card)
 assert len(cells)==4 and data['index']['ordinal']==32 and W.F(39,1024)<=W.F(projection[0])<=W.F(projection[1])<=W.F(55,1024)


def test_reserve(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.used=252*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',{})
 w.save('outcome.json',dict(passed=False),True)
 assert W.read(tmp_path/'outcome.json')['passed']is False


def test_isolated_archive_closure(tmp_path):
 for n in W.FILES:
  src=ROOT/n;dst=tmp_path/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools/reviews','tools','src'];import review_cubic_first as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
