import copy,json,os,subprocess,sys
from pathlib import Path
from fractions import Fraction as F
import pytest
ROOT=Path(os.environ.get('SECOND_QUARTER_REVIEW_ROOT',str(Path(__file__).resolve().parents[2])))
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import review_second_quarter as W
W.ROOT=ROOT


def test_union_and_whole_previous_projection():
 rows=[dict(passed=True,cell=W.cell(d,1,i),complement_dimension=11)for d in(1,2)for i in(0,1)]
 assert W.union(rows,['-1/65536','1/65536'])['total_cells']==4
 for bad in [rows[:-1],rows+[rows[0]]]:
  with pytest.raises(ValueError):W.union(bad,['0','0'])
 with pytest.raises(ValueError):W.union(rows,['0','1/64'])
 assert W.domains()[0]['endpoints'][1]==W.domains()[1]['endpoints'][0]


def test_actual_history_and_input_before_return_schema():
 p=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/second-quarter001/run');out=W.read(p/'outcome.json');ds,cells=W.tree_history(out,p)
 assert len(ds)==2 and [c['ordinal']for c in cells]==[2,3,5,6]
 assert out['native_calls']==1080 and cells[-1]['start']+126==1080
 bad=copy.deepcopy(out);bad['completed_attempts'][0]['completed']=False
 with pytest.raises(ValueError):W.tree_history(bad,p)


def test_missing_authority_rejects_before_arrays():
 class Nope:
  def tick(self):raise AssertionError('must not touch authority')
 with pytest.raises(ValueError):W.metadata(Nope(),dict(status='prepared',producer={}))


def test_reserve(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.used=252*1024**2
 with pytest.raises(ValueError):w.save('no.json',{})
 w.save('outcome.json',{'passed':False},True);assert W.read(tmp_path/'outcome.json')['passed']is False


def test_archive_closure(tmp_path):
 local=Path(__file__).parent
 packet={'tools/reviews/review_second_quarter.py':'review_second_quarter.py','tools/reviews/launch_second_quarter_review.py':'launch_second_quarter_review.py','tests/reviews/test_second_quarter_review.py':'test_second_quarter_review.py','docs/plans/native705-second-quarter-independent.md':'native705-second-quarter-independent.md','docs/plans/native705-second-quarter-independent-inputs.json':'native705-second-quarter-independent-inputs.json'}
 for n in W.FILES:
  src=ROOT/n
  if not src.exists()and n in packet:src=local/packet[n]
  target=tmp_path/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(src.read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools/reviews','tools','src'];import review_second_quarter as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
