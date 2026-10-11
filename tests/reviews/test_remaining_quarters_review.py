import copy,json,os,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(os.environ.get('REMAINING_QUARTERS_REVIEW_ROOT','/home/d7082791602/.codex/worktrees/tiny-lm-prediction-review/neural-shrimp'))
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import review_remaining_quarters as W
W.ROOT=ROOT

def prior():return dict(passed=True,parameter_endpoints=['-17/2048','15/2048'],first_and_second_union=['-49/2048','15/2048'],complement_dimension=11,prior_stable_projection=['-1/65536','1/65536'])
def rows():return [dict(passed=True,quarter=q,cell=W.cell(q,d,1,i),complement_dimension=11)for q in(3,4)for d in(1,2)for i in(0,1)]
def test_unions():
 r=rows();u=W.union(r,prior());assert u['whole_chart_endpoints']==['-49/2048','79/2048']and u['total_cells']==8
 for bad in(r[:-1],r+[r[0]]):
  with pytest.raises(ValueError):W.union(bad,prior())
 p=prior();p['prior_stable_projection']=['0','1/64']
 with pytest.raises(ValueError):W.union(r,p)
 with pytest.raises(ValueError):W.domains(2)

def test_missing_parent():
 class No:
  def tick(self):raise AssertionError('no arrays')
 with pytest.raises(ValueError):W.metadata(No(),dict(status='prepared',producer=None))

def test_reserve_and_quarter_output(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.prefix='quarters/3';w.save('domains/001/root.json',{'passed':True});assert (tmp_path/'quarters/3/domains/001/root.json').is_file();w.prefix='';w.used=508*1024**2
 with pytest.raises(ValueError):w.save('no.json',{})
 w.save('outcome.json',{'passed':False},True)

def test_archive_closure(tmp_path):
 local=Path(__file__).parent;packet={'tools/reviews/review_remaining_quarters.py':'review_remaining_quarters.py','tools/reviews/launch_remaining_quarters_review.py':'launch_remaining_quarters_review.py','tests/reviews/test_remaining_quarters_review.py':'test_remaining_quarters_review.py','docs/plans/native705-remaining-quarters-independent.md':'native705-remaining-quarters-independent.md','docs/plans/native705-remaining-quarters-independent-inputs.json':'native705-remaining-quarters-independent-inputs.json'}
 for n in W.FILES:
  src=local/packet[n]if n in packet and(local/packet[n]).exists()else ROOT/n;target=tmp_path/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(src.read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools/reviews','tools','src'];import review_remaining_quarters as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr

def test_actual_complete_tree_and_domain_prefixes():
 p=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/remaining-quarters001/run');out=W.read(p/'outcome.json');ds,cs=W.tree_history(out,p)
 assert len(ds)==4 and [c['ordinal']for c in cs]==[2,3,5,6,8,9,12,13,15,16] and out['native_calls']==2520
 assert [d['quarter']for d in ds]==[3,3,4,4]
 bad=copy.deepcopy(out);bad['completed_attempts'][0]['quarter']=4
 with pytest.raises(ValueError):W.tree_history(bad,p)
 bad=copy.deepcopy(out);bad['completed_attempts'][0]['completed']=False
 with pytest.raises(ValueError):W.tree_history(bad,p)
