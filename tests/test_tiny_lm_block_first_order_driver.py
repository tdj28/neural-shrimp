import copy,os,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import explore_tiny_lm_block_first_order as W

def test_prepared_admission_blocks_and_exact_counts(tmp_path):
 c=W.S.read(ROOT/W.CARD);c.update(status='prepared',mean_authority=None)
 with pytest.raises(ValueError):W.card_check(c,'profile')
 c.update(status='profile_admitted',mean_authority=dict(root='x',source='x',outer='x',worker='x'),source_review={'result':'PASS'})
 W.card_check(c,'profile')
 with pytest.raises(ValueError):W.card_check(c,'target')
 c['status']='target_admitted'
 with pytest.raises(ValueError):W.card_check(c,'target')
 a=W.Calls(W.Store(tmp_path/'target'),'target');b=W.Calls(W.Store(tmp_path/'profile'),'profile')
 assert a.cap=={'hvp':448}and b.cap=={'hvp':2}and a.seconds==1200 and b.seconds==60

def test_raw_budget_and_closing_reserve(tmp_path):
 s=W.Store(tmp_path/'store');s.used=W.CAP-W.RESERVE
 with pytest.raises(ValueError):s.write('blocked',b'x')
 s.write('outcome',b'x',True)

def test_full_archive_import_closure_and_isolated_launcher(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import explore_tiny_lm_block_first_order as w;import launch_tiny_lm_block_first_order as l;r=Path.cwd();a={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert a<=set(w.FILES),a-set(w.FILES);assert l.BASE==w.BASE;assert w.H.TARGET_COUNTS=={'hvp':448}"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True);assert p.returncode==0,p.stderr

def test_bound_actual_metadata_after_independent_admission(monkeypatch):
 c=W.S.read(ROOT/W.CARD)
 if c['mean_authority']is None:pytest.skip('independent baseline not yet bound; admission closed')
 def no(*a,**k):raise AssertionError('target arrays/model prohibited before profile')
 monkeypatch.setattr(W.S,'load_object',no);monkeypatch.setattr(W.N,'LMProblem',no);monkeypatch.setattr(W.P,'load_case',no)
 class Store:
  def json(self,*a,**k):pass
 class Calls:
  def check(self):pass
 result=W.metadata(Store(),Calls());assert result[3]>=.8502848679199815
