import copy,json,sys,tempfile,os,subprocess
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_clark_extent as W

def test_complete_extent_metadata_without_scientific_decoding(monkeypatch):
 card=json.loads((ROOT/W.CARD).read_text());card['status']='reviewed';read=W.read
 monkeypatch.setattr(W,'read',lambda p:card if p==ROOT/W.CARD else read(p));monkeypatch.setattr(W.M,'decode',lambda x:x)
 class Work:
  def tick(self):pass
  def save(self,*a,**kw):pass
 paths,kinds,known,binding=W.metadata(Work());assert len(paths)==2257 and len(known)==51 and kinds.count('context_attempt')==22 and kinds.count('correlated_explicit_tube_return')==32


def test_changed_independent_endpoint_identity_rejected(monkeypatch):
 card=json.loads((ROOT/W.CARD).read_text());card['status']='reviewed';card['independent_extension']['endpoint']['sha256']='0'*64;read=W.read
 monkeypatch.setattr(W,'read',lambda p:card if p==ROOT/W.CARD else read(p))
 class Work:
  def tick(self):pass
  def save(self,*a,**kw):pass
 with pytest.raises(ValueError,match='starting authority'):W.metadata(Work())


def test_closed_actual_modules_and_strict_final_caps():

 source=(ROOT/'tools/reviews/review_clark_extent.py').read_text();launcher=(ROOT/'tools/launch_clark_extent_review.py').read_text()
 assert "out['elapsed_seconds']<300"in source and "'external_seconds':315"in launcher and "['status']=='reviewed'"in launcher
 assert W.EXPECTED['paired_native_phase']==132 and W.EXPECTED['native_even_odd_phase']==294 and W.EXPECTED['clark_gradient_path_return']==480


def test_isolated_complete_archive_import(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;sys.path.insert(0,'tools/reviews');import review_clark_extent as W;assert sum(W.EXPECTED.values())==2257;assert not any('rigorous_clark_' in n for n in sys.modules);from pathlib import Path;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r)) for m in sys.modules.values() if getattr(m,'__file__',None) and Path(m.__file__).is_file() and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(W.FILES),loaded-set(W.FILES)"
 r=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=15)
 assert r.returncode==0,r.stderr
