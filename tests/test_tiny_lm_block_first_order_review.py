import json,os,subprocess,sys
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import review_tiny_lm_block_first_order as W

def test_closed_card_and_literal_gates():
 c=W.read(ROOT/W.CARD);c['status']='prepared'
 with pytest.raises(ValueError):W.card_check(c)
 c['status']='reviewed';W.card_check(c);c['producer']['source']='changed'
 with pytest.raises(ValueError):W.card_check(c)

def test_complete_actual_metadata_before_arrays(monkeypatch):
 original=W.read
 def read(p):
  c=original(p)
  if p==ROOT/W.CARD:c=dict(c,status='reviewed')
  return c
 monkeypatch.setattr(W,'read',read);monkeypatch.setattr(np,'load',lambda *_a,**_k:(_ for _ in()).throw(AssertionError('target arrays forbidden')))
 inv,out,mean=W.metadata(lambda:None)
 assert out['event_pairs']==448 and mean['retained_hvp_checked']==512 and len(inv)>10

def test_archive_closure_no_native_model(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path.insert(0,'tools');import review_tiny_lm_block_first_order as w;import launch_tiny_lm_block_first_order_review as l;r=Path.cwd();a={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert a<=set(w.FILES),a-set(w.FILES);assert not any(n.startswith(('torch','neural_shrimp','explore_tiny_lm'))for n in sys.modules);assert w.OUT.parent==l.OUT"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True);assert p.returncode==0,p.stderr
