import os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).parents[2];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_cifar_regularized as W

def test_independent_whitening_is_invariant_to_full_rank_basis():
 V=np.eye(3);H=np.diag([.01,1.,3.]);g=np.ones(3);mix=np.array([[2.,.3,0.],[0.,.5,.2],[0.,0.,4.]])
 a,b,ra=W.proposal(V,H,g);x,y,rb=W.proposal(V@mix,H@mix,g)
 assert np.linalg.norm(a-x)<1e-11 and np.linalg.norm(b-y)<1e-11 and abs(ra['forcing']-.49)<1e-12
 assert np.linalg.norm(a)<np.linalg.norm(np.linalg.solve(H,-g))
 with pytest.raises(ValueError):W.proposal(np.array([[1.],[0.]]),np.array([[1.],[0.]]),np.array([0.,1.]))

def test_actual_metadata_binds_every_native_call_before_arrays(monkeypatch):
 def forbidden(*a,**kw):raise AssertionError('target array preflight')
 monkeypatch.setattr(np,'load',forbidden);r=W.bind(lambda:None)
 assert r[-1]==dict(hvp=1,direct_gradient=5)and r[4][-1]['accepted']

def test_corrupt_native_file_or_event_order_is_rejected():
 es=[W.json.loads(v)for v in(W.PARENT/'run/events.jsonl').read_text().splitlines()];inv=W.M.read(W.PARENT/'run/manifest.json')['files'];W.events(es,inv)
 bad=[dict(e)for e in es];bad[1]=dict(bad[1],index=99)
 with pytest.raises(ValueError):W.events(bad,inv)
 bad=[dict(e)for e in es];bad[-1]=dict(bad[-1],index=99)
 with pytest.raises(ValueError):W.events(bad,inv)

def test_isolated_source_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code="import sys;from pathlib import Path;sys.path.insert(0,'tools/reviews');import review_cifar_regularized as w;r=Path.cwd();a={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert a<=set(w.FILES),a-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True);assert p.returncode==0,p.stderr
