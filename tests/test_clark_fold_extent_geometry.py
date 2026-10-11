import os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import diagnose_clark_fold_extent as W
import launch_clark_fold_extent_geometry as L

def fixture():
 cx=np.arange(72,dtype=np.float64).reshape(6,12)/100;cy=-np.roll(cx,-3,axis=0);return np.r_[cx[0],cy[0],.8],cx,cy,np.eye(12)[0]

def test_complete_phase_coordinates_and_distinct_projection():
 p,x,y,v=fixture();a=W.coordinates(p,x,y,v)
 assert a['b']>0 and a['projected_squared_over_b']<.2 and a['normalization_error']<1e-14
 assert np.allclose(np.array(a['m'])+a['amplitude']*np.array(a['q']),x[0])
 assert np.dot(a['q'],x[0]-y[0])>0

def test_literal_phase_and_partner_mutations_reject():
 p,x,y,v=fixture()
 with pytest.raises(ValueError,match='phase0'):W.coordinates(p,np.roll(x,1,axis=0),y,v)
 y=y.copy();y[1,0]+=.01
 with pytest.raises(ValueError,match='partner'):W.coordinates(p,x,y,v)

def test_exact_actual_metadata_without_array_interpretation(monkeypatch):
 monkeypatch.setattr(W.np,'load',lambda *a,**k:(_ for _ in()).throw(AssertionError('target arrays forbidden')))
 class Meta:
  def tick(self):pass
 for s in W.SPECS:
  run,result=W.bind(Meta(),s);assert len(result['points'])==s['count']

def test_full_synthetic_and_caps():
 class Profile:
  def tick(self):pass
  def save(self,n,v):self.report=v
 p=Profile();W.synthetic(p);assert p.report['passed']and p.report['points']==66
 assert W.OUT==L.OUT and W.CAPS['native_calls']==0
 text=(ROOT/'tools/launch_clark_fold_extent_geometry.py').read_text();assert "'internal_seconds':30"in text and "'external_seconds':45"in text and "'worker_rss_kib':512*1024"in text

def test_archive_import_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path.insert(0,'tools');import diagnose_clark_fold_extent as w;import launch_clark_fold_extent_geometry as l;assert w.OUT==l.OUT;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
