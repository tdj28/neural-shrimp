import copy,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import diagnose_clark_correlated_extent as W

def test_full51_exact_ratios_and_coefficient_separation():
 p=W.synthetic();o=W.diagnose(p);assert o['inclusions']==51 and o['max_residual_over_radius']==1/16 and o['max_image_over_radius']==1/16 and 0<o['exact_decoded_max_defect']<o['saved_contraction']
 assert len(o['all_coordinates'])==51 and len(o['largest'][0]['forcing_term_magnitudes'])==5
 assert W.dyad([3,-1000])*2**1000==3

def test_changed_gates_and_incomplete_radius_reject():
 p=W.synthetic();p['image_correction'][0][0]['mid']=[32,0]
 with pytest.raises(ValueError,match='gates'):W.diagnose(p)
 p=W.synthetic();p['radii'].pop()
 with pytest.raises(ValueError,match='all51'):W.diagnose(p)
 with pytest.raises(ValueError):W.ball(dict(mid=[0,0],rad=[-1,0]))

def test_actual_complete_metadata_before_decoding(monkeypatch):
 read=W.read
 def bounded(p):
  assert not p.name.endswith('.gz'),'no scientific arrays during metadata preflight'
  return read(p)
 monkeypatch.setattr(W,'read',bounded)
 class Work:
  def tick(self):pass
  def save(self,*a,**kw):pass
 co=W.bind(Work());assert co['accepted']==22 and len(co['attempts'])==32

def test_closed_source_and_exclusive_limits():
 assert all((ROOT/p).exists()for p in W.FILES)
 launcher=(ROOT/'tools/launch_clark_correlated_extent_diagnosis.py').read_text()
 assert "'output_cap_bytes':10*1024**2"in launcher and "['status']=='reviewed'"in launcher
 assert W.CAPS['native_calls']==0 and W.CAPS['seconds']==30
