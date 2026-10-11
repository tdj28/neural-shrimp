import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/reviews'))
import check_native705_chart31_boundary_hypotheses as H

def test_actual_complete_mapping():
 r=H.check();assert r['passed']and r['chart']==31 and r['independent_complement_cells']==31 and len(r['hypotheses'])==6

def test_missing_original_root_gate_rejected(monkeypatch):
 read=H.read
 def changed(p):
  r=read(p)
  if p.name=='independent-proof.json':r=copy.deepcopy(r);r['gates']['positive_rates_angle']=False
  return r
 monkeypatch.setattr(H,'read',changed)
 with pytest.raises(ValueError,match='theorem gates'):H.check()
