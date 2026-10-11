import sys
from pathlib import Path
import pytest
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from certify_census13_panel import selected,forecast,proximity

def test_complete_selection_and_duplicate_rejection():
    rows=[dict(map_id='m',component=i,certificate=dict(orbit=[[0]*13 if i<28 else[0]*72]))for i in range(61)]
    s=dict(gate=dict(qualifying=rows));a=dict(audit_gate=dict(qualifying=rows));assert selected(s,a,rows)==list(range(28))
    with pytest.raises(ValueError):selected(s,a,rows+[rows[0]])

def test_cost_counts_whole_panel_and_retention():
    assert forecast([.01,.02,.03],[100,200,300])==dict(seconds=282.,bytes=69628864)
    with pytest.raises(ValueError):forecast([.01],[100])

def test_proximity_checks_whole_box_not_only_center():
    tiny=[dict(mid=[0,0],rad=[1,-100])for _ in range(13)]
    assert Fraction(proximity(tiny,['0']*13))==Fraction(2)**-100
    tiny[2]['rad']=[1,-10]
    with pytest.raises(ValueError):proximity(tiny,['0']*13)
