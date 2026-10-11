import sys,json,gzip
from pathlib import Path
import pytest
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from certify_census72_panel import selected,forecast,proximity,validate_panel,K

def test_complete_selection_and_duplicate_rejection():
    rows=[dict(map_id='m',component=i,certificate=dict(orbit=[[0]*13 if i<28 else[0]*72]))for i in range(61)]
    s=dict(gate=dict(qualifying=rows));a=dict(audit_gate=dict(qualifying=rows));assert selected(s,a,rows)==list(range(28,61))
    with pytest.raises(ValueError):selected(s,a,rows+[rows[28]])

def test_fixed_complete_panel_no_missing_or_changed_period():
    card=json.loads((Path(__file__).resolve().parents[1]/'docs/plans/census72-panel-inputs.json').read_text())
    periods=[22,38,34,30,34,12,14,4,8,44,20,30,22,30,20,20,44,20,20,20,14,30,58,16,50,30,16,46,34,42,16,14,46]
    prepared={i:dict(period=p)for i,p in zip(card['all72_indices'],periods)}
    validate_panel(card['all72_indices'],card,prepared)
    with pytest.raises(ValueError):validate_panel(card['all72_indices'][:-1],card,prepared)
    prepared[88]['period']=48
    with pytest.raises(ValueError):validate_panel(card['all72_indices'],card,prepared)

def test_cost_includes_every_native_return_and_retention():
    assert forecast([.01,.02,.03],[100,200,300])==dict(seconds=463.2,bytes=138249728)
    with pytest.raises(ValueError):forecast([.01],[100])
    with pytest.raises(ValueError):forecast([.01,.02,float('nan')],[100,200,300])

def test_proximity_checks_whole72box():
    tiny=[dict(mid=[0,0],rad=[1,-100])for _ in range(72)]
    assert Fraction(proximity(tiny,['0']*72))==Fraction(2)**-100
    tiny[71]['rad']=[1,-10]
    with pytest.raises(ValueError):proximity(tiny,['0']*72)

def test_retained_exact_ball_serialization_roundtrip():
    raw=K.canonical_bytes(dict(input=[dict(mid=[2**767+3,-799],rad=[2**33+1,-999])],jacobian=[[dict(mid=[-7,-90],rad=[1,-150])]]))
    compressed=gzip.compress(raw,compresslevel=1,mtime=0)
    assert gzip.decompress(compressed)==raw
    assert gzip.compress(raw,compresslevel=1,mtime=0)==compressed
