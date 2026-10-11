import sys,gzip,json
from pathlib import Path
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools/reviews')]
import review_census72_ad_driver as D

def test_fixed1680_phase_cost_and_gzip_read(tmp_path):
    assert D.forecast([1.,2.,3.],[10,20,30])==dict(seconds=10140.,bytes=100800+128*1024**2)
    for value in [[],[1],[1,0,2]]:
        with pytest.raises(ValueError):D.forecast(value,[1,2,3])
    p=tmp_path/'proof.json.gz';p.write_bytes(gzip.compress(b'{"exact":"1/3"}',mtime=0));assert D.read(p)=={'exact':'1/3'}

def test_wrong_inventory_and_mutable_source_rejected(tmp_path):
    p=tmp_path/'proof.json';p.write_text('{}');m=tmp_path/'manifest.json';m.write_text(json.dumps(dict(files={'proof.json':D.entry(p)})))
    with pytest.raises(ValueError):D.inventory(tmp_path,D.entry(m)['sha256'])
    p.chmod(0o444);D.inventory(tmp_path,D.entry(m)['sha256'])
    with pytest.raises(ValueError):D.inventory(tmp_path,'0'*64)
