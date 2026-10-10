import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[2]/'tools/reviews'))
import audit_stability_coverage_saved as M
import pytest

def test_all_fixed_partitions():
 for i in range(9):
  rows=M.partition(i);assert len(rows)==8
  assert M.coverage(rows,i)==[str(M.Fraction(35+i,128)),str(M.Fraction(36+i,128))]
  assert all(M.Fraction(b)-M.Fraction(a)==M.Fraction(1,1024)for a,b in rows)

@pytest.mark.parametrize('i',[-1,9,True,None])
def test_invalid_index(i):
 with pytest.raises(ValueError):M.partition(i)

def test_no_skip_overlap_or_reordering():
 r=M.partition(8);r[3]=r[4]
 with pytest.raises(ValueError):M.coverage(r,8)
 with pytest.raises(ValueError):M.coverage(M.partition(8)[::-1],8)

def test_metadata_pin_and_path(tmp_path):
 p=tmp_path/'pins.json';v=dict(root=str(M.BASE/'test'),outer='1'*64,worker='2'*64,execution='3'*40,index=8)
 p.write_text(json.dumps(v));pin=M.sha(p.read_bytes());assert M.verify_metadata(p,pin)==v
 with pytest.raises(ValueError):M.verify_metadata(p,'0'*64)
 v['root']='/tmp/unrelated';p.write_text(json.dumps(v))
 with pytest.raises(ValueError):M.verify_metadata(p,M.sha(p.read_bytes()))

def test_omitted_source_inventory_rejected():
 with pytest.raises(ValueError,match='complete frozen source'):M.source_inventory({'source_sha256':{M.FIRST:'0'*64}},8)
