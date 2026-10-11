import sys,json
from pathlib import Path
import pytest
sys.path[:0]=[str(Path(__file__).parents[1]/'tools/reviews'),str(Path(__file__).parents[1]/'tools')]
import review_rotated705 as W
import launch_rotated705_review as L

def test_card_source_and_owned_caps():
 card=W.read(W.ROOT/W.CARD)
 assert L.OUT==W.OUT.parent and card['audit']==dict(output=str(W.OUT),seconds=600,rss_bytes=6*1024**3,as_bytes=10*1024**3,raw_bytes=1024*1024**2,precision=480,new_roots=1,native_calls=0)
 assert all((W.ROOT/n).is_file()for n in W.FILES)
 s=Path(L.__file__).read_text();assert "'internal_seconds':600"in s and "'external_seconds':620"in s and '1026*1024**2'in s

def test_own_profile_and_no_exact_rotated_parameter():
 s=Path(W.__file__).read_text();assert 'forecast=8*elapsed+120;raw=4*stored+64*1024**2'in s and 'allowed_widths=(-6,)'in s and 'known_parameter='not in s

def test_malformed_independent_matrix_rejected(tmp_path):
 p=tmp_path/'bad.json';p.write_text(json.dumps(dict(encoding='wrong',rows=[])))
 with pytest.raises(ValueError):W.matrix_file(p,705)


def test_actual_unrelated_profile_setup_and_reduction():
 from flint import arb,ctx
 with ctx.workprec(480):
  C,J,res,h,rho=W.B.synthetic_fixture(2)
  assert h==arb(1)/32 and rho==arb(1)/1024
  proof=W.H.polynomial_root(C,J,res,[arb(1)]*2,rho,h,[arb(0,rho)]*2)
  assert proof['passed'] and len(proof['coordinate_inclusions'])==2
