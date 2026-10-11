import sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_clark_endpoint_math as E

def fixture(n=2):
 C=[E.M.eye(n),arb_mat(n,n)];J=[E.M.eye(n),arb_mat(n,n),arb_mat(n,n)];J[0][0,0]=arb(3)/4
 R=[E.M.col([arb(1)/1024]*n),E.M.col([arb(1)/8]*n),arb_mat(n,1),arb_mat(n,1)]
 d=E.H.reduction(C,J,R,[arb(1)]*n,arb(1)/8);z=[arb(0)]*n;old=[d['image'][i,0]for i in range(n)];return d,z,old

def test_exact_point_iteration_shrinks_known_root_and_is_monotone():
 with ctx.workprec(512):
  d,z,old=fixture();r=E.localize(d,z,z,z,old);assert r['passed']and all(x['ratio']<.01 for x in r['rows'])
  expected=-arb(17)/768;assert r['endpoint'][0].contains(expected)
  for previous,new in zip([r['initial']]+r['history'][:-1],r['history']):assert all(E.cover(a,b)for a,b in zip(previous,new))

def test_missing_old_root_or_failed_tube_is_rejected():
 with ctx.workprec(512):
  d,z,old=fixture()
  with pytest.raises(ValueError,match='covers'):E.localize(d,z,z,z,z)
  d['passed']=False
  with pytest.raises(ValueError,match='proved'):E.localize(d,z,z,z,old)

def test_point_polynomial_noncommuting_and_full51():
 with ctx.workprec(512):
  d,z,old=fixture(51);d['defect_coefficients'][1][0,1]=arb(1)/4
  r=E.localize(d,z,z,z,old);assert len(r['endpoint'])==51 and r['point_defect'][0,1]==arb(1)/32

def test_exact_intersection_and_invalid_iteration():
 from fractions import Fraction as F
 assert E.intersection((F(-2),F(1)),(F(-1),F(3)))==(F(-1),F(1))
 with ctx.workprec(512):
  d,z,old=fixture()
  with pytest.raises(ValueError,match='bounded'):E.localize(d,z,z,z,old,iterations=17)
