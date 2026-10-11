import copy,sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'tools/reviews'),str(Path(__file__).resolve().parents[1])]
import pytest
from flint import arb,ctx
from neural_shrimp import rigorous_clark_correlated_path as P
from test_rigorous_clark_correlated_path import setup
import review_clark_correlated_acquisitions as A


def fixture(full=True):
 j,c,s,a,b=setup();e=[arb(0,arb(1)/2**24)]*11 if full else[arb(0)]*11;d=arb(0,arb(1)/512);events=[];f,J=P.coefficients(j,c,s,a,e,b,d,observe=lambda k,v:events.append((k,v)));request=dict(center=P.vector_json(c),slope=P.vector_json(s),curvature=P.vector_json(a),error=P.vector_json(e),parameter=P.ball_json(b),d=P.ball_json(d));inputs=[v for k,v in events if k=='clark_gradient_path_input'];returns=[v for k,v in events if k=='clark_gradient_path_return'];return request,inputs,returns,f,J


def test_raw_native_assembly_matches_independent_full_actual_columns():
 with ctx.workprec(480):
  for full in(False,True):
   r,i,o,f,J=fixture(full);g,K=A.assemble_path(r,i,o)
   assert all(x.contains(0)for row in(g-f).tolist()for x in row)
   assert all(x.contains(0)for a,b in zip(K,J,strict=True)for row in(a-b).tolist()for x in row)


def test_changed_native_chain_base_gradient_and_fullE_rejected():
 with ctx.workprec(480):
  r,i,o,_,_=fixture();bad=copy.deepcopy(i);bad[1]['even'][0]['value'][0][0]=P.ball_json(arb(9))
  with pytest.raises(ValueError,match='chain'):A.assemble_path(r,bad,o)
  bad=copy.deepcopy(i);bad[0]['even'][0]['gradient'][0][0][0]=P.ball_json(arb(0))
  with pytest.raises(ValueError,match='basis'):A.assemble_path(r,bad,o)
  bad=copy.deepcopy(r);bad['error'][0]=P.ball_json(arb(0,arb(1)/8))
  with pytest.raises(ValueError,match='coverage'):A.assemble_path(bad,i,o)


def test_exact_quadratic_range_retains_interior_extremum():
 from fractions import Fraction as F
 assert A.quadratic(F(0),F(0),F(2),(F(-1),F(1)),(F(0),F(0)))==(F(0),F(1))
 assert A.quadratic(F(0),F(1),F(2),(F(-1),F(1)),(F(-1,8),F(1,8)))==(F(-3,8),F(17,8))
