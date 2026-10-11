import sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/reviews'))
import review_native705_minimum_math as A
from neural_shrimp.rigorous_orbits import ball_json


def test_point_operator_noncommuting():
 with ctx.workprec(160):
  I=arb_mat([[1,0],[0,1]]);B=arb_mat([[0,arb(1)/16],[arb(1)/32,0]]);C,D,q=A.point_operator(I,[I,B,I/64],arb(1)/8,[arb(1),arb(2)])
  assert q<1
  expected=I-C*(I+B/8+I/4096)
  assert all((D[i,j]-expected[i,j]).contains(0)for i in range(2)for j in range(2))
  with pytest.raises(ValueError):A.point_operator(I,[I*3,B,I],arb(0),[arb(1)]*2)


def test_same_root_intersections_and_domain():
 with ctx.workprec(160):
  point=[arb(0),arb(0)];known=[ball_json(arb(1,arb(1)/4)),ball_json(arb(2,arb(1)/4))]
  ans=A.localize(arb_mat([[1,0],[0,1]]),arb_mat(2,2),arb_mat([[-1],[-2]]),point,known,[arb(1)]*2)
  assert all(A.R.original_ball(v).is_exact()for v in ans['box'])
  error=[ball_json(arb(0,arb(4)))]*2
  assert A.translated_domain(error,point,[known,[ball_json(v)for v in point]])
  bad=[dict(v)for v in known];bad[0]=ball_json(arb(5))
  with pytest.raises(ValueError):A.translated_domain(error,point,[bad])


def test_minimum_exact_order_and_scope():
 def row(x):return [ball_json(arb(0))]*702+[ball_json(arb(x)),ball_json(arb(1)),ball_json(arb(1))]
 rows=[row(3),row(1),row(2)];a=A.interior_minimum(rows,True);assert a['left_gap']=='2'and a['right_gap']=='1'
 with pytest.raises(ValueError):A.interior_minimum(rows,False)
 with pytest.raises(ValueError):A.interior_minimum([rows[0],rows[2],rows[1]],True)
