import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp import rigorous_polynomial_point as P

def test_nonlinear_known_root_tightens_at_negative_original_offset():
 with ctx.workprec(384):
  # H(x,t)=x²-(4+t), t=-1; original x domain[1.5,2].
  C=[arb_mat([[arb(1)/4]]),arb_mat([[0]])];D=[arb_mat([[arb(1)/8+arb(0,arb(1)/8)]])];known=[arb('1.75',arb(1)/4)];x=[arb(7)/4];proof=None
  for i in range(3):
   G=arb_mat([[x[0]*x[0]-3]])
   if i<2:x=P.finite_point_correction(C,G,x,-1,1)
   else:box,proof=P.localize_fresh_polynomial_point(C,D,G,x,known,[arb(1)],-1,1)
  assert box[0].contains(arb(3).sqrt())and box[0].rad()<known[0].rad()/1000 and proof['offset']=='-1'

def test_full_coordinates_and_actual_interval_inverse_sum():
 with ctx.workprec(256):
  C=[arb_mat([[1,0],[0,2]]),arb_mat([[0,1],[0,0]])];D=[arb_mat([[0,arb(1)/16],[0,0]])];x=[arb(1),arb(2)];known=[arb(1,arb(1)/8),arb(2,arb(1)/8)];G=arb_mat([[0],[0]])
  box,proof=P.localize_fresh_polynomial_point(C,D,G,x,known,[arb(1)]*2,0,1)
  assert all(v.contains(p)for v,p in zip(box,x))and proof['dimension']==2
  bad=arb_mat([[0],[arb('nan')]])
  with pytest.raises(ValueError):P.localize_fresh_polynomial_point(C,D,bad,x,known,[arb(1)]*2,0,1)

def test_exact_point_and_original_parameter_guards():
 C=[arb_mat([[1]])];D=[arb_mat([[0]])];G=arb_mat([[0]])
 for point,offset in [([arb(1,arb(1)/8)],0),([arb(1)],2)]:
  with pytest.raises(ValueError):P.localize_fresh_polynomial_point(C,D,G,point,[arb(1,1)],[arb(1)],offset,1)
 with pytest.raises(ValueError):P.localize_fresh_polynomial_point(C,[arb_mat([[2]])],G,[arb(1)],[arb(1,1)],[arb(1)],0,1)
