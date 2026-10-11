import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_cubic_endpoint import localize_cubic_endpoint

def fixture(n=2):
 D=[arb_mat(n,n)for _ in range(4)];D[0]=arb_mat([[arb(1)/8 if i==j else 0 for j in range(n)]for i in range(n)])
 f=[arb_mat(n,1)for _ in range(6)];f[0]=arb_mat([[arb(7)/8192]for _ in range(n)])
 return D,f,arb(1)/32,arb(1)/32,[arb(0)]*n,[arb(0,arb(1)/64)]*n,[arb(1)]*n,[arb(0,arb(1)/32)]*n

def test_conditional_point_contraction_retains_known_root():
 with ctx.workprec(256):
  box,p=localize_cubic_endpoint(*fixture());assert p['passed']and all(v.contains(arb(1)/1024)for v in box)
  assert all(v.rad()<arb((1,-18))for v in box)

def test_highest_defect_and_forcing_coefficients_not_dropped():
 with ctx.workprec(256):
  args=list(fixture(1));args[2]=args[3]=arb(1)/4;args[0][3][0,0]=8;args[1][5][0,0]=1
  box,p=localize_cubic_endpoint(*args);d=args[2];answer=(arb(7)/8192+d**5)/(1-arb(1)/8-8*d**3)
  assert box[0].contains(answer)and not box[0].contains(arb(1)/1024)

def test_invalid_domain_or_unstable_point_rejected():
 args=list(fixture());args[2]=arb(1)
 with pytest.raises(ValueError):localize_cubic_endpoint(*args)
 args=list(fixture());args[0][0][0,0]=1
 with pytest.raises(ValueError):localize_cubic_endpoint(*args)
 args=list(fixture());args[4][0]=arb(0,1)
 with pytest.raises(ValueError):localize_cubic_endpoint(*args)


def test_fresh_rhs_at_shifted_point_retains_identical_known_root():
 from neural_shrimp.rigorous_cubic_endpoint import localize_fresh_endpoint
 with ctx.workprec(256):
  D=[arb_mat([[arb(1)/8]])]+[arb_mat(1,1)for _ in range(3)]
  point=[arb(1)/1024];known=[arb(point[0],arb(1)/64)]
  box,p=localize_fresh_endpoint(D,arb_mat([[arb(1)/2]]),arb_mat(1,1),arb_mat([[arb(1)/65536]]),arb(1)/32,arb(1)/32,point,known,[arb(1)])
  assert box[0].contains(point[0]-arb(1)/114688)and box[0].rad()<arb((1,-20))
