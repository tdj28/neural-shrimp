import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/reviews'))
import pytest
from flint import arb,arb_mat,ctx
import review_clark_correlated_math as H


def test_nonnormal_noncommuting_affine_convolution_and_exact_points():
 with ctx.workprec(480):
  C=[arb_mat([[1,arb(1)/8],[0,1]]),arb_mat([[0,arb(1)/16],[arb(1)/32,0]])];J=[arb_mat([[1,-arb(1)/8],[0,1]]),arb_mat([[0,-arb(1)/16],[-arb(1)/32,0]]),arb_mat([[arb(1)/64,0],[0,arb(1)/64]])];R=[arb_mat([[arb(1)/2**20],[arb(-1)/2**20]])]+[arb_mat(2,1)for _ in range(3)];h=arb(1)/16;out=H.reduction(C,J,R,[arb(1)/64]*2,h);assert out['passed']
  for d in(-h,arb(0),h):
   c=H.evaluate(C,[arb(1),d]);j=H.evaluate(J,[arb(1),d,d*d]);D=H.evaluate(out['defect_coefficients'],[H.power(d,k)for k in range(4)]);diff=D-(H.M.eye(2)-c*j)
   assert all(x.contains(0)for row in diff.tolist()for x in row)
   f=H.evaluate(out['forcing_coefficients'],[H.power(d,k)for k in range(5)]);r=H.evaluate(R,[H.power(d,k)for k in range(4)]);assert all(x.contains(0)for row in(f+c*r).tolist()for x in row)


def test_centered_cubic_parity_and_strict_negative_gate():
 with ctx.workprec(480):
  ds=H.monomials(arb(1)/8,5);assert all(v.is_finite()for v in ds)and ds[2].contains(arb(0))and ds[2].contains(arb(1)/64)and ds[2].lower()>-arb(1)/2**30 and ds[3].contains(-arb(1)/512)and ds[3].contains(arb(1)/512)
  C=[arb_mat([[1]]),arb_mat([[0]])];J=[arb_mat([[1]]),arb_mat([[0]]),arb_mat([[0]])];R=[arb_mat([[arb(1)/8]])]+[arb_mat(1,1)for _ in range(3)];assert not H.reduction(C,J,R,[arb(1)/8],arb(1)/8)['passed']
  R[0]=arb_mat([[0]]);J[0]=arb_mat([[0]]);assert not H.reduction(C,J,R,[arb(1)/8],arb(1)/8)['passed']


def test_full51_endpoint_and_changed_phase_join_rejection():
 with ctx.workprec(480):
  n=51;C=[H.M.eye(n),arb_mat(n,n)];J=[H.M.eye(n),arb_mat(n,n),arb_mat(n,n)];R=[arb_mat(n,1)for _ in range(4)];r=[arb(1)/128]*n;h=arb(1)/64;c=[arb(i)/256 for i in range(n)];s=[arb(1)/8]*n;a=[arb(1)/16]*n;out=H.reduction(C,J,R,r,h);assert out['passed'];end=H.endpoint(out,c,s,a,1)
  assert H.join(end,h,c,s,a,arb(0),h,r)['coordinates']==51
  end[2]+=arb(1)/4;assert not H.join(end,h,c,s,a,arb(0),h,r)['passed']


def test_anchor_requires_known_nondegenerate_crossing_and_full_inverse():
 with ctx.workprec(480):
  A=H.M.eye(3);rhs=arb_mat([[arb(1)/4],[0],[arb(1)/8]]);z=[arb(0),arb(1)/2,arb(1),arb(2)];known=H.anchor(A,rhs,z,arb(3));assert len(known)==7 and known[5]==0
  with pytest.raises(ValueError):H.anchor(A,rhs,z,arb(0,1))


def test_native_primitive_and_complement_are_separate_from_critical_line():
 with ctx.workprec(480):
  cycle=[[arb(i),arb(i*i)]for i in range(6)];A=arb_mat([[1,0],[0,arb(1)/2]]);q=[arb(1),arb(0)];out=H.native_context(cycle,[A]*6,q);assert out['passed']and out['witness']==1
  constant=[cycle[0]]*6;assert not H.native_context(constant,[A]*6,q)['passed']
  unstable=arb_mat([[1,0],[0,arb(3)/2]]);assert not H.native_context(cycle,[unstable]*6,q,max_power=4)['passed']
