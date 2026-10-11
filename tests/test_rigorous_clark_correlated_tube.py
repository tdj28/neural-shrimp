from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp import rigorous_clark_correlated_tube as T


def test_noncommuting_convolution_and_odd_even_interval_powers():
 with ctx.workprec(256):
  C=[arb_mat([[1,1],[0,1]]),arb_mat([[0,1],[1,0]])];J=[arb_mat([[1,-1],[0,1]]),arb_mat([[2,1],[0,3]]),arb_mat([[0,1],[2,0]])];R=[T.col([k+1,-k])for k in range(4)];h=arb(1)/64;got=T.reduce(C,J,R,[arb(1)]*2,h)
  assert all(x.is_zero()for row in(got['defect_coefficients'][1]-(-C[0]*J[1]-C[1]*J[0])).tolist()for x in row)
  for d in[-h,arb(0),h/2,h]:
   cd=C[0]+C[1]*d;jd=J[0]+J[1]*d+J[2]*d*d;rd=R[0]+R[1]*d+R[2]*d*d+R[3]*d*d*d
   actual=T.eye(2)-cd*jd;forcing=-cd*rd
   assert all(got['whole_defect'][i,k].contains(actual[i,k])for i in range(2)for k in range(2))
   assert all(got['residual'][i,0].contains(forcing[i,0])for i in range(2))
  assert all(x.is_finite()for x in T.powers(h))


def test_scalar_common_parameter_cancellation_passes_fixed_inverse_fails():
 with ctx.workprec(256):
  # J=1+4d and C=1-4d; product defect is16d².
  h=arb(1)/8;r=[arb(1)/32];zero=arb_mat(1,1)
  got=T.reduce([arb_mat([[1]]),arb_mat([[-4]])],[arb_mat([[1]]),arb_mat([[4]]),zero],[zero]*4,r,h)
  assert got['passed']and got['contraction']<arb(1)/3
  bad=T.reduce([arb_mat([[1]]),zero],[arb_mat([[1]]),arb_mat([[16]]),zero],[zero]*4,r,h)
  assert not bad['passed']and not all(bad['coordinate_inclusions'])


def test_tube_acquires_full_same_error_and_all51_curved_root_join():
 with ctx.workprec(192):
  n=51;c=[arb(1)]*n;s=[arb(2)]*n;a=[arb(2)]*n;b=arb(1);h=arb(1)/64;r=[arb(1)/512]*n;calls=[]
  def synthetic(j,c,s,a,E,b,d,**kw):
   calls.append((list(E),d));F=arb_mat(n,4)
   for i in range(n):F[i,0]=E[i]
   return F,[T.eye(n),arb_mat(n,n),arb_mat(n,n)]
  whole,ends,data,report=T.tube([[arb(0)]*12 for _ in range(12)],c,s,a,b,h,r,acquire=synthetic)
  assert data['passed']and report['dimension']==51 and len(calls)==4
  assert all(x==0 for x in calls[0][0]+calls[1][0])and calls[2][0]==calls[3][0]and all(x.rad()>0 for x in calls[2][0])
  assert calls[2][1]==0 and calls[3][1].contains(-h)and calls[3][1].contains(h)
  for d in[-h,arb(0),h]:assert T.join([1+2*d+d*d]*n,b+d,c,s,a,b,h,r)['passed']
  bad=ends[1].copy();bad[-1]+=1;assert not T.join(bad,b+h,c,s,a,b,h,r)['passed']
  assert not T.join(ends[1],b+2*h,c,s,a,b,h,r)['passed']


def test_all51_inclusions_and_exact_data_fail_closed():
 with ctx.workprec(192):
  n=51;I=T.eye(n);zero=arb_mat(n,n);R=[arb_mat(n,1)for _ in range(4)];R[0][50,0]=1;got=T.reduce([I,zero],[I,zero,zero],R,[arb(1)/2]*n,arb(1)/16)
  assert not got['passed']and sum(got['coordinate_inclusions'])==50
  bad=[arb(1)]*n;bad[50]=arb(0)
  with pytest.raises(ValueError):T.reduce([I,zero],[I,zero,zero],R,bad,arb(1)/16)
  C=arb_mat(I);C[0,0]=arb(1,arb(1)/16)
  with pytest.raises(ValueError):T.reduce([C,zero],[I,zero,zero],R,[arb(1)]*n,arb(1)/16)
