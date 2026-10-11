import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_quadratic_preconditioner import propose_quadratic_inverse,quadratic_inverse_data
from neural_shrimp.rigorous_cubic_inverse import cubic_inverse_data
from neural_shrimp.rigorous_orbits import eye

def power(d,k):
 p=arb(1)
 for _ in range(k):p*=d
 return p

def test_quadratic_resolves_nonnormal_case_affine_cannot():
 I=eye(3);N=arb_mat([[0,4,0],[0,0,4],[0,0,0]]);Z=arb_mat(3,3);J=[I,N,Z];r=[arb_mat(3,1)for _ in range(5)];r[0][0,0]=arb((1,-20));C=propose_quadratic_inverse(I,J)
 old=cubic_inverse_data(C[:2],J,r,[arb(1)]*3,arb((1,-10)),arb(1)/2)
 new=quadratic_inverse_data(C,J,r,[arb(1)]*3,arb((1,-10)),arb(1)/2)
 assert not old['passed']and old['contraction']>=4
 assert new['passed']and new['contraction'].is_zero()and C[2]==N*N

def test_noncommuting_full_convolutions_and_sixth_degree_forcing():
 with ctx.workprec(200):
  C=[arb_mat([[1,2],[0,1]]),arb_mat([[0,1],[2,0]]),arb_mat([[2,0],[1,1]])]
  J=[arb_mat([[2,0],[1,1]]),arb_mat([[0,1],[0,2]]),arb_mat([[1,2],[3,0]])];r=[arb_mat([[k+1],[5-k]])for k in range(5)]
  ans=quadratic_inverse_data(C,J,r,[arb(1),arb(2)],arb(1),arb(1)/8)
  assert ans['forcing_coefficients'][6]==-C[2]*r[4]
  for d in [arb(-1)/8,arb(0),arb(1)/16,arb(1)/8]:
   c=sum((M*power(d,k)for k,M in enumerate(C)),arb_mat(2,2));j=sum((M*power(d,k)for k,M in enumerate(J)),arb_mat(2,2));rr=sum((v*power(d,k)for k,v in enumerate(r)),arb_mat(2,1))
   assert sum((M*power(d,k)for k,M in enumerate(ans['defect_coefficients'])),arb_mat(2,2))==eye(2)-c*j
   assert sum((v*power(d,k)for k,v in enumerate(ans['forcing_coefficients'])),arb_mat(2,1))==-c*rr

def test_full_error_and_whole_quartic_uncertainty_retained():
 I=eye(1);Z=arb_mat(1,1);J=[arb_mat([[arb(1,arb(1)/16)]]),Z,arb_mat([[arb(0,2)]])];r=[arb_mat(1,1)for _ in range(5)];ans=quadratic_inverse_data([I,Z,Z],J,r,[arb(1)],arb(1)/64,arb(1)/2)
 assert ans['contraction']>=arb(1)/16+arb(1)/2
 r[4][0,0]=arb(100);ans=quadratic_inverse_data([I,Z,I],J,r,[arb(1)],arb(1)/64,arb(1)/2)
 assert not ans['passed']and ans['forcing_coefficients'][6][0,0]==-100

def test_bad_shapes_inexact_proposal_rejected():
 I=eye(1);Z=arb_mat(1,1);r=[Z]*5
 with pytest.raises(ValueError):quadratic_inverse_data([I,Z],[I,Z,Z],r,[arb(1)],arb(1),arb(1))
 with pytest.raises(ValueError):quadratic_inverse_data([arb_mat([[arb(1,arb(1)/32)]]),Z,Z],[I,Z,Z],r,[arb(1)],arb(1),arb(1))
 with pytest.raises(ValueError):quadratic_inverse_data([I,Z,Z],[I,Z,Z],r,[arb(-1)],arb(1),arb(1))
