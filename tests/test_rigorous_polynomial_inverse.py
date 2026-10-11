from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp.rigorous_polynomial_inverse import correlated_inverse_bound
from neural_shrimp.rigorous_orbits import eye
from neural_shrimp.rigorous_eigenpair_march import decode

def power(d,k):
 v=arb(1)
 for _ in range(k):v*=d
 return v

def zeros(n):return [arb_mat(n,1)for _ in range(4)]

def matrix(rows):return arb_mat([[decode(v)for v in row]for row in rows])

def test_nonnormal_varying_inverse_proves_domain_constant_inverse_cannot():
 I=eye(2);N=arb_mat([[0,8],[0,0]]);z=arb_mat(2,2);r=zeros(2);r[0][0,0]=arb((1,-20))
 old=correlated_inverse_bound([I,z],[I,N,z],r,[arb(1)]*2,arb((1,-10)),arb((1,-1)))
 new=correlated_inverse_bound([I,-N],[I,N,z],r,[arb(1)]*2,arb((1,-10)),arb((1,-1)))
 assert not old['passed']and decode(old['contraction'])>=4
 assert new['passed']and decode(new['contraction']).is_zero()and new['coordinate_inclusions']==[True,True]


def test_noncommuting_full_polynomial_products_and_cubic_residual():
 with ctx.workprec(128):
  C=[arb_mat([[1,2],[0,1]]),arb_mat([[0,1],[2,0]])];J=[arb_mat([[2,0],[1,1]]),arb_mat([[0,1],[0,2]]),arb_mat([[1,2],[3,0]])];r=[arb_mat([[i+1],[3-i]])for i in range(4)]
  got=correlated_inverse_bound(C,J,r,[arb(1)]*2,arb(1),arb((1,-3)))
  dc=list(map(matrix,got['defect_coefficients']));rc=list(map(matrix,got['forcing_coefficients']))
  for d in [arb((1,-3)),arb((-1,-4)),arb(0)]:
   cf=C[0]+C[1]*d;jf=J[0]+J[1]*d+J[2]*d*d;rf=sum((r[i]*power(d,i) for i in range(4)),arb_mat(2,1))
   exactd=eye(2)-cf*jf;exactr=-cf*rf
   assert sum((dc[i]*power(d,i) for i in range(4)),arb_mat(2,2))==exactd
   assert sum((rc[i]*power(d,i) for i in range(5)),arb_mat(2,1))==exactr


def test_interval_remainder_and_actual_correction_radius_are_retained():
 I=eye(1);Z=arb_mat(1,1);J2=arb_mat([[arb(0,2)]]);r=zeros(1)
 got=correlated_inverse_bound([I,Z],[I,Z,J2],r,[arb(1)],arb((1,-8)),arb((1,-1)))
 assert got['passed']and decode(got['contraction'])>=arb('0.5')
 r[3][0,0]=arb(100)
 failed=correlated_inverse_bound([I,Z],[I,Z,J2],r,[arb(1)],arb((1,-8)),arb((1,-1)))
 assert not failed['passed']and failed['coordinate_inclusions']==[False]


def test_inexact_inverse_and_bad_shapes_fail_closed():
 I=eye(1);Z=arb_mat(1,1);C=arb_mat([[arb(1,arb((1,-20)))]])
 with pytest.raises(ValueError):correlated_inverse_bound([C,Z],[I,Z,Z],zeros(1),[arb(1)],arb(1),arb(1))
 with pytest.raises(ValueError):correlated_inverse_bound([I,Z],[I,Z],zeros(1),[arb(1)],arb(1),arb(1))
 with pytest.raises(ValueError):correlated_inverse_bound([I,Z],[I,Z,Z],zeros(1),[arb(-1)],arb(1),arb(1))


def test_compact_serialization_changes_no_mathematical_inequality():
 from neural_shrimp.rigorous_polynomial_inverse import correlated_inverse_data,serialize_correlated_inverse
 I=eye(2);N=arb_mat([[0,8],[0,0]]);z=arb_mat(2,2);r=zeros(2);r[0][0,0]=arb((1,-20))
 raw=correlated_inverse_data([I,-N],[I,N,z],r,[arb(1)]*2,arb((1,-10)),arb((1,-1)))
 dense=serialize_correlated_inverse(raw);compact=serialize_correlated_inverse(raw,dense=False)
 for k in ['passed','dimension','contraction','coordinate_inclusions','radii','whole_image']:assert dense[k]==compact[k]
 assert 'whole_defect'not in compact and compact['derived_matrices_retained']is False
 assert decode(compact['image_radius_ratio_upper'][0])==arb((1,-10))
