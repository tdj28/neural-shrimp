"""Conditional same-box crossing with midpoint-preconditioned validated inverses."""
from flint import arb,arb_mat
from .rigorous_eigenpair_locus import require,finite
from .rigorous_orbits import eye,midpoint_matrix,norm_inf_upper,matrix_json,vector_json,ball_json

def validated_inverse(A):
 require(A.nrows()==A.ncols()and finite(A),'finite square inverse domain')
 C=midpoint_matrix(midpoint_matrix(A).inv());D=eye(A.nrows())-C*A
 require(finite(C)and norm_inf_upper(D)<1,'strict midpoint inverse defect')
 X=(C*A).inv()*C
 require(finite(X)and all(v.contains(0)for row in(A*X-eye(A.nrows())).tolist()for v in row)and all(v.contains(0)for row in(X*A-eye(A.nrows())).tolist()for v in row),'validated two-sided inverse')
 return X

def refined_transverse_crossing(whole_J,M,w,n):
    """Conditional same-root flip derivative along the normalized unit-pair locus."""
    k=3*n+2;require(whole_J.nrows()==whole_J.ncols()==4*n+3 and len(w)==n,'whole-root differential dimensions')
    section=arb_mat([[whole_J[i,j]for j in range(k)]for i in range(k)])
    for i in range(k):section[i,n]=whole_J[i,k]
    inverse=validated_inverse(section);require(finite(inverse),'validated native section inverse')
    tangent=-inverse*arb_mat([[whole_J[i,n]]for i in range(k)])
    physical=arb_mat([[arb(1)if i==n else tangent[i,0]]for i in range(n+2)])
    border=arb_mat(n+1,n+1)
    for i in range(n):
        for j in range(n):border[i,j]=M[j,i]+int(i==j)
        border[i,n]=w[i];border[n,i]=w[i]
    border_inverse=validated_inverse(border);require(finite(border_inverse),'validated dual border inverse')
    # At the proved root (M+I)w=0 and ||w||=1, the last solution component
    # vanishes exactly; the first n give left(M+I)=0 and left.w=1.
    left=[border_inverse[i,n]for i in range(n)]
    mixed=arb_mat([[whole_J[k+i,j]for j in range(n+2)]for i in range(n)])
    derivative=(arb_mat([left])*mixed*physical)[0,0]
    return {'passed':bool(derivative.is_finite()and not derivative.contains(0)),'section':matrix_json(section),'validated_section_inverse':matrix_json(inverse),'section_tangent':matrix_json(tangent),'physical_state_rate_tangent':matrix_json(physical),'dual_border':matrix_json(border),'validated_dual_border_inverse':matrix_json(border_inverse),'critical_left':vector_json(left),'multiplier_hidden_rate_derivative':ball_json(derivative),'scope':'Conditional on same-root proved native critical equations and same-domain Jacobian. Validated fixed-hidden-rate section and nonzero total flip derivative establish a transverse flip crossing along that unit-pair locus; no normal-form criticality or global connection.'}
