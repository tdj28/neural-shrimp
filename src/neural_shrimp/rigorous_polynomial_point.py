"""Fresh residual localization of an existing polynomial-tube root.

No root is created. Caller authenticates the inverse/defect family, exact
parameter, residual, and the complete point-to-known-root segment against the
original acquired correction domain. No derivative bound is widened here.
"""
from fractions import Fraction as F
from flint import arb_mat
from .rigorous_polynomial_localization import subset,coefficients,defect_box
from .correlated_inverse_diagnosis import polynomial
from .rigorous_parameter_localization import finite,fixed_point_box
from .rigorous_connection import require
from .rigorous_orbits import ball_json,matrix_json,vector_json


def finite_point_correction(inverse,residual,point,offset,original_halfwidth):
    """One fixed finite proposal. Original-domain membership is a caller gate."""
    d,_=subset(offset,offset,original_halfwidth);n=len(point)
    coefficients(inverse,n,n,'inverse',2)
    require(all(v.is_exact()for C in inverse for row in C.tolist()for v in row),'exact inverse family')
    require(residual.nrows()==n and residual.ncols()==1 and finite(residual)and all(v.is_exact()and v.is_finite()for v in point),'exact point and actual full residual')
    correction=polynomial(inverse,d,d)*residual
    return [(point[i]-correction[i,0]).mid()for i in range(n)]


def localize_fresh_polynomial_point(inverse,defect,residual,point,known,weights,offset,original_halfwidth,*,tick=lambda:None):
    """z=root-point=-C(d)H(point,s_c+d)+D_segment(d)z.

    D encloses I-C(d)H_U on the ORIGINAL point/root segment. The exact
    rational C(d) is enclosed even if coefficient summation rounds. Known
    root existence and full coordinate/rate/mode identity remain inherited.
    """
    d,_=subset(offset,offset,original_halfwidth);n=len(weights)
    require(1<=n<=705 and len(point)==len(known)==n,'complete fixed known-root dimension')
    coefficients(inverse,n,n,'inverse',2)
    require(all(v.is_exact()for C in inverse for row in C.tolist()for v in row),'exact original inverse family')
    require(residual.nrows()==n and residual.ncols()==1 and finite(residual),'actual complete residual')
    require(all(v.is_exact()and v.is_finite()for v in point)and all(v.is_finite()for v in known),'exact finite point/known box')
    D,q,rows=defect_box(defect,weights,d,d);rhs=-polynomial(inverse,d,d)*residual;tick()
    initial=arb_mat([[known[i]-point[i]]for i in range(n)]);error,proof=fixed_point_box(D,rhs,weights,q,initial=initial);tick()
    localized=[known[i].intersection(point[i]+error[i,0])for i in range(n)]
    return localized,dict(passed=True,dimension=n,offset=str(d),point=vector_json(point),known=vector_json(known),weighted_neumann_bound=ball_json(q),weighted_row_bounds=vector_json(rows),fresh_residual=matrix_json(residual),rhs=matrix_json(rhs),refinement=proof,box=vector_json(localized),scope='Conditional same-root point localization from fresh residual and original polynomial derivative. No new root, whole-interval narrowing, or complementary stability follows alone.')
