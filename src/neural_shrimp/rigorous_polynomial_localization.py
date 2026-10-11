"""Conditional same-root enclosures retaining a polynomial inverse family.

The caller authenticates a known U(t)=p(t)+e(t), all original correction and
Jacobian domains, and common-parameter coefficient bounds for
D(t)=I-C(t) H_U and f(t)=-C(t) H(p(t),t). No root is created by these routines.
For tangent refinement the external parameter enters H affinely, with constant
H_t, so H_Ut=0. Coefficients of R' come from Taylor's theorem applied to R',
never by differentiating an interval remainder. All coordinates are retained.
"""
from fractions import Fraction as F
from flint import arb_mat
from .correlated_inverse_diagnosis import polynomial,weighted_norm
from .rigorous_parameter_localization import fixed_point_box,finite
from .rigorous_orbits import ball_json,matrix_json,vector_json
from .rigorous_connection import require


def subset(left,right,original_halfwidth):
    a,b,h=F(left),F(right),F(original_halfwidth)
    require(h>0 and -h<=a<=b<=h,'exact original closed scalar subset')
    return a,b


def coefficients(rows,n,columns,label,maximum=6):
    require(1<=len(rows)<=maximum and all(x.nrows()==n and x.ncols()==columns and finite(x)for x in rows),'complete finite '+label+' coefficients')


def defect_box(defect,weights,left,right):
    n=len(weights);require(1<=n<=705,'full bounded known-root dimension')
    coefficients(defect,n,n,'defect');D=polynomial(defect,left,right)
    bound,rows=weighted_norm(D,weights)
    require(bound.is_finite()and 0<=bound<1,'strict correlated weighted Neumann bound')
    return D,bound,rows


def localize_polynomial_root(defect,forcing,weights,original_error,left,right,original_halfwidth,*,known_image=None,tick=lambda:None):
    """Enclose the authenticated error on a subset; left==right is permitted.

    For each t, e=f(t)+D_segment(t)e. The original full-error domain must
    cover the entire p-to-root segment. A separately proved uniform root
    image may further narrow the inherited error. Neither zero containment
    nor contraction here substitutes for the caller's root existence proof.
    """
    a,b=subset(left,right,original_halfwidth);n=len(weights)
    coefficients(forcing,n,1,'preconditioned residual')
    require(original_error.nrows()==n and original_error.ncols()==1 and finite(original_error),'complete inherited correction domain')
    initial=original_error
    if known_image is not None:
        require(known_image.nrows()==n and known_image.ncols()==1 and finite(known_image),'complete independently proved root image')
        initial=arb_mat([[initial[i,0].intersection(known_image[i,0])]for i in range(n)])
    tick();D,q,rows=defect_box(defect,weights,a,b);rhs=polynomial(forcing,a,b);tick()
    error,proof=fixed_point_box(D,rhs,weights,q,initial=initial);tick()
    return error,dict(passed=True,dimension=n,original_offset=[str(a),str(b)],weighted_neumann_bound=ball_json(q),weighted_row_bounds=vector_json(rows),preconditioned_rhs=matrix_json(rhs),refinement=proof,
      scope='Conditional enclosure of the same authenticated root error using the original polynomial inverse/defect and residual domain. No new roots, acquisition domains, continuation or complementary stability theorem.')


def preconditioned_derivative_coefficients(inverse,residual_derivative):
    """Convolve -C(t)R'(t) before any scalar interval substitution."""
    n=inverse[0].nrows()if inverse else 0;require(1<=n<=705,'complete inverse family')
    coefficients(inverse,n,n,'inverse',2);coefficients(residual_derivative,n,1,'residual derivative',4)
    require(all(v.is_exact()for A in inverse for row in A.tolist()for v in row),'fixed exact inverse coefficients')
    result=[arb_mat(n,1)for _ in range(len(inverse)+len(residual_derivative)-1)]
    for i,C in enumerate(inverse):
        for j,R in enumerate(residual_derivative):result[i+j]-=C*R
    return result


def refine_polynomial_tangent(inverse,defect,residual_derivative,mixed_forcing,predictor_tangent,weights,left,right,original_halfwidth,*,tick=lambda:None):
    """e'= -C(t)[R'(t)+H_UU(segment)[e,p']] + D_root(t)e'.

    The mixed forcing encloses the entire predictor/root segment at the SAME
    parameter. D_root uses the source-authenticated original full error domain,
    not an independently acquired narrower box. A known root and uniformly
    invertible H_U justify the actual differentiable branch. H_Ut must be zero.
    """
    a,b=subset(left,right,original_halfwidth);n=len(weights)
    require(mixed_forcing.nrows()==predictor_tangent.nrows()==n and mixed_forcing.ncols()==predictor_tangent.ncols()==1 and finite(mixed_forcing)and finite(predictor_tangent),'complete native mixed forcing and predictor tangent')
    tick();D,q,rows=defect_box(defect,weights,a,b)
    forced=preconditioned_derivative_coefficients(inverse,residual_derivative)
    C=polynomial(inverse,a,b);rhs=polynomial(forced,a,b)-C*mixed_forcing;tick()
    error,proof=fixed_point_box(D,rhs,weights,q);tangent=predictor_tangent+error;require(finite(tangent),'finite full correlated tangent');tick()
    return tangent,dict(passed=True,dimension=n,original_offset=[str(a),str(b)],weighted_neumann_bound=ball_json(q),weighted_row_bounds=vector_json(rows),preconditioned_derivative_coefficients=[matrix_json(x)for x in forced],inverse_range=matrix_json(C),mixed_forcing=matrix_json(mixed_forcing),preconditioned_rhs=matrix_json(rhs),predictor_tangent=matrix_json(predictor_tangent),tangent_error=proof,tangent_box=matrix_json(tangent),
      scope='Conditional true tangent of the same known-root family, using common-parameter inverse/defect coefficients and Taylor bounds for the residual derivative. Requires constant H_parameter, complete actual segment/native-domain coverage and all mode/rate/angle coordinates. No new root or complementary stability theorem.')
