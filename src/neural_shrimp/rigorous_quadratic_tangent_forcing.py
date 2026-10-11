"""Correlated quadratic predictor forcing for a known implicit root family.

For H(U(s),s)=0 and p quadratic, e=U-p,
H_U(U)p'+H_s = (H(p,s))' + integral H_UU(p+tau*e)[e,p'] dtau.
The caller authenticates the entire predictor-root segment and fixed affine
external gauge. All physical rates, modes and squared angle remain variables.
"""
from flint import arb,arb_mat
from .rigorous_connection import require
from .rigorous_parameter_localization import fixed_point_box,finite
from .rigorous_orbits import eye,ball_json,matrix_json,vector_json


def path_residual_derivative(center_coefficients,whole_coefficients,halfwidth):
    c,w=center_coefficients,whole_coefficients
    require(c.nrows()==w.nrows()and c.ncols()==w.ncols()==4 and finite(c)and finite(w),'complete source-authenticated Taylor coefficients')
    require(halfwidth.is_exact()and halfwidth.is_finite()and halfwidth>0,'exact positive gauge halfwidth')
    d=arb(0,halfwidth)
    # Applied to R': remainder R'''(xi)*d^2/2 = 3*Taylor3(xi)*d^2.
    return arb_mat([[c[i,1]+2*c[i,2]*d+3*w[i,3]*d*d]for i in range(c.nrows())])


def balanced_directions(error,predictor_tangent,scale):
    require(len(error)==len(predictor_tangent)>0 and all(v.is_finite()for v in error+predictor_tangent),'full finite error and predictor tangent')
    require(scale.is_finite()and scale.is_exact()and scale>0,'fixed exact positive polarization scale')
    return ([a/scale+scale*b for a,b in zip(error,predictor_tangent,strict=True)],
            [a/scale-scale*b for a,b in zip(error,predictor_tangent,strict=True)])


def mixed_hessian_from_second_coefficients(plus,minus):
    require(plus.nrows()==minus.nrows()and plus.ncols()==minus.ncols()==4 and finite(plus)and finite(minus),'full native augmented path coefficients')
    # Each coefficient2 = H_UU[v,v]/2. Polarization then divides their difference by2.
    return arb_mat([[(plus[i,2]-minus[i,2])/2]for i in range(plus.nrows())])


def refine_tangent(C,J,residual_derivative,mixed_forcing,predictor_tangent,weights,*,tick=lambda:None):
    n=len(weights)
    require(1<=n<=705 and len(predictor_tangent)==n and all(v.is_finite()for v in predictor_tangent),'complete bounded native tangent')
    require(all(v.is_exact()and v.is_finite()and v>0 for v in weights),'exact positive native weights')
    require(C.nrows()==C.ncols()==J.nrows()==J.ncols()==n and finite(C)and finite(J)and all(v.is_exact()for row in C.tolist()for v in row),'exact full preconditioner and entire root Jacobian')
    require(all(a.nrows()==n and a.ncols()==1 and finite(a)for a in(residual_derivative,mixed_forcing)),'complete correlated forcing vectors')
    tick();D=eye(n)-C*J
    rows=[(sum((abs(D[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n)]
    kappa=max(rows);require(kappa.is_finite()and 0<=kappa<1,'strict full known-root Neumann bound')
    rhs=-C*(residual_derivative+mixed_forcing);error,proof=fixed_point_box(D,rhs,weights,kappa);tick()
    tangent=arb_mat([[predictor_tangent[i]+error[i,0]]for i in range(n)])
    return {'passed':True,'dimension':n,'weighted_neumann_bound':ball_json(kappa),'weighted_row_bounds':vector_json(rows),
      'residual_derivative':matrix_json(residual_derivative),'mixed_forcing':matrix_json(mixed_forcing),'preconditioned_rhs':matrix_json(rhs),
      'predictor_tangent':vector_json(predictor_tangent),'tangent_error':proof,'tangent_box':matrix_json(tangent),
      'scope':'Conditional true tangent of the authenticated quadratic known-root family. Requires exact common-parameter Taylor residual derivative, full augmented mixed Hessian on every predictor-root segment, both rates/angle/modes and fixed affine external gauge. No new root or complementary stability claim.'}
