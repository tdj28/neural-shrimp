"""Conditional known-root/implicit-tangent enclosures from saved analytic arrays.

Caller proves unique native root existence and authenticates all derivative
and predictor domains. No new model values or existence theorem are produced.
The same scalar parameter is retained in the affine root predictor.
"""
from flint import arb,arb_mat
from .rigorous_orbits import eye,ball_json,matrix_json,vector_json
from .rigorous_connection import require


def finite(A):return all(v.is_finite()for row in A.tolist()for v in row)

def weighted_norm(v,w):
    require(v.nrows()==len(w)and v.ncols()==1,'weighted vector dimension')
    return max((abs(v[i,0]).upper()/w[i]).upper()for i in range(len(w)))

def fixed_point_box(D,v,w,kappa,*,initial=None,iterations=3):
    """Enclose any x satisfying x=v+D_actual*x, ||D_actual||_w<=kappa<1.

    A common bound applies even when v,D vary together with the parameter.
    Known-root initial boxes can be intersected; three further intersections
    retain that same vector without treating interval dependence as exact.
    """
    n=len(w);require(type(iterations)is int and 0<=iterations<=3,'bounded intersection count')
    require(D.nrows()==D.ncols()==v.nrows()==n and v.ncols()==1 and finite(D)and finite(v),'finite fixed-point data')
    require(kappa.is_finite()and 0<=kappa and kappa<1,'strict weighted contraction')
    radius=(weighted_norm(v,w)/(1-kappa)).upper()
    box=arb_mat([[arb(0,(radius*w[i]).upper())]for i in range(n)])
    if initial is not None:
        require(initial.nrows()==n and initial.ncols()==1 and finite(initial),'known correction box')
        box=arb_mat([[box[i,0].intersection(initial[i,0])]for i in range(n)])
    initial_box=matrix_json(box);rows=[]
    for _ in range(iterations):
        image=v+D*box;require(finite(image),'finite fixed-point image')
        box=arb_mat([[box[i,0].intersection(image[i,0])]for i in range(n)])
        rows.append({'image':matrix_json(image),'retained':matrix_json(box)})
    return box,{'weighted_uniform_radius':ball_json(radius),'initial_box':initial_box,'intersections':rows,'box':matrix_json(box)}

def localize_parameter_root(C,J,G0,Jp,Htp,Htw,center,slope,weights,halfwidth,original_radius,*,tick=lambda:None):
    """Retain U(t)=c+s*(t-tc)+e(t), localize U(tc), enclose true U'(t).

    J contains H_U on all root/predictor mean-value segments. Jp,Htp
    contain derivatives on the affine predictor itself, while Htw contains
    the parameter partial at every actual root. These domains are caller
    prerequisites. Existence/branch identity are inherited unchanged.
    """
    n=len(center);require(1<=n<=705 and len(weights)==n,'full bounded root dimensions')
    require(all(v.is_finite()and v.is_exact()for v in center)and all(v.is_finite()and v.is_exact()and v>0 for v in weights),'exact center and positive weights')
    require(all(v.is_finite()and v.is_exact()and v>0 for v in(halfwidth,original_radius)),'positive exact parameter/radius')
    require(C.nrows()==C.ncols()==J.nrows()==J.ncols()==Jp.nrows()==Jp.ncols()==n,'full square analytic blocks')
    require(all(v.nrows()==n and v.ncols()==1 for v in(G0,Htp,Htw,slope)),'full analytic vectors')
    require(all(finite(v)for v in(C,J,Jp,G0,Htp,Htw,slope)),'finite analytic data')
    require(all(v.is_exact()for row in C.tolist()for v in row)and all(v.is_exact()for row in slope.tolist()for v in row),'fixed exact preconditioner and predictor')
    tick();D=eye(n)-C*J
    rowbounds=[(sum((abs(D[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n)]
    kappa=max(rowbounds);require(kappa.is_finite()and 0<=kappa and kappa<1,'strict full weighted Neumann bound')
    initial=arb_mat([[arb(0,(original_radius*w).upper())]for w in weights]);d=arb(0,halfwidth)
    center_rhs=-C*G0;center_error,center_proof=fixed_point_box(D,center_rhs,weights,kappa,initial=initial);tick()
    path=Jp*slope+Htp;predictor_residual=G0+path*d;whole_rhs=-C*predictor_residual
    correction,whole_proof=fixed_point_box(D,whole_rhs,weights,kappa,initial=initial);tick()
    # (I-D)(U'-s)=-C*(H_U*s+H_t). No rate or closure contribution is omitted.
    tangent_rhs=-C*(J*slope+Htw);tangent_error,tangent_proof=fixed_point_box(D,tangent_rhs,weights,kappa);tangent=slope+tangent_error;tick()
    center_box=[center[i]+center_error[i,0]for i in range(n)]
    whole_box=[center[i]+slope[i,0]*d+correction[i,0]for i in range(n)]
    require(all(v.is_finite()for v in center_box+whole_box)and finite(tangent),'finite localized domains and tangent')
    return {'passed':True,'dimension':n,'weighted_neumann_bound':ball_json(kappa),'weighted_row_bounds':vector_json(rowbounds),
      'center_preconditioned_rhs':matrix_json(center_rhs),'predictor_path_derivative':matrix_json(path),'predictor_residual':matrix_json(predictor_residual),
      'whole_preconditioned_rhs':matrix_json(whole_rhs),'tangent_preconditioned_rhs':matrix_json(tangent_rhs),
      'center_refinement':center_proof,'whole_refinement':whole_proof,'tangent_refinement':tangent_proof,
      'exact_center':vector_json(center),'exact_slope':matrix_json(slope),'parameter_halfwidth':ball_json(halfwidth),
      'center_box':vector_json(center_box),'whole_box':vector_json(whole_box),'tangent_box':matrix_json(tangent),
      'theorem':'The inherited root error obeys e=-C H(p(t),t)+(I-C A)e. A is enclosed by J on each original predictor-root segment. The integral predictor residual uses Jp*s+Htp over the common parameter interval. Weighted Neumann bounds and intersections retain this same root. Strict CJ invertibility and the inherited unique branch give Uprime; (I-D)(Uprime-s)=-C*(J*s+Ht) encloses every actual tangent, including both physical rates and all mode coordinates.',
      'scope':'Conditional localization and uniform true parameter tangent only. No new root, interval extension, derivative acquisition or complementary-attraction claim.'}
