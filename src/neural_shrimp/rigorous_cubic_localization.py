"""Conditional localization/tangents of an authenticated cubic root tube.

Caller supplies same-root full Jacobian domains and the rigorously bounded
predictor residual. All variables, including both rates and angle, are kept.
"""
from flint import arb,arb_mat
from .rigorous_parameter_localization import fixed_point_box,finite
from .rigorous_orbits import eye,ball_json,vector_json,matrix_json
from .rigorous_connection import require


def localize_cubic_root(C,J,G0,path_residual,parameter_partial,center,slope,curvature,cubic,weights,halfwidth,radius,*,tick=lambda:None):
    n=len(center)
    require(1<=n<=705 and len(weights)==len(slope)==len(curvature)==len(cubic)==n,'full cubic native dimension')
    require(all(v.is_exact()and v.is_finite()for v in center+slope+curvature+cubic+weights+[halfwidth,radius]),'exact finite cubic proposals')
    require(halfwidth>0 and radius>0 and all(v>0 for v in weights),'positive parameter and coordinate scales')
    require(C.nrows()==C.ncols()==J.nrows()==J.ncols()==n and finite(C)and finite(J),'full finite square Jacobian')
    require(all(v.is_exact()for row in C.tolist()for v in row),'exact fixed preconditioner')
    require(all(v.nrows()==n and v.ncols()==1 and finite(v)for v in(G0,path_residual,parameter_partial)),'full bounded residual and parameter partial')
    tick();D=eye(n)-C*J
    row_bounds=[(sum((abs(D[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n)]
    kappa=max(row_bounds);require(kappa.is_finite()and 0<=kappa<1,'strict full weighted cubic Neumann bound')
    initial=arb_mat([[arb(0,(radius*w).upper())]for w in weights])
    v0=-C*G0;center_error,center_proof=fixed_point_box(D,v0,weights,kappa,initial=initial);tick()
    vw=-C*path_residual;error,whole_proof=fixed_point_box(D,vw,weights,kappa,initial=initial);tick()
    d=arb(0,halfwidth);path_tangent=arb_mat([[slope[i]+curvature[i]*d+3*cubic[i]*d*d]for i in range(n)])
    vt=-C*(J*path_tangent+parameter_partial)
    tangent_error,tangent_proof=fixed_point_box(D,vt,weights,kappa);tangent=path_tangent+tangent_error;tick()
    center_box=[center[i]+center_error[i,0]for i in range(n)]
    whole_box=[center[i]+slope[i]*d+curvature[i]*d*d/2+cubic[i]*d*d*d+error[i,0]for i in range(n)]
    require(all(v.is_finite()for v in center_box+whole_box)and finite(tangent),'finite retained cubic root/tangent')
    return {'passed':True,'dimension':n,'weighted_neumann_bound':ball_json(kappa),'weighted_row_bounds':vector_json(row_bounds),
      'center_rhs':matrix_json(v0),'whole_rhs':matrix_json(vw),'tangent_rhs':matrix_json(vt),'predictor_tangent':matrix_json(path_tangent),
      'center_refinement':center_proof,'whole_refinement':whole_proof,'tangent_refinement':tangent_proof,
      'center_box':vector_json(center_box),'whole_box':vector_json(whole_box),'tangent_box':matrix_json(tangent),
      'scope':'Conditional localization of the already proved cubic predictor root family, with true parameter tangent. Exact original root, residual and full mean-value domains are caller prerequisites. No new root, extension, native derivative acquisition or complementary stability claim.'}


def cubic_segment_inputs(args,proof,original):
    """Full polynomial/root mean-value segment, covered by exact saved endpoints."""
    from fractions import Fraction as F
    from .rigorous_connection import restore_ball
    from .rigorous_quadratic_tangent_forcing import balanced_directions
    n=len(args['center']);require(proof['passed']and proof['dimension']==n and len(original)==n,'same complete localized root')
    error=[restore_ball(v[0])for v in proof['whole_refinement']['box']]
    c,s,a,b=[args[k]for k in ('center','slope','curvature','cubic')];d=arb(0,args['halfwidth'])
    predictor=[c[i]+s[i]*d+a[i]*d*d/2+b[i]*d*d*d for i in range(n)]
    pprime=[s[i]+a[i]*d+3*b[i]*d*d for i in range(n)]
    segment=[predictor[i]+arb(0,abs(error[i]).upper())for i in range(n)]
    def bounds(z):
        def value(v):return F(v[0])*F(2)**v[1]
        m,r=value(z['mid']),value(z['rad']);return m-r,m+r
    for old,new in zip(original,segment,strict=True):
        lo,hi=bounds(old);a,b=bounds(ball_json(new));require(lo<=a<=b<=hi,'exact original full-J domain covers entire cubic segment')
    plus,minus=balanced_directions(error,pprime,arb('1/256'))
    return segment,error,pprime,plus,minus
