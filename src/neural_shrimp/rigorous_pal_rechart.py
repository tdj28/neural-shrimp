"""Conditional endpoint localization and finite PAL rechart proposals.

Endpoint existence and the complete source-authenticated quadratic tube are
caller prerequisites. Rechart proposals never replace full new705 proof or
projected all-coordinate membership in a changed gauge.
"""
from fractions import Fraction
import numpy as np
from flint import arb,arb_mat
from .rigorous_eigenpair_locus import require,exact_mid
from .rigorous_orbits import exact,eye,vector_json,matrix_json,ball_json,dyadic
from .rigorous_parameter_localization import fixed_point_box,finite


def localize_endpoint(C,J,coefficients,whole_coefficients,center,slope,curvature,weights,radius,halfwidth,*,tick=lambda:None):
    n=len(center)
    require(1<=n<=705 and len(slope)==len(curvature)==len(weights)==n,'complete bounded endpoint dimension')
    require(all(v.is_finite()and v.is_exact()for v in center+slope+curvature+weights+[radius,halfwidth]),'exact finite quadratic endpoint proposals')
    require(radius>0 and halfwidth>0 and all(v>0 for v in weights),'positive exact scales')
    require(C.nrows()==C.ncols()==J.nrows()==J.ncols()==n and finite(C)and finite(J)and all(v.is_exact()for row in C.tolist()for v in row),'fixed exact full preconditioner and authenticated whole Jacobian')
    require(coefficients.nrows()==whole_coefficients.nrows()==n and coefficients.ncols()==whole_coefficients.ncols()==4 and finite(coefficients)and finite(whole_coefficients),'complete quadratic path Taylor data')
    tick();D=eye(n)-C*J
    rows=[(sum((abs(D[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n)]
    kappa=max(rows);require(kappa.is_finite()and 0<=kappa<1,'strict full weighted endpoint Neumann bound')
    h=halfwidth;residual=arb_mat([[coefficients[i,0]+coefficients[i,1]*h+coefficients[i,2]*h*h+whole_coefficients[i,3]*h*h*h]for i in range(n)])
    initial=arb_mat([[arb(0,(radius*w).upper())]for w in weights])
    error,proof=fixed_point_box(D,-C*residual,weights,kappa,initial=initial);tick()
    predictor=[center[i]+slope[i]*h+curvature[i]*h*h/2 for i in range(n)]
    endpoint=[predictor[i]+error[i,0]for i in range(n)]
    require(all(v.is_finite()for v in endpoint),'finite retained known endpoint')
    return endpoint,{'passed':True,'dimension':n,'parameter_offset':ball_json(h),'weighted_neumann_bound':ball_json(kappa),'predictor':vector_json(predictor),'residual':matrix_json(residual),'refinement':proof,'endpoint':vector_json(endpoint),'scope':'Conditional same known right-endpoint root localization using whole-tube Jacobian and authenticated cubic Taylor residual. No new acquisition or root-existence claim.'}


def rotation_proposal(endpoint,slope,curvature,old_halfwidth):
    n=len(endpoint);require(1<=n<=705 and len(slope)==len(curvature)==n,'full bounded native rechart')
    require(all(v.is_finite()for v in endpoint+slope+curvature)and old_halfwidth.is_exact()and old_halfwidth>0,'finite saved endpoint/tangent data')
    v=[slope[i]+curvature[i]*old_halfwidth for i in range(n)]
    vf=np.array([float(a.mid())for a in v]);length=float(np.linalg.norm(vf))
    require(np.isfinite(vf).all()and np.isfinite(length)and length>0,'finite nonzero local tangent proposal')
    normal=[dyadic(x/length)for x in vf];nn=[exact(x)for x in normal]
    gamma=sum((nn[i]*v[i]for i in range(n)),arb(0));require(gamma>0,'strict forward proposal orientation')
    first=[v[i]/gamma for i in range(n)];g2=sum((nn[i]*curvature[i]for i in range(n)),arb(0))
    second=[(curvature[i]-first[i]*g2)/(gamma*gamma)for i in range(n)]
    # Round all arbitrary proposal points at the current working precision.
    origin=[exact_mid(a+arb(0))for a in endpoint]
    first=[exact_mid(a+arb(0))for a in first];second=[exact_mid(a+arb(0))for a in second]
    return {'normal':normal,'origin':origin,'slope':first,'curvature':second,'gamma':ball_json(gamma),'scope':'Finite rotated-gauge quadratic seed only; all new roots and projected native anchor membership must be acquired and proved.'}


def rechart_seed(proposal,parameter):
    n=len(proposal['origin']);require(len(proposal['normal'])==len(proposal['slope'])==len(proposal['curvature'])==n,'full finite proposal')
    s=exact(parameter);require(s.is_exact()and s.is_finite(),'exact finite new parameter')
    return [exact_mid(exact(c)+exact(v)*s+exact(a)*s*s/2)for c,v,a in zip(proposal['origin'],proposal['slope'],proposal['curvature'],strict=True)]
