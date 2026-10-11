"""Exact restriction of an already proved cubic implicit root family.

The caller authenticates the original correction radii and residual Taylor
coefficients. No old Jacobian or new root is inferred from this restriction.
"""
from flint import arb, arb_mat
from .rigorous_connection import require
from .rigorous_parameter_localization import finite


def restrict_cubic(center,slope,curvature,cubic,weights,original_halfwidth,
                   radius,shift,halfwidth,center_coefficients,whole_coefficients):
    n=len(center)
    require(1<=n<=705 and len(slope)==len(curvature)==len(cubic)==len(weights)==n,'full original cubic predictor')
    require(all(v.is_exact()and v.is_finite()for v in center+slope+curvature+cubic+weights+[original_halfwidth,radius,shift,halfwidth]),'exact finite original predictor and restriction')
    require(original_halfwidth>0 and radius>0 and halfwidth>0 and all(w>0 for w in weights),'positive original and local scales')
    require(abs(shift)+halfwidth<=original_halfwidth,'exact closed subset of original root family')
    c,w=center_coefficients,whole_coefficients
    require(c.nrows()==w.nrows()==n and c.ncols()==w.ncols()==5 and finite(c)and finite(w),'authenticated full quartic residual Taylor data')
    # Original convention p(delta)=c+s*delta+a*delta^2/2+b*delta^3.
    local_center=[center[i]+slope[i]*shift+curvature[i]*(shift*shift)/2+cubic[i]*(shift*shift*shift) for i in range(n)]
    local_slope=[slope[i]+curvature[i]*shift+3*cubic[i]*(shift*shift) for i in range(n)]
    local_curvature=[curvature[i]+6*cubic[i]*shift for i in range(n)]
    require(all(v.is_exact()for v in local_center+local_slope+local_curvature),'exact recentering without coefficient rounding')
    d=arb(0,halfwidth);delta=shift+d
    # Separate Taylor theorems for R and R'. The multiplier may be rounded
    # outward; it does not enlarge the set on which old derivatives are used.
    residual=arb_mat([[c[i,0]+c[i,1]*delta+c[i,2]*(delta*delta)+c[i,3]*(delta*delta*delta)+w[i,4]*(delta*delta*delta*delta)]for i in range(n)])
    derivative=arb_mat([[c[i,1]+2*c[i,2]*delta+3*c[i,3]*(delta*delta)+4*w[i,4]*(delta*delta*delta)]for i in range(n)])
    radii=[radius*v for v in weights];require(all(v.is_exact()for v in radii),'exact inherited root correction radii')
    domain=[local_center[i]+local_slope[i]*d+local_curvature[i]*d*d/2+cubic[i]*d*d*d+arb(0,radii[i])for i in range(n)]
    tangent=[local_slope[i]+local_curvature[i]*d+3*cubic[i]*d*d for i in range(n)]
    require(all(v.is_finite()for v in domain+tangent)and finite(residual)and finite(derivative),'finite restricted predictor/forcing')
    return dict(center=local_center,slope=local_slope,curvature=local_curvature,cubic=list(cubic),weights=list(weights),halfwidth=halfwidth,radius=radius,fresh_domain=domain,path_residual=residual,residual_derivative=derivative,predictor_derivative=tangent,original_delta=delta)
