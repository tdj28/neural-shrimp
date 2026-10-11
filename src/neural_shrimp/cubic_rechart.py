"""Finite cubic endpoint reparameterization proposals, never root certificates."""
import numpy as np
from flint import arb
from .rigorous_orbits import exact,dyadic,ball_json
from .rigorous_eigenpair_locus import require,exact_mid


def rotate_cubic_endpoint(endpoint,slope,curvature,cubic,halfwidth):
    n=len(endpoint)
    require(1<=n<=705 and len(slope)==len(curvature)==len(cubic)==n,'complete cubic endpoint proposal')
    require(all(v.is_finite()for v in endpoint+slope+curvature+cubic)and halfwidth.is_exact()and halfwidth>0,'finite cubic geometry and positive exact halfwidth')
    # Curvature is p second derivative at zero; cubic is the d^3 coefficient.
    velocity=[slope[i]+curvature[i]*halfwidth+3*cubic[i]*halfwidth**2 for i in range(n)]
    acceleration=[curvature[i]+6*cubic[i]*halfwidth for i in range(n)]
    numerical=np.array([float(v.mid())for v in velocity]);length=float(np.linalg.norm(numerical))
    require(np.isfinite(numerical).all()and np.isfinite(length)and length>0,'finite nonzero forward proposal')
    normal=[dyadic(v/length)for v in numerical];nn=[exact(v)for v in normal]
    gamma=sum((nn[i]*velocity[i]for i in range(n)),arb(0));require(gamma.is_finite()and gamma>0,'positive proposed orientation')
    first=[v/gamma for v in velocity];g2=sum((nn[i]*acceleration[i]for i in range(n)),arb(0))
    second=[(acceleration[i]-first[i]*g2)/gamma**2 for i in range(n)]
    return {'normal':normal,'origin':[exact_mid(v+arb(0))for v in endpoint],
            'slope':[exact_mid(v+arb(0))for v in first],'curvature':[exact_mid(v+arb(0))for v in second],
            'gamma':ball_json(gamma),'scope':'Finite cubic endpoint rotation and quadratic seed only. The true branch, forward root and full projected known-root membership require a fresh complete interval tube. No certified tangent or inherited exact parameter in the new gauge.'}


def adaptive_width(exponent,*,root_passed,joined,other_gates,minimum=-9):
    require(type(exponent)is int and minimum<=exponent<=-5,'bounded adaptive halfwidth')
    require(all(type(v)is bool for v in(root_passed,joined,other_gates)),'literal gate decisions')
    if root_passed and joined and other_gates:return exponent,'advance_rotate'
    if not root_passed and other_gates and exponent>minimum:return exponent-1,'retry_halfwidth'
    return exponent,'stop'
