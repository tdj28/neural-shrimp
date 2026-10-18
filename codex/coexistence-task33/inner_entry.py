"""Exact scalar inner-basin construction; hypotheses require separate admission."""
from fractions import Fraction as F

def construct(radii, root_errors, lipschitz, return_power, return_upper):
    p=len(radii)
    if p<1 or len(root_errors)!=p or len(lipschitz)!=p or type(return_power)is not int or return_power<1:
        raise ValueError('invalid phase/power allocation')
    r=list(map(F,radii));e=list(map(F,root_errors));L=list(map(F,lipschitz));q=F(return_upper)
    if any(x<=0 for x in r) or any(x<0 for x in e+L) or any(a>=b for a,b in zip(e,r)) or not 0<=q<1:
        raise ValueError('missing strict interior or return contraction')
    growth=F(1);rows=[];candidates=[]
    # All intermediate domains needed by the return derivative, plus its endpoint.
    for k in range(return_power*p+1):
        i=k%p;clearance=r[i]-e[i]
        if growth: candidates.append(clearance/(2*growth))
        rows.append(dict(step=k,phase=i,growth=growth,clearance=clearance))
        growth*=L[i]
    eta=min(candidates)
    for row in rows:
        row['root_distance_upper']=row['growth']*eta
        row['strict_domain_margin']=row['clearance']-row['root_distance_upper']
        assert row['strict_domain_margin']>0
    rho=eta-e[0]
    return dict(root_centered_radius=eta,centered_entry_radius=max(F(0),rho),explicit_centered_entry_nonempty=rho>0,return_upper=q,return_margin=(1-q)*eta,intermediate_rows=rows,certificate_passed=False)
