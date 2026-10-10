"""Saved-only positive inflation proposal; never validates a changed domain."""
from fractions import Fraction
import math
import numpy as np
from flint import arb_mat,ctx
from .rigorous_eigenpair_locus import require
from .rigorous_eigenpair_march import decode
from .rigorous_orbits import matrix_json


def positive_inflation(a,B,r,observe=lambda k,v:None):
    """Inflate exact old radii by a floating positive-resolvent correction."""
    a=np.asarray(a,dtype=float);B=np.asarray(B,dtype=float);r=list(map(Fraction,r));old=np.array(list(map(float,r)))
    n=len(r)
    require(n>0 and a.shape==(n,) and B.shape==(n,n),'consistent dimensions')
    require(np.isfinite(a).all() and np.isfinite(B).all() and np.isfinite(old).all() and np.all(a>=0) and np.all(B>=0) and all(v>0 for v in r),'finite nonnegative inputs and positive radii')
    excess=np.maximum(a+B@old-old,0.)
    solution=np.linalg.solve(np.eye(n)-B,excess)
    part=lambda v:float(v) if math.isfinite(v) else None
    observe('inflation_linear_return',{'forcing':a.tolist(),'absolute_defect':B.tolist(),'old_radii':[str(v) for v in r],'excess':excess.tolist(),'solution':list(map(part,solution)),'scope':'Floating proposal return retained before validation; no widened-domain bound.'})
    require(np.isfinite(solution).all() and np.all(solution>=0),'finite nonnegative correction')
    new=[r[i]+Fraction(17,16)*Fraction.from_float(float(solution[i]))+r[i]*Fraction(2)**-60 for i in range(n)]
    maximum=max(new);exponent=-100
    while Fraction(2)**exponent<maximum:exponent+=1
    require(-100<=exponent<=-1,'bounded positive proposal')
    proposed=np.array(list(map(float,new)))
    ratios=(a+B@proposed)/proposed
    return {'radius_exponent':exponent,'weights':[str(v/Fraction(2)**exponent) for v in new],
      'exact_radii':[str(v) for v in new],'old_exact_radii':[str(v) for v in r],
      'old_domain_ratios':((a+B@old)/old).tolist(),'proposed_old_domain_ratios':ratios.tolist(),
      'linear_residual':((np.eye(n)-B)@solution-excess).tolist(),
      'changed_coordinates':sum(v>r[i]*(1+Fraction(2)**-60) for i,v in enumerate(new)),
      'maximum_relative_inflation':float(max(new[i]/r[i] for i in range(n))),
      'scope':'All old radii retained exactly; fixed17/16 positive-resolvent inflation plus relative2^-60 floor. Old-domain ratios are floating proposal diagnostics, not bounds over the changed box. New whole-domain Jacobian/path/root/primitive/anchor gates are mandatory.'}


def diagnose(chart,observe=lambda k,v:None):
    require(chart['gates']=={'uniform_normalized_eigenpair_locus':False,'primitive_period':True,'anchor_identified':True},'root-only unresolved chart')
    with ctx.workprec(chart['settings']['precision']):
        require(decode(chart['contraction'])<1,'saved contraction below one')
        C=arb_mat([[decode(v) for v in row] for row in chart['preconditioner']]);R=arb_mat([[decode(v[0])] for v in chart['predictor_residual']]);forcing=-C*R
        observe('inflation_forcing_return',{'forcing':matrix_json(forcing)})
        n=C.nrows();require(C.ncols()==n==chart['shooting_dimension'] and R.nrows()==n,'native dimensions')
        a=np.array([float(abs(forcing[i,0]).upper()) for i in range(n)])
        B=np.array([[float(abs(decode(v)).upper()) for v in row] for row in chart['defect']])
        r=[]
        for v in chart['coordinate_radii']:
            require(v['rad'][0]==0,'exact saved radius')
            r.append(Fraction(v['mid'][0])*Fraction(2)**v['mid'][1])
        result=positive_inflation(a,B,r,observe)
        result['failed_coordinate_indices']=[i for i,v in enumerate(chart['coordinate_inclusions']) if not v]
        result['summary']={k:result[k] for k in ('maximum_relative_inflation','changed_coordinates','failed_coordinate_indices')}
        result['summary'].update(old_domain_max_ratio=max(result['old_domain_ratios']),proposal_old_domain_max_ratio=max(result['proposed_old_domain_ratios']),positive=True,all_old_radii_retained=True)
        observe('inflation_proposal',result);return result
