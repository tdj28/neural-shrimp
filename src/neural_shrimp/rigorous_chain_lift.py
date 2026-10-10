"""Conditional native lift and uniqueness-domain membership of a proved chain root.

The caller must establish the same-root normalized generalized-chain identities
and native model/phase convention. Interval containment is not that proof.
"""
from fractions import Fraction
from flint import arb
from .rigorous_eigenpair_locus import phase,require,finite
from .rigorous_eigenpair_march import compact,decode
from .rigorous_orbits import exact,ball_json,vector_json,matrix_json


def native_lift(model,box,period,normal,origin,tick=lambda:None,observe=lambda k,v:None):
    n=model.n;size=2*n*period+2
    require(type(period)is int and 1<=period<=18 and len(box)==3*n+2,'native chain dimensions')
    require(len(normal)==len(origin)==size,'native gauge dimensions')
    nn=[exact(v)for v in normal];oo=[exact(v)for v in origin]
    require(all(v.is_finite() and v.is_exact()for v in nn+oo),'exact native gauge')
    require(any(v!=0 for v in nn),'nonzero native gauge')
    state=list(box[:n]);rates=list(box[n:n+2]);mode=list(box[n+2:2*n+2]);flat=[]
    require(all(v.is_finite()for v in box),'finite chain box')
    for k in range(period):
        flat.extend(state+mode);tick();value,block,rateblock=phase(model,state,mode,rates)
        observe('phase',{'index':k,'input':vector_json(state+mode+rates),'value':matrix_json(value),'block':matrix_json(block),'rates':matrix_json(rateblock)})
        require(finite(value) and finite(block) and finite(rateblock),'finite lifted native phase')
        state=[value[i,0]for i in range(n)];mode=[value[n+i,0]for i in range(n)]
    flat.extend(rates)
    parameter=sum((nn[i]*(flat[i]-oo[i])for i in range(size)),arb(0))
    require(parameter.is_finite(),'finite lifted gauge projection')
    closing=[state[i]-box[i]for i in range(n)]+[mode[i]-box[n+2+i]for i in range(n)]
    return {'schema':'neural-shrimp-conditional-chain-lift-v1','dimension':n,'period':period,'sigma':1,'shooting_dimension':size,'normal':normal,'origin':origin,'box':vector_json(flat),'parameter':ball_json(parameter),'closure_residual':vector_json(closing),'closure_contains_zero':all(v.contains(0)for v in closing),'scope':'Enclosure of the full native root conditional on the authenticated same-model normalized generalized-chain root; closure containment alone is not a root proof.'}


def quadratic_membership(lift,chart):
    """Strict membership in a proved uniform tube, conditional on lift provenance.

    The actual root's gauge is unknown but enclosed by projection. Its whole
    lifted box must lie in the uniqueness domain for every projected gauge.
    """
    c=compact(chart)
    for k in ('period','dimension','sigma','shooting_dimension','normal','origin'):
        require(lift[k]==c[k],'identical native convention '+k)
    require(lift['schema']=='neural-shrimp-conditional-chain-lift-v1' and lift['closure_contains_zero'],'conditional normalized root lift')
    size=c['shooting_dimension'];box=[decode(v)for v in lift['box']]
    require(len(box)==size,'full native lifted box')
    nn=[exact(v)for v in c['normal']];oo=[exact(v)for v in c['origin']]
    parameter=sum((nn[i]*(box[i]-oo[i])for i in range(size)),arb(0))
    center=[decode(v)for v in c['center']];slope=[decode(v)for v in c['slope']];curve=[decode(v)for v in c['curvature']]
    require(all(v.is_exact()for v in center+slope+curve),'exact tube predictor')
    delta=arb((1,c['settings']['halfwidth_exponent']));sc=exact(c['parameter_center']);d=parameter-sc
    require(c['parameter_endpoints']==[str(Fraction(c['parameter_center'])-Fraction(2)**c['settings']['halfwidth_exponent']),str(Fraction(c['parameter_center'])+Fraction(2)**c['settings']['halfwidth_exponent'])],'exact tube endpoints')
    radii=[arb((1,c['settings']['radius_exponent']))*exact(v)for v in c['weights']]
    require(len(radii)==size and all(v>0 for v in radii),'positive full-coordinate radii')
    errors=[box[i]-center[i]-slope[i]*d-curve[i]*d*d/2 for i in range(size)]
    inclusions=[bool(abs(errors[i]).upper()<radii[i])for i in range(size)]
    inside=bool(abs(d).upper()<delta)
    return {'passed':inside and all(inclusions),'parameter_inside':inside,'coordinate_inclusions':inclusions,'parameter':ball_json(parameter),'predictor_error':vector_json(errors),'coordinate_radii':vector_json(radii),'scope':'Identifies the authenticated lifted root with the connected chart root by uniqueness, conditional on exact model/phase/root provenance; no unfolding or stability claim.'}
