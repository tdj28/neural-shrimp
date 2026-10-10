"""A fixed width proposal; unchanged full-domain kernel decides every gate."""
from fractions import Fraction
from flint import ctx
from .rigorous_eigenpair_march import compact,decode
from .rigorous_eigenpair_locus import exact_mid,require
from .rigorous_orbits import exact
from .rigorous_eigenpair_path import quadratic_two_charts


def wider_inputs(parent):
    c=compact(parent);s=c['settings'];old=Fraction(2)**s['halfwidth_exponent'];sc=Fraction(c['parameter_center']);right=sc+old
    require(c['parameter_endpoints']==[str(sc-old),str(right)],'exact parent interval')
    require(-100<=s['halfwidth_exponent']<-1,'bounded width doubling')
    with ctx.workprec(s['precision']):
        # New first center is old right+new delta = old center+3old delta.
        step=exact(str(3*old));center=[decode(v) for v in c['center']];slope=[decode(v) for v in c['slope']];curvature=[decode(v) for v in c['curvature']]
        require(all(v.is_finite() and v.is_exact() for v in center+slope+curvature),'exact old predictor proposals')
        initial=[exact_mid(center[i]+slope[i]*step+curvature[i]*step*step/2) for i in range(len(center))]
        kw=dict(period=c['period'],sigma=c['sigma'],normal=c['normal'],origin=c['origin'],parameter=str(sc+3*old),weights=c['weights'],**s)
        kw['halfwidth_exponent']=s['halfwidth_exponent']+1
        kw['anchor']={'box':[decode(v) for v in c['endpoint_root_boxes'][1]],'exact_parameter':str(right),'period':c['period'],'sigma':c['sigma'],'dimension':c['dimension']}
        return initial,kw


def widen(model,parent,*,tick=lambda:None,observe=lambda k,v:None):
    initial,kw=wider_inputs(parent)
    observe('wider_proposal',{'initial':initial,'settings':{k:v for k,v in kw.items() if k!='anchor'},'parent_right':parent['parameter_endpoints'][1],'scope':'Proposal only; no saved coefficient/J enclosure reused on the wider domain. Whole parent endpoint retained by exact same-gauge anchor.'})
    return quadratic_two_charts(model,initial,**kw,tick=tick,observe=observe)
