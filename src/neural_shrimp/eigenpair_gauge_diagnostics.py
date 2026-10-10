"""Saved predictor diagnostics; deliberately not certified orbit derivatives."""
from fractions import Fraction
from flint import arb,ctx
from .rigorous_eigenpair_march import decode
from .rigorous_eigenpair_locus import require
from .rigorous_orbits import exact,ball_json,vector_json


def saved_predictor_diagnostics(chart):
    with ctx.workprec(chart['settings']['precision']):
        size=chart['shooting_dimension'];delta=exact(str(Fraction(2)**chart['settings']['halfwidth_exponent']))
        nn=[exact(v) for v in chart['normal']];ss=[decode(v) for v in chart['slope']];tt=[decode(v) for v in chart['curvature']]
        require(len(nn)==len(ss)==len(tt)==size and all(v.is_exact() and v.is_finite() for v in nn+ss+tt),'exact saved predictor data')
        rows=[]
        for name,d in (('center',arb(0)),('right_endpoint',delta)):
            tangent=[ss[i]+tt[i]*d for i in range(size)];norm=sum((v*v for v in tangent),arb(0)).sqrt();require(norm>0 and norm.is_finite(),'nonzero finite saved tangent')
            rate_norm=(tangent[-2]*tangent[-2]+tangent[-1]*tangent[-1]).sqrt();ratio=rate_norm/norm
            rows.append({'location':name,'saved_tangent':vector_json(tangent),'full_norm':ball_json(norm),'rate_projection_ratio':ball_json(ratio),'normalized_actual_rate_components':vector_json([v/norm for v in tangent[-2:]]),'normal_dot_tangent':ball_json(sum((nn[i]*tangent[i] for i in range(size)),arb(0))),'small_rate_projection_flag':bool(ratio<arb('0.0001')),'flag_threshold':'1/10000'})
        return {'rows':rows,'scope':'Arithmetic on saved predictor coefficients only; not a validated tangent to the true branch, projected regularity, simplicity, or a cusp verdict. Separate from the floating complementary multiplier diagnostic.'}
