"""Known-root localization and a full native state/mode tangent-gauge proposal.

Existence/primitive/identity come from the authenticated parent. Localization
retains that root; a fresh unchanged quadratic tube must prove the new chart.
The tangent and coordinate radii below are conditioning proposals, not claims
of critical simplicity, a regular rate projection, or bifurcation genericity.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from .rigorous_eigenpair_locus import require,finite,assemble,column,exact_mid
from .rigorous_eigenpair_march import compact,decode
from .rigorous_eigenpair_path import quadratic_two_charts
from .rigorous_orbits import exact,eye,ball_json,vector_json,matrix_json


def localization_identity(center,residual,jacobian,pre,box):
    """A known root in box is in image, whether or not C is invertible.

    Caller must enclose DH on hull(center,box); the box need not contain center.
    The candidate is returned before intersection so failed returns can survive.
    """
    size=len(center)
    require(len(box)==size and residual.nrows()==size and residual.ncols()==1,'localization dimensions')
    require(jacobian.nrows()==jacobian.ncols()==pre.nrows()==pre.ncols()==size,'square localization matrices')
    require(all(v.is_exact() and v.is_finite() for v in center) and finite(residual) and finite(jacobian) and finite(pre),'finite localization inputs')
    require(all(pre[i,j].is_exact() for i in range(size) for j in range(size)),'exact fixed preconditioner')
    error=column([box[i]-center[i] for i in range(size)])
    defect=eye(size)-pre*jacobian
    image=column(center)-pre*residual+defect*error
    return image,defect


def localize_endpoint(model,parent,*,iterations=3,tick=lambda:None,observe=lambda k,v:None):
    """Four native assemblies for three intersections, with a fixed exact center.

    The initial full endpoint box and original C are retained. Because the center
    is fixed, every new derivative domain is hull(center,current_box), even if a
    previous intersection excluded the center. No midpoint is treated as a root.
    """
    c=compact(parent);size=c['shooting_dimension'];p=c['period'];n=c['dimension']
    require(type(iterations)is int and 1<=iterations<=3,'three localization steps maximum')
    require(n==model.n and len(parent['preconditioner'])==size,'same model dimension and full parent C')
    with ctx.workprec(c['settings']['precision']):
        box=[decode(v) for v in c['endpoint_root_boxes'][1]]
        center=[v.mid() for v in box]
        pre=arb_mat([[decode(v) for v in row] for row in parent['preconditioner']])
        nn=[exact(v) for v in c['normal']];oo=[exact(v) for v in c['origin']];s=exact(c['parameter_endpoints'][1])
        observe('localization_start',{'parent_right':c['parameter_endpoints'][1],'initial_box':vector_json(box),'center':vector_json(center),'preconditioner':matrix_json(pre),'normal':c['normal'],'origin':c['origin'],'iterations':iterations})
        g,_,_=assemble(model,center,p,c['sigma'],nn,oo,s,tick,observe)
        observe('localization_center_return',{'center':vector_json(center),'residual':matrix_json(g)})
        for iteration in range(iterations):
            domain=[center[i].union(box[i]) for i in range(size)]
            _,jac,_=assemble(model,domain,p,c['sigma'],nn,oo,s,tick,observe)
            image,defect=localization_identity(center,g,jac,pre,box)
            observe('localization_image_return',{'iteration':iteration,'prior_box':vector_json(box),'domain':vector_json(domain),'jacobian':matrix_json(jac),'defect':matrix_json(defect),'image':matrix_json(image)})
            require(finite(image),'finite localization image')
            updated=[box[i].intersection(image[i,0]) for i in range(size)]
            require(all(v.is_finite() for v in updated),'finite nonempty intersection')
            observe('localization_intersection',{'iteration':iteration,'box':vector_json(updated),'scope':'Contains the same authenticated parent endpoint root; outward intersection rounding is permitted.'})
            box=updated
        result={'schema':'neural-shrimp-eigenpair-endpoint-localization-v1','passed':True,'parent_parameter':c['parameter_endpoints'][1],'iterations':iterations,'period':p,'dimension':n,'sigma':c['sigma'],'box':vector_json(box),'scope':'Conditional root containment only, inheriting parent existence, primitive period and native phase/mode identity.'}
        observe('localization_return',result)
        return result,box


def upper_fraction(v):
    require(v.is_finite() and v>0,'positive finite proposal bound')
    return Fraction(exact_mid(v.upper()))


def tangent_proposal(parent,box):
    """Exact dyadic gauge; conservative coordinate-radius/width proposals.

    A rounded normalized saved endpoint predictor derivative fixes orientation.
    t=v/(n.v) is an interval enclosure used only for radius transport; n.t and
    its residual are retained. The fresh tube computes its own S,T and gates.
    """
    c=compact(parent);size=c['shooting_dimension'];settings=c['settings']
    require(len(box)==size and all(v.is_finite() for v in box),'localized full box')
    with ctx.workprec(settings['precision']):
        delta=exact(str(Fraction(2)**settings['halfwidth_exponent']))
        vv=[(decode(c['slope'][i])+decode(c['curvature'][i])*delta).mid() for i in range(size)]
        norm=sum((v*v for v in vv),arb(0)).sqrt();require(norm.is_finite() and norm>0,'nonzero finite saved tangent')
        nn=[(v/norm).mid() for v in vv];dot=sum((nn[i]*vv[i] for i in range(size)),arb(0))
        require(dot>0,'carried positive orientation');tt=[v/dot for v in vv]
        nt=sum((nn[i]*tt[i] for i in range(size)),arb(0));require(nt.contains(1),'normalized interval tangent')
        origin=[v.mid() for v in box];proj=sum((nn[i]*(box[i]-origin[i]) for i in range(size)),arb(0))
        old=[exact(str(Fraction(2)**settings['radius_exponent']*Fraction(v))) for v in c['weights']]
        projected_radius=sum((abs(nn[i])*old[i] for i in range(size)),arb(0))
        transported=[old[i]+abs(tt[i])*projected_radius for i in range(size)]
        anchor_need=[abs(box[i]-origin[i]-tt[i]*proj).upper() for i in range(size)]
        rr=[upper_fraction(arb(17)/16*max(transported[i].upper(),2*anchor_need[i])+arb((1,-100))) for i in range(size)]
        exp=-100
        while Fraction(2)**exp<max(rr):exp+=1
        require(-100<=exp<=-1,'bounded transported radii')
        # Largest dyadic <= old halfstep length, capped at raw length 1/16.
        width=(norm*delta).lower();hexp=-4
        while arb((1,hexp))>width and hexp>=-30:hexp-=1
        require(-30<=hexp<=-4,'bounded raw arclength width')
        rate_norm=(vv[-2]*vv[-2]+vv[-1]*vv[-1]).sqrt()/norm
        result={'normal':[exact_mid(v) for v in nn],'origin':[exact_mid(v) for v in origin],'initial':[exact_mid(v) for v in origin],'parameter':'0','halfwidth_exponent':hexp,'radius_exponent':exp,'weights':[str(r/Fraction(2)**exp) for r in rr],
          'saved_endpoint_tangent':vector_json(vv),'tangent_norm':ball_json(norm),'normal_norm_squared':ball_json(sum((v*v for v in nn),arb(0))),'normal_dot_saved_tangent':ball_json(dot),'normalized_transport_tangent':vector_json(tt),'normal_dot_transport_tangent':ball_json(nt),'normalization_residual':ball_json(nt-1),'old_coordinate_radii':vector_json(old),'transported_coordinate_radii':vector_json(transported),'anchor_requirement':vector_json(anchor_need),'exact_new_radii':[str(r) for r in rr],'anchor_projection':ball_json(proj),'rate_projection_ratio':ball_json(rate_norm),
          'scope':'Raw Euclidean native state/mode/rate metric with existing units; saved predictor tangent/rate projection are descriptive proposals, not certified derivatives or genericity. Every new chart still requires fresh whole-domain proof and full correlated anchor.'}
        return result


def reanchor(model,parent,*,tick=lambda:None,observe=lambda k,v:None):
    """Exactly three localization intersections, then at most two fresh charts."""
    loc,box=localize_endpoint(model,parent,tick=tick,observe=observe)
    proposed=tangent_proposal(parent,box);observe('reanchor_proposal',proposed)
    c=compact(parent)
    kwargs={k:proposed[k] for k in ('normal','origin','parameter','halfwidth_exponent','radius_exponent','weights')}
    kwargs.update(period=c['period'],sigma=c['sigma'],newton_steps=2,precision=c['settings']['precision'],
                  anchor={'box':box,'period':c['period'],'sigma':c['sigma'],'dimension':c['dimension']},tick=tick,observe=observe)
    # No exact_parameter: the full physical box must project into the new gauge.
    result=quadratic_two_charts(model,proposed['initial'],**kwargs)
    return {'schema':'neural-shrimp-eigenpair-tangent-reanchor-v1','passed':result['passed'],'localization':loc,'proposal':proposed,'charts':[compact(v) for v in result['charts'] if v['passed']],
            'gates':[v['gates'] for v in result['charts']],'scope':'Same normalized primitive eigenpair locus, conditional on authenticated parent; no simplicity, rate-plane regularity, attraction or cusp association.'}
