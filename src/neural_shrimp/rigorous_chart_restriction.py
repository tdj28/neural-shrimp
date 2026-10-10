"""Restrict a saved curved predictor and prove a fresh full error domain.

The old H3 enclosure remains valid because the exact offset interval and the
entire value/velocity/acceleration inputs are restricted. All lower Taylor
defects stay present. A correlated anchor retains its actual enclosed gauge.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from .rigorous_orbits import exact,eye,ball_json,matrix_json,vector_json
from .rigorous_connection import require,restore_ball
from .rigorous_connection_frames import restore_matrix
from .rigorous_arclength_connection import bordered,anchor_containment

def restrict_chart(model,chart,proposal,*,tick=lambda:None,observe=lambda k,r:None):
    p=chart['statement']['period'];n=chart['statement']['phase_dimension'];size=p*n+1
    require(model.n==n and p<=18 and n<=13,'unchanged native dimensions')
    new_halfwidth=proposal['halfwidth_exponent'];old_halfwidth=chart['settings']['halfwidth_exponent']
    require(type(new_halfwidth) is int and -100<=new_halfwidth<old_halfwidth<=-1,'strict parameter restriction')
    require(len(chart['anchor']['state_box'])==size,'full possibly correlated known root anchor')
    exponent=proposal['radius_exponent'];require(type(exponent) is int and -100<=exponent<=-10,'new radius bound')
    with ctx.workprec(chart['settings']['precision']):
        center=[restore_ball(v) for v in chart['center']];nn=[exact(v) for v in chart['normal']];oo=[exact(v) for v in chart['origin']]
        slope=restore_matrix(chart['predictor_slope']);curvature=restore_matrix(chart['predictor_curvature']);c=restore_matrix(chart['preconditioner'])
        w=[exact(v) for v in proposal['exact_weights']];require(len(w)==size and all(v.is_exact() and v.is_finite() and 0<v<=1 for v in w),'positive exact new weights')
        require(all(v.is_exact() for v in center) and all(slope[i,0].is_exact() and curvature[i,0].is_exact() for i in range(size)),'same exact predictor')
        radius=arb((1,exponent));radii=[radius*v for v in w]
        require(all(exact(proposal['exact_coordinate_radii'][i])==radii[i] for i in range(size)),'exact proposed radii normalization')
        sc=exact(chart['parameter_center']);delta=arb((1,new_halfwidth));dd=arb(0,delta)
        oldd=arb(0,arb((1,old_halfwidth)))
        oldpath=[center[i]+slope[i,0]*oldd+curvature[i,0]*oldd*oldd/2 for i in range(size)]
        oldvelocity=[slope[i,0]+curvature[i,0]*oldd for i in range(size)]
        predictor=[center[i]+slope[i,0]*dd+curvature[i,0]*dd*dd/2 for i in range(size)]
        newvelocity=[slope[i,0]+curvature[i,0]*dd for i in range(size)]
        path_checks=[bool(oldpath[i].contains(predictor[i]) and oldvelocity[i].contains(newvelocity[i])) for i in range(size)]
        require(all(path_checks),'new curved path and velocity enclosed by old H3 domain')
        atcenter={k:restore_matrix(v) for k,v in chart['center_path_derivatives'].items()};h3=restore_matrix(chart['whole_path_third_derivative'])
        rr=atcenter['f']+atcenter['d']*dd+atcenter['d2']*(dd*dd/2)+h3*(dd*dd*dd/6)
        error=arb_mat([[arb(0,v)] for v in radii]);box=[predictor[i]+error[i,0] for i in range(size)]
        require(box[-1].is_finite() and box[-1]>0,'whole new hidden-rate domain positive')
        _,a,blocks=bordered(model,box,p,nn,oo,arb(sc,delta),tick)
        defect=eye(size)-c*a;image=-c*rr+defect*error
        contraction=max((sum((abs(defect[i,j]).upper()*w[j] for j in range(size)),arb(0))/w[i]).upper() for i in range(size))
        checks=[bool(abs(image[i,0]).upper()<radii[i]) for i in range(size)]
        divisors=[]
        for k in range(1,p):
            if p%k:continue
            diffs=[box[k*n+i]-box[i] for i in range(n)];sep=[i for i,v in enumerate(diffs) if v.is_finite() and not v.contains(0)]
            divisors.append({'divisor':k,'excluded':bool(sep),'separating_coordinates':sep,'differences':vector_json(diffs)})
        anchor=anchor_containment(chart['anchor'],center,slope,curvature,nn,oo,sc,delta,radii)
        root=bool(all(checks) and contraction.is_finite() and contraction<1)
        gates={'uniform_root':root,'primitive_period':all(v['excluded'] for v in divisors),'anchor_identified':anchor['passed']}
        endpoints=[]
        for sign in (-1,1):
            d=sign*delta;res=atcenter['f']+atcenter['d']*d+atcenter['d2']*(d*d/2)+h3*(d*d*d/6)
            ee=-c*res+defect*error
            endpoints.append(vector_json([center[i]+slope[i,0]*d+curvature[i,0]*d*d/2+ee[i,0] for i in range(size)]))
        result=dict(chart);result.update(passed=all(gates.values()),gates=gates,settings={**chart['settings'],'radius_exponent':exponent,'halfwidth_exponent':new_halfwidth},
          predictor_residual=matrix_json(rr),exact_parameter_endpoints=[str(Fraction(chart['parameter_center'])-Fraction(2)**new_halfwidth),str(Fraction(chart['parameter_center'])+Fraction(2)**new_halfwidth)],
          exact_weights=proposal['exact_weights'],correction_radius=ball_json(radius),coordinate_radii=vector_json(radii),
          augmented_box=vector_json(box),jacobian_tube=matrix_json(a),phase_jacobians=[matrix_json(v) for v in blocks],
          defect=matrix_json(defect),contraction=ball_json(contraction),image_correction=matrix_json(image),coordinate_inclusions=checks,
          divisors=divisors,anchor_check=anchor,endpoint_root_enclosures=endpoints,
          restriction={'proposal':proposal,'initial_settings':chart['settings'],'initial_gates':chart['gates'],
             'unchanged_exact_predictor_and_all_Taylor_coefficients':True,'new_parameter_interval':ball_json(dd),'old_parameter_interval':ball_json(oldd),
             'old_path':vector_json(oldpath),'new_path':vector_json(predictor),'old_velocity':vector_json(oldvelocity),'new_velocity':vector_json(newvelocity),
             'path_containment':path_checks,'curvature_unchanged':True,'fresh_whole_domain_jacobian':True,'maximum_adaptations':1})
        observe('restriction_gate',{'gates':gates,'contraction':ball_json(contraction),'coordinate_inclusions':sum(checks)})
        return result

