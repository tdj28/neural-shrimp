"""One deterministic error-box proposal per failed chart, then rigorous recheck.

The unchanged exact predictor and its Taylor residual do not depend on the
error box. The bordered Jacobian, Krawczyk defect, primitive exclusions and
anchor/endpoint enclosures do, and are always recomputed on the proposed box.
All floating arithmetic is proposal generation only.
"""
from fractions import Fraction
import math
import numpy as np
from scipy.linalg import solve
from flint import arb,arb_mat,ctx
from .rigorous_orbits import exact,eye,ball_json,matrix_json,vector_json
from .rigorous_connection import require,restore_ball
from .rigorous_connection_frames import restore_matrix
from .rigorous_arclength_connection import bordered,anchor_containment,arclength_tube
from .rigorous_arclength_batch import compact_chart,next_arguments


def proposed_radii(chart):
    c=restore_matrix(chart['preconditioner']);rr=restore_matrix(chart['predictor_residual']);d=restore_matrix(chart['defect'])
    n=d.nrows();old=[restore_ball(v) for v in chart['coordinate_radii']]
    require(d.ncols()==c.ncols()==c.nrows()==rr.nrows()==len(old)==n and rr.ncols()==1,'proposal dimensions')
    require(all(v.is_exact() and v>0 for v in old),'positive old coordinate radii')
    forcing=c*rr;b=np.array([[float(abs(d[i,j]).upper()) for j in range(n)] for i in range(n)])
    a=np.array([float(abs(forcing[i,0]).upper()) for i in range(n)])
    require(np.isfinite(b).all() and np.isfinite(a).all(),'finite proposal arrays')
    candidate=solve(np.eye(n)-b,a,check_finite=True)
    require(np.isfinite(candidate).all() and np.min(candidate)>=0,'nonnegative finite radii proposal')
    center=[restore_ball(v) for v in chart['center']];s=restore_matrix(chart['predictor_slope']);t=restore_matrix(chart['predictor_curvature'])
    dd=exact(chart['anchor']['exact_parameter'])-exact(chart['parameter_center'])
    needed=[abs(restore_ball(v)-(center[i]+s[i,0]*dd+t[i,0]*dd*dd/2)).upper() for i,v in enumerate(chart['anchor']['state_box'])]
    floor=min(float(v) for v in old)/16
    values=np.nextafter(np.maximum(np.maximum(2*candidate,2*np.array([float(v) for v in needed])),floor),np.inf)
    require(np.isfinite(values).all() and np.min(values)>0,'positive finite final radii proposal')
    exponent=math.ceil(math.log2(float(np.max(values))));scale=Fraction(2)**exponent
    rational=[Fraction(float(v)) for v in values];weights=[v/scale for v in rational]
    require(-100<=exponent<=-10 and all(0<v<=1 for v in weights),'fixed radius proposal cap')
    return {'radius_exponent':exponent,'exact_weights':[str(v) for v in weights],
      'exact_coordinate_radii':[str(v) for v in rational],'old_anchor_required_radii':vector_json(needed),
      'rule':'max(2*(I-abs(D))^-1*abs(CR),2*anchor_requirement,min(old radii)/16), binary64-nextup then exact dyadics',
      'scope':'untrusted proposal only; new whole-domain Jacobian and strict gates required'}


def rebox(model,chart,proposal,*,tick=lambda:None,observe=lambda k,r:None):
    p=chart['statement']['period'];n=chart['statement']['phase_dimension'];size=p*n+1
    require(model.n==n and p<=18 and n<=13,'unchanged native dimensions')
    require(chart['anchor'].get('exact_parameter') is not None,'exact predecessor gauge required')
    exponent=proposal['radius_exponent'];require(type(exponent) is int and -100<=exponent<=-10,'new radius bound')
    with ctx.workprec(chart['settings']['precision']):
        center=[restore_ball(v) for v in chart['center']];nn=[exact(v) for v in chart['normal']];oo=[exact(v) for v in chart['origin']]
        slope=restore_matrix(chart['predictor_slope']);curvature=restore_matrix(chart['predictor_curvature']);c=restore_matrix(chart['preconditioner'])
        w=[exact(v) for v in proposal['exact_weights']];require(len(w)==size and all(v.is_exact() and v.is_finite() and 0<v<=1 for v in w),'positive exact new weights')
        require(all(v.is_exact() for v in center) and all(slope[i,0].is_exact() and curvature[i,0].is_exact() for i in range(size)),'same exact predictor')
        radius=arb((1,exponent));radii=[radius*v for v in w]
        require(all(exact(proposal['exact_coordinate_radii'][i])==radii[i] for i in range(size)),'exact proposed radii normalization')
        sc=exact(chart['parameter_center']);delta=arb((1,chart['settings']['halfwidth_exponent']));dd=arb(0,delta)
        predictor=[center[i]+slope[i,0]*dd+curvature[i,0]*dd*dd/2 for i in range(size)]
        error=arb_mat([[arb(0,v)] for v in radii]);box=[predictor[i]+error[i,0] for i in range(size)]
        require(box[-1].is_finite() and box[-1]>0,'whole new hidden-rate domain positive')
        _,a,blocks=bordered(model,box,p,nn,oo,arb(sc,delta),tick)
        defect=eye(size)-c*a;rr=restore_matrix(chart['predictor_residual']);image=-c*rr+defect*error
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
        atcenter={k:restore_matrix(v) for k,v in chart['center_path_derivatives'].items()};h3=restore_matrix(chart['whole_path_third_derivative']);endpoints=[]
        for sign in (-1,1):
            d=sign*delta;res=atcenter['f']+atcenter['d']*d+atcenter['d2']*(d*d/2)+h3*(d*d*d/6)
            ee=-c*res+defect*error
            endpoints.append(vector_json([center[i]+slope[i,0]*d+curvature[i,0]*d*d/2+ee[i,0] for i in range(size)]))
        result=dict(chart);result.update(passed=all(gates.values()),gates=gates,settings={**chart['settings'],'radius_exponent':exponent},
          exact_weights=proposal['exact_weights'],correction_radius=ball_json(radius),coordinate_radii=vector_json(radii),
          augmented_box=vector_json(box),jacobian_tube=matrix_json(a),phase_jacobians=[matrix_json(v) for v in blocks],
          defect=matrix_json(defect),contraction=ball_json(contraction),image_correction=matrix_json(image),coordinate_inclusions=checks,
          divisors=divisors,anchor_check=anchor,endpoint_root_enclosures=endpoints,
          reboxing={'proposal':proposal,'initial_settings':chart['settings'],'initial_gates':chart['gates'],
             'unchanged_predictor_and_taylor_residual':True,'fresh_whole_domain_jacobian':True,'maximum_adaptations':1})
        observe('rebox_gate',{'gates':gates,'contraction':ball_json(contraction),'coordinate_inclusions':sum(checks)})
        return result


def stream_reboxed(model,jet_model,previous,*,count=16,synthetic_profile=False,force_profile_rebox=False,
                   tick=lambda:None,observe=lambda k,r:None,emit_initial=lambda i,r:None,emit_chart=lambda i,r:None):
    require(type(count) is int and 1<=count<=16,'fixed chart count')
    require(not force_profile_rebox or synthetic_profile,'forced engineering path only')
    require(previous['passed'] and all(previous['gates'].values()) or synthetic_profile,'proved predecessor')
    summaries=[];start=previous['exact_parameter_endpoints'][1];adapted=0;fold_found=False
    for index in range(count):
        tick();arguments=next_arguments(previous)
        initial=arclength_tube(model,jet_model,**arguments,**previous['settings'],tick=tick,observe=observe)
        do_rebox=force_profile_rebox or (not initial['gates']['uniform_root'] and initial['gates']['primitive_period'] and initial['gates']['anchor_identified'])
        if do_rebox:
            identity=emit_initial(index,initial);tick();proposal=proposed_radii(initial)
            result=rebox(model,initial,proposal,tick=tick,observe=observe);adapted+=1
            result['reboxing']['initial_artifact']=identity
        else:result=initial
        saved=emit_chart(index,result)
        fold_found=bool(result['passed'] and any(v['contained_in_qualified_chart'] for v in result.get('known_fold_membership',{}).get('all_cyclic_alignments',[])))
        summary={'index':index,'passed':result['passed'],'gates':result['gates'],'artifact':saved,'reboxed':do_rebox,'known_fold_identified':fold_found,
          'parameter_center':result['parameter_center'],'exact_parameter_endpoints':result['exact_parameter_endpoints'],
          'contraction':result['contraction'],'coordinate_inclusions':sum(result['coordinate_inclusions']),
          'anchor_inclusions':sum(result['anchor_check'].get('coordinate_inclusions',[])),
          'anchor_parameter_inside':result['anchor_check'].get('parameter_inside',False),
          'proper_divisors_excluded':[d['divisor'] for d in result['divisors'] if d['excluded']],
          'hidden_rate_endpoint_enclosures':[v[-1] for v in result['endpoint_root_enclosures']]}
        summaries.append(summary);observe('chart_committed',summary)
        if not synthetic_profile and (not result['passed'] or fold_found):break
        previous=compact_chart(result);del result,initial
    good=[v for v in summaries if v['passed']]
    return {'schema':'neural-shrimp-once-reboxed-arclength-v1','requested_charts':count,'completed_charts':len(summaries),
      'qualified_charts':len(good),'adapted_charts':adapted,'known_fold_identified':fold_found,
      'passed':all(v['passed'] for v in summaries) and (len(summaries)==count or fold_found),'charts':summaries,
      'new_exact_gauge_endpoints':[start,good[-1]['exact_parameter_endpoints'][1]] if good else None,
      'scope':'one deterministic error-box proposal per root-bound failure, full reacquisition; primitive branch only'}
