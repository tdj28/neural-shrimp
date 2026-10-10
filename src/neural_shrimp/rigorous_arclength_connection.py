"""Validated multiple-shooting charts in a fixed exact affine hyperplane gauge.

The curve parameter is s=n.(U-origin), U=(native phase states, actual hidden
rate). Neither normalization nor a numerical tangent is assumed exact: the
fixed dyadic normal is merely a proposal; all gates use the actual bordered
Jacobian. No attraction or fold identity follows from a successful chart alone.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from .rigorous_orbits import exact,eye,midpoint_matrix,norm_inf_upper,ball_json,matrix_json,vector_json
from .rigorous_connection import require,restore_ball,shooting
from .rigorous_connection_frames import restore_matrix,fraction_mid
from .rigorous_flips import FlipJet


def bordered(model,flat,period,normal,origin,parameter,tick=lambda:None):
    n=model.n;dim=period*n
    require(len(flat)==dim+1 and len(normal)==len(origin)==dim+1,'border dimensions')
    r,a,b,blocks=shooting(model,[flat[j*n:(j+1)*n] for j in range(period)],flat[-1],tick)
    residual=[r[i,0] for i in range(dim)]+[sum((normal[i]*(flat[i]-origin[i]) for i in range(dim+1)),arb(0))-parameter]
    jac=arb_mat([[a[i,j] if j<dim else b[i,0] for j in range(dim+1)] for i in range(dim)]+[normal])
    return arb_mat([[v] for v in residual]),jac,blocks


def arc_path(jet_model,center,slope,curvature,normal,origin,scenter,offset,period,tick=lambda:None):
    n=jet_model.n;dim=period*n;size=dim+1
    require(len(center)==size and slope.nrows()==curvature.nrows()==size,'curved path dimensions')
    jets=[FlipJet(center[i]+slope[i,0]*offset+curvature[i,0]*offset*offset/2,[],
                  slope[i,0]+curvature[i,0]*offset,[],curvature[i,0],0) for i in range(size)]
    residual=[]
    for j in range(period):
        tick();image=jet_model.jet_step(jets[j*n:(j+1)*n],jets[-1])
        residual.extend(image[i]-jets[((j+1)%period)*n+i] for i in range(n))
    gauge=sum((normal[i]*(jets[i]-origin[i]) for i in range(size)),FlipJet(0,[],0,[],0,0))
    residual.append(gauge-FlipJet(scenter+offset,[],1,[],0,0))
    return {key:arb_mat([[getattr(v,key)] for v in residual]) for key in ('f','d','d2','d3')}


def anchor_containment(anchor,center,slope,curvature,normal,origin,scenter,delta,radii):
    """Identify the known root at its actual gauge, preserving its definition.

    When s is unknown, the full anchor box yields S=n.(anchor-origin). For the
    actual root U*, s*=n.(U*-origin) belongs to S. Bounding U*-q(s*) by
    anchor-q(S) is conservative but valid. Mere overlap is never accepted.
    """
    require(anchor is not None and len(anchor['state_box'])==len(center),'full augmented anchor box')
    box=[restore_ball(v) for v in anchor['state_box']]
    projection=sum((normal[i]*(box[i]-origin[i]) for i in range(len(box))),arb(0))
    if 'exact_parameter' in anchor:
        value=exact(anchor['exact_parameter']);require(value.is_exact(),'exact common gauge parameter')
        # The predecessor certificate establishes this equation, not its box.
        interval=value;kind='prior certified root at identical exact gauge'
    else:interval=projection;kind='correlated gauge s=n.(actual anchor root-origin)'
    d=interval-scenter
    inside=bool(abs(d).upper()<=delta)
    checks=[bool(abs(box[i]-(center[i]+slope[i,0]*d+curvature[i,0]*d*d/2)).upper()<=radii[i]) for i in range(len(box))]
    return {'kind':kind,'parameter_enclosure':ball_json(interval),'projection_enclosure':ball_json(projection),
            'parameter_inside':inside,'coordinate_inclusions':checks,'passed':inside and all(checks)}


def arclength_tube(model,jet_model,initial,*,normal,origin,parameter,weights,period,
                    halfwidth_exponent=-6,radius_exponent=-22,precision=256,newton_steps=2,
                    anchor=None,tick=lambda:None,observe=lambda k,r:None):
    require(type(precision) is int and 128<=precision<=512,'precision cap')
    require(type(newton_steps) is int and 0<=newton_steps<=2,'proposal cap')
    require(all(type(x) is int and -100<=x<=-1 for x in (halfwidth_exponent,radius_exponent)),'width/radius bounds')
    n=model.n;size=period*n+1
    require(type(period) is int and 1<=period<=18 and 1<=n<=13 and jet_model.n==n,'native dimensions')
    require(len(initial)==len(normal)==len(origin)==len(weights)==size,'augmented dimensions')
    with ctx.workprec(precision):
        center=[exact(v) for v in initial];nn=[exact(v) for v in normal];oo=[exact(v) for v in origin];w=[exact(v) for v in weights]
        require(all(v.is_exact() and v.is_finite() for v in center+nn+oo+w),'fixed exact dyadic chart data')
        require(center[-1]>0,'positive initial hidden rate')
        require(any(v!=0 for v in nn) and all(v>0 and v<=1 for v in w),'nonzero normal and positive normalized weights')
        sc=exact(parameter);require(sc.is_exact() and sc.is_finite(),'exact gauge center')
        delta=arb((1,halfwidth_exponent));radius=arb((1,radius_exponent));d=arb(0,delta)
        for iteration in range(newton_steps):
            r,a,_=bordered(model,center,period,nn,oo,sc,tick);tick();step=a.solve(r)
            center=[(center[i]-step[i,0]).mid() for i in range(size)]
            observe('proposal',{'iteration':iteration,'center':vector_json(center),'residual_before':ball_json(norm_inf_upper(r))})
        residual,a,_=bordered(model,center,period,nn,oo,sc,tick);tick();pre=midpoint_matrix(a.inv())
        rhs=arb_mat([[int(i==size-1)] for i in range(size)])
        slope=midpoint_matrix(pre*rhs)
        affine=arc_path(jet_model,center,slope,arb_mat(size,1),nn,oo,sc,arb(0),period,tick)
        curvature=midpoint_matrix(-pre*affine['d2'])
        atcenter=arc_path(jet_model,center,slope,curvature,nn,oo,sc,arb(0),period,tick)
        whole=arc_path(jet_model,center,slope,curvature,nn,oo,sc,d,period,tick)
        rr=atcenter['f']+atcenter['d']*d+atcenter['d2']*(d*d/2)+whole['d3']*(d*d*d/6)
        predictor=[center[i]+slope[i,0]*d+curvature[i,0]*d*d/2 for i in range(size)]
        radii=[radius*v for v in w];error=arb_mat([[arb(0,v)] for v in radii])
        boxes=[predictor[i]+error[i,0] for i in range(size)]
        require(boxes[-1].is_finite() and boxes[-1]>0,'whole hidden-rate domain positive')
        _,abox,blocks=bordered(model,boxes,period,nn,oo,arb(sc,delta),tick)
        defect=eye(size)-pre*abox
        contraction=max((sum((abs(defect[i,j]).upper()*w[j] for j in range(size)),arb(0))/w[i]).upper() for i in range(size))
        image=-pre*rr+defect*error
        checks=[bool(abs(image[i,0]).upper()<radii[i]) for i in range(size)]
        root=bool(all(checks) and contraction.is_finite() and contraction<1)
        observe('root_gate',{'root':root,'contraction':ball_json(contraction),'max_image':ball_json(norm_inf_upper(image)),
            'max_predictor_residual':ball_json(norm_inf_upper(rr)),'hidden_rate_domain':ball_json(boxes[-1])})
        divisors=[]
        for k in range(1,period):
            if period%k:continue
            diffs=[boxes[k*n+i]-boxes[i] for i in range(n)]
            sep=[i for i,v in enumerate(diffs) if v.is_finite() and not v.contains(0)]
            divisors.append({'divisor':k,'excluded':bool(sep),'separating_coordinates':sep,'differences':vector_json(diffs)})
        joined=anchor_containment(anchor,center,slope,curvature,nn,oo,sc,delta,radii) if anchor is not None else {'passed':False}
        gates={'uniform_root':root,'primitive_period':all(v['excluded'] for v in divisors),'anchor_identified':joined['passed']}
        endpoints=[]
        for sign in (-1,1):
            dd=sign*delta;res=atcenter['f']+atcenter['d']*dd+atcenter['d2']*(dd*dd/2)+whole['d3']*(dd*dd*dd/6)
            ee=-pre*res+defect*error
            endpoints.append(vector_json([center[i]+slope[i,0]*dd+curvature[i,0]*dd*dd/2+ee[i,0] for i in range(size)]))
        return {'schema':'neural-shrimp-arclength-shooting-tube-v1','passed':all(gates.values()),'gates':gates,
            'statement':{'period':period,'phase_dimension':n,'shooting_dimension':size,'parameter':'s=n.(U-origin), U=(native states, actual hidden rate); fixed output rate'},
            'settings':{'halfwidth_exponent':halfwidth_exponent,'radius_exponent':radius_exponent,'precision':precision,'newton_steps':newton_steps},
            'parameter_center':parameter,'exact_parameter_endpoints':[str(Fraction(parameter)-Fraction(2)**halfwidth_exponent),str(Fraction(parameter)+Fraction(2)**halfwidth_exponent)],
            'normal':normal,'origin':origin,'center':vector_json(center),'predictor_slope':matrix_json(slope),'predictor_curvature':matrix_json(curvature),
            'preconditioner':matrix_json(pre),'proposal_affine_second_derivative':matrix_json(affine['d2']),
            'center_path_derivatives':{k:matrix_json(v) for k,v in atcenter.items()},'whole_path_third_derivative':matrix_json(whole['d3']),
            'predictor_residual':matrix_json(rr),'augmented_box':vector_json(boxes),'jacobian_tube':matrix_json(abox),
            'phase_jacobians':[matrix_json(v) for v in blocks],'defect':matrix_json(defect),'correction_radius':ball_json(radius),
            'exact_weights':weights,'coordinate_radii':vector_json(radii),'image_correction':matrix_json(image),'contraction':ball_json(contraction),
            'coordinate_inclusions':checks,'divisors':divisors,'anchor':anchor,'anchor_check':joined,'endpoint_root_enclosures':endpoints,
            'theorem':'Uniform cubic Taylor residual, strict weighted Krawczyk inclusion and norm below one give a locally unique continuous root for each s. The full native anchor enclosure lies in the corresponding uniqueness domain at its actual, possibly enclosed gauge.',
            'not_claimed':['attraction','fold identity','whole shrimp geometry','binary64 dynamics']}


def two_arclength_charts(model,jet_model,inputs,*,synthetic_profile=False,tick=lambda:None,observe=lambda k,r:None,**settings):
    first=arclength_tube(model,jet_model,inputs['initial'],normal=inputs['normal'],origin=inputs['origin'],parameter='0',
        weights=inputs['weights'],period=inputs['period'],anchor=inputs['anchor'],tick=tick,observe=observe,**settings)
    observe('chart_completed',{'chart':0,'result':first});charts=[first]
    if first['passed'] or synthetic_profile:
        with ctx.workprec(settings.get('precision',256)):
            delta=Fraction(2)**settings.get('halfwidth_exponent',-6);flat=[restore_ball(v) for v in first['center']]
            slope=restore_matrix(first['predictor_slope']);curv=restore_matrix(first['predictor_curvature']);d=exact(str(2*delta))
            proposed=[str(fraction_mid(flat[i]+slope[i,0]*d+curv[i,0]*d*d/2)) for i in range(len(flat))]
            anchor={'state_box':first['endpoint_root_enclosures'][1],'exact_parameter':str(delta)}
            second=arclength_tube(model,jet_model,proposed,normal=inputs['normal'],origin=inputs['origin'],parameter=str(2*delta),
                weights=inputs['weights'],period=inputs['period'],anchor=anchor,tick=tick,observe=observe,**settings)
        observe('chart_completed',{'chart':1,'result':second});charts.append(second)
    passed=len(charts)==2 and all(c['passed'] for c in charts)
    return {'schema':'neural-shrimp-two-arclength-tubes-v1','passed':passed,'charts':charts,
        'connected_parameter_endpoints':[first['exact_parameter_endpoints'][0],charts[-1]['exact_parameter_endpoints'][1]] if passed else None,
        'scope':'same native primitive branch with an explicit affine gauge; no attraction or distant fold identity'}
