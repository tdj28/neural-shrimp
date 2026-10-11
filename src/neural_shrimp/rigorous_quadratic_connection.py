"""Uniform shooting tubes with exact quadratic predictors and cubic remainder.

The unchanged explicit loss-gradient formula and directional jet arithmetic
bound the predictor residual. The unchanged C1 Hessian bounds error segments.
All cancellation coefficients, including proposal defects, remain enclosed.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from .rigorous_orbits import exact,eye,midpoint_matrix,norm_inf_upper,ball_json,matrix_json,vector_json
from .rigorous_connection import require,restore_ball,shooting
from .rigorous_connection_frames import restore_matrix,fraction_mid
from .rigorous_flips import FlipJet


def path_residual(jet_model,center,slope,curvature,rate,offset,period,tick=lambda:None):
    """Derivatives along q(d)=center+slope*d+curvature*d²/2; actual rate h+d.

    Empty base-gradient fields are intentional: only ordinary derivatives
    along this curved scalar path are needed. FlipJet.d2/d3 chain rules include
    all path-acceleration terms; these are derivatives, not Taylor coefficients.
    """
    n=jet_model.n;dim=period*n
    require(len(center)==dim and slope.nrows()==curvature.nrows()==dim,'path dimensions')
    states=[FlipJet(center[i]+slope[i,0]*offset+curvature[i,0]*offset*offset/2,[],
                    slope[i,0]+curvature[i,0]*offset,[],curvature[i,0],0) for i in range(dim)]
    parameter=FlipJet(rate+offset,[],arb(1),[],0,0);residual=[]
    for j in range(period):
        tick();image=jet_model.jet_step(states[j*n:(j+1)*n],parameter)
        residual.extend(image[i]-states[((j+1)%period)*n+i] for i in range(n))
    return {name:arb_mat([[getattr(v,name)] for v in residual]) for name in ('f','d','d2','d3')}


def quadratic_tube(model,jet_model,initial,rate,*,halfwidth_exponent=-8,radius_exponent=-24,
                   precision=256,newton_steps=2,anchor=None,tick=lambda:None,observe=lambda k,r:None):
    require(type(precision) is int and 128<=precision<=512,'precision cap')
    require(type(newton_steps) is int and 0<=newton_steps<=2,'proposal cap')
    require(all(type(x) is int and -100<=x<=-1 for x in (halfwidth_exponent,radius_exponent)),'width/radius bounds')
    p=len(initial);n=model.n;dim=p*n
    require(1<=p<=18 and 1<=n<=13 and jet_model.n==n and all(len(row)==n for row in initial),'dimension cap')
    with ctx.workprec(precision):
        h=exact(rate);delta=arb((1,halfwidth_exponent));radius=arb((1,radius_exponent));d=arb(0,delta);interval=arb(h,delta)
        require(h.is_exact() and interval>0,'exact dyadic center and positive interval')
        states=[[exact(v).mid() for v in row] for row in initial]
        for iteration in range(newton_steps):
            residual,jac,_,_=shooting(model,states,h,tick);tick();step=jac.solve(residual)
            states=[[(states[j][i]-step[j*n+i,0]).mid() for i in range(n)] for j in range(p)]
            observe('proposal',{'iteration':iteration,'states':[vector_json(row) for row in states],'residual_before':ball_json(norm_inf_upper(residual))})
        residual,jac,partial,_=shooting(model,states,h,tick);tick();pre=midpoint_matrix(jac.inv())
        slope=midpoint_matrix(-pre*partial);flat=[v for row in states for v in row]
        affine=path_residual(jet_model,flat,slope,arb_mat(dim,1),h,arb(0),p,tick)
        curvature=midpoint_matrix(-pre*affine['d2'])
        center_path=path_residual(jet_model,flat,slope,curvature,h,arb(0),p,tick)
        whole_path=path_residual(jet_model,flat,slope,curvature,h,d,p,tick)
        # Do not set H'(0) or H''(0) to zero: their enclosed proposal defects count.
        # Explicit interval products: Arb's generic ** uses logarithms and is
        # not defined on a sign-changing interval, even for integer exponents.
        square=d*d;cube=square*d
        predictor_residual=center_path['f']+center_path['d']*d+center_path['d2']*(square/2)+whole_path['d3']*(cube/6)
        predictor=[flat[i]+slope[i,0]*d+curvature[i,0]*square/2 for i in range(dim)]
        error=arb_mat([[arb(0,radius)] for _ in range(dim)])
        boxes=[predictor[i]+error[i,0] for i in range(dim)]
        phases=[boxes[j*n:(j+1)*n] for j in range(p)]
        _,abox,_,blocks=shooting(model,phases,interval,tick)
        defect=eye(dim)-pre*abox;contraction=norm_inf_upper(defect)
        image=-pre*predictor_residual+defect*error
        checks=[bool(abs(image[i,0]).upper()<radius) for i in range(dim)]
        root=bool(all(checks) and contraction.is_finite() and contraction<1)
        observe('root_gate',{'root':root,'contraction':ball_json(contraction),'max_image':ball_json(norm_inf_upper(image)),
            'max_predictor_residual':ball_json(norm_inf_upper(predictor_residual))})
        divisors=[]
        for k in range(1,p):
            if p%k:continue
            diffs=[phases[k][i]-phases[0][i] for i in range(n)]
            separating=[i for i,v in enumerate(diffs) if v.is_finite() and not v.contains(0)]
            divisors.append({'divisor':k,'excluded':bool(separating),'separating_coordinates':separating,'differences':vector_json(diffs)})
        inclusions=[]
        if anchor is not None:
            at=Fraction(anchor['rate']);a=Fraction(rate)-Fraction(2)**halfwidth_exponent;b=Fraction(rate)+Fraction(2)**halfwidth_exponent
            require(a<=at<=b and len(anchor['phase_boxes'])==p,'exact common anchor parameter')
            dd=exact(str(at-Fraction(rate)))
            for j,row in enumerate(anchor['phase_boxes']):
                require(len(row)==n,'anchor native dimensions')
                for i,value in enumerate(row):
                    k=j*n+i;predicted=flat[k]+slope[k,0]*dd+curvature[k,0]*dd*dd/2
                    inclusions.append(bool(abs(restore_ball(value)-predicted).upper()<=radius))
        gates={'uniform_root':root,'primitive_period':all(v['excluded'] for v in divisors),'anchor_identified':bool(anchor is not None and all(inclusions))}
        endpoints=[]
        for sign in (-1,1):
            dd=sign*delta
            rr=center_path['f']+center_path['d']*dd+center_path['d2']*(dd*dd/2)+whole_path['d3']*(dd*dd*dd/6)
            ee=-pre*rr+defect*error
            endpoints.append(vector_json([flat[i]+slope[i,0]*dd+curvature[i,0]*dd*dd/2+ee[i,0] for i in range(dim)]))
        return {'schema':'neural-shrimp-quadratic-shooting-tube-v1','passed':all(gates.values()),'gates':gates,
            'statement':{'period':p,'phase_dimension':n,'shooting_dimension':dim,'parameter':'actual hidden rate, fixed exact output rate'},
            'settings':{'halfwidth_exponent':halfwidth_exponent,'radius_exponent':radius_exponent,'precision':precision,'newton_steps':newton_steps},
            'parameter_center':rate,'parameter_interval':ball_json(interval),
            'exact_parameter_endpoints':[str(Fraction(rate)-Fraction(2)**halfwidth_exponent),str(Fraction(rate)+Fraction(2)**halfwidth_exponent)],
            'center':[vector_json(row) for row in states],'predictor_slope':matrix_json(slope),'predictor_curvature':matrix_json(curvature),
            'preconditioner':matrix_json(pre),'proposal_affine_second_derivative':matrix_json(affine['d2']),
            'center_path_derivatives':{k:matrix_json(v) for k,v in center_path.items()},'whole_path_third_derivative':matrix_json(whole_path['d3']),
            'predictor_residual':matrix_json(predictor_residual),'phase_boxes':[vector_json(row) for row in phases],
            'jacobian_tube':matrix_json(abox),'phase_jacobians':[matrix_json(v) for v in blocks],'defect':matrix_json(defect),
            'correction_radius':ball_json(radius),'image_correction':matrix_json(image),'contraction':ball_json(contraction),
            'coordinate_inclusions':checks,'divisors':divisors,'anchor_inclusions':inclusions,'endpoint_root_enclosures':endpoints,
            'scope':'uniform locally unique primitive branch on this closed interval, identified with supplied anchor only when all gates pass; attraction not evaluated',
            'theorem':'Cubic Taylor remainder bounds H(q(h),h); uniform Krawczyk strict inclusion and contraction on q(h)+E imply a unique continuous root. All H0/H1/H2 proposal defects are retained.',
            'not_claimed':['attraction','fold/flip connection','whole shrimp geometry','finite-precision trajectory theorem']}


def two_quadratic_charts(model,jet_model,inputs,*,synthetic_profile=False,tick=lambda:None,observe=lambda k,r:None,**settings):
    first=quadratic_tube(model,jet_model,inputs['initial'],inputs['rate'],anchor=inputs['anchor'],tick=tick,observe=observe,**settings)
    observe('chart_completed',{'chart':0,'result':first});charts=[first]
    if first['passed'] or synthetic_profile:
        with ctx.workprec(settings.get('precision',256)):
            delta=Fraction(2)**settings.get('halfwidth_exponent',-8);rate=Fraction(inputs['rate']);n=model.n;p=len(inputs['initial'])
            flat=[restore_ball(v) for row in first['center'] for v in row];s=restore_matrix(first['predictor_slope']);t=restore_matrix(first['predictor_curvature']);d=exact(str(2*delta))
            proposal=[str(fraction_mid(flat[i]+s[i,0]*d+t[i,0]*d*d/2)) for i in range(p*n)]
            ep=first['endpoint_root_enclosures'][1];anchor={'rate':str(rate+delta),'phase_boxes':[ep[j*n:(j+1)*n] for j in range(p)]}
            second=quadratic_tube(model,jet_model,[proposal[j*n:(j+1)*n] for j in range(p)],str(rate+2*delta),anchor=anchor,tick=tick,observe=observe,**settings)
        observe('chart_completed',{'chart':1,'result':second});charts.append(second)
    passed=len(charts)==2 and all(c['passed'] for c in charts)
    return {'schema':'neural-shrimp-two-quadratic-tubes-v1','passed':passed,'charts':charts,
        'connected_parameter_endpoints':[first['exact_parameter_endpoints'][0],charts[-1]['exact_parameter_endpoints'][1]] if passed else None,
        'scope':'same anchored primitive branch; no attraction, distant boundary connection or geometry proof'}
