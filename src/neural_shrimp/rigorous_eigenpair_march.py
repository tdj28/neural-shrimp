"""Bounded orchestration of unchanged quadratic tubes; floating diagnostics only."""
from fractions import Fraction
import math
import numpy as np
from flint import arb,arb_mat,ctx
from .rigorous_eigenpair_locus import require,exact_mid
from .rigorous_eigenpair_path import quadratic_tube
from .rigorous_orbits import exact,ball_json


def decode(v):
    """Conservatively restore a saved ball (constructor may enlarge radius)."""
    require(set(v)>={'mid','rad'} and all(isinstance(v[k],list) and len(v[k])==2 and all(type(a)is int for a in v[k]) for k in ('mid','rad')),'saved dyadics')
    require(v['rad'][0]>=0,'nonnegative radius')
    x=arb(tuple(v['mid']));r=arb(tuple(v['rad']));a=arb(x,r)
    require(a.is_finite(),'finite saved ball')
    return a


def compact(chart):
    require(chart['passed'] is True and chart['gates']==dict.fromkeys(('uniform_normalized_eigenpair_locus','primitive_period','anchor_identified'),True),'passed parent only')
    keys=('schema','passed','gates','period','dimension','sigma','shooting_dimension','parameter_center','parameter_endpoints','normal','origin','center','slope','curvature','weights','settings','endpoint_root_boxes')
    return {k:chart[k] for k in keys}


def next_inputs(parent):
    parent=compact(parent);p=parent['period'];n=parent['dimension'];size=2*n*p+2;s=parent['settings']
    require(parent['schema']=='neural-shrimp-quadratic-eigenpair-locus-v1' and size==parent['shooting_dimension'],'quadratic native schema')
    require(all(len(parent[k])==size for k in ('center','slope','curvature','normal','origin','weights')),'native dimensions')
    require(len(parent['endpoint_root_boxes'])==2 and all(len(v)==size for v in parent['endpoint_root_boxes']),'endpoint shape')
    delta=Fraction(2)**s['halfwidth_exponent'];sc=Fraction(parent['parameter_center']);oldright=sc+delta
    require(parent['parameter_endpoints']==[str(sc-delta),str(oldright)],'exact old gauge interval')
    center=[decode(v) for v in parent['center']];slope=[decode(v) for v in parent['slope']];curve=[decode(v) for v in parent['curvature']]
    require(all(v.is_exact() for v in center+slope+curve),'exact predictor proposals')
    dd=exact(str(2*delta));initial=[exact_mid(c+slope[i]*dd+curve[i]*dd*dd/2) for i,c in enumerate(center)]
    anchor={'box':[decode(v) for v in parent['endpoint_root_boxes'][1]],'exact_parameter':str(oldright),'period':p,'sigma':parent['sigma'],'dimension':n}
    kwargs=dict(period=p,sigma=parent['sigma'],normal=parent['normal'],origin=parent['origin'],parameter=str(sc+2*delta),weights=parent['weights'],anchor=anchor,**s)
    return initial,kwargs


def midpoint_spectrum(chart,observe=lambda k,v:None):
    """One full dense midpoint eig; no certified simplicity or stability claim."""
    n=chart['dimension'];p=chart['period'];m=2*n;size=m*p+2
    J=np.array([[float(decode(x).mid()) for x in row] for row in chart['center_jacobian']]);require(J.shape==(size,size) and np.isfinite(J).all(),'finite midpoint Jacobian')
    M=np.eye(n)
    for k in range(p):
        A=J[k*m:k*m+n,k*m:k*m+n].copy()
        if p==1:A+=np.eye(n)
        M=A@M
    values,vectors=np.linalg.eig(M)
    part=lambda v:float(v) if math.isfinite(v) else str(v)
    observe('midpoint_eig_return',{'monodromy_midpoint':M.tolist(),'eigenvalues':[[part(v.real),part(v.imag)] for v in values],'eigenvectors':[[[part(v.real),part(v.imag)] for v in row] for row in vectors],'scope':'Floating return retained before finite/residual checks.'})
    require(np.isfinite(values).all() and np.isfinite(vectors).all(),'finite eig returns')
    norm=float(np.sqrt(np.sum(abs(M)**2)))
    residuals=[float(np.sqrt(np.sum(abs(M@vectors[:,i]-values[i]*vectors[:,i])**2))/(max(1.,norm)*np.sqrt(np.sum(abs(vectors[:,i])**2)))) for i in range(n)]
    order=np.argsort(abs(values-chart['sigma']));other=float(min(abs(values[j]-chart['sigma']) for j in order[1:])) if n>1 else None
    q=np.array([float(decode(v).mid()) for v in chart['center'][n:2*n]])
    return {'monodromy_midpoint':M.tolist(),'eigenvalues':[[float(v.real),float(v.imag)] for v in values],
        'eigenvectors':[[[float(v.real),float(v.imag)] for v in row] for row in vectors],'relative_residuals':residuals,'critical_nearest_index':int(order[0]),
        'other_distance_to_critical':other,'complementary_near_critical_flag':other is not None and other<1e-4,
        'flag_threshold':1e-4,'mode_residual':float(np.sqrt(np.sum(abs(M@q-chart['sigma']*q)**2))),
        'scope':'Floating full midpoint spectrum only; not certified simple mode, attraction or bifurcation classification.'}


def proposed_radii(chart,observe=lambda k,v:None):
    """One floating positive-resolvent proposal; fresh full tube must validate it."""
    require(chart['gates']=={'uniform_normalized_eigenpair_locus':False,'primitive_period':True,'anchor_identified':True},'root-only failure for rebox')
    C=arb_mat([[decode(v) for v in row] for row in chart['preconditioner']]);R=arb_mat([[decode(v[0])] for v in chart['predictor_residual']]);forcing=-C*R
    a=np.array([float(abs(forcing[i,0]).upper()) for i in range(C.nrows())]);B=np.array([[float(abs(decode(v)).upper()) for v in row] for row in chart['defect']])
    require(np.isfinite(a).all() and np.isfinite(B).all() and np.all(a>=0) and np.all(B>=0),'finite nonnegative proposal inputs')
    solution=np.linalg.solve(np.eye(len(a))-B,a)
    observe('rebox_linear_return',{'forcing':a.tolist(),'defect_absolute':B.tolist(),'solution':[float(v) if math.isfinite(v) else None for v in solution],
        'scope':'Floating proposal only; previous derivative domain is not a proof on changed radii.'})
    require(np.isfinite(solution).all() and np.all(solution>=0),'nonnegative finite radii solve')
    radii=(17/16)*solution+max(float(np.max(solution)),2.**-100)*2.**-40
    require(np.isfinite(radii).all() and np.all(radii>0),'positive radii')
    exact_radii=[Fraction.from_float(float(v)) for v in radii];maximum=max(exact_radii)
    exponent=-100
    while Fraction(2)**exponent<maximum:exponent+=1
    require(-100<=exponent<=-1,'bounded proposed radius')
    weights=[str(v/Fraction(2)**exponent) for v in exact_radii]
    result={'radius_exponent':exponent,'weights':weights,'exact_radii':[str(v) for v in exact_radii],
        'predicted_old_domain_ratios':((a+B@radii)/radii).tolist(),'scope':'Requires a new whole-domain Jacobian and all unchanged proof/anchor gates.'}
    observe('rebox_proposal',result);return result


def march(model,parent,*,accepted_cap=4,call_cap=5,tick=lambda:None,observe=lambda k,v:None,tube=quadratic_tube):
    require(type(accepted_cap)is int and 1<=accepted_cap<=4 and call_cap==accepted_cap+1,'fixed bounded unit')
    previous=compact(parent);accepted=[];calls=0;reboxed=False;attempts=[]
    with ctx.workprec(previous['settings']['precision']):
        for index in range(accepted_cap):
            initial,kwargs=next_inputs(previous)
            while True:
                require(calls<call_cap,'kernel call ceiling');call=calls;calls+=1
                start={'call':call,'accepted_index':index,'initial':initial,'parameter':kwargs['parameter'],'radius_exponent':kwargs['radius_exponent'],'weights':kwargs['weights'],'parent_right':previous['parameter_endpoints'][1]}
                observe('march_call_start',start)
                r,_=tube(model,initial,**kwargs,tick=tick,observe=observe)
                attempt={'call':call,'gates':r['gates'],'passed':r['passed'],'parameter_endpoints':r['parameter_endpoints']};attempts.append(attempt)
                try:diagnostic=midpoint_spectrum(r,observe)
                except (ValueError,np.linalg.LinAlgError,OverflowError) as exc:diagnostic={'unresolved':str(exc),'scope':'No certified complement classification.'}
                observe('midpoint_spectrum_return',{'call':call,'diagnostic':diagnostic})
                if r['passed']:
                    saved=compact(r);observe('march_accepted',{'call':call,'index':index,'chart':saved,'diagnostic':diagnostic})
                    accepted.append(attempt);previous=saved;break
                onlyroot=r['gates']=={'uniform_normalized_eigenpair_locus':False,'primitive_period':True,'anchor_identified':True}
                if not onlyroot or reboxed:
                    return {'completed':False,'accepted':accepted,'attempts':attempts,'calls':calls,'rebox_used':reboxed,'last_chart':previous,'stop':'unresolved chart; no failed-tail advancement'}
                reboxed=True;proposal=proposed_radii(r,observe);kwargs.update(radius_exponent=proposal['radius_exponent'],weights=proposal['weights'])
                del r
    return {'completed':True,'accepted':accepted,'attempts':attempts,'calls':calls,'rebox_used':reboxed,'last_chart':previous,'stop':'fixed accepted-chart cap'}
