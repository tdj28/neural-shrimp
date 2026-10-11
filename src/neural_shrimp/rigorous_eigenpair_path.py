"""Quadratic native eigenpair paths using R[t,e]/(t^4,e^2).

Eight factorial-scaled coefficients suffice for F and D_xF q through cubic
path order. Whole-box shooting Jacobians remain the unchanged MixedJet core.
No derivative tensor, floating derivative, or omitted center defect is used.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from .rigorous_eigenpair_locus import (require,finite,column,assemble,proposal,exact_mid)
from .rigorous_orbits import exact,eye,midpoint_matrix,ball_json,vector_json,matrix_json


class PathJet:
    """a[k]=[t^k e^0], b[k]=[t^k e^1], for k=0..3."""
    __slots__=('a','b')
    def __init__(self,a,b=None):
        require(len(a)==4 and (b is None or len(b)==4),'eight coefficient shape')
        self.a=tuple(arb(v) for v in a);self.b=tuple(arb(v) for v in (b if b is not None else [0]*4))
    @classmethod
    def constant(cls,x):return cls([x,0,0,0])
    @classmethod
    def path(cls,x,s,t,q=0,qs=0,qt=0):return cls([x,s,t/2,0],[q,qs,qt/2,0])
    def __add__(self,v):
        if not isinstance(v,PathJet):v=PathJet.constant(v)
        return PathJet([x+y for x,y in zip(self.a,v.a)],[x+y for x,y in zip(self.b,v.b)])
    __radd__=__add__
    def __neg__(self):return self*-1
    def __sub__(self,v):return self+-v
    def __rsub__(self,v):return -self+v
    def __mul__(self,v):
        if not isinstance(v,PathJet):return PathJet([x*v for x in self.a],[x*v for x in self.b])
        a=[sum((self.a[i]*v.a[k-i] for i in range(k+1)),arb(0)) for k in range(4)]
        b=[sum((self.a[i]*v.b[k-i]+self.b[i]*v.a[k-i] for i in range(k+1)),arb(0)) for k in range(4)]
        return PathJet(a,b)
    __rmul__=__mul__
    def __truediv__(self,v):
        require(not isinstance(v,PathJet),'only scalar division')
        return PathJet([x/v for x in self.a],[x/v for x in self.b])
    def tanh(self):
        # The nilpotent ideal (t,e) has fifth power zero in this ring.
        c=self.a[0];q=PathJet([0,*self.a[1:]],self.b);q2=q*q;q3=q2*q;q4=q3*q
        t=c.tanh();s=1-t*t
        return PathJet.constant(t)+s*q-t*s*q2+(-2*s*s+4*t*t*s)*q3/6+(16*t*s*s-8*t*t*t*s)*q4/24
    def serialized(self):return [vector_json(self.a),vector_json(self.b)]


def path_coefficients(model,center,slope,curvature,d,period,sigma,normal,origin,parameter,
                      tick=lambda:None,observe=lambda k,r:None):
    """Taylor coefficients at arbitrary d; interval d encloses all derivatives.

    Input path is P(d+t)+e Q(d+t), with rates P_rate(d+t) and no e terms.
    Both the state and eigenvector fields are quadratic in the path coordinate.
    """
    n=model.n;m=2*n;size=m*period+2
    require(type(period) is int and 1<=period<=18 and 1<=n<=13 and type(sigma) is int and sigma in(-1,1),'native dimensions/sign')
    require(len(center)==len(slope)==len(curvature)==len(normal)==len(origin)==size,'path dimensions')
    require(all(x.is_finite() for x in center+slope+curvature+[d,parameter]),'finite path data')
    z=[PathJet.path(center[i]+slope[i]*d+curvature[i]*(d*d)/2,slope[i]+curvature[i]*d,curvature[i]) for i in range(size)]
    require(all(v.a[0]>0 for v in z[-2:]),'positive whole predictor rates')
    rows=[]
    for k in range(period):
        offset=k*m;after=((k+1)%period)*m
        x=[PathJet(z[offset+i].a,z[offset+n+i].a) for i in range(n)]
        tick();result=model.cusp_step(x,*z[-2:])
        require(len(result)==n and all(isinstance(v,PathJet) for v in result),'path return shape')
        observe('path_phase_return',{'phase':k,'d':ball_json(d),'inputs':[v.serialized() for v in x+z[-2:]],'returns':[v.serialized() for v in result]})
        require(all(a.is_finite() for v in result for a in v.a+v.b),'finite path return')
        rows.extend([[result[i].a[j]-z[after+i].a[j] for j in range(4)] for i in range(n)])
        sign=sigma if k==period-1 else 1
        rows.extend([[result[i].b[j]-sign*z[after+n+i].a[j] for j in range(4)] for i in range(n)])
    norm=(sum((v*v for v in z[n:m]),PathJet.constant(0))-1)/2
    gauge=sum((normal[i]*(z[i]-origin[i]) for i in range(size)),PathJet.constant(0))-PathJet([parameter+d,1,0,0])
    rows.extend([list(norm.a),list(gauge.a)])
    out=arb_mat(rows)
    observe('path_assembly_return',{'d':ball_json(d),'coefficients':matrix_json(out)})
    require(finite(out),'finite path coefficients')
    return out


def quadratic_anchor(anchor,center,slope,curvature,normal,origin,sc,delta,radii,period,sigma,n):
    if anchor is None:return {'passed':False,'reason':'prior root not supplied'}
    require(anchor.get('period')==period and anchor.get('sigma')==sigma and anchor.get('dimension')==n,'anchor convention')
    box=anchor['box'];require(len(box)==len(center) and all(v.is_finite() for v in box),'anchor box')
    projection=sum((normal[i]*(box[i]-origin[i]) for i in range(len(center))),arb(0))
    parameter=exact(anchor['exact_parameter']) if 'exact_parameter' in anchor else projection
    if 'exact_parameter' in anchor:require(parameter.is_exact() and parameter.is_finite(),'exact prior gauge')
    d=parameter-sc;inside=bool(abs(d).upper()<=delta)
    checks=[bool(abs(box[i]-center[i]-slope[i]*d-curvature[i]*(d*d)/2).upper()<radii[i]) for i in range(len(center))]
    return {'passed':inside and all(checks),'parameter_inside':inside,'parameter':ball_json(parameter),'projection':ball_json(projection),'coordinate_inclusions':checks,'scope':'conditional on authenticated prior root and identical native order/sign/model/gauge'}


def quadratic_tube(model,initial,*,period,sigma,normal,origin,parameter,halfwidth_exponent=-12,
                   radius_exponent=-16,weights=None,newton_steps=2,precision=256,anchor=None,
                   tick=lambda:None,observe=lambda k,r:None):
    require(type(precision) is int and 128<=precision<=512,'precision cap')
    require(type(newton_steps) is int and 0<=newton_steps<=2,'proposal cap')
    require(all(type(v) is int and -100<=v<=-1 for v in (halfwidth_exponent,radius_exponent)),'width caps')
    size=2*model.n*period+2;require(len(initial)==len(normal)==len(origin)==size,'chart dimensions')
    weights=['1']*size if weights is None else weights;require(len(weights)==size,'weight dimensions')
    with ctx.workprec(precision):
        center=[exact(v) for v in initial];nn=[exact(v) for v in normal];oo=[exact(v) for v in origin];w=[exact(v) for v in weights];sc=exact(parameter)
        require(all(v.is_finite() and v.is_exact() for v in center+nn+oo+w+[sc]),'exact finite data')
        require(any(v!=0 for v in nn) and all(v>0 and v<=1 for v in w),'gauge/weights')
        delta=arb((1,halfwidth_exponent));d=arb(0,delta);radius=arb((1,radius_exponent))
        for it in range(newton_steps):
            g,j,local=assemble(model,center,period,sigma,nn,oo,sc,tick,observe);pre,small=proposal(j,local,sigma);correction=pre*g
            center=[(center[i]-correction[i,0]).mid() for i in range(size)]
            observe('proposal',{'iteration':it,'center':vector_json(center),'residual':matrix_json(g),'correction':matrix_json(correction),'condensed_border':matrix_json(small)})
        g0,j0,local=assemble(model,center,period,sigma,nn,oo,sc,tick,observe);tick();pre,small=proposal(j0,local,sigma)
        observe('preconditioner_return',{'matrix':matrix_json(pre),'condensed_border':matrix_json(small)})
        slope_m=midpoint_matrix(pre*column([arb(0)]*(size-1)+[arb(1)]));slope=[slope_m[i,0] for i in range(size)]
        affine=path_coefficients(model,center,slope,[arb(0)]*size,arb(0),period,sigma,nn,oo,sc,tick,observe)
        curvature_m=midpoint_matrix(-pre*column([2*affine[i,2] for i in range(size)]));curvature=[curvature_m[i,0] for i in range(size)]
        observe('curvature_proposal',{'slope':vector_json(slope),'curvature':vector_json(curvature),'affine_path_coefficients':matrix_json(affine)})
        coeff=path_coefficients(model,center,slope,curvature,arb(0),period,sigma,nn,oo,sc,tick,observe)
        whole_coeff=path_coefficients(model,center,slope,curvature,d,period,sigma,nn,oo,sc,tick,observe)
        rr=column([coeff[i,0]+coeff[i,1]*d+coeff[i,2]*(d*d)+whole_coeff[i,3]*(d*d*d) for i in range(size)])
        predictor=[center[i]+slope[i]*d+curvature[i]*(d*d)/2 for i in range(size)]
        radii=[radius*v for v in w];error=column([arb(0,r) for r in radii]);box=[predictor[i]+error[i,0] for i in range(size)]
        require(all(v.is_finite() and v>0 for v in box[-2:]),'positive whole rates')
        _,jb,_=assemble(model,box,period,sigma,nn,oo,sc+d,tick,observe)
        defect=eye(size)-pre*jb
        norm=max((sum((abs(defect[i,j]).upper()*w[j] for j in range(size)),arb(0))/w[i]).upper() for i in range(size))
        image=-pre*rr+defect*error;inclusions=[bool(abs(image[i,0]).upper()<radii[i]) for i in range(size)]
        root=bool(all(inclusions) and norm.is_finite() and norm<1)
        divisors=[]
        for k in range(1,period):
            if period%k:continue
            diff=[box[k*2*model.n+i]-box[i] for i in range(model.n)]
            divisors.append({'divisor':k,'differences':vector_json(diff),'excluded':any(v.is_finite() and not v.contains(0) for v in diff)})
        ac=quadratic_anchor(anchor,center,slope,curvature,nn,oo,sc,delta,radii,period,sigma,model.n)
        gates={'uniform_normalized_eigenpair_locus':root,'primitive_period':all(v['excluded'] for v in divisors),'anchor_identified':ac['passed']}
        endpoints=[]
        for sign in (-1,1):
            dd=sign*delta;re=column([coeff[i,0]+coeff[i,1]*dd+coeff[i,2]*(dd*dd)+whole_coeff[i,3]*(dd*dd*dd) for i in range(size)])
            ee=-pre*re+defect*error
            endpoints.append([center[i]+slope[i]*dd+curvature[i]*(dd*dd)/2+ee[i,0] for i in range(size)])
        result={'schema':'neural-shrimp-quadratic-eigenpair-locus-v1','passed':all(gates.values()),'gates':gates,'period':period,'dimension':model.n,'sigma':sigma,'shooting_dimension':size,
            'parameter_center':parameter,'parameter_endpoints':[str(Fraction(parameter)-Fraction(2)**halfwidth_exponent),str(Fraction(parameter)+Fraction(2)**halfwidth_exponent)],
            'normal':normal,'origin':origin,'center':vector_json(center),'slope':vector_json(slope),'curvature':vector_json(curvature),'preconditioner':matrix_json(pre),'condensed_border':matrix_json(small),
            'center_residual':matrix_json(g0),'center_jacobian':matrix_json(j0),'affine_path_coefficients':matrix_json(affine),'quadratic_center_coefficients':matrix_json(coeff),'quadratic_whole_coefficients':matrix_json(whole_coeff),'predictor_residual':matrix_json(rr),
            'whole_box':vector_json(box),'whole_jacobian':matrix_json(jb),'defect':matrix_json(defect),'weights':weights,'coordinate_radii':vector_json(radii),'image_correction':matrix_json(image),'contraction':ball_json(norm),'coordinate_inclusions':inclusions,'divisors':divisors,'anchor_check':ac,'endpoint_root_boxes':[vector_json(v) for v in endpoints],
            'settings':{'precision':precision,'newton_steps':newton_steps,'halfwidth_exponent':halfwidth_exponent,'radius_exponent':radius_exponent},
            'scope':'normalized primitive eigenpair locus only; no critical simplicity/genericity/attraction/projected-rate regularity or distant connection'}
        observe('chart_return',result)
        return result,{'center':center,'slope':slope,'curvature':curvature,'endpoints':endpoints}


def quadratic_two_charts(model,initial,**settings):
    first,data=quadratic_tube(model,initial,**settings);results=[first]
    if first['passed']:
        with ctx.workprec(settings.get('precision',256)):
            delta=Fraction(2)**settings.get('halfwidth_exponent',-12);dd=exact(str(2*delta))
            initial=[exact_mid(c+data['slope'][i]*dd+data['curvature'][i]*(dd*dd)/2) for i,c in enumerate(data['center'])]
            kw=dict(settings,parameter=str(Fraction(settings['parameter'])+2*delta),anchor={'box':data['endpoints'][1],'exact_parameter':str(Fraction(settings['parameter'])+delta),'period':settings['period'],'sigma':settings['sigma'],'dimension':model.n})
            second,_=quadratic_tube(model,initial,**kw)
        results.append(second)
    return {'schema':'neural-shrimp-two-quadratic-eigenpair-charts-v1','passed':len(results)==2 and all(v['passed'] for v in results),'charts':results}
