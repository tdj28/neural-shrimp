"""Native multiple-shooting continuation of an actual unit conjugate pair.

Parameter t=angle**2 regularizes the double +1 endpoint. Per-phase order is
(x,q,r), then two actual rates; q.q=1 and q.r=0. Every proof gate acts on all
3*n*p+2 variables. A proposed condensed inverse is not itself a proof.
"""
from fractions import Fraction
from math import factorial
from flint import arb, arb_mat, ctx
from .rigorous_eigenpair_locus import (require, finite, column, phase as one_direction,
                                      exact_mid)
from .rigorous_orbits import (exact, eye, midpoint_matrix, ball_json,
                             vector_json, matrix_json)


def entire_pair(t, derivative=0):
    """Validated c=cos(sqrt(t)), e=sin(sqrt(t))/sqrt(t), d=t*e.

    Use entire power series also at t=0 and negative t. On |t|<=1/4,
    derivatives through order two have tail <= twice the first omitted
    absolute term: the consecutive-term ratio is <1/2 from k=21 onward.
    All polynomial and tail arithmetic is outward rounded.
    """
    require(isinstance(t,arb) and t.is_finite() and abs(t).upper()<=exact('1/4'), 'bounded entire parameter')
    require(type(derivative)is int and 0<=derivative<=2, 'entire derivative order')
    degree=20;T=abs(t).upper()
    def series(offset,j):
        result=arb(0)
        for k in range(degree,j-1,-1):
            coefficient=exact(str(Fraction((-1)**k*factorial(k),factorial(k-j)*factorial(2*k+offset))))
            result=result*t+coefficient
        k=degree+1
        tail=2*exact(str(Fraction(factorial(k),factorial(k-j)*factorial(2*k+offset))))*T**(k-j)
        return result+arb(0,tail)
    c=series(0,derivative);e=series(1,derivative)
    d=t*e if derivative==0 else t*e+derivative*series(1,derivative-1)
    return c,e,d


def terminal_matrix(n,t,derivative=0):
    require(type(n)is int and 2<=n<=13,'phase dimension')
    c,e,d=entire_pair(t,derivative);m=3*n;out=arb_mat(m,m)
    for i in range(n):
        out[i,i]=int(derivative==0)
        out[n+i,n+i]=c;out[n+i,2*n+i]=-d
        out[2*n+i,n+i]=e;out[2*n+i,2*n+i]=c
    return out


def phase(model,state,q,r,rates,tick=lambda:None,observe=lambda k,v:None,index=None):
    """Two independently seeded MixedJet passes, each direction held fixed."""
    n=model.n;m=3*n
    require(len(state)==len(q)==len(r)==n and len(rates)==2,'native phase dimensions')
    pairs=[]
    for name,direction in [('q',q),('r',r)]:
        tick();value,block,rateblock=one_direction(model,state,direction,rates)
        observe('direction_return',{'phase':index,'direction':name,'input':vector_json(state+direction+rates),
                'value':matrix_json(value),'block':matrix_json(block),'rates':matrix_json(rateblock)})
        require(finite(value)and finite(block)and finite(rateblock),'finite native direction return')
        pairs.append((value,block,rateblock))
    (vq,tq,eq),(vr,tr,er)=pairs
    value=column([vq[i,0]for i in range(2*n)]+[vr[n+i,0]for i in range(n)])
    block=arb_mat(m,m);rateblock=arb_mat(m,2)
    for i in range(2*n):
        for j in range(2*n):block[i,j]=tq[i,j]
        for j in range(2):rateblock[i,j]=eq[i,j]
    for i in range(n):
        for j in range(n):
            block[2*n+i,j]=tr[n+i,j];block[2*n+i,2*n+j]=tr[n+i,n+j]
        for j in range(2):rateblock[2*n+i,j]=er[n+i,j]
    return value,block,rateblock


def assemble(model,flat,period,t,tick=lambda:None,observe=lambda k,v:None):
    """Full H, H_z, native blocks and H_t; no dropped state/mode/rate row."""
    n=model.n;m=3*n;size=m*period+2
    require(type(period)is int and 1<=period<=18 and 2<=n<=13,'native dimensions')
    require(len(flat)==size and all(v.is_finite()for v in flat),'full native box')
    closure=terminal_matrix(n,t);closure_t=terminal_matrix(n,t,1)
    value=arb_mat(size,1);jac=arb_mat(size,size);ht=arb_mat(size,1);local=[]
    for k in range(period):
        offset=k*m;following=((k+1)%period)*m
        v,T,E=phase(model,flat[offset:offset+n],flat[offset+n:offset+2*n],
                    flat[offset+2*n:offset+m],flat[-2:],tick,observe,k)
        closing=closure if k==period-1 else eye(m)
        for i in range(m):
            value[offset+i,0]=v[i,0]-sum((closing[i,j]*flat[following+j]for j in range(m)),arb(0))
            for j in range(m):
                jac[offset+i,offset+j]+=T[i,j];jac[offset+i,following+j]-=closing[i,j]
            for j in range(2):jac[offset+i,size-2+j]=E[i,j]
        local.append((T,E))
    qt=flat[n:2*n];rt=flat[2*n:3*n]
    value[size-2,0]=(sum((x*x for x in qt),arb(0))-1)/2
    value[size-1,0]=sum((x*y for x,y in zip(qt,rt)),arb(0))
    for i in range(n):
        jac[size-2,n+i]=qt[i];jac[size-1,n+i]=rt[i];jac[size-1,2*n+i]=qt[i]
    first=column(flat[:m]);end_t=-closure_t*first
    for i in range(m):ht[(period-1)*m+i,0]=end_t[i,0]
    require(finite(value)and finite(jac)and finite(ht),'finite full assembly')
    return value,jac,local,ht


def condensed_preconditioner(local,bottom_rows,terminal):
    """Exact-dyadic proposal by elimination to (3*n+2) rows.

    No native phase inverse. The complete interval I-C*H_z is subsequently
    checked, including all eliminated rows and the non-diagonal closure.
    """
    require(bool(local),'nonempty native blocks')
    p=len(local);m=local[0][0].nrows();size=m*p+2
    require(m%3==0 and bottom_rows.nrows()==2 and bottom_rows.ncols()==size and terminal.nrows()==terminal.ncols()==m,'condensed dimensions')
    ps,qs,ns=[eye(m)],[arb_mat(m,2)],[arb_mat(m,size)]
    for k,(tt,ee)in enumerate(local):
        require(tt.nrows()==tt.ncols()==m and ee.nrows()==m and ee.ncols()==2,'native block dimensions')
        t,e=midpoint_matrix(tt),midpoint_matrix(ee)
        pp=midpoint_matrix(t*ps[-1]);qq=midpoint_matrix(t*qs[-1]+e);nn=t*ns[-1]
        for i in range(m):nn[i,k*m+i]-=1
        ps.append(pp);qs.append(qq);ns.append(midpoint_matrix(nn))
    terminal_difference=ps[-1]-midpoint_matrix(terminal)
    small=arb_mat(m+2,m+2);rhs=arb_mat(m+2,size)
    for i in range(m):
        for j in range(m):small[i,j]=terminal_difference[i,j]
        for j in range(2):small[i,m+j]=qs[-1][i,j]
        for j in range(size):rhs[i,j]=-ns[-1][i,j]
    for r in range(2):
        for k in range(p):
            weights=arb_mat([[bottom_rows[r,k*m+j].mid()for j in range(m)]])
            a,b,c=weights*ps[k],weights*qs[k],weights*ns[k]
            for j in range(m):small[m+r,j]+=a[0,j]
            for j in range(2):small[m+r,m+j]+=b[0,j]
            for j in range(size):rhs[m+r,j]-=c[0,j]
        for j in range(2):small[m+r,m+j]+=bottom_rows[r,size-2+j].mid()
        rhs[m+r,size-2+r]+=1
    require(finite(small)and finite(rhs),'finite condensed proposal')
    solved=midpoint_matrix(small.solve(rhs))
    u0=arb_mat([[solved[i,j]for j in range(size)]for i in range(m)])
    rates=arb_mat([[solved[m+i,j]for j in range(size)]for i in range(2)])
    result=arb_mat(size,size)
    for k in range(p):
        part=midpoint_matrix(ps[k]*u0+qs[k]*rates+ns[k])
        for i in range(m):
            for j in range(size):result[k*m+i,j]=part[i,j]
    for i in range(2):
        for j in range(size):result[size-2+i,j]=rates[i,j]
    require(finite(result),'finite proposed preconditioner')
    return result,midpoint_matrix(small)


def proposal(jac,local,n,t):
    size=jac.nrows()
    rows=arb_mat([[jac[size-2+i,j]for j in range(size)]for i in range(2)])
    return condensed_preconditioner(local,rows,terminal_matrix(n,t))


def native_lift(model,box,period,tick=lambda:None,observe=lambda k,v:None):
    """Conditional lift of the same proved (x,rates,q,r) Jordan root at t=0."""
    n=model.n;require(len(box)==3*n+2 and type(period)is int and 1<=period<=18,'lift dimensions')
    state=list(box[:n]);rates=list(box[n:n+2]);q=list(box[n+2:2*n+2]);r=list(box[2*n+2:]);flat=[]
    for k in range(period):
        flat.extend(state+q+r);v,_,_=phase(model,state,q,r,rates,tick,observe,k)
        state=[v[i,0]for i in range(n)];q=[v[n+i,0]for i in range(n)];r=[v[2*n+i,0]for i in range(n)]
    flat.extend(rates)
    difference=column(state+q+r)-terminal_matrix(n,arb(0))*column(flat[:3*n])
    return flat,{'dimension':n,'period':period,'shooting_dimension':len(flat),'parameter':'0',
        'box':vector_json(flat),'closure_residual':matrix_json(difference),
        'closure_contains_zero':all(difference[i,0].contains(0)for i in range(3*n)),
        'scope':'Conditional on the authenticated same-model normalized exact Jordan root. Zero containment is not a root proof.'}


def anchor_test(anchor,center,slope,tc,delta,radii,n,period):
    if anchor is None:return {'passed':False,'reason':'prior exact root not supplied'}
    size=len(center)
    require(anchor['dimension']==n and anchor['period']==period and len(anchor['box'])==size,'same native anchor convention')
    ta=exact(anchor['parameter']);require(ta.is_finite()and ta.is_exact(),'exact anchor parameter')
    box=anchor['box'];require(all(v.is_finite()for v in box),'finite full anchor')
    d=ta-tc;inside=bool(abs(d).upper()<=delta)
    checks=[bool(abs(box[i]-center[i]-slope[i,0]*d).upper()<radii[i])for i in range(size)]
    return {'passed':inside and all(checks),'parameter_inside':inside,'parameter':ball_json(ta),'coordinate_inclusions':checks,
            'scope':'Conditional on externally established same-model full native root provenance.'}


def unit_pair_tube(model,initial,*,period,parameter,halfwidth_exponent=-12,
                   radius_exponent=-16,weights=None,newton_steps=2,precision=256,
                   anchor=None,tick=lambda:None,observe=lambda k,v:None):
    """Uniform affine C1 Krawczyk tube; complementary spectrum is separate.

    H(P(t),t) lies in H(U,tc)+(H_z(P(I),I)*S+H_t(P(I),I))*(t-tc).
    All derivatives cover the entire predictor/box, including true t dependence.
    """
    require(type(precision)is int and 128<=precision<=512,'precision cap')
    require(type(newton_steps)is int and 0<=newton_steps<=2,'proposal cap')
    require(all(type(v)is int and -100<=v<=-1 for v in(halfwidth_exponent,radius_exponent)),'width caps')
    n=model.n;size=3*n*period+2
    require(len(initial)==size,'chart dimensions')
    weights=['1']*size if weights is None else weights
    require(len(weights)==size,'full coordinate weights')
    with ctx.workprec(precision):
        center=[exact(v)for v in initial];w=[exact(v)for v in weights];tc=exact(parameter)
        require(all(v.is_finite()and v.is_exact()for v in center+w+[tc]),'exact finite chart data')
        require(all(v>0 and v<=1 for v in w),'positive bounded weights')
        delta=arb((1,halfwidth_exponent));radius=arb((1,radius_exponent));d=arb(0,delta);ti=tc+d
        # The exact advertised interval is nonnegative. Its Arb evaluation
        # enclosure can extend slightly below zero; the entire kernel covers it.
        require(tc-delta>=0 and tc+delta<=exact('1/8'),'nonnegative exact chart parameter')
        for iteration in range(newton_steps):
            g,j,local,_=assemble(model,center,period,tc,tick,observe);pre,small=proposal(j,local,n,tc)
            correction=pre*g;center=[(center[i]-correction[i,0]).mid()for i in range(size)]
            observe('newton_proposal',{'iteration':iteration,'center':vector_json(center),'residual':matrix_json(g),'correction':matrix_json(correction)})
        g0,j0,local,ht0=assemble(model,center,period,tc,tick,observe)
        tick();pre,small=proposal(j0,local,n,tc);slope=midpoint_matrix(-pre*ht0)
        observe('preconditioner_return',{'matrix':matrix_json(pre),'condensed_border':matrix_json(small)})
        predictor=[center[i]+slope[i,0]*d for i in range(size)]
        _,jp,_,htp=assemble(model,predictor,period,ti,tick,observe)
        path_derivative=jp*slope+htp;rr=g0+path_derivative*d
        radii=[radius*v for v in w];error=column([arb(0,r)for r in radii]);box=[predictor[i]+error[i,0]for i in range(size)]
        require(all(v.is_finite()and v>0 for v in box[-2:]),'whole actual rates positive')
        _,jb,localb,htb=assemble(model,box,period,ti,tick,observe)
        defect=eye(size)-pre*jb
        bound=max((sum((abs(defect[i,j]).upper()*w[j]for j in range(size)),arb(0))/w[i]).upper()for i in range(size))
        image=-pre*rr+defect*error
        inclusions=[bool(abs(image[i,0]).upper()<radii[i])for i in range(size)]
        root=bool(all(inclusions)and bound.is_finite()and bound<1)
        divisors=[]
        for k in range(1,period):
            if period%k:continue
            diff=[box[k*3*n+i]-box[i]for i in range(n)]
            divisors.append({'divisor':k,'differences':vector_json(diff),'excluded':any(v.is_finite()and not v.contains(0)for v in diff)})
        rbox=box[2*n:3*n];r_nonzero=any(not v.contains(0)for v in rbox)
        ac=anchor_test(anchor,center,slope,tc,delta,radii,n,period)
        gates={'uniform_normalized_unit_pair':root,'primitive_period':all(v['excluded']for v in divisors),
               'realified_frame_independent':r_nonzero,'anchor_identified':ac['passed']}
        endpoint=[]
        for sign in(-1,1):
            dd=sign*delta;ee=-pre*(g0+path_derivative*dd)+defect*error
            endpoint.append([center[i]+slope[i,0]*dd+ee[i,0]for i in range(size)])
        result={'schema':'neural-shrimp-native-unit-pair-tube-v1','passed':all(gates.values()),'gates':gates,
            'dimension':n,'period':period,'shooting_dimension':size,'parameter_center':parameter,
            'parameter_endpoints':[str(Fraction(parameter)-Fraction(2)**halfwidth_exponent),str(Fraction(parameter)+Fraction(2)**halfwidth_exponent)],
            'center':vector_json(center),'slope':matrix_json(slope),'preconditioner':matrix_json(pre),'condensed_border':matrix_json(small),
            'center_residual':matrix_json(g0),'predictor_jacobian':matrix_json(jp),'predictor_parameter_derivative':matrix_json(htp),
            'path_derivative':matrix_json(path_derivative),'predictor_residual':matrix_json(rr),'whole_box':vector_json(box),
            'whole_jacobian':matrix_json(jb),'whole_parameter_derivative':matrix_json(htb),'defect':matrix_json(defect),
            'weights':weights,'coordinate_radii':vector_json(radii),'image_correction':matrix_json(image),'contraction':ball_json(bound),
            'coordinate_inclusions':inclusions,'divisors':divisors,'anchor_check':ac,'endpoint_root_boxes':[vector_json(v)for v in endpoint],
            'settings':{'precision':precision,'newton_steps':newton_steps,'halfwidth_exponent':halfwidth_exponent,'radius_exponent':radius_exponent},
            'statement':'Conditional connected primitive orbit branch carrying a normalized realified unit pair. For t>0 the eigenvalues are exp(±i sqrt(t)); at t=0 a length-two +1 Jordan chain. Complement stability, criticality, stability sides and whole-window boundary topology require separate gates.'}
        observe('chart_return',result)
        return result,{'center':center,'slope':slope,'endpoints':endpoint,'whole_native_blocks':localb}
