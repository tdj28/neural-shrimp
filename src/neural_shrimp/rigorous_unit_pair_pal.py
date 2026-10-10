"""Full native unit-pair pseudo-arclength curves, including squared angle.

Per phase (x,q,r), then two physical rates and t=angle**2. The fixed gauge
adds one equation, giving 3*n*p+3 coordinates. Condensation only proposes a
finite preconditioner; every proof gate retains the complete system.
"""
from fractions import Fraction
from math import factorial
from flint import arb, arb_mat, ctx
from .rigorous_unit_pair import assemble as fixed_angle_assemble, terminal_matrix
from .rigorous_eigenpair_path import PathJet
from .rigorous_eigenpair_locus import require, finite, column
from .rigorous_orbits import exact, eye, midpoint_matrix, vector_json, matrix_json, ball_json


def assemble(model, flat, period, normal, origin, parameter, tick=lambda:None,
             observe=lambda k,v:None):
    size=3*model.n*period+3
    require(len(flat)==len(normal)==len(origin)==size,'full PAL dimensions')
    value,jac,blocks,ht=fixed_angle_assemble(model,flat[:-1],period,flat[-1],tick,observe)
    out=arb_mat(size,1);full=arb_mat(size,size);local=[];m=3*model.n
    for i in range(size-1):
        out[i,0]=value[i,0]
        for j in range(size-1):full[i,j]=jac[i,j]
        full[i,size-1]=ht[i,0]
    out[size-1,0]=sum((normal[j]*(flat[j]-origin[j])for j in range(size)),arb(0))-parameter
    for j in range(size):full[size-1,j]=normal[j]
    for k,(T,_)in enumerate(blocks):
        local.append((T,arb_mat([[full[k*m+i,size-3+j]for j in range(3)]for i in range(m)])))
    require(finite(out)and finite(full),'finite full PAL assembly')
    return out,full,local


def condensed_preconditioner(local,bottom,terminal):
    """Eliminate native rows into the three-global-variable border.

    t is treated as a global variable: its native column includes the
    derivative of the final closure. No phase block is inverted.
    """
    require(bool(local),'nonempty phases')
    p=len(local);m=local[0][0].nrows();size=m*p+3
    require(m%3==0 and bottom.nrows()==3 and bottom.ncols()==size and terminal.nrows()==terminal.ncols()==m,'PAL border dimensions')
    ps,qs,ns=[eye(m)],[arb_mat(m,3)],[arb_mat(m,size)]
    for k,(tt,ee)in enumerate(local):
        require(tt.nrows()==tt.ncols()==m and ee.nrows()==m and ee.ncols()==3,'PAL phase block dimensions')
        t,e=midpoint_matrix(tt),midpoint_matrix(ee)
        pp=midpoint_matrix(t*ps[-1]);qq=midpoint_matrix(t*qs[-1]+e);nn=t*ns[-1]
        for i in range(m):nn[i,k*m+i]-=1
        ps.append(pp);qs.append(qq);ns.append(midpoint_matrix(nn))
    difference=ps[-1]-midpoint_matrix(terminal);small=arb_mat(m+3,m+3);rhs=arb_mat(m+3,size)
    for i in range(m):
        for j in range(m):small[i,j]=difference[i,j]
        for j in range(3):small[i,m+j]=qs[-1][i,j]
        for j in range(size):rhs[i,j]=-ns[-1][i,j]
    for r in range(3):
        for k in range(p):
            weights=arb_mat([[bottom[r,k*m+j].mid()for j in range(m)]])
            a,b,c=weights*ps[k],weights*qs[k],weights*ns[k]
            for j in range(m):small[m+r,j]+=a[0,j]
            for j in range(3):small[m+r,m+j]+=b[0,j]
            for j in range(size):rhs[m+r,j]-=c[0,j]
        for j in range(3):small[m+r,m+j]+=bottom[r,size-3+j].mid()
        rhs[m+r,size-3+r]+=1
    solved=midpoint_matrix(small.solve(rhs));u0=arb_mat([[solved[i,j]for j in range(size)]for i in range(m)])
    globals_=arb_mat([[solved[m+i,j]for j in range(size)]for i in range(3)]);result=arb_mat(size,size)
    for k in range(p):
        part=midpoint_matrix(ps[k]*u0+qs[k]*globals_+ns[k])
        for i in range(m):
            for j in range(size):result[k*m+i,j]=part[i,j]
    for i in range(3):
        for j in range(size):result[size-3+i,j]=globals_[i,j]
    require(finite(result),'finite full PAL preconditioner proposal')
    return result,midpoint_matrix(small)


def proposal(jac,local,n,t):
    size=jac.nrows();bottom=arb_mat([[jac[size-3+i,j]for j in range(size)]for i in range(3)])
    return condensed_preconditioner(local,bottom,terminal_matrix(n,t))


def entire_path(t):
    """c(t),e(t),d(t) through path order3, including interval derivative tails.

    Taylor-compose at the possibly interval t.a[0]. For scalar derivatives
    j<=3 of the entire series on |t|<=1/4, the ratio of successive absolute
    tail terms after k=21 is <1/2; twice the first omitted term encloses it.
    No mode derivative enters t.
    """
    require(isinstance(t,PathJet)and all(v.is_zero()for v in t.b),'angle has no mode direction')
    require(t.a[0].is_finite()and abs(t.a[0]).upper()<=exact('1/4'),'bounded path angle')
    T=abs(t.a[0]).upper();v=PathJet([0,*t.a[1:]]);powers=[PathJet.constant(1),v,v*v,v*v*v]
    def derivative(offset,j):
        result=arb(0)
        for k in range(20,j-1,-1):
            a=Fraction((-1)**k*factorial(k),factorial(k-j)*factorial(2*k+offset))
            result=result*t.a[0]+exact(str(a))
        tail=2*exact(str(Fraction(factorial(21),factorial(21-j)*factorial(42+offset))))*T**(21-j)
        return result+arb(0,tail)
    def compose(offset):
        return sum((powers[j]*derivative(offset,j)/factorial(j)for j in range(4)),PathJet.constant(0))
    c,e=compose(0),compose(1)
    return c,e,t*e


def path_coefficients(model,center,slope,curvature,d,period,normal,origin,parameter,
                      tick=lambda:None,observe=lambda k,v:None):
    n=model.n;m=3*n;size=m*period+3
    require(len(center)==len(slope)==len(curvature)==len(normal)==len(origin)==size,'full PAL path dimensions')
    require(all(v.is_finite()for v in center+slope+curvature+[d,parameter]),'finite PAL path')
    z=[PathJet.path(center[i]+slope[i]*d+curvature[i]*d*d/2,slope[i]+curvature[i]*d,curvature[i])for i in range(size)]
    require(z[-3].a[0]>0 and z[-2].a[0]>0,'positive predictor rates')
    c,e,dd=entire_path(z[-1]);rows=[]
    for k in range(period):
        offset=k*m;after=((k+1)%period)*m;returns=[]
        for name,index in [('q',n),('r',2*n)]:
            x=[PathJet(z[offset+i].a,z[offset+index+i].a)for i in range(n)]
            tick();ret=model.cusp_step(x,*z[-3:-1])
            observe('pal_path_phase_return',{'phase':k,'direction':name,'d':ball_json(d),'inputs':[v.serialized()for v in x+z[-3:-1]],'returns':[v.serialized()for v in ret]})
            require(len(ret)==n and all(isinstance(v,PathJet)and all(a.is_finite()for a in v.a+v.b)for v in ret),'finite PAL path return')
            returns.append(ret)
        first,second=returns
        rows.extend([[first[i].a[j]-z[after+i].a[j]for j in range(4)]for i in range(n)])
        for group,ret in [(n,first),(2*n,second)]:
            for i in range(n):
                close=z[after+group+i]
                if k==period-1:
                    close=c*z[n+i]-dd*z[2*n+i]if group==n else e*z[n+i]+c*z[2*n+i]
                rows.append([ret[i].b[j]-close.a[j]for j in range(4)])
    norm=(sum((v*v for v in z[n:2*n]),PathJet.constant(0))-1)/2
    orth=sum((z[n+i]*z[2*n+i]for i in range(n)),PathJet.constant(0))
    gauge=sum((normal[i]*(z[i]-origin[i])for i in range(size)),PathJet.constant(0))-PathJet([parameter+d,1,0,0])
    rows.extend([list(norm.a),list(orth.a),list(gauge.a)])
    out=arb_mat(rows);observe('pal_path_assembly_return',{'d':ball_json(d),'coefficients':matrix_json(out)})
    require(finite(out),'finite PAL path coefficients');return out


def anchor_test(anchor,center,slope,curvature,normal,origin,sc,delta,radii,n,period):
    if anchor is None:return {'passed':False,'reason':'prior root missing'}
    size=len(center)
    require(anchor['dimension']==n and anchor['period']==period and len(anchor['box'])==size,'full native PAL anchor')
    box=anchor['box'];require(all(v.is_finite()for v in box),'finite anchor box')
    projection=sum((normal[i]*(box[i]-origin[i])for i in range(size)),arb(0))
    parameter=exact(anchor['exact_parameter'])if 'exact_parameter'in anchor else projection
    if 'exact_parameter'in anchor:require(parameter.is_finite()and parameter.is_exact(),'exact prior gauge parameter')
    d=parameter-sc;inside=bool(abs(d).upper()<=delta)
    checks=[bool(abs(box[i]-center[i]-slope[i]*d-curvature[i]*d*d/2).upper()<radii[i])for i in range(size)]
    return {'passed':inside and all(checks),'parameter_inside':inside,'projection':ball_json(projection),'parameter':ball_json(parameter),'coordinate_inclusions':checks,'scope':'Conditional on independently established same-model native full-root provenance and matching gauge if exact_parameter is supplied.'}


def quadratic_tube(model,initial,*,period,normal,origin,parameter,halfwidth_exponent=-12,
                   radius_exponent=-16,weights=None,newton_steps=2,precision=256,
                   anchor=None,tick=lambda:None,observe=lambda k,v:None):
    require(type(precision)is int and 128<=precision<=512,'precision cap')
    require(type(newton_steps)is int and 0<=newton_steps<=2,'proposal cap')
    require(all(type(v)is int and -100<=v<=-1 for v in(halfwidth_exponent,radius_exponent)),'width caps')
    size=3*model.n*period+3;require(len(initial)==len(normal)==len(origin)==size,'full PAL chart dimensions')
    weights=['1']*size if weights is None else weights;require(len(weights)==size,'full coordinate weights')
    with ctx.workprec(precision):
        center=[exact(v)for v in initial];nn=[exact(v)for v in normal];oo=[exact(v)for v in origin];w=[exact(v)for v in weights];sc=exact(parameter)
        require(all(v.is_finite()and v.is_exact()for v in center+nn+oo+w+[sc]),'exact finite chart data')
        require(any(v!=0 for v in nn)and all(0<v<=1 for v in w),'nonzero gauge and positive bounded weights')
        delta=arb((1,halfwidth_exponent));d=arb(0,delta);radius=arb((1,radius_exponent));rhs=column([arb(0)]*(size-1)+[arb(1)])
        for it in range(newton_steps):
            g,j,local=assemble(model,center,period,nn,oo,sc,tick,observe);pre,small=proposal(j,local,model.n,center[-1]);correction=pre*g
            center=[(center[i]-correction[i,0]).mid()for i in range(size)]
            observe('pal_newton_proposal',{'iteration':it,'center':vector_json(center),'value':matrix_json(g),'jacobian':matrix_json(j),'correction':matrix_json(correction)})
        g0,j0,local=assemble(model,center,period,nn,oo,sc,tick,observe);tick();pre,small=proposal(j0,local,model.n,center[-1]);slope_m=midpoint_matrix(pre*rhs);slope=[slope_m[i,0]for i in range(size)]
        observe('pal_preconditioner_return',{'matrix':matrix_json(pre),'condensed_border':matrix_json(small)})
        affine=path_coefficients(model,center,slope,[arb(0)]*size,arb(0),period,nn,oo,sc,tick,observe)
        curvature_m=midpoint_matrix(-pre*column([2*affine[i,2]for i in range(size)]));curvature=[curvature_m[i,0]for i in range(size)]
        coeff=path_coefficients(model,center,slope,curvature,arb(0),period,nn,oo,sc,tick,observe)
        whole_coeff=path_coefficients(model,center,slope,curvature,d,period,nn,oo,sc,tick,observe)
        residual=column([coeff[i,0]+coeff[i,1]*d+coeff[i,2]*d*d+whole_coeff[i,3]*d*d*d for i in range(size)])
        predictor=[center[i]+slope[i]*d+curvature[i]*d*d/2 for i in range(size)]
        radii=[radius*v for v in w];error=column([arb(0,r)for r in radii]);box=[predictor[i]+error[i,0]for i in range(size)]
        require(box[-3]>0 and box[-2]>0 and abs(box[-1]).upper()<=exact('1/4'),'whole physical rates and entire angle domain')
        _,jb,localb=assemble(model,box,period,nn,oo,sc+d,tick,observe);defect=eye(size)-pre*jb
        bound=max((sum((abs(defect[i,j]).upper()*w[j]for j in range(size)),arb(0))/w[i]).upper()for i in range(size))
        image=-pre*residual+defect*error;inc=[bool(abs(image[i,0]).upper()<radii[i])for i in range(size)]
        root=bool(all(inc)and bound.is_finite()and bound<1);divisors=[]
        for k in range(1,period):
            if period%k:continue
            diff=[box[k*3*model.n+i]-box[i]for i in range(model.n)]
            divisors.append({'divisor':k,'differences':vector_json(diff),'excluded':any(not v.contains(0)for v in diff)})
        ac=anchor_test(anchor,center,slope,curvature,nn,oo,sc,delta,radii,model.n,period)
        gates={'uniform_full_native_unit_pair_locus':root,'primitive_period':all(v['excluded']for v in divisors),'realified_frame_independent':any(not v.contains(0)for v in box[2*model.n:3*model.n]),'anchor_identified':ac['passed']}
        endpoints=[]
        for sign in(-1,1):
            dd=sign*delta;rr=column([coeff[i,0]+coeff[i,1]*dd+coeff[i,2]*dd*dd+whole_coeff[i,3]*dd*dd*dd for i in range(size)])
            ee=-pre*rr+defect*error;endpoints.append([center[i]+slope[i]*dd+curvature[i]*dd*dd/2+ee[i,0]for i in range(size)])
        result={'schema':'neural-shrimp-native-unit-pair-pal-v1','passed':all(gates.values()),'gates':gates,'dimension':model.n,'period':period,'shooting_dimension':size,'parameter_center':parameter,'parameter_endpoints':[str(Fraction(parameter)-Fraction(2)**halfwidth_exponent),str(Fraction(parameter)+Fraction(2)**halfwidth_exponent)],'normal':normal,'origin':origin,'center':vector_json(center),'slope':vector_json(slope),'curvature':vector_json(curvature),'preconditioner':matrix_json(pre),'condensed_border':matrix_json(small),'center_residual':matrix_json(g0),'center_jacobian':matrix_json(j0),'affine_path_coefficients':matrix_json(affine),'quadratic_center_coefficients':matrix_json(coeff),'quadratic_whole_coefficients':matrix_json(whole_coeff),'predictor_residual':matrix_json(residual),'whole_box':vector_json(box),'whole_jacobian':matrix_json(jb),'defect':matrix_json(defect),'weights':weights,'coordinate_radii':vector_json(radii),'image_correction':matrix_json(image),'coordinate_inclusions':inc,'contraction':ball_json(bound),'divisors':divisors,'anchor_check':ac,'endpoint_root_boxes':[vector_json(v)for v in endpoints],'settings':{'precision':precision,'newton_steps':newton_steps,'halfwidth_exponent':halfwidth_exponent,'radius_exponent':radius_exponent},'scope':'Conditional connected primitive native normalized unit-pair locus; when t>0 its pair is exp(±i sqrt(t)). Whole-domain positivity of t, complementary spectrum, normal-form criticality, stability sides and whole-window topology require separate qualification.'}
        observe('pal_chart_return',result)
        return result,{'center':center,'slope':slope,'curvature':curvature,'endpoints':endpoints,'whole_native_blocks':localb}
