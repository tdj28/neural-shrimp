"""Independent saved-ball reductions for quadratic weighted shooting tubes.

No model, gradient, Hessian, jet acquisition or proposal solve is called.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json
import review_joined_period18_math as M


def original_ball(record):
    M.interval(record);mid=arb(tuple(record['mid']));m,e=record['rad']
    if m:
        M.need(m.bit_length()<=30,'saved magnitude precision')
        value=arb(mid,arb((m,e))-arb((1,e+m.bit_length()-32)))
    else:value=mid
    got=ball_json(value)
    M.need(got['mid']==record['mid']and got['rad']==record['rad'],'exact serialized ball reconstruction')
    return value

def matrix(rows):return arb_mat([[original_ball(v)for v in row]for row in rows])
def exact(q):
    q=Fraction(q);return arb(q.numerator)/q.denominator

def same(a,rows,why):
    M.need(a.nrows()==len(rows)and all(a.ncols()==len(r)for r in rows),why+' dimensions')
    for i,row in enumerate(rows):
        for j,v in enumerate(row):
            got=ball_json(a[i,j]);M.need(got['mid']==v['mid']and got['rad']==v['rad'],why+' exact ball identity')

def check_chart(chart,anchor,weighted,precision=384):
    p,n=chart['statement']['period'],chart['statement']['phase_dimension'];N=p*n;settings=chart['settings']
    M.need(N==chart['statement']['shooting_dimension']and len(chart['center'])==p,'native shooting dimension')
    hc=Fraction(chart['parameter_center']);delta=Fraction(2)**settings['halfwidth_exponent'];radius=Fraction(2)**settings['radius_exponent']
    M.need(list(map(Fraction,chart['exact_parameter_endpoints']))==[hc-delta,hc+delta],'exact chart parameter interval')
    lo,hi=M.interval(chart['parameter_interval']);M.need(lo<=hc-delta and hc+delta<=hi,'whole parameter domain')
    base=[M.point(v)for row in chart['center']for v in row];S=[M.point(row[0])for row in chart['predictor_slope']];T=[M.point(row[0])for row in chart['predictor_curvature']]
    weights=list(map(Fraction,chart['exact_weights']))if weighted else [Fraction(1)]*N
    M.need(len(weights)==len(base)==len(S)==len(T)==N and all(w>0 and w<=1 and w.denominator&(w.denominator-1)==0 for w in weights),'positive exact dyadic coordinate weights')
    radii=[radius*w for w in weights];M.need(M.point(chart['correction_radius'])==radius,'fixed overall radius')
    if weighted:M.need([M.point(v)for v in chart['coordinate_radii']]==radii,'all coordinate radii')
    for i in range(N):
        blo,bhi=M.interval(chart['phase_boxes'][i//n][i%n]);spread=abs(S[i])*delta
        # An independent simple exact range bound; no quadratic optimization.
        lower=base[i]-spread+min(0,T[i])*delta*delta/2-radii[i]
        upper=base[i]+spread+max(0,T[i])*delta*delta/2+radii[i]
        M.need(blo<=lower and upper<=bhi,'C1 domain covers full curved predictor plus error')
    M.need(len(chart['phase_jacobians'])==p,'all native derivatives')
    for row in chart['preconditioner']:
        for value in row:M.point(value)
    with ctx.workprec(precision):
        A=arb_mat(N,N)
        for j,records in enumerate(chart['phase_jacobians']):
            block=matrix(records);M.need(block.nrows()==block.ncols()==n,'native block dimensions')
            for a in range(n):
                for b in range(n):A[j*n+a,j*n+b]+=block[a,b]
                A[j*n+a,((j+1)%p)*n+a]-=1
        C=matrix(chart['preconditioner']);d=arb(0,exact(delta));rr=exact(radius);w=[exact(v)for v in weights]
        H={k:matrix(v)for k,v in chart['center_path_derivatives'].items()};third=matrix(chart['whole_path_third_derivative'])
        M.need(set(H)=={'f','d','d2','d3'}and all(v.nrows()==N and v.ncols()==1 for v in [*H.values(),third]),'all Taylor coefficient defects retained')
        square=d*d;cube=square*d
        R=H['f']+H['d']*d+H['d2']*(square/2)+third*(cube/6)
        D=M.ident(N)-C*A;E=arb_mat([[arb(0,rr*wi)]for wi in w]);forcing=-C*R;propagated=D*E;K=forcing+propagated
        contract=max((sum((abs(D[i,j]).upper()*w[j]for j in range(N)),arb(0))/w[i]).upper()for i in range(N))
        inclusion=[bool(abs(K[i,0]).upper()<rr*w[i])for i in range(N)]
        root=bool(contract.is_finite()and contract<1 and all(inclusion))
        if precision==256:
            same(A,chart['jacobian_tube'],'native shooting assembly');same(R,chart['predictor_residual'],'cubic Taylor residual');same(D,chart['defect'],'Krawczyk defect');same(K,chart['image_correction'],'Krawczyk correction image')
            M.need(inclusion==chart['coordinate_inclusions'],'all saved inclusion decisions')
            # Weighted norm includes explicit outward upper at each row.
            if weighted:M.need(ball_json(contract)['mid']==chart['contraction']['mid']and ball_json(contract)['rad']==chart['contraction']['rad'],'saved weighted contraction identity')
        primitive=[]
        M.need([x['divisor']for x in chart['divisors']]==[k for k in range(1,p)if p%k==0],'every proper divisor')
        for k in range(1,p):
            if p%k:continue
            # Exact serialized endpoint subtraction, not overlap with producer assertion.
            sep=[]
            for i in range(n):
                a,b=M.interval(chart['phase_boxes'][k][i]);c,e=M.interval(chart['phase_boxes'][0][i])
                if a-e>0 or b-c<0:sep.append(i)
            primitive.append({'divisor':k,'separating_coordinates':sep})
        inherited=True
        M.need(anchor is not None and len(anchor['phase_boxes'])==p and all(len(row)==n for row in anchor['phase_boxes']),'complete original/common-parameter anchor')
        at=Fraction(anchor['rate']);M.need(hc-delta<=at<=hc+delta,'common anchor parameter');dd=at-hc
        for i in range(N):
            pred=base[i]+S[i]*dd+T[i]*dd*dd/2;a,b=M.interval(anchor['phase_boxes'][i//n][i%n])
            inherited &= pred-radii[i]<=a<=b<=pred+radii[i]
        gates={'uniform_root':root,'primitive_period':all(x['separating_coordinates']for x in primitive),'anchor_identified':bool(inherited)}
        if precision==256:M.need(gates==chart['gates']and all(gates.values())==chart['passed'],'original mathematical gate decisions')
        endpoints=[]
        for sign in(-1,1):
            d0=sign*exact(delta);res=H['f']+H['d']*d0+H['d2']*(d0*d0/2)+third*(d0*d0*d0/6)
            image=-C*res+D*E
            values=[exact(base[i])+exact(S[i])*d0+exact(T[i])*d0*d0/2+image[i,0]for i in range(N)]
            saved=chart['endpoint_root_enclosures'][0 if sign<0 else 1]
            if precision==256:same(arb_mat([[v]for v in values]),[[v]for v in saved],'endpoint root enclosures')
            endpoints.append([ball_json(v)for v in values])
        return {'passed':all(gates.values()),'gates':gates,'precision':precision,'coordinate_count':N,'contraction_upper':ball_json(contract),'maximum_forcing_ratio':ball_json(max((abs(forcing[i,0]).upper()/(rr*w[i])).upper()for i in range(N))),'maximum_propagated_ratio':ball_json(max((abs(propagated[i,0]).upper()/(rr*w[i])).upper()for i in range(N))),'maximum_image_ratio':ball_json(max((abs(K[i,0]).upper()/(rr*w[i])).upper()for i in range(N))),'primitive':primitive,'recomputed_endpoints':endpoints}


def check_join(result,inputs,weighted,tick=lambda:None):
    charts=result['charts'];M.need(1<=len(charts)<=2,'bounded chart count');rows=[];anchor=inputs['anchor'];previous=None
    expected=Fraction(inputs['rate']);delta=Fraction(2)**charts[0]['settings']['halfwidth_exponent']
    for index,chart in enumerate(charts):
        M.need(Fraction(chart['parameter_center'])==expected+2*index*delta,'fixed adjacent chart center')
        if index:
            saved=charts[0]['endpoint_root_enclosures'][1];p,n=chart['statement']['period'],chart['statement']['phase_dimension']
            anchor={'rate':str(expected+delta),'phase_boxes':[saved[j*n:(j+1)*n]for j in range(p)]}
        original=check_chart(chart,anchor,weighted,256);tick()
        if index:
            p,n=chart['statement']['period'],chart['statement']['phase_dimension'];fresh=previous['recomputed_endpoints'][1]
            anchor={'rate':str(expected+delta),'phase_boxes':[fresh[j*n:(j+1)*n]for j in range(p)]}
        fresh=check_chart(chart,anchor,weighted,384);tick();rows.append({'chart':index,'original_precision':original,'independent_higher_precision':fresh});previous=fresh
    passed=len(rows)==2 and all(r['independent_higher_precision']['passed']for r in rows)
    if result['passed']:M.need(passed,'claimed complete joined primitive branch')
    if passed:M.need(result['connected_parameter_endpoints']==[str(expected-delta),str(expected+3*delta)],'complete closed connected interval')
    return {'passed':passed,'charts':rows,'exact_parameter_endpoints':result['connected_parameter_endpoints'],'scope':'saved Taylor/C1 enclosures independently reduced; no new derivative acquisition or attraction claim'}
