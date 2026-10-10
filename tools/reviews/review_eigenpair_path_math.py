"""Independent reductions and native assembly checks of saved eigenpair tubes.

No producer proof, derivative, proposal or model call. Analytic native blocks
and path jets remain shared inputs; Arb is a shared arithmetic dependency.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json,vector_json
import review_joined_period18_math as M
import review_quadratic_connection_math as R
need=M.need

def vector(rows):return [R.original_ball(v) for v in rows]
def identity(a,b,why):
    need(len(a)==len(b),why+' shape')
    R.same(arb_mat([[v] for v in a]),[[v] for v in b],why)

def rebuild_whole(chart,events):
    """Independently place retained local derivative blocks in native order."""
    p,n=chart['period'],chart['dimension'];m=2*n;N=m*p+2
    need(len(events)==p,'all whole-domain native returns')
    box=chart['whole_box'];A=arb_mat(N,N)
    for k,e in enumerate(events):
        need(e['phase']==k,'phase order')
        need(e['input']==box[k*m:(k+1)*m]+box[-2:],'entire exact derivative domain')
        B,T=R.matrix(e['block']),R.matrix(e['rates'])
        need(B.nrows()==B.ncols()==m and T.nrows()==m and T.ncols()==2,'local derivative dimensions')
        for i in range(m):
            for j in range(m):A[k*m+i,k*m+j]+=B[i,j]
            A[k*m+i,((k+1)%p)*m+i]-=chart['sigma'] if k==p-1 and i>=n else 1
            for j in range(2):A[k*m+i,N-2+j]=T[i,j]
    for i in range(n):A[N-2,n+i]=R.original_ball(box[n+i])
    for i in range(N):A[N-1,i]=R.exact(chart['normal'][i])
    return A

def path_inputs(chart,d):
    c,s,t=map(vector,(chart['center'],chart['slope'],chart['curvature']))
    return [[c[i]+s[i]*d+t[i]*(d*d)/2,s[i]+t[i]*d,t[i]/2,arb(0)] for i in range(len(c))]

def rebuild_coefficients(chart,events,d,curvature=True):
    p,n=chart['period'],chart['dimension'];m=2*n;N=m*p+2
    need(len(events)==p,'all path phase returns')
    cc=chart if curvature else dict(chart,curvature=[ball_json(arb(0))]*N)
    z=path_inputs(cc,d);rows=[]
    for k,e in enumerate(events):
        need(e['phase']==k and M.interval(e['d'])==M.interval(ball_json(d)),'path coordinate and phase')
        before=k*m;after=((k+1)%p)*m
        expected=[[vector_json(z[before+i]),vector_json(z[before+n+i])] for i in range(n)]+[[vector_json(v),vector_json([arb(0)]*4)] for v in z[-2:]]
        need(e['inputs']==expected,'whole native path input coefficients')
        returns=e['returns'];need(len(returns)==n,'path return shape')
        for part in (0,1):
            sign=chart['sigma'] if part and k==p-1 else 1
            for i in range(n):
                vals=vector(returns[i][part]);rows.append([vals[j]-(sign*z[after+part*n+i][j] if part else z[after+i][j]) for j in range(4)])
    norm=[(sum((sum((z[n+i][a]*z[n+i][j-a] for a in range(j+1)),arb(0)) for i in range(n)),arb(0))-arb(int(j==0)))/2 for j in range(4)]
    normal=[R.exact(v) for v in chart['normal']];origin=[R.exact(v) for v in chart['origin']]
    gauge=[sum((normal[i]*(z[i][j]-(origin[i] if j==0 else 0)) for i in range(N)),arb(0))+(-1*(R.exact(chart['parameter_center'])+d) if j==0 else -int(j==1)) for j in range(4)]
    return arb_mat(rows+[norm,gauge])

def reduction(C,A,H,W,weights,radius,delta):
    N=C.nrows();d=arb(0,R.exact(delta));rho=R.exact(radius);w=[R.exact(v) for v in weights]
    E=arb_mat([[arb(0,rho*v)] for v in w]);D=M.ident(N)-C*A
    residual=arb_mat([[H[i,0]+H[i,1]*d+H[i,2]*(d*d)+W[i,3]*(d*d*d)] for i in range(N)])
    K=-C*residual+D*E
    norm=max((sum((abs(D[i,j]).upper()*w[j] for j in range(N)),arb(0))/w[i]).upper() for i in range(N))
    inclusions=[bool(abs(K[i,0]).upper()<rho*w[i]) for i in range(N)]
    return {'root':bool(norm.is_finite() and norm<1 and all(inclusions)),'norm':norm,'inclusions':inclusions,'D':D,'E':E,'K':K,'residual':residual}

def check_chart(chart,anchor,events,precision,expected_settings=None):
    p,n=chart['period'],chart['dimension'];N=2*n*p+2
    need(p==18 and n==13 and chart['sigma']==1 and chart['shooting_dimension']==N==470,'native convention')
    need(chart['schema']=='neural-shrimp-quadratic-eigenpair-locus-v1','quadratic chart schema')
    settings=chart['settings'];need(settings==(expected_settings or {'precision':256,'newton_steps':2,'halfwidth_exponent':-15,'radius_exponent':-18}),'explicit fixed settings')
    center=Fraction(chart['parameter_center']);delta=Fraction(2)**settings['halfwidth_exponent'];radius=Fraction(2)**settings['radius_exponent']
    need(list(map(Fraction,chart['parameter_endpoints']))==[center-delta,center+delta],'parameter endpoints')
    c,s,t=[[M.point(v) for v in chart[k]] for k in ('center','slope','curvature')]
    weights=list(map(Fraction,chart['weights']));radii=[radius*w for w in weights]
    need(len(weights)==len(c)==len(s)==len(t)==len(anchor['box'])==N and all(0<w<=1 and w.denominator&(w.denominator-1)==0 for w in weights),'all positive dyadic weights')
    need([M.point(v) for v in chart['coordinate_radii']]==radii,'exact radii')
    for row in chart['preconditioner']:
        for v in row:M.point(v)
    for i,b in enumerate(chart['whole_box']):
        lo,hi=M.interval(b);a=abs(s[i])*delta
        need(lo<=c[i]-a+min(0,t[i])*delta*delta/2-radii[i] and hi>=c[i]+a+max(0,t[i])*delta*delta/2+radii[i],'full predictor/error domain')
    need(all(M.interval(v)[0]>0 for v in chart['whole_box'][-2:]),'positive rates')
    phases=[v for kind,v in events if kind=='phase_return'];paths=[v for kind,v in events if kind=='path_phase_return'];assemblies=[v for kind,v in events if kind=='path_assembly_return']
    need(len(phases)==4*p and len(paths)==3*p and len(assemblies)==3,'complete 4C1/3jet phase inventory')
    with ctx.workprec(256):
        A=rebuild_whole(chart,phases[-p:]);R.same(A,chart['whole_jacobian'],'whole native block assembly')
        for i,key in enumerate(('affine_path_coefficients','quadratic_center_coefficients','quadratic_whole_coefficients')):
            d=arb(0,R.exact(delta)) if i==2 else arb(0)
            coefficients=rebuild_coefficients(chart,paths[i*p:(i+1)*p],d,i!=0)
            R.same(coefficients,assemblies[i]['coefficients'],'retained path assembly')
            R.same(coefficients,chart[key],'chart coefficient binding')
    with ctx.workprec(precision):
        A=rebuild_whole(chart,phases[-p:]);C=R.matrix(chart['preconditioner']);H=R.matrix(chart['quadratic_center_coefficients']);W=R.matrix(chart['quadratic_whole_coefficients'])
        r=reduction(C,A,H,W,weights,radius,delta)
        if precision==256:
            for value,key in [(r['residual'],'predictor_residual'),(r['D'],'defect'),(r['K'],'image_correction')]:R.same(value,chart[key],'original '+key)
            identity([r['norm']],[chart['contraction']],'weighted norm')
            need(r['inclusions']==chart['coordinate_inclusions'],'all inclusion decisions')
        proper=[k for k in range(1,p) if p%k==0];need([v['divisor'] for v in chart['divisors']]==proper,'all divisors')
        primitive=[]
        for k in proper:
            coords=[]
            for i in range(n):
                a,b=M.interval(chart['whole_box'][k*2*n+i]);x,y=M.interval(chart['whole_box'][i])
                if a-y>0 or b-x<0:coords.append(i)
            primitive.append({'divisor':k,'separating_coordinates':coords})
        parameter=anchor_parameter(chart,anchor);dd=parameter-R.exact(center);need(abs(dd).upper()<=R.exact(delta),'whole shared gauge enclosure')
        join=[bool(abs(R.original_ball(anchor['box'][i])-R.exact(c[i])-R.exact(s[i])*dd-R.exact(t[i])*(dd*dd)/2).upper()<R.exact(radii[i])) for i in range(N)]
        gates={'uniform_normalized_eigenpair_locus':r['root'],'primitive_period':all(v['separating_coordinates'] for v in primitive),'anchor_identified':all(join)}
        if precision==256:need(gates==chart['gates'] and all(gates.values())==chart['passed'],'original gates')
        ends=[]
        for sign in (-1,1):
            dd=R.exact(sign*delta);rr=arb_mat([[H[i,0]+H[i,1]*dd+H[i,2]*(dd*dd)+W[i,3]*(dd*dd*dd)] for i in range(N)])
            k=-C*rr+r['D']*r['E'];endpoint=[R.exact(c[i])+R.exact(s[i])*dd+R.exact(t[i])*(dd*dd)/2+k[i,0] for i in range(N)]
            if precision==256:identity(endpoint,chart['endpoint_root_boxes'][0 if sign<0 else 1],'endpoint root enclosure')
            ends.append(vector_json(endpoint))
        return {'passed':all(gates.values()),'gates':gates,'precision':precision,'contraction':ball_json(r['norm']),'inclusions':sum(r['inclusions']),'anchor_coordinates':sum(join),'primitive':primitive,'recomputed_endpoints':ends}


def anchor_parameter(chart,anchor):
    """Known-root box implies a correlated gauge enclosure, never its midpoint."""
    need(len(chart['normal'])==len(chart['origin'])==len(anchor['box']),'full anchor projection')
    if 'exact_parameter' in anchor:return R.exact(Fraction(anchor['exact_parameter']))
    return sum((R.exact(a)*(R.original_ball(b)-R.exact(o)) for a,b,o in zip(chart['normal'],anchor['box'],chart['origin'],strict=True)),arb(0))
