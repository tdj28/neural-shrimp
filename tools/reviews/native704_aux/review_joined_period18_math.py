"""Independent saved-ball reductions for phase-labelled shooting chart joins.

No model derivative, continuation, eigensolver or SVD is called here. The map
Jacobians and residual enclosures are accepted only after external source/input
binding by the accompanying runner. Arb arithmetic remains shared.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json,matrix_json

def need(ok,why):
    if not ok:raise ValueError(why)
def pair(x):
    need(isinstance(x,list) and len(x)==2 and all(type(v)is int for v in x),'dyadic pair')
    return Fraction(x[0])*Fraction(2)**x[1]
def interval(x):
    m,r=pair(x['mid']),pair(x['rad']);need(r>=0,'nonnegative radius');return m-r,m+r
def stored(x):
    interval(x);return arb(arb(tuple(x['mid'])),arb(tuple(x['rad'])))
def mat(rows):return arb_mat([[stored(x) for x in row] for row in rows])
def ident(n):return arb_mat([[int(i==j) for j in range(n)] for i in range(n)])
def norm(a):return max(sum((abs(a[i,j]).upper() for j in range(a.ncols())),arb(0)) for i in range(a.nrows()))
def point(x):
    a,b=interval(x);need(a==b,'exact point required');return a
def contains(outer,inner,strict=False):
    a,b=interval(outer);c,d=interval(inner)
    return a<c and d<b if strict else a<=c and d<=b

def audit_chart(r,anchor,expected_center,settings):
    p,n=18,13;N=p*n
    need(r['settings']==settings and r['parameter_center']==expected_center,'fixed chart/settings')
    need(r['statement']['period']==p and r['statement']['phase_dimension']==n and r['statement']['shooting_dimension']==N,'full18x13 dimensions')
    need(all(r['gates'][k] for k in ['uniform_root','primitive_period','anchor_identified']),'branch gates')
    delta=Fraction(2)**settings['halfwidth_exponent'];rho=Fraction(2)**settings['radius_exponent'];center_h=Fraction(expected_center)
    need(list(map(Fraction,r['exact_parameter_endpoints']))==[center_h-delta,center_h+delta],'exact closed parameter endpoints')
    aa,bb=interval(r['parameter_interval']);need(aa<=center_h-delta and center_h+delta<=bb,'parameter interval covers declared range')
    need(point(r['correction_radius'])==rho and rho>0,'declared correction radius')
    base=[point(v) for row in r['center'] for v in row];slope=[point(row[0]) for row in r['predictor_slope']]
    need(len(base)==len(slope)==N,'all phase coordinates')
    for j in range(p):
        for i in range(n):
            k=j*n+i;lo,hi=interval(r['phase_boxes'][j][i]);spread=abs(slope[k])*delta+rho
            need(lo<=base[k]-spread and base[k]+spread<=hi,'phase box covers affine uniqueness domain')
    need(len(r['phase_jacobians'])==p and [d['divisor'] for d in r['divisors']]==[1,2,3,6,9],'complete derivative/divisor inventory')
    with ctx.workprec(384):
        C=mat(r['preconditioner']);A=mat(r['jacobian_tube']);S=mat(r['predictor_slope']);B=mat(r['parameter_partial_tube']);H=mat(r['residual_center'])
        need(C.nrows()==C.ncols()==A.nrows()==A.ncols()==N and S.nrows()==B.nrows()==H.nrows()==N,'saved matrix dimensions')
        for row in r['preconditioner']:
            for x in row:point(x)
        blocks=[mat(row) for row in r['phase_jacobians']]
        # Rebuild shooting derivative block positions independently.
        rebuilt=arb_mat(N,N)
        for j,J in enumerate(blocks):
            need(J.nrows()==J.ncols()==n,'native derivative dimensions')
            for a in range(n):
                for b in range(n):rebuilt[j*n+a,j*n+b]+=J[a,b]
                rebuilt[j*n+a,((j+1)%p)*n+a]-=1
        need(all(rebuilt[i,j].overlaps(A[i,j]) for i in range(N) for j in range(N)),'shooting Jacobian block assembly')
        A=rebuilt
        D=ident(N)-C*A;contract=norm(D);need(contract<1,'independent uniform contraction')
        d=arb(0,arb(delta.numerator)/delta.denominator);rr=arb(rho.numerator)/rho.denominator;E=arb_mat([[arb(0,rr)] for _ in range(N)])
        path=A*S+B;K=-C*(H+path*d)+D*E
        need(all(abs(K[i,0]).upper()<rr for i in range(N)),'independent strict uniform Krawczyk inclusion')
        need(all(max(abs(a),abs(b))<rho for row in r['image_correction'] for a,b in [interval(row[0])]),'stored strict image inclusion')
        primitive=[]
        for div in (1,2,3,6,9):
            separating=[i for i in range(n) if not(stored(r['phase_boxes'][div][i])-stored(r['phase_boxes'][0][i])).contains(0)]
            need(bool(separating),'whole-chart primitive exclusion');primitive.append({'divisor':div,'coordinates':separating})
        # Rebuild the two endpoint root enclosures, the objects used for the join.
        endpoint_checks=[]
        for side in (-1,1):
            dd=side*arb(delta.numerator)/delta.denominator
            image=-C*(H+path*dd)+D*E
            endpoint=[arb(base[i].numerator)/base[i].denominator+S[i,0]*dd+image[i,0] for i in range(N)]
            saved=r['endpoint_root_enclosures'][0 if side<0 else 1]
            need(len(saved)==N,'full endpoint inventory')
            # The recomputed enclosures may differ by outward rounding. For
            # joining we use these new valid enclosures directly as well.
            endpoint_checks.append([ball_json(v) for v in endpoint])
        ah=Fraction(anchor['rate']);need(center_h-delta<=ah<=center_h+delta,'anchor parameter lies in chart')
        need(len(anchor['phase_boxes'])==p and all(len(row)==n for row in anchor['phase_boxes']),'native anchor inventory')
        for j in range(p):
            for i in range(n):
                k=j*n+i;lo,hi=interval(anchor['phase_boxes'][j][i]);pred=base[k]+slope[k]*(ah-center_h)
                need(pred-rho<=lo<=hi<=pred+rho,'entire anchor enclosure in uniqueness domain')
        return {'passed':True,'contraction':ball_json(contract),'image_norm':ball_json(norm(K)),'anchor_coordinates':N,'primitive':primitive,'recomputed_endpoints':endpoint_checks,'scope':'Uniform root and primitive branch only; saved analytic enclosures/Arb trusted, no stability inference.'}

def audit_join(charts,inp,settings):
    need(len(charts)==2,'two declared charts');h=Fraction(inp['rate']);w=Fraction(2)**settings['halfwidth_exponent']
    first=audit_chart(charts[0],inp['anchor'],str(h),settings)
    saved=charts[0]['endpoint_root_enclosures'][1]
    anchor={'rate':str(h+w),'phase_boxes':[saved[j*13:(j+1)*13] for j in range(18)]}
    second=audit_chart(charts[1],anchor,str(h+2*w),settings)
    # Also join the newly independently reduced endpoint, rather than merely
    # trusting the producer endpoint arithmetic.
    other=first['recomputed_endpoints'][1];base=[point(x) for row in charts[1]['center'] for x in row];slope=[point(row[0]) for row in charts[1]['predictor_slope']];rho=Fraction(2)**settings['radius_exponent']
    for k,x in enumerate(other):
        lo,hi=interval(x);prediction=base[k]-slope[k]*w;need(prediction-rho<=lo<=hi<=prediction+rho,'independently recomputed endpoint joins chart two')
    return {'passed':True,'charts':[first,second],'original_anchor_coordinates':234,'joined_endpoint_coordinates':234,'exact_interval':[str(h-w),str(h+3*w)],'scope':'Same original primitive18 branch across two closed intervals, conditional only on hash-bound original anchor and derivative enclosures. Stability separate.'}


def audit_positive(matrix,record):
    need(record['passed'],'recorded positivity gate')
    n=matrix.nrows();basis=mat(record['basis']);inverse=basis.inv()
    need(basis.nrows()==basis.ncols()==n,'congruence dimension')
    for row in record['basis']:
        for value in row:point(value)
    # A point basis whose interval inverse exists plus this verified defect is
    # nonsingular; no numerical Cholesky is repeated or trusted as a gate.
    pre=mat(record['inverse']);defect=norm(ident(n)-pre*basis);need(defect<1,'independent congruence inverse witness')
    transformed=inverse*matrix*inverse.transpose()
    margins=[transformed[i,i].lower()-sum((abs(transformed[i,j]).upper() for j in range(n) if j!=i),arb(0)) for i in range(n)]
    need(all(v>0 for v in margins),'independent positive Gershgorin margins')
    return {'passed':True,'inverse_defect':ball_json(defect),'minimum_row_margin':ball_json(min(margins))}

def audit_periodic(witness):
    need(witness['passed'] and witness['point_passed'] and witness['phase_count']==18 and witness['dimension']==13 and witness['cyclic_P_final_index']==0,'full periodic witness')
    with ctx.workprec(384):
        blocks=[mat(x) for x in witness['interval_J']];rows=witness['phases'];need(len(rows)==len(blocks)==18,'all18 phase inequalities')
        metrics=[mat(row['P']) for row in rows]
        for j,row in enumerate(rows):
            need(row['phase']==j,'native phase order')
            for a in range(13):
                for b in range(13):need(point(row['P'][a][b])==point(row['P'][b][a]),'fixed symmetric exact metric')
        summaries=[]
        for j,(J,P) in enumerate(zip(blocks,metrics)):
            nextP=metrics[(j+1)%18];J0=arb_mat([[J[a,b].mid() for b in range(13)] for a in range(13)]);E=J-J0
            decrease=P-J0.transpose()*nextP*J0-J0.transpose()*nextP*E-E.transpose()*nextP*J0-E.transpose()*nextP*E
            summaries.append({'phase':j,'P':audit_positive(P,rows[j]['P_positive']),'D':audit_positive(decrease,rows[j]['interval_D_positive'])})
        return {'passed':True,'phases':summaries,'theorem':'All18 exact cyclic metrics are positive and all native phase decreases are positive for every enclosed derivative; strict contraction over one period follows. Orbit identity is a separate join check.'}
