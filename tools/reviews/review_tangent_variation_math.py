"""Independent saved arithmetic for the implicit second-derivative refinement.

Known-root and first-tangent proofs are upstream prerequisites. Analytic phase
Jacobian/total-derivative acquisitions remain shared; no model calls here.
"""
from flint import arb,arb_mat,ctx
import review_correlated_slice_math as L
from neural_shrimp.rigorous_orbits import ball_json,matrix_json,vector_json
need=L.need


def solve(C,A,b,weights,saved=None):
    n=A.nrows();need(A.ncols()==C.nrows()==C.ncols()==b.nrows()==len(weights)==n and b.ncols()==1,'complete implicit solve')
    w=[L.exact(v) for v in weights];need(all(v.is_exact() and v>0 for v in w),'positive exact weights')
    need(all(v.is_finite() for m in (C,A,b) for row in m.tolist() for v in row),'finite complete implicit inputs')
    D=L.M.ident(n)-C*A
    need(all(v.is_finite() for row in D.tolist() for v in row),'finite defect')
    q=max((sum((abs(D[i,j]).upper()*w[j] for j in range(n)),arb(0))/w[i]).upper() for i in range(n))
    need(q.is_finite() and q<1,'strict uniform inverse')
    v=C*b;Dv=D*v
    rho=(max((abs(Dv[i,0]).upper()/w[i]).upper() for i in range(n))/(1-q)).upper()
    need(rho.is_finite() and rho>=0,'finite inverse tail')
    Y=arb_mat([[arb(v[i,0],w[i]*rho)] for i in range(n)])
    if saved is not None:
        need(saved['passed'] and saved['weighted_neumann_bound']==ball_json(q) and saved['weighted_tail_radius']==ball_json(rho),'saved strict inverse and tail')
        L.eq(v,saved['preconditioned_rhs'],'interval-valued RHS retained');L.eq(Y,saved['initial_box'],'full inverse-tail box');need(len(saved['refinements'])==3,'three inverse intersections')
    for k in range(3):
        image=v+D*Y;Y=arb_mat([[Y[i,0].intersection(image[i,0])] for i in range(n)])
        if saved is not None:
            L.eq(image,saved['refinements'][k]['image'],'implicit image');L.eq(Y,saved['refinements'][k]['retained'],'implicit retained enclosure')
    if saved is not None:L.eq(Y,saved['box'],'implicit final enclosure')
    return Y,{'passed':True,'weighted_neumann_bound':ball_json(q),'weighted_tail_radius':ball_json(rho)}


def forcing(jacs,primes,v,h,hidden,rows=None):
    p=len(jacs);need(p>0 and len(primes)==p,'complete phases');n=jacs[0].nrows();N=p*n+1
    need(v.nrows()==N and v.ncols()==1 and 0<hidden<=n and h.is_finite() and h>0,'actual full state/rate direction')
    out=arb_mat(N,1)
    if rows is not None:need(len(rows)==p,'complete curvature rows')
    for j in range(p):
        J,K=jacs[j],primes[j];need(J.nrows()==J.ncols()==K.nrows()==K.ncols()==n,'native phase dimensions')
        u=arb_mat([[v[j*n+i,0]] for i in range(n)])
        B=arb_mat([[((arb(int(i==k))-J[i,k])/h) if i<hidden else arb(0) for k in range(n)] for i in range(n)])
        bp=-B*u;c=K*u+bp*v[N-1,0]
        need(all(a.is_finite() for row in c.tolist() for a in row),'finite curvature forcing')
        for i in range(n):out[j*n+i,0]=c[i,0]
        if rows is not None:
            need(rows[j]['phase']==j,'native curvature ordering')
            for value,key in [(B,'hidden_hessian_rows'),(bp,'parameter_column_derivative'),(c,'forcing')]:L.eq(value,rows[j][key],'independent '+key)
    need(out[N-1,0].is_zero(),'linear gauge curvature zero')
    return out


def audit(chart,start,whole_jac,center_jac,base_tangent,physical,phases,record,actual):
    with ctx.workprec(256):
        p,n=chart['statement']['period'],chart['statement']['phase_dimension'];N=p*n+1
        need(n==13 and p==18 and record['passed'],'same native branch dimensions')
        C=L.dm(start['preconditioner']);A=L.dm(whole_jac);Ac=L.dm(center_jac);v=L.dm(base_tangent['derivative_box'])
        need(vector_json([v[i,0] for i in range(N)])==physical['actual_uniform_tangent'],'same first total-derivative direction')
        h=L.decoded(physical['actual_whole_box'][-1]);weights=chart['exact_weights'];half=L.exact(actual['halfwidth'])
        need(half.is_exact() and half>0 and actual['exact_endpoints']==physical['exact_endpoints'],'same exact closed parameter slice')
        from fractions import Fraction
        need(Fraction(actual['halfwidth'])==(Fraction(actual['exact_endpoints'][1])-Fraction(actual['exact_endpoints'][0]))/2,'exact halfwidth')
        center,cr=solve(C,Ac,arb_mat([[int(i==N-1)] for i in range(N)]),weights,record['center_derivative'])
        need(len(phases)==p and [x['phase'] for x in phases]==list(range(p)),'original native first derivatives')
        b=forcing([L.dm(x['whole_J_mixed_diagnostic']) for x in phases],[L.dm(x['whole_total_Jprime']) for x in phases],v,h,9,record['native_forcing_rows']);L.eq(b,record['forcing'],'full curvature forcing with both rate cross terms')
        second,sr=solve(C,A,-b,weights,record['second_derivative'])
        path=center+second*arb(0,half);refined=arb_mat([[v[i,0].intersection(path[i,0])] for i in range(N)])
        L.eq(path,record['centered_derivative_enclosure'],'integrated tangent enclosure');L.eq(refined,record['refined_derivative_box'],'old/new tangent intersection')
        need(len(actual['actual_whole_box'])==len(actual['actual_uniform_tangent'])==N,'complete actual refined derivative domains')
        for i in range(N):
            L.contains(actual['actual_whole_box'][i],physical['actual_whole_box'][i],'second MixedJet retains original whole state/rate')
            L.contains(actual['actual_uniform_tangent'][i],record['refined_derivative_box'][i][0],'second MixedJet retains true refined tangent')
        return {'passed':True,'center_inverse':cr,'second_inverse':sr,'coordinates':N,'actual_refined_domains_retained':True,'scope':'Independent256 implicit second derivative/center integration, conditional on known smooth root, uniform first tangent and source-bound analytic phase derivatives.'}
