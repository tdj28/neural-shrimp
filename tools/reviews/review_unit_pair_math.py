"""Independent native704 unit-pair assembly and Krawczyk reduction.

Only retained analytic direction acquisitions are shared. No producer proof,
terminal-series, lift, model, derivative, or inverse-proposal routine is called.
"""
from fractions import Fraction as F
from math import factorial
from flint import arb,arb_mat,ctx
import review_joined_period18_math as M
import review_quadratic_connection_math as R
from neural_shrimp.rigorous_orbits import ball_json,vector_json
need=M.need

def same_ball(a,b):return M.interval(a)==M.interval(b)
def contains(outer,inner,why):
    a,b=M.interval(outer);c,d=M.interval(inner);need(a<=c<=d<=b,why)

def terminal(n,t,derivative=0):
    """Independent degree24 direct sums with a proved geometric remainder."""
    need(type(n)is int and 2<=n<=13 and type(derivative)is int and 0<=derivative<=2,'terminal dimensions/order')
    need(t.is_finite() and abs(t).upper()<=R.exact(F(1,4)),'entire-series domain')
    def series(offset,j):
        total=arb(0)
        for k in range(j,25):
            a=F((-1)**k*factorial(k),factorial(k-j)*factorial(2*k+offset))
            total+=R.exact(a)*t**(k-j)
        # For k>=25 and j<=2, the consecutive absolute-term ratio is
        # ((k+1)/(k+1-j))*|t|/((2k+2+offset)*(2k+1+offset))<1/2.
        k=25;tail=2*R.exact(F(factorial(k),factorial(k-j)*factorial(2*k+offset)))*abs(t).upper()**(k-j)
        return total+arb(0,tail)
    c=series(0,derivative);e=series(1,derivative);d=t*e
    if derivative:d+=derivative*series(1,derivative-1)
    out=arb_mat(3*n,3*n)
    for i in range(n):
        out[i,i]=int(derivative==0);out[n+i,n+i]=c;out[n+i,2*n+i]=-d;out[2*n+i,n+i]=e;out[2*n+i,2*n+i]=c
    return out

def rebuild(flat,events,n,p,t):
    """Place all native state/mode/rate blocks; independently impose S(t)."""
    m=3*n;N=m*p+2
    need(2<=n<=13 and 1<=p<=18 and len(flat)==N and len(events)==2*p,'full native dimensions')
    need(all(v.is_finite() for v in flat),'finite native box')
    closing=terminal(n,t);closing_t=terminal(n,t,1)
    H=arb_mat(N,1);J=arb_mat(N,N);Ht=arb_mat(N,1)
    for k in range(p):
        off=k*m;nxt=((k+1)%p)*m;q,r=events[2*k:2*k+2]
        need(q['phase']==r['phase']==k and q['direction']=='q' and r['direction']=='r','fixed native q/r order')
        for e,direction in ((q,1),(r,2)):
            actual=flat[off:off+n]+flat[off+direction*n:off+(direction+1)*n]+flat[-2:]
            need(len(e['input'])==2*n+2,'full actual direction domain')
            for outer,inner in zip(e['input'],actual,strict=True):contains(outer,ball_json(inner),'native acquisition covers complete actual domain')
            need(len(e['value'])==len(e['block'])==len(e['rates'])==2*n and all(len(a)==1 for a in e['value']) and all(len(a)==2*n for a in e['block']) and all(len(a)==2 for a in e['rates']),'native derivative shapes')
        for i in range(n):
            need(same_ball(q['value'][i][0],r['value'][i][0]),'same state return in both directions')
            for j in range(n):
                need(same_ball(q['block'][i][j],r['block'][i][j]) and same_ball(q['block'][i][j],q['block'][n+i][n+j]) and same_ball(r['block'][i][j],r['block'][n+i][n+j]),'same full native DF')
                need(M.point(q['block'][i][n+j])==M.point(r['block'][i][n+j])==0,'state independent of auxiliary direction')
            for j in range(2):need(same_ball(q['rates'][i][j],r['rates'][i][j]),'same physical-rate state derivatives')
        vq,vr=R.matrix(q['value']),R.matrix(r['value']);Bq,Br=R.matrix(q['block']),R.matrix(r['block']);Eq,Er=R.matrix(q['rates']),R.matrix(r['rates'])
        V=arb_mat([[vq[i,0]] for i in range(2*n)]+[[vr[n+i,0]] for i in range(n)])
        S=closing if k==p-1 else M.ident(m)
        for i in range(m):
            H[off+i,0]=V[i,0]-sum((S[i,j]*flat[nxt+j] for j in range(m)),arb(0))
            for j in range(m):J[off+i,nxt+j]-=S[i,j]
        for i in range(2*n):
            for j in range(2*n):J[off+i,off+j]+=Bq[i,j]
            for j in range(2):J[off+i,N-2+j]=Eq[i,j]
        for i in range(n):
            for j in range(n):J[off+2*n+i,off+j]+=Br[n+i,j];J[off+2*n+i,off+2*n+j]+=Br[n+i,n+j]
            for j in range(2):J[off+2*n+i,N-2+j]=Er[n+i,j]
    q=flat[n:2*n];r=flat[2*n:3*n]
    H[N-2,0]=(sum((v*v for v in q),arb(0))-1)/2;H[N-1,0]=sum((x*y for x,y in zip(q,r,strict=True)),arb(0))
    for i in range(n):J[N-2,n+i]=q[i];J[N-1,n+i]=r[i];J[N-1,2*n+i]=q[i]
    for i in range(m):Ht[(p-1)*m+i,0]=-sum((closing_t[i,j]*flat[j] for j in range(m)),arb(0))
    return H,J,Ht

def contraction(C,J,G0,Jp,Htp,slope,weights,radius,delta):
    N=C.nrows();need(C.ncols()==J.nrows()==J.ncols()==Jp.nrows()==Jp.ncols()==N,'square full reduction')
    need(G0.nrows()==Htp.nrows()==slope.nrows()==len(weights)==N and G0.ncols()==Htp.ncols()==slope.ncols()==1,'full vectors')
    w=[R.exact(v) for v in weights];rho=R.exact(radius);d=arb(0,R.exact(delta))
    need(rho>0 and R.exact(delta)>0 and all(x.is_finite() and x>0 and x<=1 for x in w),'positive finite radii')
    D=M.ident(N)-C*J;E=arb_mat([[arb(0,rho*x)] for x in w]);path=Jp*slope+Htp;residual=G0+path*d;K=-C*residual+D*E
    bound=max((sum((abs(D[i,j]).upper()*w[j] for j in range(N)),arb(0))/w[i]).upper() for i in range(N))
    checks=[bool(abs(K[i,0]).upper()<rho*w[i]) for i in range(N)]
    return {'passed':bool(bound.is_finite() and bound<1 and all(checks)),'bound':bound,'checks':checks,'D':D,'E':E,'path':path,'G0':G0,'K':K}

def check_chart(chart,anchor,events,precision=384):
    need(chart['schema']=='neural-shrimp-native-unit-pair-tube-v1' and chart['dimension']==13 and chart['period']==18 and chart['shooting_dimension']==704,'fixed native704 convention')
    settings=chart['settings'];need(settings=={'precision':256,'newton_steps':2,'halfwidth_exponent':-16,'radius_exponent':-11},'fixed measured finite-curve design')
    n,p,N=13,18,704;tc=F(chart['parameter_center']);delta=F(1,65536);radius=F(1,2048)
    need(tc==delta and chart['parameter_endpoints']==['0','1/32768'],'exact nonnegative finite interval')
    c=[M.point(v)for v in chart['center']];s=[M.point(v[0])for v in chart['slope']];w=list(map(F,chart['weights']));radii=[radius*x for x in w]
    need(len(c)==len(s)==len(w)==len(anchor)==N and all(0<x<=1 for x in w),'complete positive weighted coordinates')
    need([M.point(v)for v in chart['coordinate_radii']]==radii,'exact native radii')
    for row in chart['preconditioner']:
        for v in row:M.point(v)
    need(len(events)==216,'six complete36-direction groups, no discarded final acquisition')
    groups=[events[k:k+36] for k in range(0,216,36)]
    with ctx.workprec(precision):
        U=[R.exact(v)for v in c];S=arb_mat([[R.exact(v)]for v in s]);d=arb(0,R.exact(delta));ti=R.exact(tc)+d
        P=[U[i]+S[i,0]*d for i in range(N)];W=[P[i]+arb(0,R.exact(radii[i])) for i in range(N)]
        for old,new in zip(chart['whole_box'],W,strict=True):contains(old,ball_json(new),'retained full correction domain')
        need(all(v>0 for v in W[-2:]),'whole actual rates positive')
        G0,_,_=rebuild(U,groups[3],n,p,R.exact(tc));_,Jp,Htp=rebuild(P,groups[4],n,p,ti);_,J,_=rebuild(W,groups[5],n,p,ti)
        C=R.matrix(chart['preconditioner']);r=contraction(C,J,G0,Jp,Htp,S,w,radius,delta)
        proper=[1,2,3,6,9];need([v['divisor'] for v in chart['divisors']]==proper,'every proper native divisor')
        primitive=[]
        for k in proper:
            coordinates=[i for i in range(n) if not (W[k*3*n+i]-W[i]).contains(0)]
            primitive.append({'divisor':k,'separating_coordinates':coordinates})
        nonzero=[i for i in range(n) if not W[2*n+i].contains(0)]
        checks=[]
        for i,box in enumerate(anchor):
            low,high=M.interval(box);prediction=c[i]-s[i]*tc
            checks.append(prediction-radii[i]<low<=high<prediction+radii[i])
        ends=[]
        for sign in (-1,1):
            dd=R.exact(sign*delta);K=-C*(G0+r['path']*dd)+r['D']*r['E']
            ends.append(vector_json([U[i]+S[i,0]*dd+K[i,0] for i in range(N)]))
        gates={'uniform_normalized_unit_pair':r['passed'],'primitive_period':all(v['separating_coordinates'] for v in primitive),'realified_frame_independent':bool(nonzero),'anchor_identified':all(checks)}
        return {'passed':all(gates.values()),'gates':gates,'precision':precision,'inclusions':sum(r['checks']),'anchor_coordinates':sum(checks),'contraction':ball_json(r['bound']),'primitive':primitive,'nonzero_r_coordinates':nonzero,'recomputed_endpoints':ends,'scope':'Independent384 native block/terminal-series/parameter-path and weighted full704 Krawczyk arithmetic using source-authenticated shared analytic acquisitions. Prior exact normalized Jordan root supplies anchor existence. No complementary attraction or NS criticality claim.'}

def check_lift(root_box,lift,events,n=13,p=18):
    """Exact serialized-domain coverage of the previously proved Jordan root."""
    need(len(root_box)==3*n+2 and len(events)==2*p and lift['dimension']==n and lift['period']==p and lift['shooting_dimension']==3*n*p+2 and lift['parameter']=='0','full conditional native lift')
    rates=root_box[n:n+2];expected=root_box[:n]+root_box[n+2:2*n+2]+root_box[2*n+2:];flat=[]
    need(all(M.interval(v)[0]>0 for v in rates),'same positive root rates')
    for k in range(p):
        q,r=events[2*k:2*k+2];need(q['phase']==r['phase']==k and q['direction']=='q'and r['direction']=='r','native lift phase order')
        need(len(q['input'])==len(r['input'])==2*n+2 and len(q['value'])==len(r['value'])==2*n and all(len(v)==1 for row in(q,r)for v in row['value']),'complete actual lift domains/returns')
        for row,which in ((q,1),(r,2)):
            wanted=expected[:n]+expected[which*n:(which+1)*n]+rates
            for outer,inner in zip(row['input'],wanted,strict=True):contains(outer,inner,'each lift phase covers preceding native values and fixed root rates')
        need(all(same_ball(x,y)for x,y in zip(q['input'][:n],r['input'][:n])) and all(same_ball(x[0],y[0])for x,y in zip(q['value'][:n],r['value'][:n])),'same state shared across both direction acquisitions')
        flat+=q['input'][:2*n]+r['input'][n:2*n]
        expected=[v[0]for v in q['value']]+[v[0]for v in r['value'][n:]]
    flat+=rates;need(len(lift['box'])==len(flat),'all704 native lift coordinates')
    for outer,inner in zip(lift['box'],flat,strict=True):contains(outer,inner,'saved lift covers entire actual input domains')
    return {'passed':True,'native_coordinates':len(flat),'phase_count':p,'scope':'Conditional root closure and normalization supplied by the separately audited exact Jordan root, not by zero containment of a return.'}
