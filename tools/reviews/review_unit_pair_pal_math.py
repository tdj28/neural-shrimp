"""Independent polynomial and full705 Krawczyk audit from retained acquisitions.

No native model, producer PathJet, predictor or proof routine is called.
Shared analytic direction/path derivative arrays and Arb remain dependencies.
"""
from fractions import Fraction as F
from math import factorial
from flint import arb,arb_mat,ctx
import review_unit_pair_math as U
import review_joined_period18_math as M
import review_quadratic_connection_math as R
from neural_shrimp.rigorous_orbits import ball_json,vector_json
need=M.need

def poly(v):
    need(len(v)==4,'four Taylor coefficients');return list(v)
def add(a,b):return [a[i]+b[i]for i in range(4)]
def scale(a,b):
    if isinstance(b,F):b=R.exact(b)
    return [v*b for v in a]
def mul(a,b):return [sum((a[j]*b[i-j]for j in range(i+1)),arb(0))for i in range(4)]
def const(a):return [a,arb(0),arb(0),arb(0)]
def terminal_series(t,offset):
    """Independent degree26 direct derivative sums and geometric tails."""
    need(offset in (0,1) and len(t)==4 and abs(t[0]).upper()<=R.exact(F(1,4)),'bounded entire composition')
    h=[arb(0),*t[1:]];powers=[const(arb(1))]
    for _ in range(3):powers.append(mul(powers[-1],h))
    result=const(arb(0));T=abs(t[0]).upper()
    for j in range(4):
        value=sum((R.exact(F((-1)**k*factorial(k),factorial(k-j)*factorial(2*k+offset)))*t[0]**(k-j)for k in range(j,27)),arb(0))
        # k>=27, j<=3, |t|<=1/4: ratio <=(28/25)/4/(55*56)<1/2.
        tail=2*R.exact(F(factorial(27),factorial(27-j)*factorial(54+offset)))*T**(27-j)
        value+=arb(0,tail);result=add(result,scale(powers[j],value/factorial(j)))
    return result

def rebuild(flat,events,n,p,normal,origin,parameter):
    N=3*n*p+3;need(len(flat)==len(normal)==len(origin)==N,'complete PAL coordinates')
    H,J,Ht=U.rebuild(flat[:-1],events,n,p,flat[-1]);out=arb_mat(N,1);jac=arb_mat(N,N)
    for i in range(N-1):
        out[i,0]=H[i,0]
        for j in range(N-1):jac[i,j]=J[i,j]
        jac[i,N-1]=Ht[i,0]
    out[N-1,0]=sum((normal[i]*(flat[i]-origin[i])for i in range(N)),arb(0))-parameter
    for i in range(N):jac[N-1,i]=normal[i]
    return out,jac

def path_coefficients(center,slope,curvature,d,events,n,p,normal,origin,sc):
    N=3*n*p+3;m=3*n;need(len(center)==len(slope)==len(curvature)==len(normal)==len(origin)==N and len(events)==2*p,'full path dimensions')
    z=[[center[i]+slope[i]*d+curvature[i]*d*d/2,slope[i]+curvature[i]*d,curvature[i]/2,arb(0)]for i in range(N)]
    c=terminal_series(z[-1],0);e=terminal_series(z[-1],1);te=mul(z[-1],e);rows=[]
    for k in range(p):
        off=k*m;nxt=((k+1)%p)*m;returns=[]
        for j,name in [(1,'q'),(2,'r')]:
            event=events[2*k+j-1];need(event['phase']==k and event['direction']==name,'fixed path phase and direction')
            U.contains(event['d'],ball_json(d),'full path parameter domain')
            expected=[(z[off+i],z[off+j*n+i])for i in range(n)]+[(z[-3],const(arb(0))),(z[-2],const(arb(0)))]
            need(len(event['inputs'])==n+2 and len(event['returns'])==n,'full path acquisition shape')
            for old,(a,b)in zip(event['inputs'],expected,strict=True):
                need(set(old)=={'a','b'} and len(old['a'])==len(old['b'])==4,'full actual input jet')
                for outer,inner in zip(old['a']+old['b'],a+b,strict=True):U.contains(outer,ball_json(inner),'each retained native path derivative domain')
            for item in event['returns']:need(set(item)=={'a','b'} and len(item['a'])==len(item['b'])==4,'full return jet')
            returns.append([([R.original_ball(v)for v in item['a']],[R.original_ball(v)for v in item['b']])for item in event['returns']])
        for i in range(n):
            need(all(U.same_ball(a,b)for a,b in zip(events[2*k]['returns'][i]['a'],events[2*k+1]['returns'][i]['a'],strict=True)),'same state Taylor returns')
            rows.append(add(returns[0][i][0],scale(z[nxt+i],-1)))
        for group,ret in [(n,returns[0]),(2*n,returns[1])]:
            for i in range(n):
                close=z[nxt+group+i]
                if k==p-1:close=add(mul(c,z[n+i]),scale(mul(te,z[2*n+i]),-1))if group==n else add(mul(e,z[n+i]),mul(c,z[2*n+i]))
                rows.append(add(ret[i][1],scale(close,-1)))
    norm=const(arb(-1));orth=const(arb(0));gauge=scale([sc+d,arb(1),arb(0),arb(0)],-1)
    for i in range(n):norm=add(norm,mul(z[n+i],z[n+i]));orth=add(orth,mul(z[n+i],z[2*n+i]))
    for i in range(N):gauge=add(gauge,scale(add(z[i],const(-origin[i])),normal[i]))
    return arb_mat(rows+[scale(norm,F(1,2)),orth,gauge])

def anchor_membership(anchor,c,s,a,normal,origin,sc,delta,radii):
    need(len(anchor)==len(c)==len(s)==len(a)==len(normal)==len(origin)==len(radii),'all anchor coordinates')
    projection=sum((normal[i]*(R.original_ball(anchor[i])-origin[i])for i in range(len(c))),arb(0));d=projection-sc
    checks=[bool(abs(R.original_ball(anchor[i])-c[i]-s[i]*d-a[i]*d*d/2).upper()<radii[i])for i in range(len(c))]
    inside=bool(abs(d).upper()<=delta)
    return dict(passed=inside and all(checks),parameter_inside=inside,coordinates=sum(checks),projection=ball_json(projection))

def check_chart(chart,directions,paths,anchor=None,precision=384,tick=lambda:None):
    need(chart['schema']=='neural-shrimp-native-unit-pair-pal-v1' and chart['dimension']==13 and chart['period']==18 and chart['shooting_dimension']==705,'full native705 chart')
    n,p,N=13,18,705;settings=chart['settings'];need(settings['precision']==256 and settings['newton_steps']==2 and -100<=settings['halfwidth_exponent']<=-1 and -100<=settings['radius_exponent']<=-1,'supported original design')
    sc=F(chart['parameter_center']);delta=F(2)**settings['halfwidth_exponent'];radius=F(2)**settings['radius_exponent']
    need(chart['parameter_endpoints']==[str(sc-delta),str(sc+delta)],'exact parameter interval')
    data=[[M.point(v)for v in chart[k]]for k in ['center','slope','curvature']];normal=list(map(F,chart['normal']));origin=list(map(F,chart['origin']));w=list(map(F,chart['weights']))
    need(all(len(v)==N for v in data+[normal,origin,w])and all(0<x<=1 for x in w)and any(normal),'complete fixed exact proposal')
    need([M.point(v)for v in chart['coordinate_radii']]==[radius*x for x in w],'exact radius weights')
    need(len(directions)==144 and len(paths)==108,'all retained native acquisitions')
    for row in chart['preconditioner']:
        for v in row:M.point(v)
    need(len(chart['preconditioner'])==N and all(len(v)==N for v in chart['preconditioner']),'full exact preconditioner')
    with ctx.workprec(precision):
        c,s,a=[[R.exact(v)for v in row]for row in data];nn=[R.exact(v)for v in normal];oo=[R.exact(v)for v in origin];tc=R.exact(sc);dd=R.exact(delta);d=arb(0,dd);rr=[R.exact(radius*v)for v in w]
        predictor=[c[i]+s[i]*d+a[i]*d*d/2 for i in range(N)];whole=[predictor[i]+arb(0,rr[i])for i in range(N)]
        for old,new in zip(chart['whole_box'],whole,strict=True):U.contains(old,ball_json(new),'retained whole domain covers recomputed exact predictor')
        need(all(v>0 for v in whole[-3:])and abs(whole[-1]).upper()<=R.exact(F(1,4)),'positive physical rates/angle on entire domain')
        g0,j0=rebuild(c,directions[72:108],n,p,nn,oo,tc);tick();_,J=rebuild(whole,directions[108:],n,p,nn,oo,tc+d);tick()
        coeff=path_coefficients(c,s,a,arb(0),paths[36:72],n,p,nn,oo,tc);tick();wc=path_coefficients(c,s,a,d,paths[72:],n,p,nn,oo,tc);tick()
        C=R.matrix(chart['preconditioner']);residual=arb_mat([[coeff[i,0]+coeff[i,1]*d+coeff[i,2]*d*d+wc[i,3]*d*d*d]for i in range(N)])
        D=M.ident(N)-C*J;E=arb_mat([[arb(0,v)]for v in rr]);K=-C*residual+D*E;tick()
        bound=max((sum((abs(D[i,j]).upper()*R.exact(w[j])for j in range(N)),arb(0))/R.exact(w[i])).upper()for i in range(N))
        checks=[bool(abs(K[i,0]).upper()<rr[i])for i in range(N)]
        proper=[1,2,3,6,9];primitive=[dict(divisor=k,separating_coordinates=[i for i in range(n)if not(whole[k*3*n+i]-whole[i]).contains(0)])for k in proper]
        nonzero=[i for i in range(n)if not whole[2*n+i].contains(0)]
        ac=dict(passed=False,reason='No independently proved anchor supplied')if anchor is None else anchor_membership(anchor,c,s,a,nn,oo,tc,dd,rr)
        endpoints=[]
        for sign in (-1,1):
            x=sign*dd;v=arb_mat([[coeff[i,0]+coeff[i,1]*x+coeff[i,2]*x*x+wc[i,3]*x*x*x]for i in range(N)]);ke=-C*v+D*E
            endpoints.append(vector_json([c[i]+s[i]*x+a[i]*x*x/2+ke[i,0]for i in range(N)]))
        gates=dict(uniform_root=bool(all(checks)and bound.is_finite()and bound<1),primitive_period=all(bool(v['separating_coordinates'])for v in primitive),frame_independent=bool(nonzero),positive_angle=True,anchor_identified=ac['passed'])
        return dict(passed=all(gates.values()),gates=gates,precision=precision,inclusions=sum(checks),contraction=ball_json(bound),anchor=ac,primitive=primitive,recomputed_endpoints=endpoints,scope='Independent direct polynomial/terminal-series/native block and full705 Krawczyk arithmetic. Shared source-authenticated native derivatives and Arb. Connected only with separately proved full anchor; no complementary attraction, criticality or whole-window topology.')
