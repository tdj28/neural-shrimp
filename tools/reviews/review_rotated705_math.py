"""Independent same-root endpoint localization and native shooting residual."""
from flint import arb,arb_mat
import review_cubic705_math as H
need,R,M,U=H.need,H.R,H.M,H.U

def residual(point,period,normal,origin,parameter,events):
    N=len(point);n=(N-3)//(3*period);m=3*n
    need(2<=n<=13 and 1<=period<=18 and N==m*period+3 and len(normal)==len(origin)==N and len(events)==2*period,'complete native point/gauge/events')
    need(all(x.is_finite()and x.is_exact()for x in point+normal+origin+[parameter]),'literal exact point and gauge')
    t=point[-1];need(t>0 and t<arb(1)/4 and all(x>0 for x in point[-3:-1]),'positive physical rates and bounded nonzero angle')
    angle=t.sqrt();c=angle.cos();e=angle.sin()/angle;d=t*e;values=[]
    for k in range(period):
        off=m*k;after=m*((k+1)%period);pair=[]
        for mode,name in ((n,'q'),(2*n,'r')):
            ev=events[2*k+(name=='r')]
            need(ev['phase']==k and ev['direction']==name,'literal native phase/direction order')
            expected=point[off:off+n]+point[off+mode:off+mode+n]+point[-3:-1]
            need(len(ev['input'])==2*n+2 and len(ev['value'])==2*n and all(len(x)==1 for x in ev['value']),'entire recorded native point and return')
            for a,b in zip(ev['input'],expected,strict=True):need(M.point(a)==M.point(H.ball_json(b)),'exact actual native input')
            pair.append([R.original_ball(x[0])for x in ev['value']])
        q,r=pair
        for i in range(n):need(U.same_ball(events[2*k]['value'][i][0],events[2*k+1]['value'][i][0]),'same q/r native state return')
        values += [q[i]-point[after+i]for i in range(n)]
        if k+1<period:
            values += [q[n+i]-point[after+n+i]for i in range(n)]
            values += [r[n+i]-point[after+2*n+i]for i in range(n)]
        else:
            values += [q[n+i]-c*point[n+i]+d*point[2*n+i]for i in range(n)]
            values += [r[n+i]-e*point[n+i]-c*point[2*n+i]for i in range(n)]
    values += [(sum((point[n+i]**2 for i in range(n)),arb(0))-1)/2,sum((point[n+i]*point[2*n+i]for i in range(n)),arb(0)),sum((normal[i]*(point[i]-origin[i])for i in range(N)),arb(0))-parameter]
    out=arb_mat([[v]for v in values]);need(H.finite(out),'finite independently assembled full residual');return out

def localize(D,rhs,weights,initial):
    """Conditional known-error enclosure for e=rhs+D_actual e, |D_actual|<=|D|.

    Caller authenticates the whole point-to-known-root derivative domain.
    Weighted Neumann radius is rigorous before intersections; each intersection
    retains the inherited exact root. This function does not prove existence.
    """
    n=len(weights);need(1<=n<=705 and D.nrows()==D.ncols()==n and rhs.nrows()==n and rhs.ncols()==1 and len(initial)==n,'complete localization dimensions')
    need(H.finite(D)and H.finite(rhs)and all(w.is_exact()and w.is_finite()and w>0 for w in weights)and all(v.is_finite()for v in initial),'finite positive complete localization data')
    q=max((sum((abs(D[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n));need(q<1,'strict whole-domain weighted Neumann bound')
    radius=(max((abs(rhs[i,0]).upper()/weights[i]).upper()for i in range(n))/(1-q)).upper()
    error=[initial[i].intersection(arb(0,(radius*weights[i]).upper()))for i in range(n)]
    for _ in range(3):
        image=rhs+D*arb_mat([[v]for v in error]);error=[error[i].intersection(image[i,0])for i in range(n)]
    need(all(v.is_finite()for v in error),'finite retained same-root enclosure');return error,q,radius

def segment_in_domain(predictor,error,point,known):
    need(len(predictor)==len(error)==len(point)==len(known),'complete segment')
    for p,e,x,y in zip(predictor,error,point,known,strict=True):
        pl=M.point(H.ball_json(p));lo,hi=M.interval(e)
        for value in(x,y):
            a,b=M.interval(H.ball_json(value));need(pl+lo<=a<=b<=pl+hi,'literal entire original-error segment')
    return True
