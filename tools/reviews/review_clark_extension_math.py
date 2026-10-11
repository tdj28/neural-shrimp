"""Independent endpoint assembly/curvature from shared native MixedJet returns.

The exact source-authenticated caller supplies all actual base/direction inputs;
these old per-phase outputs did not separately serialize their input jets.
No producer augmented-system or inverse/curvature reducer is imported.
"""
from flint import arb,arb_mat
import review_clark_correlated_acquisitions as A
H,M,need=A.H,A.M,A.need

def pair(z,b,direction,rows):
 n=(len(z)-2)//2;N=2*n+2;K=N-1
 need(2<=n<=12 and len(z)==N and len(direction)==N+1 and len(rows)==3 and [r['phase']for r in rows]==[0,1,2],'complete native paired inputs/phase order')
 last=rows[-1]['even']+rows[-1]['odd'];need(len(last)==2*n and all(len(x['g'])==len(x['m'])==N+1 for x in last),'complete native derivative returns')
 value=[];grad=[];der=[];mixed=[]
 for i,x in enumerate(last):
  coord=i if i<n else i+1;sgn=-1 if i<n else 1
  value.append(-M.decode(x['f'])+sgn*z[coord]);grad.append([-M.decode(v)+sgn*int(k==coord)for k,v in enumerate(x['g'])]);der.append(-M.decode(x['d'])+sgn*direction[coord]);mixed.append([-M.decode(v)for v in x['m']])
 q=z[n+1:N-1];v=direction[n+1:N-1];value.append((sum((x*x for x in q),arb(0))-1)/2)
 grad.append([q[k-n-1]if n+1<=k<N-1 else arb(0)for k in range(N+1)]);der.append(sum((x*y for x,y in zip(q,v,strict=True)),arb(0)));mixed.append([v[k-n-1]if n+1<=k<N-1 else arb(0)for k in range(N+1)])
 return M.col(value),arb_mat(grad),M.col(der),arb_mat(mixed)

def explicit(request,rows):
 unknown=[M.decode(v)for v in request['input']];b=M.decode(request['squared_amplitude']);N=(len(unknown)+1)//2;K=N-1;n=(N-2)//2;need(N+K==len(unknown),'complete explicit unknown')
 z=unknown[:N];v=unknown[N:];f,g,d,m=pair(z,b,v+[arb(0),arb(1)],rows);value=arb_mat(f.tolist()+d.tolist()+[[v[n]]]);J=arb_mat([[g[i,k]for k in range(N)]+[arb(0)]*K for i in range(K)]+[[m[i,k]for k in range(N)]+[g[i,k]for k in range(K)]for i in range(K)]+[[arb(0)]*N+[arb(int(k==n))for k in range(K)]]);hb=arb_mat([[g[i,N]]for i in range(K)]+[[m[i,N]]for i in range(K)]+[[arb(0)]])
 need(M.finite(value)and M.finite(J)and M.finite(hb),'finite independently assembled full explicit family');return value,J,hb

def fixed_gain_jacobian(J,hb):
 n=J.nrows();N=(n+1)//2;K=N-1;need(n==J.ncols()==hb.nrows()and hb.ncols()==1,'complete fixedgain columns');columns=list(range(K))+list(range(N,n));return arb_mat([[J[i,k]for k in columns]+[hb[i,0]]for i in range(n)])

def paired_curvature(z,b,first_rows,second_rows,saved_v,emit=lambda *_:None):
 N=len(z);K=N-1;n=(N-2)//2;need(b.is_finite()and b>0,'positive exact paired root support')
 f,g,_,_=pair(z,b,[arb(0)]*(N+1),first_rows);hu=arb_mat([[g[i,k]for k in range(K)]for i in range(K)]);hb=M.col([g[i,N]for i in range(K)]);inverse=M.inverse(hu,emit);v=-inverse*hb
 need(len(saved_v)==K and all(A.covers(raw,(A.dyadic([int(x)for x in v[i,0].lower().man_exp()]),A.dyadic([int(x)for x in v[i,0].upper().man_exp()])))for i,raw in enumerate(saved_v)),'own paired tangent covered by actual second acquisition direction')
 acquired=[M.decode(x)for x in saved_v];f2,g2,d2,mixed=pair(z,b,acquired+[arb(0),arb(1)],second_rows)
 need(all(x.contains(0)for x in M.vec(f))and v[n,0].contains(0)and all(x.contains(0)for x in M.vec(d2)),'already established exact paired/fold identities consistency')
 # m_i = D_(v,0,1) grad H_i; multiply by own (U_b,0,1).
 total=M.col([mixed[i,N]+sum((mixed[i,k]*v[k,0]for k in range(K)),arb(0))for i in range(K)]);second=-inverse*total;bb=second[n,0];aa=4*b*bb
 need(bb.is_finite()and not bb.contains(0),'strict physical endpoint curvature')
 return dict(passed=True,paired_jacobian=hu,paired_inverse=inverse,paired_tangent=v,mixed_total=total,second_tangent=second,leak_bb=bb,leak_aa=aa,scope='Conditional on independently proved same positive full51 root. Separate paired inverse and total fixed-gain physical curvature; native identity and complementary qualification separate.')
