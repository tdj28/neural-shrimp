"""Binary64 saved full-error polynomial diagnostics; no certified new domain."""
import math
import numpy as np
from .unit_pair_conditioning import middle_radius,positive_scaling
from .rigorous_eigenpair_locus import require

def pair(rows):
 n=len(rows);m=len(rows[0]);a=np.empty((n,m));r=np.empty_like(a)
 for i,row in enumerate(rows):
  require(len(row)==m,'complete rectangular saved coefficients')
  for j,v in enumerate(row):a[i,j],r[i,j]=middle_radius(v)
 return a,r

def add(*args):return sum((v[0]for v in args)),sum((v[1]for v in args))
def neg(a):return -a[0],a[1]
def mul(a,b):return a[0]@b[0],np.abs(a[0])@b[1]+a[1]@np.abs(b[0])+a[1]@b[1]
def scalar(a,m,r=0.):return a[0]*m,np.abs(a[0])*r+a[1]*(abs(m)+r)

def reconstruct(returns,inputs,normal,n=13,p=18,sigma=-1):
 N=2*n*p+2;m=2*n;require(len(returns)==len(inputs)==p and len(normal)==N and sigma in(-1,1),'complete literal native blocks')
 out=[(np.zeros((N,N)),np.zeros((N,N)))for _ in range(3)]
 for phase,(ret,inp)in enumerate(zip(returns,inputs,strict=True)):
  require(ret['phase']==inp['phase']==phase and len(ret['returns'])==n and len(inp['inputs'])==n+2,'native phase/dimension identity');off=phase*m;nxt=((phase+1)%p)*m
  for k in range(3):
   A,R=out[k]
   for i,v in enumerate(ret['returns']):
    require(set(v)=={'value','gradient'}and len(v['gradient'])==n+2,'full held-mode returned partials')
    for j,g in enumerate(v['gradient']):
     aa,ar=middle_radius(g[0][k]);bb,br=middle_radius(g[1][k]);col=off+j if j<n else N-2+j-n
     A[off+i,col]+=aa;R[off+i,col]+=ar;A[off+n+i,col]+=bb;R[off+n+i,col]+=br
     if j<n:A[off+n+i,off+n+j]+=aa;R[off+n+i,off+n+j]+=ar
    if k==0:A[off+i,nxt+i]-=1;A[off+n+i,nxt+n+i]-=sigma if phase==p-1 else 1
   if phase==0:
    for j in range(n):A[N-2,n+j],R[N-2,n+j]=middle_radius(inp['inputs'][j]['value'][1][k])
 for j,value in enumerate(normal):out[0][0][N-1,j]=float(value)
 require(all(np.isfinite(v).all()for ar in out for v in ar),'finite saved Jacobian proposal');return out

def polynomial(C,J,R,h):
 n=C[0][0].shape[0];I=(np.eye(n),np.zeros((n,n)));dc=[add(I,neg(mul(C[0],J[0]))),neg(add(mul(C[0],J[1]),mul(C[1],J[0]))),neg(add(mul(C[0],J[2]),mul(C[1],J[1]))),neg(mul(C[1],J[2]))];rc=[(np.zeros((n,1)),np.zeros((n,1)))for _ in range(5)]
 for i in range(2):
  for j in range(4):rc[i+j]=add(rc[i+j],neg(mul(C[i],R[j])))
 def whole(values,width):
  terms=[]
  for i,a in enumerate(values):
   if i==0:terms.append(a)
   elif i%2:terms.append(scalar(a,0,width**i))
   else:terms.append(scalar(a,width**i/2,width**i/2))
  return add(*terms)
 return dc,rc,whole(dc,h),whole(rc,h),whole

def budget(C,J,residual,weights,rho,h):
 w=np.asarray(weights,float);require(np.isfinite(w).all()and np.all(w>0)and rho>0 and h>0,'positive finite original scales');dc,rc,D,f,whole=polynomial(C,J,residual,h);B=np.abs(D[0])+D[1];force=(np.abs(f[0])+f[1])[:,0];r=rho*w;old=(B@w)/w;image=(force+B@r)/r;scale=positive_scaling(B);proposal=np.maximum(np.asarray(scale['weights']),2.**-20);proposal/=proposal.max();newq=float(np.max((B@proposal)/proposal));top=np.argsort(old)[-10:][::-1]
 answer=dict(scope='Saved binary64 interval-style majorants and scaling proposals only. Floating rounding is not certified. Every changed error box requires new native acquisition; frozen-coefficient width comparisons are not proofs.',dimension=len(w),original_weights=w.tolist(),original_max_weighted_row=float(old.max()),original_max_image_ratio=float(image.max()),dominant_rows=[dict(coordinate=int(i),row_ratio=float(old[i]),image_ratio=float(image[i]),weight=float(w[i]))for i in top],scaling=scale,dyadic_weight_proposal=[str(float(v).as_integer_ratio()[0])+'/'+str(float(v).as_integer_ratio()[1])for v in proposal],proposal_same_old_domain_max_row=newq,proposal_same_old_domain_max_image=float(np.max((force+B@(rho*proposal))/(rho*proposal))),forcing_order_max_ratios=[float(np.max((np.abs(v[0][:,0])+v[1][:,0])*h**i/r))for i,v in enumerate(rc)],defect_order_max_rows=[float(np.max(((np.abs(v[0])+v[1])*h**i@w)/w))for i,v in enumerate(dc)],unchanged_coefficients_width_proposals=[])
 for factor in(1.,.5,.25):
  d=whole(dc,h*factor);ff=whole(rc,h*factor);bb=np.abs(d[0])+d[1];fm=(np.abs(ff[0])+ff[1])[:,0];answer['unchanged_coefficients_width_proposals'].append(dict(factor=factor,original_row=float(np.max(bb@w/w)),proposed_row=float(np.max(bb@proposal/proposal)),proposed_image=float(np.max((fm+bb@(rho*proposal))/(rho*proposal)))))
 if newq<.95:
  solution=np.linalg.solve(np.eye(len(w))-B,force);answer['resolvent_proposal']=dict(nonnegative=bool(np.isfinite(solution).all()and np.all(solution>=0)),radii=solution.tolist(),max_linear_residual=float(np.max(np.abs((np.eye(len(w))-B)@solution-force))))
 require(all(math.isfinite(v)for v in [answer['original_max_weighted_row'],answer['original_max_image_ratio'],newq]),'finite numerical diagnostics');return answer
