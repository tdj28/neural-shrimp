"""Independent cubic705 native-return reductions, with degree4 residual remainder.

No producer residual/gradient assembly or proof reduction is invoked. Native
arrays and Arb remain trusted shared dependencies. Domain containment is checked
for every acquired input, including all correction coordinates. Cubic entries
are degree3 coefficients; curvature entries are second derivatives.
"""
from math import factorial,comb
from fractions import Fraction as F
from flint import arb,arb_mat
from review_correlated705_math import P,U,R,M,need,finite,mid,as_path,scalar_path,check_path,first_terminal
from neural_shrimp.rigorous_orbits import ball_json,vector_json

def jacobian_coefficients(c,s,a,b,error,d,events,n,p,normal,tick=lambda:None):
 N=3*n*p+3;m=3*n;need(2<=n<=13 and 1<=p<=18 and all(len(v)==N for v in(c,s,a,b,error,normal))and len(events)==2*p,'complete native gradient-path geometry')
 need(all(v.is_exact()and v.is_finite()for v in c+s+a+b+normal)and all(v.is_finite()and v.contains(0)for v in error),'fixed exact predictor and constant correction domain')
 z=[[c[i]+s[i]*d+a[i]*d*d/2+b[i]*d*d*d+error[i],s[i]+a[i]*d+3*b[i]*d*d,a[i]/2+3*b[i]*d,b[i]]for i in range(N)];need(z[-3][0]>0 and z[-2][0]>0,'whole actual positive rates')
 cs=P.terminal_series(z[-1],0);es=P.terminal_series(z[-1],1);ds=P.mul(z[-1],es);cp,ep,dp=first_terminal(z[-1]);J=[arb_mat(N,N)for _ in range(3)]
 for phase in range(p):
  off=phase*m;after=((phase+1)%p)*m;both=[]
  for group,name in((n,'q'),(2*n,'r')):
   event=events[2*phase+(name=='r')];need(event['phase']==phase and event['direction']==name,'literal native q/r order');U.contains(event['d'],ball_json(d),'actual parameter domain')
   inputs=event['inputs'];rets=event['returns'];need(len(inputs)==n+2 and len(rets)==n,'complete native gradient acquisition')
   expected=[[z[off+i],z[off+group+i]]for i in range(n)]+[[z[-3],[arb(0)]*4],[z[-2],[arb(0)]*4]]
   for i,(old,values)in enumerate(zip(inputs,expected,strict=True)):
    need(set(old)=={'value','gradient'}and len(old['gradient'])==n+2,'actual base gradient shape');check_path(old['value'],values,'same predictor/correction/rates/direction')
    for j,grad in enumerate(old['gradient']):check_path(grad,scalar_path(arb(int(i==j))),'literal base basis; held mode')
   parsed=[]
   for item in rets:
    need(set(item)=={'value','gradient'}and len(item['gradient'])==n+2,'full returned base gradients');parsed.append(dict(value=as_path(item['value']),gradient=[as_path(g)for g in item['gradient']]))
   both.append(parsed)
  q,r=both
  for i in range(n):
   # Mode holdings cannot change any state/base-only coefficient.
   for k in range(4):
    need(U.same_ball(events[2*phase]['returns'][i]['value'][0][k],events[2*phase+1]['returns'][i]['value'][0][k]),'identical q/r state path')
    for j in range(n+2):need(U.same_ball(events[2*phase]['returns'][i]['gradient'][j][0][k],events[2*phase+1]['returns'][i]['gradient'][j][0][k]),'identical full q/r base gradient')
   for degree in range(3):
    for j in range(n):
     J[degree][off+i,off+j]+=q[i]['gradient'][j][0][degree]
     for group,ret in((n,q),(2*n,r)):
      J[degree][off+group+i,off+j]+=ret[i]['gradient'][j][1][degree]
      J[degree][off+group+i,off+group+j]+=ret[i]['gradient'][j][0][degree]
    for rate in range(2):
     J[degree][off+i,N-3+rate]+=q[i]['gradient'][n+rate][0][degree]
     for group,ret in((n,q),(2*n,r)):J[degree][off+group+i,N-3+rate]+=ret[i]['gradient'][n+rate][1][degree]
    J[degree][off+i,after+i]-=int(degree==0)
    if phase+1<p:
     for group in(n,2*n):J[degree][off+group+i,after+group+i]-=int(degree==0)
    else:
     for row,col,poly,sign in((n,n,cs,-1),(n,2*n,ds,1),(2*n,n,es,-1),(2*n,2*n,cs,-1)):J[degree][off+row+i,after+col+i]+=sign*poly[degree]
     J[degree][off+n+i,N-1]+=(-P.mul(cp,z[n+i])[degree]+P.mul(dp,z[2*n+i])[degree])
     J[degree][off+2*n+i,N-1]+=(-P.mul(ep,z[n+i])[degree]-P.mul(cp,z[2*n+i])[degree])
  tick()
 for degree in range(3):
  for i in range(n):J[degree][N-3,n+i]=z[n+i][degree];J[degree][N-2,n+i]=z[2*n+i][degree];J[degree][N-2,2*n+i]=z[n+i][degree]
 for i in range(N):J[0][N-1,i]=normal[i]
 need(all(finite(v)for v in J),'finite full independent Jacobian coefficients');return J


def polynomial_root(C0,J,residual,weights,radius,h,error,tick=lambda:None,emit=lambda *_:None):
 N=len(weights);need(1<=N<=705 and len(J)==3 and len(residual)==5 and len(error)==N,'full polynomial dimensions')
 need(C0.nrows()==C0.ncols()==N and all(v.is_exact()and v.is_finite()for row in C0.tolist()for v in row),'finite exact C0 proposal')
 need(all(v.nrows()==v.ncols()==N and finite(v)for v in J)and all(v.nrows()==N and v.ncols()==1 and finite(v)for v in residual),'full finite native coefficients')
 need(all(v.is_exact()and v.is_finite()and v>0 for v in weights+[radius,h]),'positive exact proof scales');radii=[radius*w for w in weights]
 for e,r in zip(error,radii,strict=True):need(e.is_finite()and e.lower()<=-r and e.upper()>=r,'actual acquired correction domain covers requested radii')
 C1=mid(-C0*mid(J[1])*C0);C=[C0,C1];emit('inverse_linear_proposal',C1);tick()
 D=[arb_mat(N,N)for _ in range(4)];D[0]=M.ident(N)
 for i in range(2):
  for j in range(3):D[i+j]-=C[i]*J[j];tick()
 forcing=[arb_mat(N,1)for _ in range(6)]
 for i in range(2):
  for j in range(5):forcing[i+j]-=C[i]*residual[j]
 powers=[arb(1),arb(0,h),arb(h*h/2,h*h/2),arb(0,h*h*h),arb(h*h*h*h/2,h*h*h*h/2),arb(0,h*h*h*h*h)]
 wholeD=sum((v*powers[k]for k,v in enumerate(D)),arb_mat(N,N));wholeF=sum((v*powers[k]for k,v in enumerate(forcing)),arb_mat(N,1));E=arb_mat([[v]for v in error]);image=wholeF+wholeD*E;tick()
 need(finite(wholeD)and finite(image),'finite full polynomial image');bound=max((sum((abs(wholeD[i,j]).upper()*weights[j]for j in range(N)),arb(0))/weights[i]).upper()for i in range(N));checks=[bool(abs(image[i,0]).upper()<radii[i])for i in range(N)]
 for k,v in enumerate(D):emit(f'defect_coefficient_{k}',v)
 for k,v in enumerate(forcing):emit(f'forcing_coefficient_{k}',v)
 endpoints=[]
 for d in(-h,h):
  rr=sum((residual[k]*(d**k)for k in range(5)),arb_mat(N,1));endpoints.append(-(C0+C1*d)*rr+wholeD*E)
 report=dict(passed=bool(bound.is_finite()and bound<1 and all(checks)),dimension=N,contraction=ball_json(bound),coordinate_inclusions=checks,radii=vector_json(radii),image_correction=vector_json([image[i,0]for i in range(N)]),endpoint_corrections=[vector_json([e[i,0]for i in range(N)])for e in endpoints],scope='Independent quartic-residual full noncommuting affine-preconditioner/Taylor convolution with common parameter parity. Native derivative-domain and original known-root authority remain caller obligations; no complementary stability claim.')
 emit('correlated_root',report);return report

def add5(a,b):return [a[i]+b[i]for i in range(5)]
def scale5(a,b):return [v*b for v in a]
def mul5(a,b):return [sum((a[j]*b[k-j]for j in range(k+1)),arb(0))for k in range(5)]
def const5(a):return [a,*[arb(0)]*4]
def terminal5(t,offset):
 need(offset in (0,1)and len(t)==5 and abs(t[0]).upper()<=R.exact(F(1,4)),'bounded entire quartic composition')
 nil=[arb(0),*t[1:]];powers=[const5(arb(1))]
 for _ in range(4):powers.append(mul5(powers[-1],nil))
 T=abs(t[0]).upper();answer=const5(arb(0))
 for j in range(5):
  # Direct binomial derivative/j! coefficients. For k>=27,j<=4 and
  # |t0|<=1/4 the absolute term ratio is <(28/24)/4/(55*56)<1/2.
  coefficient=sum((R.exact(F((-1)**k*comb(k,j),factorial(2*k+offset)))*t[0]**(k-j)for k in range(j,27)),arb(0))
  tail=2*R.exact(F(comb(27,j),factorial(54+offset)))*T**(27-j)
  answer=add5(answer,scale5(powers[j],coefficient+arb(0,tail)))
 return answer

def residual_coefficients(c,s,a,b,d,events,n,p,normal,origin,sc):
 N=3*n*p+3;m=3*n
 need(2<=n<=13 and 1<=p<=18 and all(len(v)==N for v in(c,s,a,b,normal,origin))and len(events)==2*p,'full cubic path dimensions')
 need(all(v.is_exact()and v.is_finite()for v in c+s+a+b+normal+origin+[sc])and d.is_finite(),'finite exact cubic predictor')
 z=[[c[i]+s[i]*d+a[i]*d*d/2+b[i]*d*d*d,s[i]+a[i]*d+3*b[i]*d*d,a[i]/2+3*b[i]*d,b[i],arb(0)]for i in range(N)]
 need(z[-3][0]>0 and z[-2][0]>0,'positive actual rates')
 co=terminal5(z[-1],0);si=terminal5(z[-1],1);tsi=mul5(z[-1],si);rows=[]
 for k in range(p):
  off=k*m;nxt=((k+1)%p)*m;returns=[]
  for group,name in((n,'q'),(2*n,'r')):
   event=events[2*k+(name=='r')];need(event['phase']==k and event['direction']==name,'literal quartic phase/direction order');U.contains(event['d'],ball_json(d),'whole acquired quartic parameter domain')
   expected=[(z[off+i],z[off+group+i])for i in range(n)]+[(z[-3],const5(arb(0))),(z[-2],const5(arb(0)))]
   need(len(event['inputs'])==n+2 and len(event['returns'])==n,'full quartic native shapes')
   for old,(u,v)in zip(event['inputs'],expected,strict=True):
    need(type(old)is list and len(old)==2 and all(type(row)is list and len(row)==5 for row in old),'full actual quartic input')
    for outer,inner in zip(old[0]+old[1],u+v,strict=True):U.contains(outer,ball_json(inner),'each actual native quartic input coefficient')
   for item in event['returns']:need(type(item)is list and len(item)==2 and all(type(row)is list and len(row)==5 for row in item),'complete native quartic return')
   returns.append([[[R.original_ball(v)for v in row]for row in item]for item in event['returns']])
  for i in range(n):
   need(all(U.same_ball(u,v)for u,v in zip(events[2*k]['returns'][i][0],events[2*k+1]['returns'][i][0],strict=True)),'identical q/r state quartic returns');rows.append(add5(returns[0][i][0],scale5(z[nxt+i],-1)))
  for group,ret in((n,returns[0]),(2*n,returns[1])):
   for i in range(n):
    close=z[nxt+group+i]
    if k+1==p:close=add5(mul5(co,z[n+i]),scale5(mul5(tsi,z[2*n+i]),-1))if group==n else add5(mul5(si,z[n+i]),mul5(co,z[2*n+i]))
    rows.append(add5(ret[i][1],scale5(close,-1)))
 norm=const5(arb(-1));orth=const5(arb(0));gauge=[-sc-d,arb(-1),arb(0),arb(0),arb(0)]
 for i in range(n):norm=add5(norm,mul5(z[n+i],z[n+i]));orth=add5(orth,mul5(z[n+i],z[2*n+i]))
 for i in range(N):gauge=add5(gauge,scale5(add5(z[i],const5(-origin[i])),normal[i]))
 out=arb_mat(rows+[scale5(norm,arb(1)/2),orth,gauge]);need(finite(out),'finite full independent quartic residual');return out

def cubic_membership(anchor,c,s,a,b,normal,origin,sc,h,radii,known_parameter=None):
 N=len(c);need(all(len(v)==N for v in(anchor,s,a,b,normal,origin,radii))and all(v.is_exact()and v.is_finite()and v>0 for v in radii),'all exact positive anchor coordinates/radii')
 point=[R.original_ball(v)for v in anchor];projection=sum((normal[i]*(point[i]-origin[i])for i in range(N)),arb(0))
 if known_parameter is None:d=projection-sc
 else:
  parameter=R.exact(F(known_parameter));need(projection.contains(parameter),'same-gauge exact parameter consistent with actual root enclosure');d=parameter-sc
 checks=[bool(abs(point[i]-c[i]-s[i]*d-a[i]*d*d/2-b[i]*d*d*d).upper()<radii[i])for i in range(N)];inside=bool(abs(d).upper()<=h)
 return dict(passed=inside and all(checks),parameter_inside=inside,coordinates=sum(checks),coordinate_inclusions=checks,projection=ball_json(projection),known_parameter=known_parameter,scope='Full strict membership; a known exact scalar requires caller-proved identical affine gauges and previous endpoint existence.')
