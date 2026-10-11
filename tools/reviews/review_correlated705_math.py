"""Independent correlated705 Jacobian/residual polynomial root reductions.

Only authenticated native gradient/path returns and finite C0 are proposals.
No producer model, gradient-path assembly or correlated proof code is called.
"""
from pathlib import Path
import sys
from fractions import Fraction as F
from flint import arb,arb_mat
sys.path.insert(0,str(Path(__file__).with_name('native704_aux')))
import review_unit_pair_pal_math as P
import review_pal_rechart_math as L
U,R,M,need=P.U,P.R,P.M,P.need
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json


def finite(a):return all(v.is_finite()for row in a.tolist()for v in row)
def mid(a):return arb_mat([[v.mid()for v in row]for row in a.tolist()])
def as_path(v):
 need(type(v)is list and len(v)==2 and all(type(row)is list and len(row)==4 for row in v),'actual full path pair')
 return [[R.original_ball(a)for a in row]for row in v]
def scalar_path(v):return [[v,arb(0),arb(0),arb(0)],[arb(0)]*4]
def check_path(old,expected,why):
 need(type(old)is list and len(old)==2 and all(type(v)is list and len(v)==4 for v in old),'complete '+why)
 for row,actual in zip(old,expected,strict=True):
  for outer,inner in zip(row,actual,strict=True):U.contains(outer,ball_json(inner),why)


def first_terminal(t):
 # Direct entire derivatives supplied by the separately reviewed degree26
 # Taylor helper. Extract derivatives at t0, then compose with this path.
 base=[t[0],arb(1),arb(0),arb(0)];out=[]
 for offset in(0,1):
  b=P.terminal_series(base,offset);out.append([b[1],2*b[2]*t[1],2*b[2]*t[2]+3*b[3]*t[1]*t[1],arb(0)])
 cp,ep=out;e=P.terminal_series(t,1);dp=P.add(e,P.mul(t,ep));return cp,ep,dp


def jacobian_coefficients(c,s,a,error,d,events,n,p,normal,tick=lambda:None):
 N=3*n*p+3;m=3*n;need(2<=n<=13 and 1<=p<=18 and all(len(v)==N for v in(c,s,a,error,normal))and len(events)==2*p,'complete native gradient-path geometry')
 need(all(v.is_exact()and v.is_finite()for v in c+s+a+normal)and all(v.is_finite()and v.contains(0)for v in error),'fixed exact predictor and constant correction domain')
 z=[[c[i]+s[i]*d+a[i]*d*d/2+error[i],s[i]+a[i]*d,a[i]/2,arb(0)]for i in range(N)];need(z[-3][0]>0 and z[-2][0]>0,'whole actual positive rates')
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
 N=len(weights);need(1<=N<=705 and len(J)==3 and len(residual)==4 and len(error)==N,'full polynomial dimensions')
 need(C0.nrows()==C0.ncols()==N and all(v.is_exact()and v.is_finite()for row in C0.tolist()for v in row),'finite exact C0 proposal')
 need(all(v.nrows()==v.ncols()==N and finite(v)for v in J)and all(v.nrows()==N and v.ncols()==1 and finite(v)for v in residual),'full finite native coefficients')
 need(all(v.is_exact()and v.is_finite()and v>0 for v in weights+[radius,h]),'positive exact proof scales');radii=[radius*w for w in weights]
 for e,r in zip(error,radii,strict=True):need(e.is_finite()and e.lower()<=-r and e.upper()>=r,'actual acquired correction domain covers requested radii')
 C1=mid(-C0*mid(J[1])*C0);C=[C0,C1];emit('inverse_linear_proposal',C1);tick()
 D=[arb_mat(N,N)for _ in range(4)];D[0]=M.ident(N)
 for i in range(2):
  for j in range(3):D[i+j]-=C[i]*J[j];tick()
 forcing=[arb_mat(N,1)for _ in range(5)]
 for i in range(2):
  for j in range(4):forcing[i+j]-=C[i]*residual[j]
 powers=[arb(1),arb(0,h),arb(h*h/2,h*h/2),arb(0,h*h*h),arb(h*h*h*h/2,h*h*h*h/2)]
 wholeD=sum((v*powers[k]for k,v in enumerate(D)),arb_mat(N,N));wholeF=sum((v*powers[k]for k,v in enumerate(forcing)),arb_mat(N,1));E=arb_mat([[v]for v in error]);image=wholeF+wholeD*E;tick()
 need(finite(wholeD)and finite(image),'finite full polynomial image');bound=max((sum((abs(wholeD[i,j]).upper()*weights[j]for j in range(N)),arb(0))/weights[i]).upper()for i in range(N));checks=[bool(abs(image[i,0]).upper()<radii[i])for i in range(N)]
 for k,v in enumerate(D):emit(f'defect_coefficient_{k}',v)
 for k,v in enumerate(forcing):emit(f'forcing_coefficient_{k}',v)
 endpoints=[]
 for d in(-h,h):
  rr=sum((residual[k]*(d**k)for k in range(4)),arb_mat(N,1));endpoints.append(-(C0+C1*d)*rr+wholeD*E)
 report=dict(passed=bool(bound.is_finite()and bound<1 and all(checks)),dimension=N,contraction=ball_json(bound),coordinate_inclusions=checks,radii=vector_json(radii),image_correction=vector_json([image[i,0]for i in range(N)]),endpoint_corrections=[vector_json([e[i,0]for i in range(N)])for e in endpoints],scope='Independent full noncommuting affine-preconditioner/Taylor convolution with common parameter parity. Native derivative-domain and original known-root authority remain caller obligations; no complementary stability claim.')
 emit('correlated_root',report);return report
