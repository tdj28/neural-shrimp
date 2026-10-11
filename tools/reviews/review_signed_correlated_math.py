"""Independent signed native gradient/path assembly; no producer imports.

Reuse independently authored generic polynomial/anchor reductions unchanged.
The held correction domain is present in both zero/whole Jacobian passes.
"""
from flint import arb,arb_mat
import review_correlated705_math as H
P,R,U,need=H.P,H.R,H.U,H.need
from neural_shrimp.rigorous_orbits import ball_json

def jacobian(c,s,a,error,d,events,n,p,sigma,normal,tick=lambda:None):
 N=2*n*p+2;m=2*n;need(1<=n<=13 and 1<=p<=18 and sigma in(-1,1)and all(len(v)==N for v in(c,s,a,error,normal))and len(events)==p,'full signed gradient geometry')
 need(all(v.is_exact()and v.is_finite()for v in c+s+a+normal)and all(v.is_finite()and v.contains(0)for v in error),'exact predictor/gauge and full constant error')
 z=[[c[i]+s[i]*d+a[i]*d*d/2+error[i],s[i]+a[i]*d,a[i]/2,arb(0)]for i in range(N)];need(z[-2][0]>0 and z[-1][0]>0,'whole physical rates');J=[arb_mat(N,N)for _ in range(3)]
 for phase,event in enumerate(events):
  off=phase*m;nxt=((phase+1)%p)*m;need(event['phase']==phase,'literal signed phase order');U.contains(event['d'],ball_json(d),'same acquired parameter domain');inputs=event['inputs'];rets=event['returns'];need(len(inputs)==n+2 and len(rets)==n,'complete gradient input/output')
  expected=[[z[off+i],z[off+n+i]]for i in range(n)]+[[z[-2],[arb(0)]*4],[z[-1],[arb(0)]*4]]
  for i,(old,want)in enumerate(zip(inputs,expected,strict=True)):
   need(set(old)=={'value','gradient'}and len(old['gradient'])==n+2,'full base input gradient');H.check_path(old['value'],want,'whole predictor/error/mode/rates')
   for j,g in enumerate(old['gradient']):H.check_path(g,H.scalar_path(arb(int(i==j))),'independent base basis with held mode')
  rows=[]
  for item in rets:
   need(set(item)=={'value','gradient'}and len(item['gradient'])==n+2,'complete returned base gradients');rows.append(dict(value=H.as_path(item['value']),gradient=[H.as_path(g)for g in item['gradient']]))
  for k in range(3):
   for i in range(n):
    for j in range(n):
     J[k][off+i,off+j]+=rows[i]['gradient'][j][0][k];J[k][off+n+i,off+j]+=rows[i]['gradient'][j][1][k];J[k][off+n+i,off+n+j]+=rows[i]['gradient'][j][0][k]
    for rate in range(2):
     J[k][off+i,N-2+rate]+=rows[i]['gradient'][n+rate][0][k];J[k][off+n+i,N-2+rate]+=rows[i]['gradient'][n+rate][1][k]
    J[k][off+i,nxt+i]-=int(k==0);J[k][off+n+i,nxt+n+i]-=(sigma if phase==p-1 else 1)*int(k==0)
  tick()
 for k in range(3):
  for i in range(n):J[k][N-2,n+i]=z[n+i][k]
 for i in range(N):J[0][N-1,i]=normal[i]
 need(all(H.finite(v)for v in J),'finite complete signed Jacobian');return J

def residual(c,s,a,d,events,n,p,sigma,normal,origin,sc,tick=lambda:None):
 N=2*n*p+2;m=2*n;need(all(len(v)==N for v in(c,s,a,normal,origin))and len(events)==p and sigma in(-1,1),'complete signed path geometry')
 z=[[c[i]+s[i]*d+a[i]*d*d/2,s[i]+a[i]*d,a[i]/2,arb(0)]for i in range(N)];rows=[]
 for phase,event in enumerate(events):
  off=phase*m;nxt=((phase+1)%p)*m;need(event['phase']==phase,'literal path phase');U.contains(event['d'],ball_json(d),'same path parameter domain');need(len(event['inputs'])==n+2 and len(event['returns'])==n,'entire native path input/return')
  expected=[[z[off+i],z[off+n+i]]for i in range(n)]+[[z[-2],[arb(0)]*4],[z[-1],[arb(0)]*4]]
  for old,want in zip(event['inputs'],expected,strict=True):H.check_path(old,want,'actual zero-error residual path')
  rets=[H.as_path(v)for v in event['returns']]
  for part in(0,1):
   sign=sigma if part==1 and phase+1==p else 1
   for i in range(n):rows.append([rets[i][part][k]-sign*z[nxt+part*n+i][k]for k in range(4)])
  tick()
 square=[sum((P.mul(z[n+i],z[n+i])[k]for i in range(n)),arb(0))for k in range(4)];norm=P.scale(P.add(square,P.const(arb(-1))),arb(1)/2)
 gauge=[sum((normal[i]*(z[i][k]-(origin[i]if k==0 else 0))for i in range(N)),arb(0))-(sc+d if k==0 else arb(int(k==1)))for k in range(4)]
 out=arb_mat(rows+[norm,gauge]);need(H.finite(out),'finite full signed residual');return out

def anchor(box,c,s,a,normal,origin,sc,h,radii,exact_parameter=None):
 projection=R.exact(exact_parameter)if exact_parameter is not None else sum((normal[i]*(R.original_ball(box[i])-origin[i])for i in range(len(c))),arb(0));d=projection-sc;inside=bool(abs(d).upper()<=h);checks=[bool(abs(R.original_ball(b)-c[i]-s[i]*d-a[i]*d*d/2).upper()<radii[i])for i,b in enumerate(box)]
 return dict(passed=inside and all(checks),parameter_inside=inside,coordinates=sum(checks),projection=ball_json(projection))
