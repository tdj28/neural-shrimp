"""Independent saved point residual and exact-domain known-root localization.

The original existence proof is external authority. These reductions consume
source-authenticated native MixedJet returns but never import the producer's
refinement or signed-shooting assembly. Arb/native derivatives remain shared.
"""
from fractions import Fraction as F
from flint import arb,arb_mat
import review_signed_correlated as S
import review_known_endpoint_math as L
from neural_shrimp.rigorous_orbits import ball_json,vector_json
R,M,U,need=S.R,S.M,S.U,S.need

def fixed_domain(chart,domain,known,point,parameter):
 n=len(point);need(n>0 and all(len(v)==n for v in(known,domain['error'],chart['center'],chart['slope'],chart['curvature'])),'all fixed-parameter coordinates')
 d=F(parameter)-F(chart['parameter_center']);lo,hi=map(F,chart['parameter_endpoints']);need(lo<=F(parameter)<=hi,'covered exact parameter');bounds=[]
 for i in range(n):
  p=M.point(chart['center'][i])+M.point(chart['slope'][i])*d+M.point(chart['curvature'][i])*d*d/2;a,b=M.interval(domain['error'][i]);lower,upper=p+a,p+b
  for z in(known[i],point[i]):
   x,y=M.interval(z);need(lower<=x<=y<=upper,'literal original fixed-parameter domain')
  bounds.append(ball_json(arb(R.exact((lower+upper)/2),R.exact((upper-lower)/2))))
 return bounds

def point_residual(point,normal,origin,parameter,events,n,p,sigma=-1,tick=lambda:None):
 N=2*n*p+2;need(1<=n<=13 and 1<=p<=18 and sigma in(-1,1)and all(len(v)==N for v in(point,normal,origin))and len(events)==p,'full signed point geometry')
 need(all(z.is_exact()and z.is_finite()for z in point+normal+origin)and point[-2]>0 and point[-1]>0,'exact actual point and physical rates');out=[]
 for phase,event in enumerate(events):
  off=2*n*phase;nxt=2*n*((phase+1)%p);inputs=event['inputs'];returns=event['returns'];need(event['phase']==phase and len(inputs)==n+2 and len(returns)==n,'literal complete native point phase')
  values=point[off:off+n]+point[-2:];directions=point[off+n:off+2*n]+[arb(0),arb(0)]
  for i,z in enumerate(inputs):
   need(set(z)=={'kind','f','g','d','m'}and z['kind']=='mixed'and len(z['g'])==len(z['m'])==n+2,'full native MixedJet input')
   U.contains(z['f'],ball_json(values[i]),'actual state/rates');U.contains(z['d'],ball_json(directions[i]),'actual signed direction')
   for j in range(n+2):U.contains(z['g'][j],ball_json(arb(int(i==j))),'actual base basis');U.contains(z['m'][j],ball_json(arb(0)),'zero mixed input')
  for z in returns:need(set(z)=={'kind','f','g','d','m'}and z['kind']=='mixed'and len(z['g'])==len(z['m'])==n+2 and all(R.original_ball(v).is_finite()for v in[z['f'],z['d'],*z['g'],*z['m']]),'full finite native MixedJet return')
  out.extend(R.original_ball(z['f'])-point[nxt+i]for i,z in enumerate(returns));sign=sigma if phase==p-1 else 1
  out.extend(R.original_ball(z['d'])-sign*point[nxt+n+i]for i,z in enumerate(returns));tick()
 out.append((sum((v*v for v in point[n:2*n]),arb(0))-1)/2);out.append(sum((normal[i]*(point[i]-origin[i])for i in range(N)),arb(0))-parameter)
 return arb_mat([[v]for v in out])

def localize(C,J,residual,point,known,chart,domain,parameter,weights):
 whole=fixed_domain(chart,domain,known,vector_json(point),parameter)
 result=L.localize(C,J,residual,point,known,whole,weights)
 fixed_domain(chart,domain,result['box'],vector_json(point),parameter)
 return result
