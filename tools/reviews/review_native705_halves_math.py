"""Independent closed-half restriction and fresh-domain localization of known705 roots.

No producer restriction/localization routine is imported. Exact old root
identity is a caller prerequisite; the actual native acquisitions are shared.
"""
from fractions import Fraction as F
from flint import arb,arb_mat,ctx
import review_native705_complement_math as H
R,M,P,need=H.R,H.M,H.P,H.need
ball_json,vector_json,matrix_json=H.ball_json,H.vector_json,H.matrix_json


def geometry(unit):
 need(type(unit)is int and unit in(1,2),'exact two closed halves')
 center=F(351+64*(unit-1),4096);h=F(1,128)
 return center,[center-h,center+h]


def translate(c,v,a,w,H0,rho,shift,h,cc,wc):
 n=len(c);need(n>0 and len(v)==len(a)==len(w)==n,'complete predictor')
 need(all(x.is_exact()and x.is_finite()for x in c+v+a+w+[H0,rho,shift,h])and H0>0 and rho>0 and h>0 and all(x>0 for x in w),'exact finite predictor/scales')
 need(abs(shift)+h<=H0,'closed subset of proved parameter interval')
 need(cc.nrows()==wc.nrows()==n and cc.ncols()==wc.ncols()==4 and H.finite(cc)and H.finite(wc),'all original cubic residual coefficients')
 newc=[c[i]+v[i]*shift+a[i]*shift**2/2 for i in range(n)];newv=[v[i]+a[i]*shift for i in range(n)]
 need(all(x.is_exact()for x in newc+newv),'no rounded predictor recentering')
 d=arb(0,h);delta=shift+d;r=[rho*x for x in w]
 need(all(x.is_exact()for x in r),'exact inherited coordinate radii')
 pred=[newc[i]+newv[i]*d+a[i]*d*d/2 for i in range(n)]
 domain=[pred[i]+arb(0,r[i])for i in range(n)]
 # Two separate Taylor theorems at the OLD center. Never differentiate a remainder.
 residual=arb_mat([[cc[i,0]+cc[i,1]*delta+cc[i,2]*delta*delta+wc[i,3]*delta**3]for i in range(n)])
 derivative=arb_mat([[cc[i,1]+2*cc[i,2]*delta+3*wc[i,3]*delta*delta]for i in range(n)])
 return dict(c=newc,v=newv,a=a,w=w,h=h,r=r,pred=pred,domain=domain,residual=residual,rp=derivative,delta=delta)


def mixed_direction(inp,out,n=13):
 """Rebuild a held-q/r direction block directly from actual MixedJet returns."""
 need(inp['kind']==out['kind']=='full_jacobian'and inp['stage']==out['stage']and type(inp['phase'])is int and 0<=inp['phase']<18 and inp['direction']in('q','r'),'actual mixed acquisition identity')
 x,y=inp['inputs'],out['returns'];need(len(x)==n+2 and len(y)==n,'full acquired MixedJet dimensions')
 for i,z in enumerate(x):
  need(set(z)=={'f','g','d','m'}and len(z['g'])==len(z['m'])==n+2,'complete actual input jet')
  need(all(M.point(v)==int(k==i)for k,v in enumerate(z['g']))and all(M.point(v)==0 for v in z['m']),'actual independent base and held-mode seeds')
  if i>=n:need(M.point(z['d'])==0,'zero rate directional seed')
 for z in y:
  need(set(z)=={'f','g','d','m'}and len(z['g'])==len(z['m'])==n+2,'complete actual returned jet')
  need(all(R.original_ball(v).is_finite()for v in [z['f'],z['d']]+z['g']+z['m']),'finite actual derivative values')
 zero=ball_json(arb(0));block=[];rates=[]
 for z in y:block.append(z['g'][:n]+[zero]*n);rates.append(z['g'][n:])
 for z in y:block.append(z['m'][:n]+z['g'][:n]);rates.append(z['m'][n:])
 return dict(phase=inp['phase'],direction=inp['direction'],input=[z['f']for z in x[:n]]+[z['d']for z in x[:n]]+[z['f']for z in x[n:]],value=[[z['f']]for z in y]+[[z['d']]for z in y],block=block,rates=rates)


def localize(chart,old_paths,fresh,C,unit,tick=lambda:None):
 center,ends=geometry(unit);need(chart['schema']=='neural-shrimp-native-unit-pair-pal-v1'and(chart['dimension'],chart['period'],chart['shooting_dimension'])==(13,18,705),'full native705 chart')
 need(chart['parameter_center']=='383/4096'and chart['parameter_endpoints']==['319/4096','447/4096']and chart['settings']=={'precision':256,'newton_steps':2,'halfwidth_exponent':-6,'radius_exponent':-10},'exact original known wide chart')
 c,v,a=[[R.exact(M.point(z))for z in chart[k]]for k in ('center','slope','curvature')];w=[R.exact(F(z))for z in chart['weights']];nn=[R.exact(F(z))for z in chart['normal']];oo=[R.exact(F(z))for z in chart['origin']];n=705
 need(all(len(z)==n for z in(c,v,a,w,nn,oo))and all(x>0 for x in w),'all705 exact coefficients')
 need([M.point(z)for z in chart['coordinate_radii']]==[F(1,1024)*F(z)for z in chart['weights']],'literal inherited coordinate radii')
 sc=R.exact(F(383,4096));oldh=arb((1,-6));h=arb((1,-7));newsc=R.exact(center);shift=newsc-sc
 need(len(old_paths)==108 and len(fresh)==72,'complete old108 path and fresh72 MixedJet groups')
 cc=P.path_coefficients(c,v,a,arb(0),old_paths[36:72],13,18,nn,oo,sc);tick();wc=P.path_coefficients(c,v,a,arb(0,oldh),old_paths[72:],13,18,nn,oo,sc);tick()
 data=translate(c,v,a,w,oldh,arb((1,-10)),shift,h,cc,wc);del cc,wc
 g0,_=P.rebuild(data['c'],fresh[:36],13,18,nn,oo,newsc);tick();_,J=P.rebuild(data['domain'],fresh[36:],13,18,nn,oo,newsc+arb(0,h));tick()
 need(C.nrows()==C.ncols()==n and all(z.is_exact()and z.is_finite()for row in C.tolist()for z in row),'exact finite proposed full705 inverse')
 D=M.ident(n)-C*J;k=max((sum((abs(D[i,j]).upper()*w[j]for j in range(n)),arb(0))/w[i]).upper()for i in range(n));need(k.is_finite()and 0<=k<1,'strict independently recomputed new-domain Neumann bound')
 initial=arb_mat([[arb(0,r)]for r in data['r']]);ce=H.H.enclosure(D,-C*g0,w,k,initial,iterations=6);tick();we=H.H.enclosure(D,-C*data['residual'],w,k,initial,iterations=6);tick()
 centerbox=[data['c'][i]+ce[i,0]for i in range(n)];whole=[data['pred'][i]+we[i,0]for i in range(n)];error=[we[i,0]for i in range(n)];pprime=[data['v'][i]+a[i]*arb(0,h)for i in range(n)];segment=[data['pred'][i]+arb(0,abs(error[i]).upper())for i in range(n)]
 # Mean-value segments stay in the actual recentered root-domain expression.
 for outer,inner in zip(data['domain'],segment,strict=True):H.H.contains(ball_json(outer),inner)
 need(all(z.is_finite()for z in centerbox+whole+segment)and all(z>0 for z in centerbox[-3:]+whole[-3:]+segment[-3:]),'same known root with positive both rates/angle')
 result=dict(C=C,D=D,kappa=k,weights=w,center=centerbox,whole=whole,error=error,segment=segment,pprime=pprime,rp=data['rp'],normal=nn,origin=oo,parameter=newsc,halfwidth=h)
 report=dict(passed=True,precision=ctx.prec,dimension=705,own_intersections=6,parameter_endpoints=list(map(str,ends)),weighted_neumann_bound=ball_json(k),center=vector_json(centerbox),whole=vector_json(whole),error=vector_json(error),segment=vector_json(segment),predictor_tangent=vector_json(pprime),residual_derivative=matrix_json(data['rp']),scope='Independent exact restriction of same previously proved root, original Taylor residual and fresh-domain full705 derivative reconstruction. No new root acquisition.')
 return result,report
