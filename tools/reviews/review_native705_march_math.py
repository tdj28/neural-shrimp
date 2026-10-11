"""Same independent half reduction on exact cells of the fixed24 known roots."""
from fractions import Fraction as F
from flint import arb,arb_mat,ctx
import review_native705_halves_math as Q
H,R,M,P,need=Q.H,Q.R,Q.M,Q.P,Q.need
ball_json,vector_json,matrix_json=Q.ball_json,Q.vector_json,Q.matrix_json
translate=Q.translate

def geometry(chart,cell):
 sc=F(chart['parameter_center']);H0=F(1,128);depth=cell['depth'];index=cell['index'];need(type(depth)is int and 0<=depth<=3 and type(index)is int and 0<=index<2**depth,'finite canonical dyadic cell')
 h=H0/2**depth;c=sc+(2*index+1-2**depth)*h;ends=[c-h,c+h]
 need(cell==dict(center=str(c),depth=depth,endpoints=list(map(str,ends)),halfwidth=str(h),index=index,shift=str(c-sc)),'literal exact cell geometry')
 need(chart['parameter_endpoints']==list(map(str,[sc-H0,sc+H0]))and sc-H0<=ends[0]<ends[1]<=sc+H0,'exact subset of same chart gauge');return c,ends

def closed_unions(rows,charts):
 need([r['ordinal']for r in charts]==list(range(7,31))and len(rows)==48,'all fixed24 original roots and48 leaves')
 result=[]
 for old in charts:
  ordinal=old['ordinal'];selected=sorted([r for r in rows if r['ordinal']==ordinal],key=lambda r:F(r['cell']['endpoints'][0]));need(selected and all(r['passed']and r['same_root']and r['complement_dimension']==11 for r in selected),'all leaves independently passed')
  for r in selected:geometry(dict(parameter_center=old['settings']['parameter'],parameter_endpoints=old['parameter_endpoints']),r['cell'])
  ends=[[F(x)for x in r['cell']['endpoints']]for r in selected];need(ends[0][0]==F(old['parameter_endpoints'][0])and ends[-1][1]==F(old['parameter_endpoints'][1])and all(a[1]==b[0]for a,b in zip(ends,ends[1:])),'closed exact union without gaps or overlapping interiors')
  result.append(dict(ordinal=ordinal,passed=True,parameter_endpoints=old['parameter_endpoints'],leaves=len(selected),same_gauge_within_original_chart=True,same_gauge_with_predecessor=old['same_gauge'],independent_original_root=old['independent']))
 return result

def localize(chart,old_paths,fresh,C,cell,tick=lambda:None):
 center,ends=geometry(chart,cell);need(chart['schema']=='neural-shrimp-native-unit-pair-pal-v1'and(chart['dimension'],chart['period'],chart['shooting_dimension'])==(13,18,705),'full native705 chart')
 need(chart['settings']=={'precision':256,'newton_steps':2,'halfwidth_exponent':-7,'radius_exponent':-10},'exact original known rotated chart')
 c,v,a=[[R.exact(M.point(z))for z in chart[k]]for k in ('center','slope','curvature')];w=[R.exact(F(z))for z in chart['weights']];nn=[R.exact(F(z))for z in chart['normal']];oo=[R.exact(F(z))for z in chart['origin']];n=705
 need(all(len(z)==n for z in(c,v,a,w,nn,oo))and all(x>0 for x in w),'all705 exact coefficients')
 need([M.point(z)for z in chart['coordinate_radii']]==[F(1,1024)*F(z)for z in chart['weights']],'literal inherited coordinate radii')
 sc=R.exact(F(chart['parameter_center']));oldh=arb((1,-7));h=R.exact(F(cell['halfwidth']));newsc=R.exact(center);shift=newsc-sc
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
 result=dict(fresh_center=data['c'],fresh_domain=data['domain'],C=C,D=D,kappa=k,weights=w,center=centerbox,whole=whole,error=error,segment=segment,pprime=pprime,rp=data['rp'],normal=nn,origin=oo,parameter=newsc,halfwidth=h)
 report=dict(passed=True,precision=ctx.prec,dimension=705,own_intersections=6,parameter_endpoints=list(map(str,ends)),weighted_neumann_bound=ball_json(k),center=vector_json(centerbox),whole=vector_json(whole),error=vector_json(error),segment=vector_json(segment),predictor_tangent=vector_json(pprime),residual_derivative=matrix_json(data['rp']),scope='Independent exact restriction of same previously proved root, original Taylor residual and fresh-domain full705 derivative reconstruction. No new root acquisition.')
 return result,report
