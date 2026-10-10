"""Independent saved-ball reduction for fixed-gauge native shooting charts.

Native state Jacobians, the hidden-rate partial column and path jets are saved
source-bound enclosures. No model derivative or proposal solve is called here.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json
import review_joined_period18_math as M
from review_quadratic_connection_math import original_ball,matrix,exact,same

def chart_check(c,anchor,normal,origin,weights,precision=384):
 p,n=c['statement']['period'],c['statement']['phase_dimension'];N=p*n+1
 M.need(c['statement']['shooting_dimension']==N and c['normal']==normal and c['origin']==origin and c['exact_weights']==weights,'fixed native gauge and weights')
 sc=Fraction(c['parameter_center']);delta=Fraction(2)**c['settings']['halfwidth_exponent'];radius=Fraction(2)**c['settings']['radius_exponent']
 M.need(c['exact_parameter_endpoints']==[str(sc-delta),str(sc+delta)],'exact gauge interval')
 base=[M.point(x)for x in c['center']];S=[M.point(x[0])for x in c['predictor_slope']];T=[M.point(x[0])for x in c['predictor_curvature']];nn=list(map(Fraction,normal));oo=list(map(Fraction,origin));ww=list(map(Fraction,weights))
 M.need(all(len(v)==N for v in(base,S,T,nn,oo,ww,c['augmented_box']))and any(nn)and all(0<w<=1 and w.denominator&(w.denominator-1)==0 for w in ww),'complete exact chart geometry')
 radii=[radius*w for w in ww];M.need(M.point(c['correction_radius'])==radius and [M.point(x)for x in c['coordinate_radii']]==radii,'exact coordinate radii')
 for i in range(N):
  lo,hi=M.interval(c['augmented_box'][i]);spread=abs(S[i])*delta
  M.need(lo<=base[i]-spread+min(0,T[i])*delta*delta/2-radii[i] and base[i]+spread+max(0,T[i])*delta*delta/2+radii[i]<=hi,'whole curved predictor plus error domain')
 M.need(M.interval(c['augmented_box'][-1])[0]>0,'positive entire hidden-rate domain')
 M.need([d['divisor']for d in c['divisors']]==[k for k in range(1,p)if p%k==0],'all proper divisors')
 for key,target in [('f',sum(nn[i]*(base[i]-oo[i])for i in range(N))-sc),('d',sum(nn[i]*S[i]for i in range(N))-1),('d2',sum(nn[i]*T[i]for i in range(N))),('d3',Fraction(0))]:
  lo,hi=M.interval(c['center_path_derivatives'][key][-1][0]);M.need(lo<=target<=hi,'exact gauge Taylor coefficient '+key)
 M.need(M.interval(c['whole_path_third_derivative'][-1][0])[0]<=0<=M.interval(c['whole_path_third_derivative'][-1][0])[1],'zero gauge third derivative')
 with ctx.workprec(precision):
  saved=matrix(c['jacobian_tube']);M.need(saved.nrows()==saved.ncols()==N,'full bordered matrix')
  A=arb_mat(N,N)
  M.need(len(c['phase_jacobians'])==p,'all native blocks')
  for j,records in enumerate(c['phase_jacobians']):
   block=matrix(records);M.need(block.nrows()==block.ncols()==n,'native block dimensions')
   for a in range(n):
    for b in range(n):A[j*n+a,j*n+b]+=block[a,b]
    A[j*n+a,((j+1)%p)*n+a]-=1
    # This source-bound analytic partial is not derivable from DF alone.
    A[j*n+a,N-1]=saved[j*n+a,N-1]
  for i in range(N):M.need(M.point(c['jacobian_tube'][-1][i])==nn[i],'exact gauge border row');A[N-1,i]=exact(nn[i])
  C=matrix(c['preconditioner']);M.need(C.nrows()==C.ncols()==N,'preconditioner dimensions')
  for row in c['preconditioner']:
   for v in row:M.point(v)
  H={key:matrix(v)for key,v in c['center_path_derivatives'].items()};third=matrix(c['whole_path_third_derivative'])
  M.need(set(H)=={'f','d','d2','d3'}and all(v.nrows()==N and v.ncols()==1 for v in [*H.values(),third]),'full path jets')
  d=arb(0,exact(delta));R=H['f']+H['d']*d+H['d2']*(d*d/2)+third*(d*d*d/6);D=M.ident(N)-C*A
  E=arb_mat([[arb(0,exact(r))]for r in radii]);K=-C*R+D*E
  contraction=max((sum((abs(D[i,j]).upper()*exact(ww[j])for j in range(N)),arb(0))/exact(ww[i])).upper()for i in range(N))
  inclusion=[bool(abs(K[i,0]).upper()<exact(radii[i]))for i in range(N)];root=bool(contraction<1 and all(inclusion))
  if precision==256:
   same(A,c['jacobian_tube'],'native bordered assembly');same(R,c['predictor_residual'],'all curved Taylor terms');same(D,c['defect'],'bordered defect');same(K,c['image_correction'],'bordered image')
   M.need(inclusion==c['coordinate_inclusions'],'all235 inclusion decisions');same(arb_mat([[contraction]]),[[c['contraction']]],'weighted contraction')
  primitive=[]
  for k in range(1,p):
   if p%k:continue
   sep=[]
   for i in range(n):
    a,b=M.interval(c['augmented_box'][k*n+i]);lo,hi=M.interval(c['augmented_box'][i])
    if a-hi>0 or b-lo<0:sep.append(i)
   primitive.append({'divisor':k,'coordinates':sep})
  M.need(anchor is not None and len(anchor['state_box'])==N,'known full augmented anchor')
  # The original producer constructor rounds radius outwards. Recreate that
  # conservative arithmetic at256; the stronger join uses exact saved balls.
  B=[M.stored(v)if precision==256 else original_ball(v)for v in anchor['state_box']]
  projection=sum((exact(nn[i])*(B[i]-exact(oo[i]))for i in range(N)),arb(0))
  if 'exact_parameter'in anchor:
   parameter=exact(anchor['exact_parameter']);kind='prior exact gauge'
  else:parameter=projection;kind='correlated gauge'
  off=parameter-exact(sc);inside=bool(abs(off).upper()<=exact(delta))
  joined=[bool(abs(B[i]-(exact(base[i])+exact(S[i])*off+exact(T[i])*off*off/2)).upper()<=exact(radii[i]))for i in range(N)]
  gates={'uniform_root':root,'primitive_period':all(x['coordinates']for x in primitive),'anchor_identified':inside and all(joined)}
  if precision==256:M.need(gates==c['gates']and joined==c['anchor_check']['coordinate_inclusions']and inside==c['anchor_check']['parameter_inside'],'saved anchor and proof gate decisions')
  endpoints=[]
  for side in(-1,1):
   a=side*exact(delta);r=H['f']+H['d']*a+H['d2']*(a*a/2)+third*(a*a*a/6);image=-C*r+D*E
   end=arb_mat([[exact(base[i])+exact(S[i])*a+exact(T[i])*a*a/2+image[i,0]]for i in range(N)])
   if precision==256:same(end,[[v]for v in c['endpoint_root_enclosures'][0 if side<0 else 1]],'same-gauge endpoint enclosure')
   endpoints.append([ball_json(end[i,0])for i in range(N)])
  return {'passed':all(gates.values()),'gates':gates,'precision':precision,'contraction':ball_json(contraction),'max_image_ratio':ball_json(max((abs(K[i,0]).upper()/exact(radii[i])).upper()for i in range(N))),'anchor_kind':kind,'anchor_projection':ball_json(projection),'anchor_parameter':ball_json(parameter),'coordinate_count':N,'primitive':primitive,'fresh_endpoints':endpoints}
