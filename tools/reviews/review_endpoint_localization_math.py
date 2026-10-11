"""Independent fixed-gauge known-root localization reduction; no map calls."""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import vector_json,ball_json
import review_joined_period18_math as M
from review_localization_math import original_ball,original_matrix,eq,exact

def contains(outer,inner,why):
 a,b=M.interval(outer);c,d=M.interval(inner);M.need(a<=c<=d<=b,why)

def audit(parent,start,stages,result,precision=256,fresh_anchor=None):
 p,n=parent['statement']['period'],parent['statement']['phase_dimension'];N=p*n+1
 M.need(parent['passed'] and all(parent['gates'].values()),'qualified inherited root')
 M.need(start['old_parameter']==parent['exact_parameter_endpoints'][1] and start['old_normal']==parent['normal'] and start['old_origin']==parent['origin'],'unchanged exact gauge')
 M.need(result['passed'] and result['scientific_claim'] and result['iterations']==3 and result['precision']==256,'fixed complete localization')
 M.need(result['statement']==parent['statement'] and result['parent_gates']==parent['gates'],'inherited native statement')
 nn=list(map(Fraction,parent['normal']));oo=list(map(Fraction,parent['origin']));z=[M.point(v) for v in start['center']]
 M.need(len(z)==len(nn)==len(oo)==len(start['initial_box'])==N,'full native dimensions')
 for a,b in zip(start['initial_box'],parent['endpoint_root_enclosures'][1]):contains(a,b,'entire known endpoint retained')
 for i in range(N):M.need(z[i]==M.pair(start['initial_box'][i]['mid']),'exact midpoint center')
 def jac(records,blocks):
  saved_records=records
  A=original_matrix(records);M.need(A.nrows()==A.ncols()==N and len(blocks)==p,'full bordered Jacobian')
  assembled=arb_mat(N,N)
  for j,records in enumerate(blocks):
   J=original_matrix(records);M.need(J.nrows()==J.ncols()==n,'native block shape')
   for a in range(n):
    for b in range(n):assembled[j*n+a,j*n+b]+=J[a,b]
    assembled[j*n+a,((j+1)%p)*n+a]-=1
    assembled[j*n+a,N-1]=A[j*n+a,N-1]
  for i in range(N):assembled[N-1,i]=exact(nn[i])
  eq(assembled,saved_records,'native assembly with source-bound rate partial')
  return A
 summaries=[]
 with ctx.workprec(precision):
  C=original_matrix(start['preconditioner']);G=original_matrix(start['center_residual'])
  M.need(C.nrows()==C.ncols()==N and G.nrows()==N and G.ncols()==1,'center/preconditioner shape')
  for row in start['preconditioner']:
   for v in row:M.point(v)
  gauge=sum(nn[i]*(z[i]-oo[i]) for i in range(N))-Fraction(start['old_parameter'])
  glo,ghi=M.interval(start['center_residual'][-1][0]);M.need(glo<=gauge<=ghi,'exact center gauge residual enclosed')
  jac(start['center_jacobian'],start['center_phase_jacobians'])
  originalE=arb_mat([[original_ball(v)-exact(z[i])]for i,v in enumerate(start['initial_box'])]);eq(originalE,start['initial_error'],'full recentering')
  if fresh_anchor is not None:
   M.need(len(fresh_anchor)==N,'fresh full anchor')
   for a,b in zip(start['initial_box'],fresh_anchor):contains(a,b,'fresh audited endpoint within initial domain')
   E=arb_mat([[original_ball(v)-exact(z[i])] for i,v in enumerate(fresh_anchor)])
  else:E=originalE
  saved_prior=start['initial_error']
  for k,row in enumerate(stages):
   M.need(row['iteration']==k and row['prior_error']==saved_prior,'ordered retained correction chain')
   M.need(len(row['derivative_domain'])==N,'complete derivative domain')
   for i in range(N):
    lo,hi=M.interval(row['prior_error'][i][0]);a,b=M.interval(row['derivative_domain'][i])
    M.need(a<=z[i]+min(0,lo) and z[i]+max(0,hi)<=b,'whole mean-value segment coverage')
    elo,ehi=M.interval(ball_json(E[i,0]));M.need(a<=z[i]+min(0,elo) and z[i]+max(0,ehi)<=b,'fresh mean-value segment coverage')
   M.need(M.interval(row['derivative_domain'][-1])[0]>0,'positive hidden rate domain')
   J=jac(row['jacobian'],row['phase_jacobians']);D=M.ident(N)-C*J;K=-C*G+D*E
   updated=arb_mat([[E[i,0].intersection(K[i,0])]for i in range(N)])
   if precision==256:
    eq(E,row['prior_error'],'exact prior error');eq(D,row['defect'],'defect');eq(K,row['image'],'containment image');eq(updated,row['retained_error'],'outward intersection')
   summaries.append({'iteration':k,'max_radius':ball_json(max(updated[i,0].rad().upper()for i in range(N)))})
   E=updated;saved_prior=row['retained_error']
  M.need(len(summaries)==3,'all three saved intersections')
  final=[exact(z[i])+E[i,0]for i in range(N)]
  if precision==256:eq(arb_mat([[v]for v in final]),[[v]for v in result['final_state_box']],'final inherited root enclosure')
  return {'passed':True,'precision':precision,'intersections':summaries,'final_state_box':vector_json(final),'exact_gauge':start['old_parameter'],'scope':'same already proved primitive root; intersection containment, no new existence/attraction or interval extension; source-bound analytic derivatives'}
