"""Bounded finite signed native predictor refresh; no proof from proposals."""
from flint import arb,arb_mat
from neural_shrimp.rigorous_eigenpair_locus import assemble,proposal,require,finite
from neural_shrimp.rigorous_eigenpair_path import path_coefficients
from neural_shrimp.rigorous_orbits import vector_json,matrix_json,midpoint_matrix


def refresh_signed_proposal(model,initial,*,period,sigma,normal,origin,parameter,newton_steps=2,tick=lambda:None,observe=lambda k,v:None):
 n=model.n;N=2*n*period+2
 require(type(newton_steps)is int and 0<=newton_steps<=2 and type(sigma)is int and sigma in(-1,1),'fixed signed Newton proposal budget')
 require(all(len(v)==N for v in [initial,normal,origin])and all(v.is_finite()and v.is_exact()for v in initial+normal+origin+[parameter]),'exact full signed native inputs')
 center=list(initial);rhs=arb_mat([[arb(i==N-1)]for i in range(N)])
 for step in range(newton_steps+1):
  observe('signed_proposal_input',dict(iteration=step,center=vector_json(center),normal=vector_json(normal),origin=vector_json(origin),parameter=vector_json([parameter])[0],period=period,sigma=sigma))
  value,jac,local=assemble(model,center,period,sigma,normal,origin,parameter,tick,observe)
  tick();pre,small=proposal(jac,local,sigma);require(finite(pre),'finite signed inverse proposal');correction=pre*value
  observe('signed_proposal_return',dict(iteration=step,value=matrix_json(value),correction=matrix_json(correction),condensed_border=matrix_json(small),final_center=step==newton_steps))
  if step<newton_steps:
   require(all(abs(correction[i,0]).upper()<max(arb(1),abs(center[i]).upper())/8 for i in range(N)),'bounded signed proposal correction')
   center=[(center[i]-correction[i,0]).mid()for i in range(N)]
 tangent=midpoint_matrix(pre*rhs);slope=[tangent[i,0]for i in range(N)]
 observe('signed_affine_proposal_input',dict(center=vector_json(center),slope=vector_json(slope),normal=vector_json(normal),origin=vector_json(origin),parameter=vector_json([parameter])[0]))
 affine=path_coefficients(model,center,slope,[arb(0)]*N,arb(0),period,sigma,normal,origin,parameter,tick,observe)
 tick();curve=-pre*arb_mat([[2*affine[i,2]]for i in range(N)]);curvature=[curve[i,0].mid()for i in range(N)]
 record=dict(center=vector_json(center),slope=vector_json(slope),curvature=vector_json(curvature),preconditioner_exact=[[[int(v)for v in x.mid().man_exp()]for x in row]for row in pre.tolist()],newton_steps=newton_steps,native_calls=period*(newton_steps+2),algebra_ticks=newton_steps+2,scope='Finite predictor/inverse proposals only. Fresh full-domain signed root/primitive/known-root join remains required.')
 observe('signed_predictor_return',record);return dict(center=center,slope=slope,curvature=curvature,C0=pre,record=record)
