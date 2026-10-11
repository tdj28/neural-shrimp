"""Exact signed-half critical family and bounded gain-parameter tube.

The caller authenticates a known exact organizer or critical root before
using its box as an identity anchor. Interval overlap never provides that
root authority. Native phases are literal; only the normalized mode may
have either recorded orientation.
"""
from flint import arb,arb_mat
from . import rigorous_clark_generalized_flip as A
from .rigorous_orbits import eye,midpoint_matrix,ball_json,vector_json,matrix_json
require=A.require

def finite(a):return all(v.is_finite()for row in a.tolist()for v in row)
def column(v):return arb_mat([[x]for x in v])

def critical(j,z,parameter,observe=lambda *_:None):
 """H=(G-x,(DG+I)q,(q.q-1)/2,g-s), z=(x,leak,q,gain)."""
 n=len(j);N=2*n+2
 require(1<=n<=16 and len(z)==N and all(len(row)==n for row in j),'complete critical family')
 require(all(v.is_finite()for v in z)and all(v.is_finite()for row in j for v in row)and z[-1]>0 and 0<z[n]<=1 and parameter.is_finite(),'actual physical critical inputs')
 v=[A.Dual.variable(t,N,k)for k,t in enumerate(z)];x,d,q,g=v[:n],v[n],v[n+1:-1],v[-1]
 paths=A.signed_path([x,q],j,g,d,observe)
 eq=[paths[0][i]-x[i]for i in range(n)]+[paths[1][i]+q[i]for i in range(n)]+[(sum((t*t for t in q),v[0]*0)-1)/2,g-parameter]
 h=column([v.f for v in eq]);J=arb_mat([list(v.g)for v in eq]);mat=arb_mat([[J[i,k]+int(i==k)for k in range(n)]for i in range(n)])
 require(finite(h)and finite(J),'finite full critical family')
 observe('critical_family_return',dict(parameter=ball_json(parameter),input=vector_json(z),value=matrix_json(h),jacobian=matrix_json(J),signed_monodromy=matrix_json(mat)))
 return h,J,dict(monodromy=mat)

def anchor(box,center,slope,parameter,halfwidth,radii):
 """Project a supplied known root through its exact physical gain."""
 N=len(center);require(all(len(v)==N for v in(box,slope,radii))and N>=2,'full known-root projection')
 d=box[-1]-parameter;error=[box[i]-center[i]-slope[i]*d for i in range(N)]
 inside=bool(abs(d).upper()<halfwidth);checks=[bool(abs(v).upper()<radii[i])for i,v in enumerate(error)]
 return dict(passed=bool(inside and all(checks)),parameter_inside=inside,coordinates=sum(checks),coordinate_inclusions=checks,parameter_offset=ball_json(d),error=vector_json(error),scope='Conditional membership of the caller-authenticated exact critical root, with parameter equal to its actual gain; no existence follows from this projection alone.')

def tube(evaluate,center,parameter,halfwidth,radii,*,observe=lambda *_:None,tick=lambda:None):
 """Three exact acquisitions; mean-value residual along a finite predictor.

For |d|<=h, p(d)=c+v*d. Residual is enclosed by
H(c,s0)+(J_predictor*v-e_gain)*d. The separate whole-tube
Jacobian encloses every p(d)+E before the fixed C contraction test.
"""
 N=len(center);require(2<=N<=34 and len(radii)==N and all(v.is_exact()and v.is_finite()for v in center+[parameter,halfwidth]+radii)and halfwidth>0 and all(v>0 for v in radii)and center[-1]==parameter,'fixed exact tube geometry')
 h0,J0,_=evaluate(center,parameter);tick();require(h0.nrows()==N and h0.ncols()==1 and J0.nrows()==J0.ncols()==N and finite(h0)and finite(J0),'full center family')
 C=midpoint_matrix(J0.inv());forcing=column([arb(0)]*(N-1)+[arb(1)]);v=[(C*forcing)[i,0].mid()for i in range(N)];v[-1]=arb(1)
 d=arb(0,halfwidth);predictor=[center[i]+v[i]*d for i in range(N)];error=[arb(0,r)for r in radii];whole=[a+b for a,b in zip(predictor,error)]
 _,Jp,_=evaluate(predictor,parameter+d);tick();_,Jw,data=evaluate(whole,parameter+d);tick()
 require(Jp.nrows()==Jp.ncols()==Jw.nrows()==Jw.ncols()==N and finite(Jp)and finite(Jw),'full predictor and whole derivative domains')
 D=eye(N)-C*Jw;residual=-C*h0-(C*Jp*column(v)-C*forcing)*d;image=residual+D*column(error)
 bound=max((sum((abs(D[i,k]).upper()*radii[k]for k in range(N)),arb(0))/radii[i]).upper()for i in range(N))
 checks=[bool(abs(image[i,0]).upper()<radii[i])for i in range(N)];passed=bool(finite(D)and finite(image)and bound.is_finite()and bound<1 and all(checks))
 report=dict(passed=passed,dimension=N,center=vector_json(center),slope=vector_json(v),parameter_center=ball_json(parameter),halfwidth=ball_json(halfwidth),radii=vector_json(radii),actual_error=vector_json(error),predictor_box=vector_json(predictor),whole_box=vector_json(whole),center_value=matrix_json(h0),center_jacobian=matrix_json(J0),predictor_jacobian=matrix_json(Jp),whole_jacobian=matrix_json(Jw),preconditioner=matrix_json(C),defect=matrix_json(D),preconditioned_residual=matrix_json(residual),image_correction=matrix_json(image),coordinate_inclusions=checks,contraction=ball_json(bound),scope='Uniform unique critical root at every physical gain in the fixed interval, conditional on complete authenticated family acquisitions. Known organizer and endpoint identities require separate strict full-coordinate joins.')
 observe('critical_tube_return',report);return whole,v,data,report

def native_transport(j,box,steps=6,observe=lambda *_:None):
 """Literal F^steps/DF^steps inside one full six-phase native cycle.

Conditional on an exact G=-F3 fixed root and its normalized critical -1
mode in box. Oddness gives G F^k=F^k G and transports that mode; normalize
by the positive norm. The caller must prove root authority and joins.
"""
 n=len(j);N=2*n+2;require(1<=n<=16 and len(box)==N and steps==6 and all(len(row)==n for row in j),'fixed literal full-six-phase return')
 x,d,g=box[:n],box[n],box[-1];q=box[n+1:-1];orbit=[list(x)];modes=[list(q)];js=[]
 require(g>0 and 0<d<1,'positive native physical parameters')
 for phase in range(6):
  t=[v.tanh()for v in orbit[-1]];J=arb_mat([[(1-d)*int(i==k)+d*g*j[i][k]*(1-t[k]*t[k])for k in range(n)]for i in range(n)]);end=[(1-d)*orbit[-1][i]+d*g*sum((j[i][k]*t[k]for k in range(n)),arb(0))for i in range(n)];mode=J*column(modes[-1]);orbit.append(end);modes.append([mode[i,0]for i in range(n)]);js.append(J)
  observe('native_transport_phase',dict(phase=phase,state=vector_json(end),jacobian=matrix_json(J),mode=vector_json(modes[-1])))
 norm=sum((v*v for v in modes[steps]),arb(0)).sqrt();require(norm.is_finite()and norm>0,'strict nonzero literal transported mode')
 qq=[v/norm for v in modes[steps]];result=orbit[steps]+[d]+qq+[g]
 require(all((orbit[6][i]-x[i]).contains(0)and(modes[6][i]-q[i]).contains(0)for i in range(n)),'known native6 state/mode consistency')
 report=dict(phase=steps,normalization=ball_json(norm),input=vector_json(box),transported=vector_json(result),orbit=[vector_json(v)for v in orbit],modes=[vector_json(v)for v in modes],native_jacobians=[matrix_json(v)for v in js],scope='Literal full-native phase6 and positive mode normalization, conditional on the exact signed critical root. No state sign/phase search; mode orientation is a separately recorded caller choice.')
 observe('native_transport_return',report);return result,report
