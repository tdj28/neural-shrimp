"""Conditional full signed eigenpair tube with a correlated inverse proposal."""
from fractions import Fraction
from flint import arb,arb_mat
from .rigorous_signed_gradient_path import signed_jacobian_path
from neural_shrimp.rigorous_eigenpair_path import path_coefficients,quadratic_anchor
from neural_shrimp.rigorous_polynomial_inverse import correlated_inverse_data,serialize_correlated_inverse
from neural_shrimp.rigorous_eigenpair_locus import require,finite,exact_mid
from neural_shrimp.rigorous_orbits import exact,vector_json,matrix_json,midpoint_matrix


def signed_correlated_tube(model,center,slope,curvature,C0,*,period,sigma,normal,origin,parameter,halfwidth_exponent,radius_exponent,weights,anchor=None,tick=lambda:None,observe=lambda k,v:None):
 n=model.n;size=2*n*period+2
 require(type(sigma)is int and sigma in(-1,1),'literal signed native closure')
 require(all(len(v)==size for v in [center,slope,curvature,normal,origin,weights]),'full signed native proposals')
 require(all(v.is_finite()and v.is_exact()for v in center+slope+curvature+normal+origin+weights),'exact finite full proposals')
 require(C0.nrows()==C0.ncols()==size and finite(C0)and all(v.is_exact()for row in C0.tolist()for v in row),'finite exact full inverse proposal')
 require(all(type(k)is int and -100<=k<=-1 for k in [halfwidth_exponent,radius_exponent])and all(v>0 for v in weights),'bounded dyadic widths and positive weights')
 sc=exact(parameter);h=arb((1,halfwidth_exponent));rho=arb((1,radius_exponent));d=arb(0,h);require(sc.is_finite()and sc.is_exact(),'exact gauge parameter')
 radii=[rho*w for w in weights];error=[arb(0,v.upper())for v in radii]
 predictor=[center[i]+slope[i]*d+curvature[i]*d*d/2 for i in range(size)];box=[predictor[i]+error[i]for i in range(size)]
 require(box[-2]>0 and box[-1]>0,'whole actual rates positive')
 observe('signed_correlated_domain',dict(center=vector_json(center),slope=vector_json(slope),curvature=vector_json(curvature),error=vector_json(error),whole_box=vector_json(box),parameter=parameter,halfwidth_exponent=halfwidth_exponent,radius_exponent=radius_exponent,sigma=sigma))
 jzero=signed_jacobian_path(model,center,slope,curvature,error,arb(0),period,sigma,normal,tick=tick,observe=observe,retain_assembly=False)
 jwhole=signed_jacobian_path(model,center,slope,curvature,error,d,period,sigma,normal,tick=tick,observe=observe,retain_assembly=False)
 J=[jzero[0],jzero[1],jwhole[2]];C1=midpoint_matrix(-C0*midpoint_matrix(J[1])*C0)
 coeff=path_coefficients(model,center,slope,curvature,arb(0),period,sigma,normal,origin,sc,tick,observe)
 whole=path_coefficients(model,center,slope,curvature,d,period,sigma,normal,origin,sc,tick,observe)
 residual=[arb_mat([[coeff[i,k]if k<3 else whole[i,3]]for i in range(size)])for k in range(4)]
 bound=correlated_inverse_data([C0,C1],J,residual,weights,rho,h,tick=tick)
 divisors=[]
 for k in range(1,period):
  if period%k:continue
  diff=[box[2*n*k+i]-box[i]for i in range(n)];divisors.append(dict(divisor=k,differences=vector_json(diff),excluded=any(not v.contains(0)for v in diff)))
 ac=quadratic_anchor(anchor,center,slope,curvature,normal,origin,sc,h,radii,period,sigma,n)
 gates=dict(uniform_normalized_eigenpair_locus=bound['passed'],primitive_period=all(v['excluded']for v in divisors),anchor_identified=ac['passed']);endpoints=[];E=arb_mat([[v]for v in error])
 for dd in [-h,h]:
  rr=residual[0]+residual[1]*dd+residual[2]*dd*dd+residual[3]*dd*dd*dd
  ee=-(C0+C1*dd)*rr+bound['whole_defect']*E
  endpoints.append([center[i]+slope[i]*dd+curvature[i]*dd*dd/2+ee[i,0]for i in range(size)])
 result=dict(schema='neural-shrimp-correlated-signed-eigenpair-v1',passed=all(gates.values()),gates=gates,dimension=n,period=period,sigma=sigma,shooting_dimension=size,parameter_center=parameter,parameter_endpoints=[str(Fraction(parameter)-Fraction(2)**halfwidth_exponent),str(Fraction(parameter)+Fraction(2)**halfwidth_exponent)],normal=[exact_mid(v)for v in normal],origin=[exact_mid(v)for v in origin],center=vector_json(center),slope=vector_json(slope),curvature=vector_json(curvature),residual_coefficients=[matrix_json(v)for v in residual],whole_box=vector_json(box),weights=[exact_mid(v)for v in weights],settings=dict(halfwidth_exponent=halfwidth_exponent,radius_exponent=radius_exponent),bound=serialize_correlated_inverse(bound,dense=False),divisors=divisors,anchor_check=ac,endpoint_root_boxes=[vector_json(v)for v in endpoints],inverse_polynomial_encoding='exact-dyadic-pairs',inverse_polynomial_exact=[[[[int(a)for a in v.mid().man_exp()]for v in row]for row in M.tolist()]for M in [C0,C1]],scope='Conditional full normalized primitive signed-multiplier locus. Exact model/source/same-domain and known-root identity are caller obligations; critical simplicity, complementary stability, nondegeneracy and global closure are separate.')
 observe('signed_correlated_chart_return',result);return result
