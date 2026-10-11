"""Full signed native eigenpair Jacobian Taylor enclosures on curved tubes.

The zero/whole-parameter acquisitions must use the identical full correction
box E. The error is held constant along the formal path, never set to zero.
The signed terminal condition is supplied literally; no phase identification.
"""
from flint import arb,arb_mat
from neural_shrimp.rigorous_gradient_path import GradientPathJet
from neural_shrimp.rigorous_eigenpair_path import PathJet
from neural_shrimp.rigorous_eigenpair_locus import require,finite
from neural_shrimp.rigorous_orbits import matrix_json,ball_json


def signed_jacobian_path(model,center,slope,curvature,error,d,period,sigma,normal,*,tick=lambda:None,observe=lambda k,v:None,retain_assembly=True):
 n=model.n;m=2*n;size=m*period+2
 require(type(period)is int and 1<=period<=18 and 1<=n<=13 and type(sigma)is int and sigma in(-1,1),'native period/dimension/sign')
 require(type(retain_assembly)is bool and all(len(v)==size for v in [center,slope,curvature,error,normal]),'full signed shooting dimensions')
 require(all(v.is_finite()and v.is_exact()for v in center+slope+curvature+normal),'exact finite predictor and gauge')
 require(d.is_finite()and all(v.is_finite()and v.contains(0)for v in error),'whole fixed correction domain')
 z=[PathJet.path(center[i]+slope[i]*d+curvature[i]*d*d/2+error[i],slope[i]+curvature[i]*d,curvature[i])for i in range(size)]
 require(z[-2].a[0]>0 and z[-1].a[0]>0,'whole actual rates positive')
 J=[arb_mat(size,size)for _ in range(3)]
 for phase in range(period):
  off=m*phase;after=m*((phase+1)%period)
  x=[GradientPathJet(PathJet(z[off+i].a,z[off+n+i].a),[int(i==j)for j in range(n+2)])for i in range(n)]
  rates=[GradientPathJet(z[size-2+i],[int(j==n+i)for j in range(n+2)])for i in range(2)]
  observe('signed_gradient_path_input',dict(phase=phase,d=ball_json(d),inputs=[v.serialized()for v in x+rates]));tick();ret=model.cusp_step(x,*rates)
  require(len(ret)==n and all(isinstance(v,GradientPathJet)and len(v.g)==n+2 for v in ret),'full signed native gradient return')
  observe('signed_gradient_path_return',dict(phase=phase,returns=[v.serialized()for v in ret]))
  require(all(a.is_finite()for v in ret for p in [v.v,*v.g]for a in p.a+p.b),'finite signed gradient coefficients')
  for k in range(3):
   for i in range(n):
    for j in range(n):
     J[k][off+i,off+j]+=ret[i].g[j].a[k]
     J[k][off+n+i,off+j]+=ret[i].g[j].b[k]
     J[k][off+n+i,off+n+j]+=ret[i].g[j].a[k]
    for a in range(2):
     J[k][off+i,size-2+a]+=ret[i].g[n+a].a[k]
     J[k][off+n+i,size-2+a]+=ret[i].g[n+a].b[k]
    J[k][off+i,after+i]-=int(k==0)
    J[k][off+n+i,after+n+i]-=(sigma if phase==period-1 else 1)*int(k==0)
   if phase==0:
    for i in range(n):J[k][size-2,n+i]=z[n+i].a[k]
 for j in range(size):J[0][size-1,j]=normal[j]
 require(all(finite(v)for v in J),'finite full signed Jacobian Taylor coefficients')
 ans=dict(d=ball_json(d),coefficient_orders=[0,1,2],shape=[size,size],sigma=sigma,derived_assembly_retained=retain_assembly)
 if retain_assembly:ans['coefficients']=[matrix_json(v)for v in J]
 observe('signed_gradient_path_assembly_return',ans);return J
