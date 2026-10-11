"""Independent saved scalar arithmetic for two fixed quadratic CIFAR steps."""
import math
import numpy as np

def need(ok,message):
 if not ok:raise ValueError(message)
def vec(a,n=None):
 a=np.asarray(a);need(a.dtype==np.float64 and a.ndim==1 and (n is None or len(a)==n)and np.isfinite(a).all(),'finite float64 vector');return a
def dot(a,b):return math.fsum(float(x)*float(y)for x,y in zip(a,b,strict=True))
def norm(a):return math.sqrt(dot(a,a))
def close(a,b):return math.isclose(float(a),float(b),rel_tol=5e-11,abs_tol=1e-24)
def path(x,p,a,alpha):return np.array([(float(xx)+alpha*float(pp))+(alpha*alpha)*float(aa)for xx,pp,aa in zip(x,p,a,strict=True)],dtype=np.float64)
def arithmetic(x,g,origin,p,hp,pred,a,ha,predha,states,gradients):
 x=vec(x);n=len(x);g,origin,p,hp,pred,a,ha,predha=[vec(v,n)for v in(g,origin,p,hp,pred,a,ha,predha)]
 need(len(states)==len(gradients)==2,'both fixed points');ng=norm(g);pn=norm(p);scale=max(1.,norm(origin));need(ng>0 and pn>0,'nonzero tangent/gradient')
 forcing=norm(g+hp)/ng;action=norm(pred-hp)/max(ng,norm(hp),1e-12);slope=dot(g,hp)
 need(forcing<=.5 and action<=1e-8 and slope<0 and norm(x-origin)/scale<=.05,'old tangent qualification')
 ae=norm(ha-predha)/max(ng,norm(ha),norm(predha),1e-12);need(ae<=1e-8,'new fixed acceleration action')
 phi=dot(g,g)/2;rows=[]
 for alpha,y,gy in zip([1/32,1/64],states,gradients,strict=True):
  y=vec(y,n);gy=vec(gy,n);need(np.array_equal(y,path(x,p,a,alpha)),'all actual quadratic path coordinates')
  trust=norm(y-origin)/scale;curve=alpha*norm(a)/pn;need(trust<=.05 and curve<=.2,'original trust and bounded curvature')
  merit=dot(gy,gy)/2;bound=phi+1e-4*alpha*slope
  rows.append(dict(alpha=alpha,relative_origin_distance=trust,correction_to_primary_step=curve,admitted=True,accepted=merit<=bound,merit=merit,merit_ratio=merit/phi,armijo_bound=bound,gradient_norm=norm(gy),relative_gradient=norm(gy)/max(1.,norm(y))))
 return dict(tangent=dict(true_forcing=forcing,physical_action_relative_error=action,slope=slope),acceleration_action_relative_error=ae,trials=rows,accepted_index=next((i for i,r in enumerate(rows)if r['accepted']),None))
