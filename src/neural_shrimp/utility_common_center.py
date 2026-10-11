"""One-step paired loss contrast after subtracting its actual-center gradient."""
import math
import numpy as np
from .utility_paired_geometry import paired

def require(ok,message):
    if not ok:raise ValueError(message)

def reduce_contrast(loss_plus,loss_minus,gradient_center,v,gradient_original):
    g,v,g0=map(lambda x:np.asarray(x,dtype=float),(gradient_center,v,gradient_original))
    require(g.shape==v.shape==g0.shape and g.ndim==1 and all(np.isfinite(x).all() for x in (g,v,g0)),'finite gradient/vector shapes')
    require(type(loss_plus)in(float,int) and type(loss_minus)in(float,int) and math.isfinite(loss_plus)and math.isfinite(loss_minus),'finite scalar losses')
    dot=lambda a,b:math.fsum(float(x)*float(y)for x,y in zip(a,b,strict=True))
    delta=math.fsum([loss_plus,-loss_minus]);pred=2*dot(g,v);old=2*dot(g0,v);residual=math.fsum([loss_plus,-loss_minus,-pred]);rounding_scale=float(64*np.finfo(float).eps*max(1.,abs(loss_plus),abs(loss_minus),abs(pred)));norm=math.hypot(*v)
    require(norm>0 and all(math.isfinite(x)for x in (delta,pred,old,residual,rounding_scale)),'finite nonzero contrast')
    return dict(loss_plus=loss_plus,loss_minus=loss_minus,loss_difference_plus_minus=delta,center_first_order_prediction=pred,original_state_first_order_prediction=old,first_order_center_shift=pred-old,centered_residual=residual,absolute_centered_residual=abs(residual),arithmetic_reporting_scale=rounding_scale,contrast_above_reporting_scale=abs(delta)>rounding_scale,residual_above_reporting_scale=abs(residual)>rounding_scale,residual_fraction_of_contrast=abs(residual)/abs(delta)if abs(delta)>rounding_scale else None,contrast_vector_norm=norm,centered_residual_over_vector_norm_cubed=residual/norm**3,scope='Finite paired one-step loss minus ordinary actual-center first-order prediction. Reporting scale is heuristic float64 arithmetic, not a validated error bound or geometry effect.')

def evaluate(theta,g,rates,gradient,loss,emit=lambda k,v:None,check=lambda:None):
    t=np.asarray(theta,dtype=float);g=np.asarray(g,dtype=float);require(t.shape==g.shape and t.ndim==1 and np.isfinite(t).all(),'finite original state')
    geometry=paired(g,np.asarray(rates,dtype=float));require(geometry['qualified'],'fixed paired feasibility')
    s=geometry['scaled_gradient'];v=geometry['orthogonal_component'];z=t-(1-geometry['kappa'])*s;plus=z+v;minus=z-v
    require(all(np.isfinite(a).all()for a in (z,plus,minus)),'finite actual arms')
    emit('paired_states',dict(original=t,original_gradient=g,rates=np.asarray(rates),center=z,plus=plus,minus=minus,contrast=v,geometry=geometry))
    check();emit('operation_start',dict(kind='gradient_center',state=z));gc=np.asarray(gradient(z),dtype=float);emit('operation_return',dict(kind='gradient_center',value=gc));check();require(gc.shape==g.shape and np.isfinite(gc).all(),'finite returned gradient')
    values=[]
    for name,state in [('loss_plus',plus),('loss_minus',minus)]:
        check();emit('operation_start',dict(kind=name,state=state));value=float(loss(state));emit('operation_return',dict(kind=name,value=np.array(value)));check();require(math.isfinite(value),'finite returned loss');values.append(value)
    result=reduce_contrast(*values,gc,v,g);emit('contrast_reduction',result);check();return result
