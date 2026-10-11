"""Saved same-state orbit residual/line/secant identities, never a new model call."""
import math
import numpy as np
from .cifar_phase_scaling import dot,transform,scales,diagonal

def norm(v):return math.sqrt(dot(v,v))
def need(v,m):
    if not v:raise ValueError(m)
def assess(theta,one,two,p,ap,z,az,eta,ridge):
    theta,one,two,p,ap,z,az=[np.asarray(v,dtype=np.float64)for v in [theta,one,two,p,ap,z,az]];n=len(theta)
    need(n>0 and all(v.shape==(n,)and np.isfinite(v).all()for v in [theta,one,two,p,ap])and z.shape==az.shape==(2*n,)and np.isfinite(z).all()and np.isfinite(az).all(),'finite complete same-state arrays')
    need(np.array_equal(p,z[:n])and eta>0 and ridge>0,'same full physical direction and positive map constants')
    r=two-theta;delta=one-theta;rr=dot(r,r);aa=dot(ap,ap);slope=dot(r,ap)
    need(rr>0 and aa>0 and dot(delta,delta)>0,'nonzero residual/action/phase difference')
    beta=-slope/aa;clipped=max(0.,min(1.,beta));best=r+clipped*ap;projection=dot(r,delta)/dot(delta,delta);transverse=r-projection*delta
    # Exact identities of the recorded GD phase maps; no midpoint Hessian claim.
    g0=-delta/eta;g1=(delta-r)/eta;secant=g1-g0;curvature=dot(delta,secant)/dot(delta,delta);perp=secant-curvature*delta;threshold=2/eta
    residual=az-np.r_[np.zeros(n),-r];phase=transform(residual);scale=max(1.,norm(theta));weights,_=scales(eta,ridge)
    return dict(relative_closure=norm(r)/scale,relative_phase_separation=norm(delta)/scale,physical_forcing=norm(r+ap)/norm(r),line=dict(slope=slope,action_squared=aa,unconstrained_beta=beta,bounded_beta=clipped,best_predicted_forcing=norm(best)/norm(r),predicted_relative_closure=norm(best)/scale,relative_step=clipped*norm(p)/scale,forcing_point5_reachable=bool(norm(best)/norm(r)<=.5),scope='Best scalar in[0,1] of this saved p under its actual same-state linear action. Nonlinear merit untested.'),phase_residual=dict(mean_norm=norm(phase[:n]),alternating_norm=norm(phase[n:]),mean_energy_fraction=dot(phase[:n],phase[:n])/dot(residual,residual),scaled_relative_norm=norm(diagonal(phase,weights))/norm(diagonal(transform(np.r_[np.zeros(n),-r]),weights))),cycle_geometry=dict(residual_parallel_to_phase_fraction=projection**2*dot(delta,delta)/rr,residual_transverse_fraction=dot(transverse,transverse)/rr,mean_endpoint_gradient_norm=norm(g0+g1)/2,endpoint_gradient_difference_norm=norm(secant),secant_curvature=curvature,flip_linear_threshold=threshold,curvature_to_threshold=curvature/threshold,relative_secant_transverse_defect=norm(perp)/norm(secant),scope='Endpoint-gradient secant inferred from the two saved GD maps. Neither actual midpoint gradient/Hessian nor a proved flip/eigenvalue/branch.'),available_full_period_directions=1,scope='All identities use one current state. No old Jacobian actions transferred, no new point/periodic orbit or stability assertion.')
