"""One saved inexact-Newton direction, accepted only by measured gradient merit.

The old forcing<=.5 failure remains unchanged. For exact H, eta<1 implies
 g.T H p <= -(1-eta)||g||^2, so a backtracked gradient-merit trial is justified.
These floating diagnostics do not certify a stationary point or training orbit.
"""
import numpy as np
from .cifar_amplitude import vector
SETTINGS=dict(forcing_limit=.9,armijo=1e-4,trust_relative=.05,trials=13,stationarity=1e-10)
def direction_gate(x,g,p,hp):
 x,g,p,hp=[vector(v)for v in (x,g,p,hp)]
 if not all(v.shape==x.shape for v in (g,p,hp)):raise ValueError('full identical coordinate dimensions')
 norm=float(np.linalg.norm(g))
 if not np.isfinite(norm)or norm==0:raise ValueError('positive finite original gradient')
 forcing=float(np.linalg.norm(g+hp))/norm;slope=float(np.dot(g,hp));merit=.5*norm*norm
 return dict(forcing=forcing,old_forcing_pass=forcing<=.5,slope=slope,initial_merit=merit,
             cauchy_descent_upper=-(1-forcing)*norm*norm,
             passed=bool(np.isfinite([forcing,slope,merit]).all()and forcing<=SETTINGS['forcing_limit']and slope<0))
def trial_gate(origin,candidate,g,alpha,gate):
 x,y,g=[vector(v)for v in (origin,candidate,g)]
 if y.shape!=x.shape or g.shape!=x.shape or not 0<alpha<=1:raise ValueError('same full trial dimensions and positive bounded alpha')
 distance=float(np.linalg.norm(y-x))/max(1.,float(np.linalg.norm(x)))
 merit=.5*float(np.dot(g,g));bound=gate['initial_merit']+SETTINGS['armijo']*alpha*gate['slope']
 return dict(alpha=alpha,relative_origin_distance=distance,within_trust=distance<=SETTINGS['trust_relative'],merit=merit,armijo_bound=bound,
             accepted=bool(gate['passed']and distance<=SETTINGS['trust_relative']and np.isfinite([merit,bound]).all()and merit<=bound),
             relative_gradient=float(np.linalg.norm(g))/max(1.,float(np.linalg.norm(y))))
