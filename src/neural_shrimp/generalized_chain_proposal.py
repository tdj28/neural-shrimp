"""Saved-array seed only; no native evaluation or event certificate."""
import math
import numpy as np
from .rigorous_eigenpair_locus import require
from .rigorous_eigenpair_march import decode
from .rigorous_orbits import exact,dyadic


def from_accepted_chart(chart,spectrum):
    require(chart['passed'] is True and chart['gates']==dict.fromkeys(('uniform_normalized_eigenpair_locus','primitive_period','anchor_identified'),True),'qualified native root required')
    require(chart['period']==18 and chart['dimension']==13 and chart['sigma']==1 and chart['shooting_dimension']==470,'fixed native primitive18 +1 system')
    require(len(chart['center'])==470,'full native center')
    center=[decode(v)for v in chart['center']]
    require(all(v.is_exact() and v.is_finite()for v in center),'exact finite saved predictor')
    require(center[-2]>0 and center[-1]>0,'positive actual rates')
    q=np.array([float(v)for v in center[13:26]])
    matrix=np.array(spectrum['monodromy_midpoint']);values=np.array([complex(*v)for v in spectrum['eigenvalues']]);vectors=np.array([[complex(*v)for v in row]for row in spectrum['eigenvectors']])
    residuals=spectrum['relative_residuals'];known=spectrum['critical_nearest_index']
    require(matrix.shape==(13,13) and values.shape==(13,) and vectors.shape==(13,13),'full saved spectrum shapes')
    require(np.isfinite(matrix).all() and np.isfinite(values).all() and np.isfinite(vectors).all(),'finite saved spectrum')
    require(type(known)is int and 0<=known<13 and abs(values[known]-1)<1e-6,'identified known+1 mode')
    require(len(residuals)==13 and all(type(v)in(int,float) and math.isfinite(v) and 0<=v<1e-8 for v in residuals),'full saved eigen residual gates')
    norm=max(1.,float(np.linalg.norm(matrix)))
    require(all(np.linalg.norm(vectors[:,i])>0 and np.linalg.norm(matrix@vectors[:,i]-values[i]*vectors[:,i])/(norm*np.linalg.norm(vectors[:,i]))<1e-8 for i in range(13)),'recomputed full saved eigen residuals')
    qnorm=float(q@q);require(abs(qnorm-1)<1e-8 and np.linalg.norm(matrix@q-q)<1e-8,'native phase0 mode normalization/residual')
    other=min((i for i in range(13)if i!=known),key=lambda i:abs(values[i]-1));mu=values[other]
    require(1e-8<abs(mu-1)<=.03 and abs(mu.imag)<1e-12,'real near-neutral candidate, separated numerically from known1')
    u=vectors[:,other];require(np.max(abs(u.imag))<1e-12,'real companion direction')
    u=u.real;overlap=float(q@u)
    require(abs(overlap)>1e-3,'nonorthogonal companion proposal')
    delta=float(mu.real-1);r=(u*(qnorm/overlap)-q)/delta
    require(np.isfinite(r).all() and np.linalg.norm(r)<1024,'bounded finite generalized direction')
    residual=float(np.linalg.norm((matrix-np.eye(13))@r-q-delta*r))
    require(residual<1e-7 and abs(float(q@r))<1e-7,'saved near-chain consistency')
    seed=center[:13]+center[-2:]+center[13:26]+[exact(dyadic(v))for v in r]
    return seed,{'companion_index':other,'multiplier':float(mu.real),'known_index':known,'gap':abs(delta),'mode_overlap':overlap,'q_norm_squared':qnorm,'r_norm':float(np.linalg.norm(r)),'near_chain_residual':residual,'orthogonality_residual':abs(float(q@r)),'scope':'Floating seed from authenticated saved arrays only; not a Jordan-chain proof, event bracket, actual orbit derivative, or certified branch connection.'}
