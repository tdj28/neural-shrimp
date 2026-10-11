import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/reviews'))
import numpy as np
import pytest
import cifar_nonlinearity_math as M


def test_linear_model_no_remainder_and_positive_trust_root():
 x=np.array([1.,0.]);p=np.array([0.,-.01]);g=np.array([0.,1.]);hp=np.array([0.,-1.]);r=M.analyze(x,x,g,p,hp,[(x+p,g+hp)])
 assert r['rows'][0]['nonlinear_norm']==0 and r['rows'][0]['actual_to_predicted_decrease']==1
 assert r['numerical_trust_alpha']==pytest.approx(5,rel=1e-14) and r['accepted_alpha']==1


def test_nonlinear_cancellation_and_empirical_order_retained():
 x=np.array([1.,0.]);g=np.array([0.,1.]);p=np.array([0.,.01]);hp=-g
 # Full and half steps fail, quarter accepts; exact quadratic gradient remainder.
 pairs=[(x+a*p,g+a*hp+3*a*a*g)for a in [1.,.5,.25]]
 r=M.analyze(x,x,g,p,hp,pairs)
 assert r['accepted_alpha']==.25
 assert all(abs(v['quadratic_remainder_coefficient']-3)<1e-14 for v in r['rows'])
 assert r['rows'][1]['successive_remainder_order']==2
 assert r['rows'][0]['actual_to_predicted_decrease']<0
 assert all(abs(v['merit_decomposition_error'])<1e-14 for v in r['rows'])


def test_trust_skips_never_invent_nonlinear_measurements():
 x=np.array([1.,0.]);p=np.array([0.,.08]);g=np.array([0.,1.]);hp=-g
 r=M.analyze(x,x,g,p,hp,[None,(x+p/2,g+hp/2)])
 assert not r['rows'][0]['nonlinear_observed']and r['numerical_trust_alpha']==pytest.approx(.625,rel=1e-14)
 with pytest.raises(ValueError):M.analyze(x,x,g,p,hp,[(x+p,g+hp)])
 with pytest.raises(ValueError):M.analyze(x,x,g,p,hp,[None,(x+p/2+1e-6,g+hp/2)])


def test_first_acceptance_and_full_finite_shape_are_mandatory():
 x=np.array([1.,0.]);p=np.array([0.,.01]);g=np.array([0.,1.]);hp=-g
 with pytest.raises(ValueError):M.analyze(x,x,g,p,hp,[(x+p,g+hp),(x+p/2,g+hp/2)])
 with pytest.raises(ValueError):M.analyze(x,x,g,p,np.array([float('nan'),0]),[])
