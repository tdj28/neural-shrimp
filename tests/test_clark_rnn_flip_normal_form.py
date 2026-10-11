import math
import numpy as np
import pytest
from neural_shrimp import clark_rnn_flip_normal_form as N
from neural_shrimp.clark_rnn_symmetry import g_jets
from neural_shrimp.clark_rnn_boundaries import third_directional

def test_tanh_series_against_closed_fifth_derivative():
 x=np.zeros((6,3));x[0]=[-.4,0.,.7];x[1]=[.3,-.2,.5];t=np.tanh(x[0]);q=x[1]
 derivatives=[t,1-t*t,-2*t*(1-t*t),-2+8*t*t-6*t**4,16*t-40*t**3+24*t**5,16-136*t*t+240*t**4-120*t**6]
 expected=np.array([v*q**k/math.factorial(k)for k,v in enumerate(derivatives)])
 assert np.allclose(N.tanh_series5(x),expected,rtol=2e-14,atol=2e-15)

def test_full_signed_path_matches_independent_first_second_third_jets():
 rng=np.random.default_rng(6301);j=rng.normal(size=(3,3))*.2;x=rng.normal(size=3)*.3;q=rng.normal(size=3);q/=np.linalg.norm(q);coeff=np.zeros((6,3));coeff[0]=x;coeff[1]=q
 out=N.signed_half_series(coeff,j,1.2,.8);end,u,v,_,_=g_jets(x,j,1.2,.8,3);third,_=third_directional(x,q,j,1.2,.8,3)
 assert np.allclose(out[0],end,atol=1e-15)and np.allclose(out[1],u[:,:3]@q,atol=1e-15)
 assert np.allclose(2*out[2],np.einsum('ijk,j,k->i',v[:,:3,:3],q,q),atol=1e-15)and np.allclose(6*out[3],-third,atol=1e-15)

def test_quadratic_scalar_flip_known_cubic_and_quintic():
 a=.4
 def evaluate(h):return -h+a*N.multiply(h,h)
 r=N.normal_form5(np.zeros(1),np.array([[-1.]]),np.ones(1),np.ones(1),evaluate)
 assert abs(r['cubic']-a*a)<1e-14 and abs(r['quintic']-5*a**4/4)<1e-14 and r['invariance_residual']<1e-14
 assert abs(r['parameterization'][4,0]-5*a**3/8)<1e-14

def test_nonlinear_stable_coordinate_change_and_degenerate_flip():
 cubic=0.;quintic=-.7;a=.2;b=-.3;c=.4;rho=.5
 def phi(s):return a*N.multiply(s,s)+b*N.multiply(N.multiply(s,s),s)+c*N.multiply(N.multiply(s,s),N.multiply(s,s))
 def evaluate(h):
  s=h[:,0];f=-s+cubic*N.multiply(N.multiply(s,s),s)+quintic*N.multiply(N.multiply(N.multiply(s,s),N.multiply(s,s)),s)
  return np.column_stack([f,rho*(h[:,1]-phi(s))+phi(f)])
 r=N.normal_form5(np.zeros(2),np.diag([-1.,rho]),np.array([1.,0.]),np.array([1.,0.]),evaluate)
 assert abs(r['cubic'])<1e-14 and abs(r['quintic']-quintic)<1e-14 and np.allclose(r['parameterization'][2:5,1],[a,b,c])and r['invariance_residual']<1e-14
 flipped=N.normal_form5(np.zeros(2),np.diag([-1.,rho]),np.array([-1.,0.]),np.array([-1.,0.]),evaluate)
 assert flipped['cubic']==r['cubic']and flipped['quintic']==r['quintic']

def test_signed_three_tanh_map_known_odd_coefficients_and_call_counts():
 rows=[];calls=[];solves=[]
 def evaluate(h):calls.append(h.copy());return N.signed_half_series(h,np.ones((1,1)),1.,1.,observe=lambda k,v:rows.append((k,v)))
 def solve(a,b,tag):solves.append(tag);return np.linalg.solve(a,b)
 r=N.normal_form5(np.zeros(1),np.array([[-1.]]),np.ones(1),np.ones(1),evaluate,solve=solve)
 assert abs(r['cubic']-1)<1e-14 and abs(r['quintic']+7/5)<1e-14 and r['invariance_residual']<1e-14
 assert len(calls)==5 and len(rows)==15 and solves==['even2','odd3','even4','odd5']

def test_wrong_actual_mode_and_nonfinite_path_rejected():
 with pytest.raises(ValueError):N.normal_form5(np.zeros(1),np.array([[1.]]),np.ones(1),np.ones(1),lambda h:h)
 with pytest.raises(ValueError):N.normal_form5(np.zeros(1),np.array([[-1.]]),np.ones(1),np.ones(1),lambda h:np.full_like(h,np.nan))
