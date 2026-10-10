import numpy as np
import pytest
from neural_shrimp.unit_pair_flip import augmented,spectrum_report,crossing_derivative

class Normal:
 n=4
 def cusp_step(self,z,h,o):
  x,y,v,w=z
  return [x+y,y+h-1+x*x*.5+(o-1)*y+2*x*y,(-.9+x+h*.1)*v+.2*v*v,.2*w]


def test_all_native_augmented_jacobian_columns():
 m=Normal();k=14;z=np.linspace(-.02,.03,k);z[4:6]=[1.001,.97]
 y=np.r_[z,.1,[.2,.3,.8,.1]];v,J,data=augmented(m,y,3);eps=1e-6
 for i in range(len(y)):
  dy=np.eye(len(y))[i]*eps
  fd=(augmented(m,y+dy,3)[0]-augmented(m,y-dy,3)[0])/(2*eps)
  np.testing.assert_allclose(J[:,i],fd,atol=2e-8,rtol=2e-6)
 assert data['w_phases'].shape==(4,4)


def test_native_pair_flip_and_remaining_stability_not_conflated():
 angle=.2;c,s=np.cos(angle),np.sin(angle)
 M=np.array([[c,-s,0,0],[s,c,0,0],[0,0,-1,0],[0,0,0,.5]])
 w=np.array([0,0,1,0.]);r=spectrum_report(M,w,angle);assert r['passed']and r['remaining_max_modulus']==.5
 M[3,3]=1.1;assert not spectrum_report(M,w,angle)['gates']['remaining_attracting']
 M[3,3]=-.999999;assert not spectrum_report(M,w,angle)['gates']['simple_flip']


def test_total_crossing_derivative_includes_state_motion_and_actual_rate():
 n=4;k=14;J=np.zeros((19,19));J[:k,:k]=np.eye(k);J[:k,k]=0;J[n,k]=-1
 J[0,n]=.3;J[k+2,0]=2;J[k+2,n]=-.7
 M=np.diag([.5,.4,-1,.2]);w=np.array([0,0,1,0.])
 r=crossing_derivative(J,w,M,n)
 assert r['passed']and r['multiplier_hidden_rate_derivative']==pytest.approx(-1.3)
