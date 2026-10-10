import numpy as np
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_unit_pair import entire_pair
from neural_shrimp.rigorous_orbits import exact,dyadic
from neural_shrimp.unit_pair_flip import augmented as floating
from neural_shrimp.rigorous_unit_pair_flip import augmented,root_proof,complement,transverse_crossing

class Normal:
 n=4
 def cusp_step(self,z,h,o):
  x,y,v,w=z;return[x+y,y+h-1+x*x*.5+(o-1)*y+2*x*y,(-.9+x+h*.1)*v+.2*v*v,.2*w]


def fixture():
 c,e,d=entire_pair(exact('1/10'))
 class Rotation:
  n=4
  def cusp_step(self,z,h,o):
   x,y,v,w=z;return[o*(c*x+e*y),o*(-d*x+c*y),-h*v+v*v*v,w*arb('0.2')]
 y=list(map(exact,['0']*4+['1','1']+['1','0','0','0']+['0','1','0','0']+['1/10']+['0','0','1','0']))
 return Rotation(),y


def test_all_interval_native_blocks_match_separate_float_and_fd_checked_kernel():
 with ctx.workprec(384):
  y=np.r_[np.linspace(-.02,.03,14),.1,[.2,.3,.8,.1]];y[4:6]=[1.001,.97]
  a,J,data=augmented(Normal(),[exact(dyadic(v))for v in y],3);f,b,_=floating(Normal(),y,3)
  np.testing.assert_allclose([float(a[i,0].mid())for i in range(19)],f,atol=2e-14)
  np.testing.assert_allclose([[float(J[i,j].mid())for j in range(19)]for i in range(19)],b,atol=2e-14)


def test_nonsingular_normal_form_strict_root_complement_and_total_crossing():
 with ctx.workprec(384):
  model,y=fixture();answer,data=root_proof(model,y,1,radius_exponent=-40,newton_steps=1)
  assert answer['passed']and all(answer['coordinate_inclusions'])and all(answer['gates'].values())
  result=complement(data['monodromy'],y[6:10],y[10:14],y[15:],y[14]);assert result['passed']
  _,J,data=augmented(model,y,1);cross=transverse_crossing(J,data['monodromy'],y[15:],4)
  assert cross['passed']and cross['multiplier_hidden_rate_derivative']['mid']==[-1,0]


def test_unstable_complement_and_zero_crossing_not_accepted():
 with ctx.workprec(256):
  model,y=fixture();_,J,data=augmented(model,y,1);M=data['monodromy'];M[3,3]=arb('1.2')
  assert not complement(M,y[6:10],y[10:14],y[15:],y[14])['passed']
  for i in range(14,18):
   for j in range(6):J[i,j]=0
  assert not transverse_crossing(J,M,y[15:],4)['passed']


def test_singular_critical_frame_is_rejected_and_caps_precede_native_call():
 with ctx.workprec(256):
  model,y=fixture();_,_,data=augmented(model,y,1)
  with pytest.raises((ValueError,ZeroDivisionError)):complement(data['monodromy'],y[6:10],y[6:10],y[15:],y[14])
  def stop():raise TimeoutError('owned native cap')
  with pytest.raises(TimeoutError):augmented(model,y,18,stop)
