import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_quadratic_localization import localize_quadratic_root
from neural_shrimp.rigorous_connection import restore_ball


def data():
 h=arb('1/128');d=arb(0,h);rho=arb('1/128');x=1+d/2-d*d/8
 return dict(C=arb_mat([[arb('1/2'),0],[0,1]]),J=arb_mat([[2*(x+arb(0,rho)),0],[0,1]]),G0=arb_mat(2,1),
   path_residual=arb_mat([[-d*d*d/8+d*d*d*d/64],[0]]),parameter_partial=arb_mat([[-1],[-2*d]]),
   center=[arb(1),arb(0)],slope=[arb('1/2'),arb(0)],curvature=[arb('-1/4'),arb(2)],weights=[arb(1),arb(1)],halfwidth=h,radius=rho)


def test_nonlinear_quadratic_root_and_true_tangent():
 with ctx.workprec(256):
  r=localize_quadratic_root(**data());w=[restore_ball(v)for v in r['whole_box']];v=[restore_ball(row[0])for row in r['tangent_box']]
  for k in range(-8,9):
   t=arb(k)/1024;x=(1+t).sqrt()
   assert w[0].contains(x)and w[1].contains(t*t)
   assert v[0].contains(1/(2*x))and v[1].contains(2*t)
  assert restore_ball(r['center_box'][0]).rad()<arb('1e-60')


def test_nonlinear_path_defect_is_retained():
 with ctx.workprec(256):
  a=data();a['G0'][0,0]=arb('1/1000000');r=localize_quadratic_root(**a)
  assert not restore_ball(r['center_rhs'][0][0]).contains(0)


def test_unproved_inverse_and_inexact_curvature_rejected():
 with ctx.workprec(256):
  a=data();a['J']=arb_mat(2,2)
  with pytest.raises(ValueError,match='Neumann'):localize_quadratic_root(**a)
  a=data();a['curvature'][0]=arb('-1/4',arb('1/1000'))
  with pytest.raises(ValueError,match='exact finite'):localize_quadratic_root(**a)
