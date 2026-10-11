import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_parameter_localization import localize_parameter_root,fixed_point_box
from neural_shrimp.rigorous_orbits import exact,eye
from neural_shrimp.rigorous_connection import restore_ball


def fixture():
 h=arb('1/128');d=arb(0,h);c=[arb(1),arb(0)];s=arb_mat([[arb('1/2')],[0]]);w=[arb(1),arb(1)];rho=arb('1/64')
 x=1+d/2+arb(0,rho);J=arb_mat([[2*x,0],[0,1]]);Jp=arb_mat([[2*(1+d/2),0],[0,1]]);Ht=arb_mat([[-1],[-2*d]])
 return dict(C=arb_mat([[arb('1/2'),0],[0,1]]),J=J,G0=arb_mat(2,1),Jp=Jp,Htp=Ht,Htw=Ht,center=c,slope=s,weights=w,halfwidth=h,original_radius=rho)

def test_nonlinear_known_parameter_root_and_true_tangent():
 with ctx.workprec(256):
  a=fixture();r=localize_parameter_root(**a);assert r['passed']
  cb=[restore_ball(v)for v in r['center_box']];wb=[restore_ball(v)for v in r['whole_box']];vb=[restore_ball(v[0])for v in r['tangent_box']]
  assert cb[0].contains(1)and cb[1].contains(0)and cb[0].rad()<arb('1e-50')
  for k in range(-8,9):
   t=arb(k)/1024;x=(1+t).sqrt();truth=[x,t*t];velocity=[1/(2*x),2*t]
   assert all(wb[i].contains(truth[i])and vb[i].contains(velocity[i])for i in range(2))
  assert wb[1].rad()<a['original_radius']/10


def test_coupled_exact_linear_root_has_exact_tangent():
 with ctx.workprec(256):
  J=arb_mat([[1,1],[0,1]]);C=arb_mat([[1,-1],[0,1]]);s=arb_mat([[2],[-3]]);Ht=-(J*s)
  r=localize_parameter_root(C,J,arb_mat(2,1),J,Ht,Ht,[arb(0),arb(0)],s,[arb(1),arb('1/2')],arb('1/8'),arb('1/64'))
  assert all(restore_ball(v[0]).contains(s[i,0])and restore_ball(v[0]).rad()<arb('1e-50')for i,v in enumerate(r['tangent_box']))


def test_singular_or_inexact_proposals_fail_closed():
 with ctx.workprec(256):
  a=fixture();a['J']=arb_mat([[0,0],[0,1]])
  with pytest.raises(ValueError,match='Neumann'):localize_parameter_root(**a)
  a=fixture();a['C'][0,0]=arb('1/2','1/1000')
  with pytest.raises(ValueError,match='fixed exact'):localize_parameter_root(**a)
  a=fixture();a['weights'][0]=arb(0)
  with pytest.raises(ValueError,match='positive weights'):localize_parameter_root(**a)


def test_interval_rhs_is_not_treated_as_its_midpoint():
 with ctx.workprec(256):
  D=arb_mat([[arb('1/4')]]);v=arb_mat([[arb(1,arb('1/8'))]])
  box,r=fixed_point_box(D,v,[arb(1)],arb('1/4'))
  assert box[0,0].contains(arb('7/6'))and box[0,0].contains(arb('3/2'))
