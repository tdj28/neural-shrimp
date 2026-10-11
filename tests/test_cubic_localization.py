import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_cubic_localization import localize_cubic_root,cubic_segment_inputs
from neural_shrimp.rigorous_connection import restore_ball
from neural_shrimp.rigorous_orbits import vector_json

def data():
 h=arb('1/128');d=arb(0,h);rho=arb('1/128');p=1+d/2-d*d/8+d*d*d/16
 return dict(C=arb_mat([[arb('1/2'),0],[0,1]]),J=arb_mat([[2*(p+arb(0,rho)),0],[0,1]]),G0=arb_mat(2,1),path_residual=arb_mat([[5*d*d*d*d/64-d*d*d*d*d/64+d*d*d*d*d*d/256],[0]]),parameter_partial=arb_mat([[-1],[-3*d*d]]),center=[arb(1),arb(0)],slope=[arb('1/2'),arb(0)],curvature=[arb('-1/4'),arb(0)],cubic=[arb('1/16'),arb(1)],weights=[arb(1),arb(1)],halfwidth=h,radius=rho)

def original(a):
 d=arb(0,a['halfwidth']);return vector_json([c+s*d+v*d*d/2+b*d*d*d+arb(0,a['radius']*w)for c,s,v,b,w in zip(a['center'],a['slope'],a['curvature'],a['cubic'],a['weights'])])

def test_cubic_true_nonlinear_root_and_tangent():
 with ctx.workprec(384):
  a=data();p=localize_cubic_root(**a);whole=[restore_ball(x)for x in p['whole_box']];v=[restore_ball(x[0])for x in p['tangent_box']]
  for k in range(-8,9):
   t=arb(k)/1024;x=(1+t).sqrt();assert whole[0].contains(x)and whole[1].contains(t*t*t)
   assert v[0].contains(1/(2*x))and v[1].contains(3*t*t)
  assert restore_ball(p['center_box'][0]).rad()<arb('1e-100')

def test_full_cubic_segment_tangent_and_polarized_directions():
 with ctx.workprec(384):
  a=data();p=localize_cubic_root(**a);segment,error,v,plus,minus=cubic_segment_inputs(a,p,original(a))
  for k in range(-8,9):
   t=arb(k)/1024;assert segment[0].contains((1+t).sqrt())and segment[1].contains(t*t*t)and v[1].contains(3*t*t)
  assert all(z.is_finite()for z in plus+minus)

def test_wrong_original_serialized_domain_and_nonexact_cubic_rejected():
 with ctx.workprec(384):
  a=data();p=localize_cubic_root(**a);domain=original(a);domain[0]={'mid':[1,0],'rad':[0,0]}
  with pytest.raises(ValueError,match='exact original'):cubic_segment_inputs(a,p,domain)
  a['cubic'][0]=arb('1/16','1/10000')
  with pytest.raises(ValueError,match='exact finite'):localize_cubic_root(**a)

def test_quadratic_special_case_preserves_prior_formulas():
 from neural_shrimp.rigorous_quadratic_localization import localize_quadratic_root
 with ctx.workprec(384):
  a=data();a['cubic']=[arb(0),arb(0)];old={k:v for k,v in a.items()if k!='cubic'};x=localize_cubic_root(**a);y=localize_quadratic_root(**old)
  for k in ['whole_box','center_box','tangent_box','predictor_tangent','weighted_neumann_bound']:assert x[k]==y[k]
