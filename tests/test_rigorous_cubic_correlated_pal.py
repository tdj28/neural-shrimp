import math
import mpmath as mp
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_cubic_gradient_path import cubic_jacobian_coefficients
from neural_shrimp.rigorous_cubic_inverse import cubic_inverse_data
from neural_shrimp.rigorous_cubic_correlated_pal import cubic_correlated_tube
from neural_shrimp.rigorous_unit_pair_pal import quadratic_tube
from neural_shrimp.rigorous_orbits import exact,eye
from neural_shrimp.rigorous_eigenpair_march import decode
from test_rigorous_cubic_unit_pair import Model,fixture,residual,mpv
from test_rigorous_unit_pair import Polynomial,Primitive
from test_rigorous_unit_pair_pal import fixture as curve_fixture


def test_full_cubic_gradient_jacobian_against_independent_nested_derivatives():
 with ctx.workprec(256),mp.workdps(85):
  c,s,a,b,n=fixture();d=arb(1)/64;J=cubic_jacobian_coefficients(Model(),c,s,a,b,[arb(0)]*9,d,1,n)
  cc,ss,aa,bb,nn=[list(map(mpv,row))for row in(c,s,a,b,n)]
  for order in range(3):
   for i in range(9):
    for j in range(9):
     def f(t):
      z=[cc[k]+ss[k]*t+aa[k]*t*t/2+bb[k]*t*t*t for k in range(9)]
      return mp.diff(lambda u:residual([z[k]+(u if k==j else 0)for k in range(9)],nn)[i],0)
     ref=mp.diff(f,mp.mpf(1)/64,order)/math.factorial(order)
     assert(J[order][i,j]-arb(mp.nstr(ref,80))).contains(0),(order,i,j)


def test_cubic_full_error_d0_and_whole_second_derivative_are_preserved():
 with ctx.workprec(256):
  c,s,a,b,n=fixture();h=arb(1)/64;radius=arb((1,-16));E=[arb(0,radius)]*9;events=[]
  J0=cubic_jacobian_coefficients(Model(),c,s,a,b,E,arb(0),1,n,observe=lambda k,v:events.append((k,v)))
  J2=cubic_jacobian_coefficients(Model(),c,s,a,b,E,arb(0,h),1,n)
  assert sum(k=='gradient_path_input'for k,v in events)==2
  for sign in [-1,1]:
   cc=[v+sign*radius for v in c]
   point=cubic_jacobian_coefficients(Model(),cc,s,a,b,[arb(0)]*9,arb(0),1,n)
   for order in [0,1]:assert all(J0[order][i,j].contains(point[order][i,j])for i in range(9)for j in range(9))
   point=cubic_jacobian_coefficients(Model(),cc,s,a,b,[arb(0)]*9,sign*h,1,n)
   assert all(J2[2][i,j].contains(point[2][i,j])for i in range(9)for j in range(9))


def test_quartic_forcing_keeps_nonzero_fifth_order_and_all_coordinate_gates():
 with ctx.workprec(160):
  I=eye(2);N=arb_mat([[0,2],[0,0]]);Z=arb_mat(2,2);r=[arb_mat(2,1)for _ in range(5)];r[4][1,0]=1;h=arb(1)/4
  bound=cubic_inverse_data([I,-N],[I,N,Z],r,[arb(1)]*2,arb(1)/64,h)
  assert bound['passed']and len(bound['forcing_coefficients'])==6 and bound['forcing_coefficients'][5][0,0]==2
  assert bound['whole_image'][0,0].contains(2*h*h*h*h*h)and bound['whole_image'][0,0].contains(-2*h*h*h*h*h)
  r[0][0,0]=1
  bad=cubic_inverse_data([I,-N],[I,N,Z],r,[arb(1)]*2,arb(1)/64,h)
  assert not bad['passed']and not all(bad['coordinate_inclusions'])


@pytest.mark.parametrize('model,period',[(Polynomial(),1),(Primitive(),2)])
def test_full_cubic_curve_primitive_known_anchor_and_exact_analytic_endpoints(model,period):
 with ctx.workprec(256):
  ini,kw=curve_fixture(model,period);kw.update(parameter='3/1024',halfwidth_exponent=-11)
  old,data=quadratic_tube(model,ini,**kw);N=len(ini);C=arb_mat([[decode(v)for v in row]for row in old['preconditioner']])
  # Full finite proposal from retained actual center residual, not a proof of cancellation.
  cubic_m=-C*arb_mat([[decode(row[3])]for row in old['quadratic_center_coefficients']]);b=[cubic_m[i,0].mid()for i in range(N)]
  args=dict(period=period,normal=[exact(v)for v in kw['normal']],origin=[arb(0)]*N,parameter='3/1024',halfwidth_exponent=-11,radius_exponent=-12,weights=[arb(1)]*N,anchor={'box':data['endpoints'][0],'exact_parameter':'5/2048','dimension':model.n,'period':period})
  ticks=[];events=[];proof=cubic_correlated_tube(model,data['center'],data['slope'],data['curvature'],b,C,**args,tick=lambda:ticks.append(1),observe=lambda k,v:events.append((k,v)))
  assert proof['passed'],proof['gates'];assert len(ticks)==8*period+2 and len(proof['residual_coefficients'])==5
  assert sum(k=='cubic_path_input'for k,v in events)==sum(k=='cubic_path_return'for k,v in events)==4*period
  if period==1:
   for d,endpoint in zip([exact('5/2048'),exact('7/2048')],proof['endpoint_root_boxes']):
    with ctx.workprec(512):x=2*(d.sqrt().cos()-1);h=1-x*x/2;o=1-x
    assert decode(endpoint[0]).contains(x)and decode(endpoint[-3]).contains(h)and decode(endpoint[-2]).contains(o)
  args['anchor']['box'][0]+=1
  bad=cubic_correlated_tube(model,data['center'],data['slope'],data['curvature'],b,C,**args)
  assert not bad['passed']and not bad['gates']['anchor_identified']
