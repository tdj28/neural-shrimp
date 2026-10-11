import hashlib,subprocess
from pathlib import Path
from fractions import Fraction as F
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_quadratic_tangent_forcing import *
from neural_shrimp.rigorous_unit_pair_pal import path_coefficients


def test_exact_original_path_source_copies():
 root=Path(__file__).parents[1]
 for n in ['rigorous_eigenpair_locus','rigorous_eigenpair_path','rigorous_unit_pair','rigorous_unit_pair_pal']:
  p='src/neural_shrimp/'+n+'.py'
  assert hashlib.sha256((root/p).read_bytes()).digest()==hashlib.sha256(subprocess.check_output(['git','show','b03d990f:'+p],cwd=root)).digest()


@pytest.mark.parametrize("power",[1,3])
def test_full_augmented_mixed_hessian_with_both_rates_modes_and_angle(power):
 class Model:
  n=2
  def cusp_step(self,x,a,b):return [a*x[0],b*x[1]]if power==1 else[a*x[0]*x[0]*x[0],b*x[1]*x[1]*x[1]]
 with ctx.workprec(384):
  z=list(map(arb,[2,3,1,2,3,4,5,6,0]));e=[arb(str(F(i+1,32)))for i in range(9)];p=[arb(str(F(10-i,64)))for i in range(9)]
  vp,vm=balanced_directions(e,p,arb('1/256'));zero=[arb(0)]*9;normal=[arb(0)]*8+[arb(1)]
  cp=path_coefficients(Model(),z,vp,zero,arb(0),1,normal,zero,arb(0));cm=path_coefficients(Model(),z,vm,zero,arb(0),1,normal,zero,arb(0));got=mixed_hessian_from_second_coefficients(cp,cm)
  expected=[]
  for i in range(2):
   x=z[i];a=z[6+i];m=power
   expected.append(m*x**(m-1)*(e[6+i]*p[i]+p[6+i]*e[i])+a*m*(m-1)*x**(m-2)*e[i]*p[i])
  def mode_native(i,k):
   x=z[i];a=z[6+i];q=z[k+i];m=power
   return m*((m-1)*x**(m-2)*q*(e[6+i]*p[i]+p[6+i]*e[i])+x**(m-1)*(e[6+i]*p[k+i]+p[6+i]*e[k+i])+a*(m-1)*(m-2)*x**(m-3)*e[i]*p[i]*q+a*(m-1)*x**(m-2)*(e[i]*p[k+i]+p[i]*e[k+i]))
  for i in range(2):expected.append(mode_native(i,2)+e[8]*p[2+i]/2+p[8]*e[2+i]/2-e[8]*p[8]*z[2+i]/12+e[8]*p[4+i]+p[8]*e[4+i]-e[8]*p[8]*z[4+i]/3)
  for i in range(2):expected.append(mode_native(i,4)+e[8]*p[4+i]/2+p[8]*e[4+i]/2-e[8]*p[8]*z[4+i]/12+(e[8]*p[2+i]+p[8]*e[2+i])/6-e[8]*p[8]*z[2+i]/60)
  expected += [sum(e[2+i]*p[2+i]for i in range(2)),sum(e[2+i]*p[4+i]+e[4+i]*p[2+i]for i in range(2)),arb(0)]
  assert all(got[i,0].contains(v)for i,v in enumerate(expected))


def test_quadratic_predictor_residual_derivative_uses_taylor_factor_three():
 with ctx.workprec(256):
  c=arb_mat([[1,2,3,4]]);w=arb_mat([[0,0,0,arb(4,arb('1/8'))]]);a=path_residual_derivative(c,w,arb('1/16'))[0,0]
  for sign in [-1,0,1]:
   t=arb(sign)/16;assert a.contains(2+6*t+12*t*t)


def test_known_nonlinear_root_tangent_retained_with_small_error_forcing():
 # H(u,s)=u^2-(1+s), p(s)=1+s/2, |s|<=1/16.
 with ctx.workprec(256):
  d=arb(0,arb('1/16'));p=1+d/2;root=(1+d).sqrt();e=root-p
  # Broad e/root boxes suffice for validity; R'=s/2 and H''[e,p']=e.
  out=refine_tangent(arb_mat([[arb('1/2')]]),arb_mat([[2*root]]),arb_mat([[d/2]]),arb_mat([[e]]),[arb('1/2')],[arb(1)])
  b=out['tangent_box'][0][0]
  box=arb(arb(tuple(b["mid"])),arb(tuple(b["rad"])))
  for t in ['-1/16','0','1/16']:assert box.contains(1/(2*(1+arb(t)).sqrt()))


def test_polarization_rejects_inexact_or_nonpositive_scale():
 for scale in [arb(0),arb(-1),arb(1,arb('1/8'))]:
  with pytest.raises(ValueError):balanced_directions([arb(1)],[arb(2)],scale)
