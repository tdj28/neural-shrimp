import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp.rigorous_signed_correlated import signed_correlated_tube
from neural_shrimp.rigorous_eigenpair_locus import assemble
from neural_shrimp.rigorous_orbits import exact,midpoint_matrix
class Map:
 n=1
 def cusp_step(self,x,h,o):return[(o-exact('3/2'))*x[0]+h-exact('1/4')]
def fixture():
 c=list(map(exact,['0','1','1/4','1/2']));s=list(map(exact,['1/2','0','1','0']));a=[arb(0)]*4;normal=[arb(0),arb(0),arb(1),arb(0)];origin=c.copy();_,J,_=assemble(Map(),c,1,-1,normal,origin,arb(0));C=midpoint_matrix(J.inv());return c,s,a,normal,origin,C

def test_exact_nonlinear_augmented_curve_with_fullerror_and_anchor():
 with ctx.workprec(192):
  c,s,a,n,o,C=fixture();h=exact('1/16');anchor=dict(box=[c[i]-s[i]*h for i in range(4)],exact_parameter='-1/16',period=1,sigma=-1,dimension=1);events=[]
  r=signed_correlated_tube(Map(),c,s,a,C,period=1,sigma=-1,normal=n,origin=o,parameter='0',halfwidth_exponent=-4,radius_exponent=-14,weights=[arb(1)]*4,anchor=anchor,observe=lambda k,v:events.append((k,v)))
  assert r['passed']and all(r['bound']['coordinate_inclusions'])and all(r['anchor_check']['coordinate_inclusions'])
  g=[v for k,v in events if k=='signed_gradient_path_input'];assert len(g)==2
  assert g[0]['inputs'][0]['value'][0][0]['rad'][0]>0 and g[0]['inputs'][0]['value'][1][0]['rad'][0]>0
  assert len(r['inverse_polynomial_exact'])==2 and r['parameter_endpoints']==['-1/16','1/16']

def test_false_root_and_wrong_sign_anchor_cannot_pass():
 with ctx.workprec(192):
  c,s,a,n,o,C=fixture();bad=c.copy();bad[0]+=1
  r=signed_correlated_tube(Map(),bad,s,a,C,period=1,sigma=-1,normal=n,origin=o,parameter='0',halfwidth_exponent=-4,radius_exponent=-14,weights=[arb(1)]*4)
  assert not r['passed']and not r['gates']['uniform_normalized_eigenpair_locus']and not r['gates']['anchor_identified']
  with pytest.raises(ValueError,match='anchor convention'):signed_correlated_tube(Map(),c,s,a,C,period=1,sigma=-1,normal=n,origin=o,parameter='0',halfwidth_exponent=-4,radius_exponent=-14,weights=[arb(1)]*4,anchor=dict(box=c,period=1,sigma=1,dimension=1))
