import sys,copy
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/reviews'))
import review_clark_bridge_math as B

def ball(v):return dict(mid=list(v.mid().man_exp()),rad=list(v.rad().man_exp()))
def matrix(m):return [[ball(v)for v in row]for row in m.tolist()]
def fixture():
 c=[arb(1),arb(1)];v=[arb(2),arb(1)];r=[arb('1/32'),arb('1/1024')];h=arb('1/16');d=arb(0,h);e=[arb(0,z)for z in r];p=[c[i]+v[i]*d for i in range(2)];w=[p[i]+e[i]for i in range(2)]
 return dict(center=[ball(z)for z in c],slope=[ball(z)for z in v],radii=[ball(z)for z in r],parameter_center=ball(c[-1]),halfwidth=ball(h),actual_error=[ball(z)for z in e],predictor_box=[ball(z)for z in p],whole_box=[ball(z)for z in w],center_value=matrix(arb_mat(2,1)),center_jacobian=matrix(arb_mat([[1,-2],[0,1]])),predictor_jacobian=matrix(arb_mat([[1,-2*p[-1]],[0,1]])),whole_jacobian=matrix(arb_mat([[1,-2*w[-1]],[0,1]])))
def test_independent_nonlinear_uniform_and_exact_known_root_join():
 with ctx.workprec(512):
  t=B.uniform(fixture());assert t['passed']
  for i in[-7,0,7]:
   s=arb(1)+arb(i)/128;assert B.join([s*s,s],t)['passed']
  assert not B.join([arb(1),arb(2)],t)['passed']
def test_actual_domain_and_image_failure_rejected():
 with ctx.workprec(512):
  d=fixture();d['whole_box'][0]=ball(arb(1))
  with pytest.raises(ValueError,match='coverage'):B.uniform(d)
  d=fixture();d['center_value']=matrix(arb_mat([[1],[0]]))
  with pytest.raises(ValueError,match='strict'):B.uniform(d)
def test_exact_endpoint_guard_rejects_outward_escape():
 with ctx.workprec(512):
  raw=ball(arb(1,arb('1/16')));lo,hi=B.endpoints(raw)
  assert B.covered(raw,lo)and B.covered(raw,hi)
  assert not B.covered(raw,hi+arb((1,-300)))
def test_native_orientation_unique_and_matching():
 def rows(s):return [dict(sign=x,join=dict(passed=x==s,coordinates=26))for x in(1,-1)]
 assert B.choose(rows(1),rows(1))==1
 with pytest.raises(ValueError):B.choose(rows(1),rows(-1))
