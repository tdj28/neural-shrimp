import importlib.util,json
from fractions import Fraction as F
from pathlib import Path
from flint import arb,arb_mat,ctx
import pytest
s=importlib.util.spec_from_file_location('E',Path(__file__).resolve().parents[1]/'tools/clark_endpoint_localization.py');E=importlib.util.module_from_spec(s);s.loader.exec_module(E)

def fixture(n=2):
 z=[arb(0)]*n;r=[arb(1)]*n;D=arb_mat([[arb(int(i==j))/4 for j in range(n)]for i in range(n)]);f=arb_mat([[arb(1)/8]for i in range(n)]);image=f+D*arb_mat([[arb(0,1)]for i in range(n)])
 data=dict(passed=True,dimension=n,defect_coefficients=[D]+[arb_mat(n,n)]*3,forcing_coefficients=[f]+[arb_mat(n,1)]*4,image=image)
 return data,z,z,z,r,arb(1)/16,[image[i,0]for i in range(n)]

def test_known_linear_root_and_monotone_endpoint():
 with ctx.workprec(384):
  args=fixture();v=E.localize(*args)
  assert v['passed']and v['restricted']and len(v['history'])==16
  assert all(E.contains(E.bounds(x),(F(1,6),F(1,6)))for x in v['endpoint'])
  assert all(E.contains(E.bounds(old),E.bounds(new))for old,new in zip(args[-1],v['endpoint']))

def test_actual_full51_synthetic_and_finite_serialization():
 with ctx.workprec(384):
  v=E.localize(*fixture(51));assert v['restricted']and len(v['endpoint'])==51
  def b(x):return dict(mid=list(map(int,x.mid().man_exp())),rad=list(map(int,x.rad().man_exp())))
  payload=json.dumps(dict(endpoint=[b(x)for x in v['endpoint']],history=[dict((k,[b(x)for x in xs])for k,xs in row.items())for row in v['history']]))
  assert len(payload)<1024**2

def test_reject_unproved_or_uncovered_and_nonfinite():
 args=list(fixture());args[0]=dict(args[0],passed=False)
 with pytest.raises(ValueError,match='already proved'):E.localize(*args)
 args=list(fixture());args[-1]=[arb(0)]*2
 with pytest.raises(ValueError,match='covers'):E.localize(*args)
 args=list(fixture());args[0]['defect_coefficients'][0][0,0]=arb('nan')
 with pytest.raises(ValueError,match='finite'):E.localize(*args)

def test_huge_exact_dyadic_restoration_is_outward_at_working_precision():
 with ctx.workprec(512):
  a=F(1)+F(1,2**16000);b=a+F(1,2**500);v=E.restore((a,b))
  assert E.contains(E.bounds(v),(a,b))and v.mid().bits()<=512
  json.dumps(dict(mid=list(map(int,v.mid().man_exp())),rad=list(map(int,v.rad().man_exp()))))

def test_endpoint_uses_higher_precision_than_native_predictor_coefficients():
 with ctx.workprec(512):
  args=list(fixture());args[1]=[arb(1)]*2;args[2]=[arb((1,-400))]*2
  p=[args[1][i]+args[2][i]*args[5]for i in range(2)]
  args[-1]=[p[i]+args[0]['image'][i,0]for i in range(2)]
 with ctx.workprec(384):
  with pytest.raises(ValueError,match='exact polynomial endpoint'):E.localize(*args)
 with ctx.workprec(512):
  result=E.localize(*args);assert result['restricted']
  assert all(E.contains(E.bounds(x),(F(1)+F(1,2**404)+F(1,6),)*2)for x in result['endpoint'])
