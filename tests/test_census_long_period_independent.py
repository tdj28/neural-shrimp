import copy,importlib.util
from pathlib import Path
import pytest
from flint import arb,ctx

def module():
 p=Path(__file__).parents[1]/'tools/audit_census_long_period.py';s=importlib.util.spec_from_file_location('census_review',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fixture(m):
 b=m.C.ball_json
 with ctx.workprec(128):
  z=b(arb(0));quarter=b(arb(1)/4);radius=b(arb(0,arb((1,-20))))
  return {'box':[radius]*2,'center':[z]*2,'settings':{'radius_exponent':-20},'preconditioner':[[b(-arb(16)/15),z],[z,b(-arb(16)/15)]],'residual':[[z],[z]],'jacobians':[[[quarter,z],[z,quarter]]]*2,'orbit_enclosures':[[radius]*2,[b(arb(2)),b(arb(3))],[radius]*2],'attraction_witness_power':1}

def test_independent_strict_root_divisors_and_power():
 m=module()
 with ctx.workprec(384):
  r=fixture(m);out=m.saved_arithmetic(r,period=2,n=2)
  assert out['strict_inclusions']==2 and out['attraction_power']==1 and out['divisors'][0]['divisor']==1
  for changed in ['residual','divisor','power','radius']:
   bad=copy.deepcopy(r)
   if changed=='residual':bad['residual'][0][0]=m.C.ball_json(arb(2))
   if changed=='divisor':bad['orbit_enclosures'][1]=bad['box']
   if changed=='power':bad['attraction_witness_power']=3
   if changed=='radius':bad['box'][0]['rad']=[1,-19]
   with pytest.raises(ValueError):m.saved_arithmetic(bad,period=2,n=2)

def test_exact_all72_strict_membership_not_overlap():
 m=module();inner=[{'mid':[1,0],'rad':[1,-288]}]*72;outer=[{'mid':[1,0],'rad':[1,-286]}]*72
 assert all(m.contains(outer,inner))
 with pytest.raises(ValueError):m.contains(inner,outer)
 with pytest.raises(ValueError):m.contains(inner,inner)
 with pytest.raises(ValueError):m.contains(outer[:-1],inner)

def test_exact_center_no_newton_and_precise_cost():
 m=module();center=[{'mid':[3,-400],'rad':[0,0]}]*72;inputs={'approximation':['0']*72,'period':58}
 assert m.fresh_inputs(inputs,center)['approximation']==[str(m.F(3)*m.F(2)**-400)]*72
 assert inputs['approximation']==['0']*72
 assert m.forecast([.25,.3,.2])==pytest.approx(99.6)
 with pytest.raises(ValueError):m.forecast([.2,.3])
 with pytest.raises(ValueError):m.fresh_inputs(inputs,[{'mid':[1,0],'rad':[1,-400]}]*72)

def test_nonfinite_negative_ball_rejected():
 m=module()
 for bad in [{'mid':[0,0],'rad':[-1,0]},{'finite':False,'mid':[0,0],'rad':[0,0]}]:
  with pytest.raises(ValueError):m.bounds(bad)
