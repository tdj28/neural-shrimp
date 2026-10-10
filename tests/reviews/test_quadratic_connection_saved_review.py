import sys
from pathlib import Path
from copy import deepcopy
import pytest
from flint import arb,ctx
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'tools/reviews'),str(Path(__file__).resolve().parents[1])]
from test_rigorous_quadratic_connection import QuadraticBranch,TwoCycle
from neural_shrimp.rigorous_orbits import ball_json
from neural_shrimp.rigorous_quadratic_connection import two_quadratic_charts
from neural_shrimp.rigorous_weighted_connection import two_weighted_quadratic_charts
import review_quadratic_connection_math as R


def fixture(weighted=False,nonlinear=False):
 with ctx.workprec(256):
  model=TwoCycle()if nonlinear else QuadraticBranch();initial=[['0'],['-1']]if nonlinear else[['1']]
  inp={'initial':initial,'rate':'1','anchor':{'rate':'1','phase_boxes':[[ball_json(arb(v))for v in row]for row in initial]}}
  settings=dict(halfwidth_exponent=-10,radius_exponent=-18)
  result=(two_weighted_quadratic_charts(model,model,inp,weights=['1','1/2']if nonlinear else['1/8'],**settings)if weighted else two_quadratic_charts(model,model,inp,**settings))
 return inp,result

@pytest.mark.parametrize('weighted',[False,True])
@pytest.mark.parametrize('nonlinear',[False,True])
def test_independent_saved_taylor_root_join(weighted,nonlinear):
 i,r=fixture(weighted,nonlinear);assert r['passed'];assert R.check_join(r,i,weighted)['passed']

@pytest.mark.parametrize('mutation',['h0','h1','h2','h3','image','endpoint','anchor','weight','domain','primitive','center'])
def test_proof_mutations_rejected(mutation):
 i,r=fixture(True,True);c=r['charts'][0];one=ball_json(arb(1))
 if mutation in ('h0','h1','h2'):c['center_path_derivatives'][{'h0':'f','h1':'d','h2':'d2'}[mutation]][0][0]=one
 if mutation=='h3':c['whole_path_third_derivative'][0][0]=one
 if mutation=='image':c['image_correction'][0][0]=one
 if mutation=='endpoint':c['endpoint_root_enclosures'][1][0]=one
 if mutation=='anchor':i['anchor']['phase_boxes'][0][0]=one
 if mutation=='weight':c['exact_weights'][0]='-1'
 if mutation=='domain':c['phase_boxes'][0][0]=ball_json(arb(0))
 if mutation=='primitive':c['divisors']=[]
 if mutation=='center':r['charts'][1]['parameter_center']='2'
 with pytest.raises(ValueError):R.check_join(r,i,True)

@pytest.mark.parametrize('bits',[128,256,384])
def test_exact_decoder_nontrivial_radius(bits):
 with ctx.workprec(bits):
  saved={'mid':[3,-10],'rad':[879609305,-43],'display':'ignored'};got=ball_json(R.original_ball(saved));assert got['mid']==saved['mid']and got['rad']==saved['rad']


def test_every_bound_python_source_parses_and_driver_imports():
 import ast
 import review_weighted_connection as driver
 root=Path(__file__).resolve().parents[2]
 for name in driver.FILES:
  if name.endswith('.py'):ast.parse((root/name).read_text(),filename=name)
 assert len(driver.UNITS)==2
