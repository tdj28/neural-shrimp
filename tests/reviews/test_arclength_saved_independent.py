import ast,sys
from copy import deepcopy
from pathlib import Path
import pytest
from flint import arb
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'tools/reviews'),str(Path(__file__).resolve().parents[1])]
from test_rigorous_arclength_connection import Fold,inputs
from neural_shrimp.rigorous_arclength_connection import two_arclength_charts
from neural_shrimp.rigorous_orbits import ball_json
import review_arclength_connection_math as R

def fixture():
 i=inputs();v=two_arclength_charts(Fold(),Fold(),i);assert v['passed'];return i,v

def test_independent_fold_chart_and_stronger_endpoint_join():
 i,v=fixture();previous=None
 for k,c in enumerate(v['charts']):
  a=c['anchor'];assert R.chart_check(c,a,i['normal'],i['origin'],i['weights'],256)['passed']
  if previous:a={'state_box':previous['fresh_endpoints'][1],'exact_parameter':v['charts'][k-1]['exact_parameter_endpoints'][1]}
  previous=R.chart_check(c,a,i['normal'],i['origin'],i['weights'],384);assert previous['passed']

@pytest.mark.parametrize('field',['gauge_row','normal','origin','weight','partial','h0','h1','h2','h3','image','endpoint','domain','anchor','exact_gauge'])
def test_independent_mutation_rejected(field):
 i,v=fixture();c=v['charts'][0];a=deepcopy(c['anchor']);one=ball_json(arb(1))
 if field=='gauge_row':c['jacobian_tube'][-1][0]=ball_json(arb(2))
 if field=='normal':c['normal'][0]='2'
 if field=='origin':c['origin'][0]='2'
 if field=='weight':c['exact_weights'][0]='-1'
 if field=='partial':c['jacobian_tube'][0][-1]=one
 if field in ('h0','h1','h2','h3'):c['center_path_derivatives'][{'h0':'f','h1':'d','h2':'d2','h3':'d3'}[field]][-1][0]=one
 if field=='image':c['image_correction'][0][0]=one
 if field=='endpoint':c['endpoint_root_enclosures'][1][0]=one
 if field=='domain':c['augmented_box'][-1]=ball_json(arb(0))
 if field=='anchor':a['state_box'][1]=ball_json(arb(2))
 if field=='exact_gauge':a['exact_parameter']='1'
 with pytest.raises(ValueError):R.chart_check(c,a,['1','0'],['0','1'],['1','1'],256)

def test_native_period_two_divisor_and_phase_order():
 from test_rigorous_quadratic_connection import TwoCycle
 i={'initial':['0','-1','1'],'normal':['0','0','1'],'origin':['0','-1','1'],'weights':['1']*3,'period':2,'anchor':{'state_box':[ball_json(arb(0)),ball_json(arb(-1)),ball_json(arb(1))]}}
 v=two_arclength_charts(TwoCycle(),TwoCycle(),i,halfwidth_exponent=-10,radius_exponent=-18)
 assert R.chart_check(v['charts'][0],i['anchor'],i['normal'],i['origin'],i['weights'])['passed']
 c=deepcopy(v['charts'][0]);c['divisors']=[]
 with pytest.raises(ValueError):R.chart_check(c,i['anchor'],i['normal'],i['origin'],i['weights'])


def test_bound_review_python_sources_parse():
 import review_arclength_connection as driver
 root=Path(__file__).resolve().parents[2]
 for name in driver.FILES:
  if name.endswith('.py'):ast.parse((root/name).read_text(),filename=name)


def test_restricted_saved_Taylor_decoder_replay():
 from neural_shrimp.rigorous_chart_restriction import restrict_chart
 from fractions import Fraction
 i,v=fixture();old=v['charts'][0]
 proposal={'halfwidth_exponent':-10,'radius_exponent':-18,'exact_weights':i['weights'],'exact_coordinate_radii':[str(Fraction(2)**-18*Fraction(w)) for w in i['weights']]}
 c=restrict_chart(Fold(),old,proposal)
 assert c['passed']
 assert R.chart_check(c,c['anchor'],c['normal'],c['origin'],c['exact_weights'],256,decoded_taylor=True)['passed']
 assert R.chart_check(c,c['anchor'],c['normal'],c['origin'],c['exact_weights'],384,decoded_taylor=True)['passed']
