import copy,sys
from pathlib import Path
import pytest
from flint import ctx,arb
sys.path[:0]=[str(Path(__file__).parents[2]/'tools/reviews'),str(Path(__file__).parents[1])]
from test_rigorous_arclength_localization import parent
from test_rigorous_arclength_connection import Fold
from neural_shrimp.rigorous_arclength_localization import localize_endpoint
from neural_shrimp.rigorous_orbits import ball_json,vector_json
import review_endpoint_localization_math as R

def fixture(shift=False):
 with ctx.workprec(256):
  p=parent()
  if shift:p['endpoint_root_enclosures'][1]=vector_json([arb('129/256',arb('1/128')),arb('321/256',arb('1/128'))])
  r=localize_endpoint(Fold(),p)
  return p,r['start'],r['intersections'],r

@pytest.mark.parametrize('shift',[False,True])
def test_independent_intersections_and_stronger_precision(shift):
 p,s,rows,r=fixture(shift)
 for prec in(256,384):assert R.audit(p,s,rows,r,prec)['passed']

@pytest.mark.parametrize('field',['gauge','center','domain','residual','jacobian','image','intersection','order','final','preconditioner'])
def test_corruption_rejected(field):
 p,s,rows,r=fixture();one=ball_json(arb(1))
 if field=='gauge':s['old_parameter']='1'
 if field=='center':s['center'][0]=one
 if field=='domain':rows[0]['derivative_domain'][0]=one
 if field=='residual':s['center_residual'][0][0]=one
 if field=='jacobian':rows[0]['jacobian'][0][0]=one
 if field=='image':rows[0]['image'][0][0]=one
 if field=='intersection':rows[0]['retained_error'][0][0]=one
 if field=='order':rows[1]['prior_error'][0][0]=one
 if field=='final':r['final_state_box'][0]=one
 if field=='preconditioner':s['preconditioner'][0][0]=one
 with pytest.raises(ValueError):R.audit(p,s,rows,r)
