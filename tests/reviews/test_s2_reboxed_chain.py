import sys,copy,importlib.util
from fractions import Fraction
from pathlib import Path
import pytest
from flint import ctx,arb
sys.path[:0]=[str(Path(__file__).parents[2]/'tools/reviews'),str(Path(__file__).parents[1])]
import review_reboxed_arclength_math as M
from test_rigorous_arclength_connection import Fold,inputs
from neural_shrimp.rigorous_arclength_connection import two_arclength_charts
from neural_shrimp.rigorous_arclength_rebox import rebox,proposed_radii
from neural_shrimp.rigorous_orbits import ball_json

def fixture():
 with ctx.workprec(256):
  c=two_arclength_charts(Fold(),Fold(),inputs())['charts'][-1]
  p=proposed_radii(c);r=rebox(Fold(),c,p)
  assert r['passed'];return c,r

def test_conservative_decode_replay_and_stronger_independent_reduction():
 c,r=fixture()
 for precision in(256,384):assert M.chart_check(r,r['anchor'],r['normal'],r['origin'],r['exact_weights'],precision)['passed']

@pytest.mark.parametrize('field',['residual','jet','domain','image','endpoint','radii','weight'])
def test_reboxed_math_mutation_rejected(field):
 c,r=fixture();one=ball_json(arb(1))
 if field=='residual':r['predictor_residual'][0][0]=one
 if field=='jet':r['center_path_derivatives']['f'][0][0]=one
 if field=='domain':r['augmented_box'][0]=one
 if field=='image':r['image_correction'][0][0]=one
 if field=='endpoint':r['endpoint_root_enclosures'][1][0]=one
 if field=='radii':r['coordinate_radii'][0]=one
 if field=='weight':r['exact_weights'][0]='0'
 with pytest.raises(ValueError):M.chart_check(r,r['anchor'],r['normal'],r['origin'],r['exact_weights'],256)

def test_old_charts_require_old_reducer():
 c,r=fixture()
 with pytest.raises(ValueError,match='reboxed'):M.chart_check(c,c['anchor'],c['normal'],c['origin'],c['exact_weights'],256)

def test_nonzero_interval_taylor_residual_uses_the_extra_decode():
 from flint import arb_mat
 class Quartic(Fold):
  def step(self,x,h):
   z=x[0];return [z+z*z+z*z*z+z*z*z*z-(h-1)],arb_mat([[1+2*z+3*z*z+4*z*z*z]]),[arb(-1)]
  def jet_step(self,x,h):
   z=x[0];return [z+z*z+z*z*z+z*z*z*z-(h-1)]
 with ctx.workprec(256):
  m=Quartic();c=two_arclength_charts(m,m,inputs(),halfwidth_exponent=-10,radius_exponent=-22)['charts'][-1]
  assert c['predictor_residual'][0][0]['rad'][0]>0
  # Exercise decoder algebra with prescribed exact radii, independently of
  # the floating proposal heuristic and its radius-cap admission.
  proposal={'radius_exponent':-22,'exact_weights':['1','1'],
            'exact_coordinate_radii':[str(Fraction(2)**-22)]*2}
  r=rebox(m,c,proposal);assert r['passed']
  for precision in(256,384):assert M.chart_check(r,r['anchor'],r['normal'],r['origin'],r['exact_weights'],precision)['passed']
