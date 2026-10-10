from copy import deepcopy
import sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/reviews'))
import review_localization_math as review
from neural_shrimp.rigorous_orbits import ball_json,matrix_json
from neural_shrimp.rigorous_root_localization import localize_root

class Affine:
 n=1
 def step(self,x,h):return [(x[0]+h)/2],arb_mat([[arb('1/2')]]),[arb('1/2')]

def fixture():
 with ctx.workprec(256):
  chart={'gates':{'uniform_root':True,'primitive_period':True},'statement':{'period':1,'phase_dimension':1},'parameter_center':'1','exact_parameter_endpoints':['3/4','5/4'],'center':[[ball_json(arb(1))]],'predictor_slope':matrix_json(arb_mat([[arb(1)]])),'preconditioner':matrix_json(arb_mat([[arb(-2)]])),'image_correction':matrix_json(arb_mat([[arb(0,arb('1/1024'))]]))}
  loc=localize_root(Affine(),chart,'1','9/8')
 return chart,loc

def test_independent_saved_affine_reduction():
 c,l=fixture();r=review.audit_localization(c,l,['1','9/8']);assert r['passed'] and len(r['refinements'])==3

@pytest.mark.parametrize('where',['initial','domain','image','intersection','final','primitive','chain'])
def test_tampering_rejected(where):
 c,l=fixture();zero=ball_json(arb(0))
 if where=='initial':l['initial_correction'][0][0]=zero
 if where=='domain':l['refinements'][0]['derivative_phase_boxes'][0][0]=zero
 if where=='image':l['refinements'][0]['image'][0][0]=ball_json(arb(1))
 if where=='intersection':l['refinements'][0]['retained_correction'][0][0]=ball_json(arb(1))
 if where=='final':l['phase_boxes'][0][0]=zero
 if where=='primitive':l['inherited_primitive_period']=False
 if where=='chain':l['refinements']=l['refinements'][::-1]
 with pytest.raises(ValueError):review.audit_localization(c,l,['1','9/8'])

def test_hull_zero_required_even_when_prior_excludes_it():
 c,l=fixture();l['initial_correction'][0][0]=ball_json(arb('1/1000','1/10000'));l['refinements'][0]['prior_correction']=deepcopy(l['initial_correction']);l['refinements'][0]['predictor_error_segment']=deepcopy(l['initial_correction'])
 with pytest.raises(ValueError):review.audit_localization(c,l,['1','9/8'])

@pytest.mark.parametrize('tamper',[False,True])
def test_parent_image_dependency_not_just_strict_radius(tamper):
 c,_=fixture();c.update(settings={'radius_exponent':-10,'halfwidth_exponent':-3},phase_jacobians=[matrix_json(arb_mat([[arb('.5')]]))],jacobian_tube=matrix_json(arb_mat([[arb('-.5')]])),parameter_partial_tube=matrix_json(arb_mat([[arb('.5')]])),residual_center=matrix_json(arb_mat([[arb(0)]])),image_correction=matrix_json(arb_mat([[arb('1e-20')if tamper else arb(0)]])))
 if tamper:
  with pytest.raises(ValueError,match='inherited parent image'):review.parent_image(c)
 else:assert review.parent_image(c)['passed']
