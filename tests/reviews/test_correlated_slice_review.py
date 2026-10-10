import sys
from pathlib import Path
from copy import deepcopy
import pytest
from flint import ctx,arb
sys.path[:0]=[str(Path(__file__).parents[2]/'tools/reviews'),str(Path(__file__).parents[1])]
from test_rigorous_stability_slice import chart
from test_rigorous_arclength_slices import StableFold
from neural_shrimp.rigorous_stability_slice import stability_slice
from neural_shrimp.rigorous_orbits import ball_json
from review_correlated_slice_math import localization,tangent

def fixture():
 with ctx.workprec(256):
  c=chart();e=[];r=stability_slice(StableFold(),StableFold(),c,'1/32','1/16',observe=lambda k,v:e.append((k,v)))
 return c,e,r

def check(c,e,r):
 d=dict(e);whole=[v for k,v in e if k=='whole_slice_intersection'];center=[v for k,v in e if k=='center_slice_intersection']
 a=localization(c,d['whole_slice_start'],whole,r['whole_localization']);b=localization(c,d['center_slice_start'],center,r['center_localization'])
 t=tangent(d['whole_slice_start']['preconditioner'],whole[-1]['jacobian'],c['exact_weights'],r['uniform_tangent'],d['physical_inputs']['actual_uniform_tangent'])
 return a,b,t

def test_nonpoint_shifted_curved_branch_and_implicit_rate_tangent():
 c,e,r=fixture();a,b,t=check(c,e,r);assert a['passed'] and b['passed'] and t['passed']

@pytest.mark.parametrize('kind',['initial','domain','gauge','image','tangent'])
def test_corrupted_containment_or_derivative_is_rejected(kind):
 c,e,r=fixture()
 if kind=='initial':next(v for k,v in e if k=='whole_slice_start')['initial_error'][0][0]=ball_json(arb(1))
 if kind=='domain':next(v for k,v in e if k=='whole_slice_intersection')['derivative_domain'][0]=ball_json(arb(0))
 if kind=='gauge':next(v for k,v in e if k=='whole_slice_intersection')['jacobian'][-1][0]=ball_json(arb(0))
 if kind=='image':next(v for k,v in e if k=='whole_slice_intersection')['image'][0][0]=ball_json(arb(10))
 if kind=='tangent':next(v for k,v in e if k=='physical_inputs')['actual_uniform_tangent'][0]=ball_json(arb(0))
 with pytest.raises(ValueError):check(c,e,r)
