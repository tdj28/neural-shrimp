import sys
from pathlib import Path
from fractions import Fraction
import pytest
from flint import arb
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/reviews'))
import review_s2_fold_connection as R
from neural_shrimp.rigorous_orbits import ball_json

def chart():
 z=ball_json(arb(0));one=ball_json(arb(1));rad=ball_json(arb((1,-10)))
 return {'normal':['0']*234+['1'],'origin':['0']*235,'parameter_center':'0','settings':{'halfwidth_exponent':-10},'center':[z]*235,'predictor_slope':[[z] for _ in range(234)]+[[one]],'predictor_curvature':[[z] for _ in range(235)],'coordinate_radii':[rad]*235}

def test_full_state_and_rate_membership():
 c=chart();z=ball_json(arb(0));b=[z]*235
 assert all(q['passed'] for q in R.membership(c,b,384))
 b[117]=ball_json(arb((1,-9)))
 assert not any(q['passed'] for q in R.membership(c,b,384))

def test_strict_parameter_edge_rejected():
 c=chart();b=[ball_json(arb(0))]*234+[ball_json(arb((1,-10)))]
 assert not any(q['passed'] for q in R.membership(c,b,384))

def test_no_tolerance_in_full_box_containment():
 a=ball_json(arb(0,arb((1,-20))));b=ball_json(arb((1,-19)))
 assert R.contains(a,a) and not R.contains(a,b)

def test_missing_native_coordinate_rejected():
 with pytest.raises(ValueError):R.membership(chart(),[ball_json(arb(0))]*234,384)
