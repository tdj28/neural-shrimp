import sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools/reviews')]
from review_pal_rechart_math import endpoint_residual,L,R
from neural_shrimp.rigorous_orbits import ball_json

def test_endpoint_uses_only_whole_cubic_remainder():
    with ctx.workprec(128):
        c=arb_mat([[1,2,3,999],[-3,4,5,888]]);w=arb_mat([[99,99,99,4],[99,99,99,6]])
        v=endpoint_residual(c,w,arb(1)/2)
        assert v[0,0]==arb(13)/4 and v[1,0]==arb(1)
        with pytest.raises(ValueError):endpoint_residual(c,w,arb(0))

def test_known_root_mean_value_domain_is_mandatory():
    with ctx.workprec(128):
        box=lambda x,r:ball_json(arb(x,r))
        good=L.localize(arb_mat([[1]]),arb_mat([[1]]),arb_mat([[-1]]),[arb(0)],[box(1,.1)],[box(0,2)],[arb(1)])
        assert good['passed']and R.original_ball(good['box'][0]).contains(arb(1))
        with pytest.raises(ValueError):L.localize(arb_mat([[1]]),arb_mat([[1]]),arb_mat([[-1]]),[arb(0)],[box(1,.1)],[box(0,.2)],[arb(1)])
