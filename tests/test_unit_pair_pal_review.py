import sys
from pathlib import Path
from fractions import Fraction as F
import pytest
from flint import arb,ctx
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools/reviews')]
import review_unit_pair_pal_math as A
from neural_shrimp.rigorous_orbits import vector_json

def test_direct_convolution_retains_all_cubic_terms():
    assert A.mul([1,2,3,4],[5,6,7,8])==[5,16,34,60]
def test_entire_series_at_origin_derivatives_through_three():
    with ctx.workprec(256):
        t=[arb(0),arb(1),arb(0),arb(0)]
        for offset in (0,1):
            values=A.terminal_series(t,offset)
            from math import factorial
            for j,v in enumerate(values):
                lo,hi=A.M.interval(A.ball_json(v));assert lo<=F((-1)**j,factorial(2*j+offset))<=hi
def test_anchor_uses_full_parameter_projection_and_strict_box():
    c=[arb(0)]*3;s=[arb(1),arb(0),arb(0)];a=[arb(0)]*3;normal=s;rr=[arb(1)]*3
    good=vector_json([arb(0),arb(0),arb(0)])
    assert A.anchor_membership(good,c,s,a,normal,c,arb(0),arb(1),rr)['passed']
    bad=vector_json([arb(0),arb(1),arb(0)])
    assert not A.anchor_membership(bad,c,s,a,normal,c,arb(0),arb(1),rr)['passed']
    outside=vector_json([arb(2),arb(0),arb(0)])
    assert not A.anchor_membership(outside,c,s,a,normal,c,arb(0),arb(1),rr)['passed']
