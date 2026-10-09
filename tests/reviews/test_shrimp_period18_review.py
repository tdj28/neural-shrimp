"""Synthetic exact-endpoint controls; no model or target proof."""
import importlib.util
from fractions import Fraction
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('period18_review',Path(__file__).parents[2]/'tools/reviews/review_shrimp_period18_certificate.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_signed_dyadic_endpoints():
    assert m.pair_fraction([-3,-2])==Fraction(-3,4)
    assert m.ball_interval({'mid':[-3,-2],'rad':[1,-3]})==(Fraction(-7,8),Fraction(-5,8))

def test_exact_farther_endpoint_and_normalization():
    lo,hi=m.ball_interval({'mid':[5,-2],'rad':[1,-4]});seed=Fraction(2)
    assert max(abs(lo-seed),abs(hi-seed))/max(Fraction(1),abs(seed))==Fraction(13,32)

def test_negative_radius_rejected():
    with pytest.raises(ValueError,match='nonnegative'):m.ball_interval({'mid':[1,0],'rad':[-1,-4]})
