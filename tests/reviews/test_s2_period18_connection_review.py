import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[2]/"tools/reviews/review_s2_period18_connection.py"
s=importlib.util.spec_from_file_location("cr",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def b(mid,rad):return {"mid":[mid,-8],"rad":[rad,-8]}
def test_strict_inclusion_excludes_touching_and_overlapping_boxes():
 assert m.subset(b(0,1),b(0,2))
 assert not m.subset(b(1,1),b(0,2))
 assert not m.subset(b(2,1),b(0,2))
 assert m.subset(b(1,1),b(0,2),strict=False)
def test_negative_radius_and_inexact_pair_rejected():
 with pytest.raises(ValueError):m.interval(b(0,-1))
 with pytest.raises(ValueError):m.pair([1.0,-8])
