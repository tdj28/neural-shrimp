import importlib.util
from pathlib import Path
import pytest

def runner():
 p=Path(__file__).parents[1]/'tools/localize_clark_rnn_n12_flip_cubic.py';s=importlib.util.spec_from_file_location('local_cubic',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_strict_bisection_preserves_original_opposite_signs():
 m=runner();left={'gain':1.,'cubic':.2};right={'gain':2.,'cubic':-.2}
 for c in[.1,-.1]:
  mid={'gain':1.5,'cubic':c};a,b=m.split_bracket(left,right,mid)
  assert a['gain']<b['gain']and a['cubic']>0>b['cubic']
  assert (a is mid)==(c>0)and(b is mid)==(c<0)

@pytest.mark.parametrize('g,c',[(1.,.1),(2.,-.1),(1.5,0.),(1.5,float('nan'))])
def test_degenerate_or_outside_midpoint_abstains(g,c):
 m=runner()
 with pytest.raises(ValueError):m.split_bracket({'gain':1.,'cubic':.2},{'gain':2.,'cubic':-.2},{'gain':g,'cubic':c})

def test_fixed_resolution_and_caps():
 m=runner();assert m.ITERATIONS==24 and m.TAGS==('point01','point02')
 assert (4.116836158869373-4.071142010060755)/2**m.ITERATIONS<3e-9
