import importlib.util
from pathlib import Path
import pytest

def module():
 p=Path(__file__).parents[1]/'tools/certify_native_unit_pair_flip.py';s=importlib.util.spec_from_file_location('rigorous_flip_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_fixed_whole_root_cost_admission():
 m=module();r=m.forecast([.4,.5],[900000,900000],50000);assert r['admitted']and r['native_calls']==324
 assert not m.forecast([4.,4.],[900000,900000],50000)['admitted']
 assert not m.forecast([.5,.5],[4000000,4000000],50000)['admitted']
 with pytest.raises(ValueError):m.forecast([.4],[900000],50000)


def test_unproved_or_wrong_dimension_root_cannot_enter_conditional_qualification():
 m=module()
 with pytest.raises(ValueError):m.qualify_same_root({'passed':False})
 with pytest.raises(ValueError):m.qualify_same_root({'passed':True,'augmented_dimension':41})
