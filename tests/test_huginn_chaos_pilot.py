import importlib.util
from pathlib import Path
import math
import pytest
spec=importlib.util.spec_from_file_location('pilot',Path(__file__).parents[1]/'tools/huginn_chaos_pilot.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)

def test_linear_growth():
    assert pilot.growth_log(.5e-4,1e-4)==pytest.approx(math.log(.5))
    assert pilot.growth_log(2e-4,1e-4)==pytest.approx(math.log(2))

@pytest.mark.parametrize('distance,radius',[(0,1),(-1,1),(float('nan'),1),(1,0),(1,float('inf'))])
def test_precision_failure(distance,radius):
    with pytest.raises(ValueError):pilot.growth_log(distance,radius)
