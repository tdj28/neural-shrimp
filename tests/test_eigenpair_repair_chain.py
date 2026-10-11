import copy,importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from neural_shrimp.rigorous_eigenpair_inflation_march import march

def driver():
    p=Path(__file__).parents[1]/'tools/continue_eigenpair_repair_chain.py';s=importlib.util.spec_from_file_location('repair_chain',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def args(m,unit=5):
    return SimpleNamespace(unit=unit,expected_right=str(m.Fraction(16*(unit-1)+23,128)),parent=m.T.BASE/f'attempt-{unit-1:03d}-wider-march',**m.FIRST_PINS)

def test_one_source_inventory_transition_only():
    m=driver();assert m.parent_inventory(5)==m.F.SOURCES
    assert m.parent_inventory(6)==m.SOURCES and m.parent_inventory(64)==m.SOURCES
    for unit in [4,65,5.,True]:
        with pytest.raises(ValueError):m.parent_inventory(unit)

def test_original_repair_core_and_parent_validation_unchanged():
    m=driver();assert m.march is march and m.validate_parent is m.C.validate_parent
    assert m.march is not m.F.march
    assert set(m.F.SOURCES)<=set(m.SOURCES)
    assert all((Path(__file__).parents[1]/p).is_file() for p in m.SOURCES)

def test_exact_forecast_parent_is_required():
    m=driver();a=args(m);m.install(a)
    for key in m.FIRST_PINS:
        b=copy.copy(a);setattr(b,key,'0'*len(getattr(a,key)))
        with pytest.raises(ValueError):m.install(b)

@pytest.mark.parametrize('key,value',[('unit',4),('expected_right','86/128'),('parent',Path('/tmp/foreign')),('parent_source','bad')])
def test_invalid_parent_before_outputs(key,value):
    m=driver();a=args(m);setattr(a,key,value)
    with pytest.raises(ValueError):m.install(a)

def test_later_endpoint_and_owned_output_are_exact():
    m=driver();a=args(m,6);a.parent_source='a'*40;m.install(a)
    a.expected_right='87/128'
    with pytest.raises(ValueError):m.install(a)
