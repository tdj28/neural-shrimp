import importlib.util
from pathlib import Path
import pytest
from flint import ctx
ROOT=Path(__file__).parents[1]
def load(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
P=load('profile_native_unit_pair');L=load('launch_native_unit_pair_profile')

def test_no_target_mode_and_same_owned_root():
 assert P.BASE==L.BASE
 assert 'argparse'not in (ROOT/'tools/profile_native_unit_pair.py').read_text()
 assert P.SETTINGS['period']==18

def test_synthetic_lift_full704_and_both_native_calls_accounted():
 calls=[]
 with ctx.workprec(256):
  initial,anchor,lift=P.fixture(lambda:calls.append(1))
 assert len(initial)==len(anchor['box'])==704 and len(calls)==36
 assert lift['closure_contains_zero']and anchor['parameter']=='0'
 assert all(name in P.SOURCES for name in ('src/neural_shrimp/rigorous_unit_pair.py','tools/launch_native_unit_pair_profile.py','src/neural_shrimp/models.py'))

def test_projection_strict_time_memory_and_storage():
 assert P.projection([1,2],[.01]*12,100000,1000)['resource_admitted']
 assert not P.projection([100,100],[1]*12,100000,1000)['resource_admitted']
 assert not P.projection([1,1],[.01]*12,4*1024**2,1000)['resource_admitted']
 assert not P.projection([1,1],[.01]*12,100000,300*1024**2)['resource_admitted']
@pytest.mark.parametrize('bad',[0,-1,True,float('inf'),float('nan')])
def test_invalid_measurement_rejected(bad):
 with pytest.raises(ValueError):P.projection([bad,1],[.01]*12,100000,1000)
