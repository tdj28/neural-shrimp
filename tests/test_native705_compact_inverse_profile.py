import importlib.util
from pathlib import Path
import pytest
from flint import ctx
ROOT=Path(__file__).parents[1]
def load(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=load('profile_native705_compact_inverse');L=load('launch_native705_compact_inverse_profile')

def test_exclusive_synthetic_source_closure_and_matching_roots():
 assert P.BASE==L.BASE and P.OUT==L.BASE/'attempt001-profile/run'
 assert len(P.SOURCES)==len(set(P.SOURCES))and all((ROOT/n).is_file()for n in P.SOURCES)
 assert 'src/neural_shrimp/rigorous_gradient_path.py'in P.SOURCES and 'src/neural_shrimp/rigorous_polynomial_inverse.py'in P.SOURCES
 assert 'argparse'not in(ROOT/'tools/profile_native705_compact_inverse.py').read_text()

def test_actual13D_gradient_pair_has_full_retained_inputs_and_outputs():
 from neural_shrimp.rigorous_cusps import TwoRateCuspMap
 with ctx.workprec(256):
  inp,x,q,r,rates=P.P.gd_fixture();calls=[];events=[];P.gradient_pair(TwoRateCuspMap(inp),x,q,r,rates,tick=lambda:calls.append(1),observe=lambda k,v:events.append((k,v)))
 assert len(calls)==2 and [k for k,v in events]==['profile_gradient_input','profile_gradient_return']*2
 assert all(len(v['returns'])==13 and all(len(row['gradient'])==15 for row in v['returns'])for k,v in events if k=='profile_gradient_return')

@pytest.mark.parametrize('bad',[0,-1,True,float('nan'),float('inf')])
def test_invalid_forecast_values_fail_closed(bad):
 with pytest.raises(ValueError):P.projection([bad,1],[.01]*12,[.01]*12,100000,1000)

def test_measured_time_memory_storage_strict_gates():
 assert P.projection([1,2],[.01]*12,[.01]*12,100000,1000)['resource_admitted']
 assert not P.projection([100,100],[1]*12,[1]*12,100000,1000)['resource_admitted']
 assert not P.projection([1,2],[.01]*12,[.01]*12,4*1024**2,1000)['resource_admitted']
 assert not P.projection([1,2],[.01]*12,[.01]*12,100000,400*1024**2)['resource_admitted']


def test_profile_compact_mode_is_fixed_and_historical_root_distinct():
 assert P.SETTINGS["compact_evidence"]is True and P.BASE.name=="correlated002"
 assert "compact=True"in(ROOT/"tools/profile_native705_compact_inverse.py").read_text()
