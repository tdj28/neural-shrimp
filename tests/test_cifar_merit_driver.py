import importlib.util
from pathlib import Path
import pytest
def module(name,file):
 p=Path(__file__).parents[1]/'tools'/file;s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module('merit_driver','explore_cifar_merit_lsmr.py');l=module('merit_launcher','launch_cifar_merit_lsmr.py')
def test_measured_gate_counts_all_six_steps():
 old=dict(status='qualification_complete',qualification_passed=True,pairings=[dict(passed=True,seconds=[.05,.05])]*3);events=[dict(kind='operation_returned',operation='action',seconds=1.),dict(kind='operation_returned',operation='direct_gradient',seconds=.01)]
 out=m.measured_gate(old,events);assert out['outer_cap']==6 and out['inner_cap']==200 and 265<out['forecast_seconds']<270
 old['pairings'][0]['seconds']=[.2,.2]
 with pytest.raises(ValueError):m.measured_gate(old,events)
def test_launcher_success_and_partial_count_failure():
 caps=m.CAPS;row=dict(status='merit_pilot_complete',comparison_qualified=True,accepted_orbit=False,spectrum_requested=False,operation_caps=caps,operation_attempted={k:0 for k in caps},operation_returned={k:0 for k in caps},steps=[],accepted_steps=0)
 assert l.scientific_complete(row)
 row['operation_attempted']['hvp']=1;assert not l.scientific_complete(row)
 assert '320s'in l.command(Path('/tmp/merit'), '0'*40)
