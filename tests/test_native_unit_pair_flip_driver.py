import importlib.util
from pathlib import Path
import pytest

def module():
 p=Path(__file__).parents[1]/'tools/localize_native_unit_pair_flip.py';s=importlib.util.spec_from_file_location('flip_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_full_native_measured_counts_and_forecast():
 m=module();times={k:[.1]*6 for k in m.CAPS};sizes={k:[50000]*6 for k in m.CAPS}
 r=m.forecast(times,sizes,50000);assert r['admitted']and r['native_phase_calls']==5256
 times['augmented'][0]=2;assert not m.forecast(times,sizes,50000)['admitted']
 with pytest.raises(ValueError):m.forecast({k:[.1]*5 for k in m.CAPS},sizes,50000)


def test_no_numerical_trial_is_relabelled_rigorous():
 m=module();assert set(m.CAPS)=={'augmented','section','adjacent'}
 assert m.PINS['execution']=='f70f3e6a9696c89f4aed0c2398dbfd3b78ec1020'
