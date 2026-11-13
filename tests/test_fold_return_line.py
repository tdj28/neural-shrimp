import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('fold_return_line',Path(__file__).parents[1]/'tools/fold_return_line.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_reference_match_requires_order_not_only_set_membership():
    orbit=np.array([[1.,0.],[0.,1.],[-1.,-1.]])
    assert m.reference_match(orbit[np.arange(150)%3],orbit)['passed']
    assert not m.reference_match(orbit[-np.arange(150)%3],orbit)['passed']
    wrong=orbit[np.arange(150)%3].copy();wrong[-50,0]+=.01
    assert not m.reference_match(wrong,orbit)['passed']


def test_reference_match_rejects_missing_and_nonfinite_evidence():
    orbit=np.array([[1.,0.],[0.,1.],[-1.,-1.]])
    with pytest.raises(ValueError):m.reference_match(orbit,orbit)
    h=orbit[np.arange(150)%3].copy();h[-1]=np.nan
    with pytest.raises(ValueError):m.reference_match(h,orbit)


def test_repeated_visits_include_phase_switch_and_ignore_grazing():
    v=np.full(40,4.);t=np.zeros(40)
    v[2:12:2]=.1;v[19:29:2]=-.2;v[34:38:2]=.1
    events=m.channel_visits(v,t,1.)
    assert [(x['entry'],x['last_inside'],x['phase']) for x in events]==[(2,10,0),(19,27,1)]
    t[23]=.1
    assert len(m.channel_visits(v,t,1.))==1


def test_visit_does_not_become_two_due_to_two_step_sampling():
    v=np.full(20,4.);v[1::2]=0.
    events=m.channel_visits(v,np.zeros(20),1.)
    assert len(events)==1 and events[0]['left_censored'] and events[0]['right_censored']
