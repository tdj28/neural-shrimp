import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('fold_return_cycle_map',Path(__file__).parents[1]/'tools/fold_return_cycle_map.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_full_passage_counts_initial_confirmed_samples_and_keeps_phase():
    v=np.full(26,100.);v[0:16:2]=np.linspace(-.5,.8,8);v[16::2]=[1.1,1.5,2.,2.8,3.4]
    r=m.audit_departure(v,np.zeros(26),1.)
    assert r=={'kind':'next_natural_exit','update':24,'v':3.4,'transverse':0.,'channel_readings':10}


def test_short_or_recrossing_passage_is_not_silently_accepted():
    v=np.zeros(26);v[4::2]=4.
    assert m.audit_departure(v,np.zeros(26),1.)['kind']=='short_passage'
    v=np.zeros(26);v[16]=1.2;v[20::2]=4
    assert m.audit_departure(v,np.zeros(26),1.)['kind']=='channel_recrossing'
    with pytest.raises(ValueError):m.audit_departure(np.ones(8)*4,np.zeros(8),1.)
