import importlib.util
from pathlib import Path
import pytest

def driver():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_whole_chart_stability.py';spec=importlib.util.spec_from_file_location('full_chart_stability',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_single_full_chart_rejects_gap_or_subslice():
    m=driver();ends=['17/64','35/128'];assert m.single_coverage([ends],ends)==ends
    for rows in [[],[['17/64','69/256']],[ends,ends]]:
        with pytest.raises(ValueError):m.single_coverage(rows,ends)

def test_profile_never_opens_target(monkeypatch):
    m=driver();monkeypatch.setattr(m,'ORIGINAL_INPUT',lambda mode:{'synthetic':True});monkeypatch.setattr(m,'completed_adjacent',lambda:pytest.fail('target opened'))
    assert m.input_for('profile')=={'synthetic':True}

def test_wrapper_restores_both_coverage_and_source(monkeypatch,tmp_path):
    m=driver();saved=m.base.sources,m.base.input_for,m.base.coverage
    def call(*a):
        assert m.base.sources is m.sources
        m.base.input_for('profile')
        assert m.base.coverage is m.single_coverage
        raise OSError('durable save')
    monkeypatch.setattr(m,'ORIGINAL_INPUT',lambda mode:{'synthetic':True});monkeypatch.setattr(m.base,'execute',call)
    with pytest.raises(OSError):m.execute(tmp_path,'profile')
    assert (m.base.sources,m.base.input_for,m.base.coverage)==saved
