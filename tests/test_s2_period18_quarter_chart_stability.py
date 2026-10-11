import importlib.util
from pathlib import Path
from fractions import Fraction
import pytest

def module():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_quarter_chart_stability.py';s=importlib.util.spec_from_file_location('quarter_chart',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_exact_complete_quarters_and_reject_gaps():
    m=module();ends=['17/64','35/128'];rows=m.partition(ends);assert m.coverage(rows,ends)==ends
    assert all(Fraction(b)-Fraction(a)==Fraction(1,512)for a,b in rows)
    for bad in [rows[:-1],rows[::-1],[rows[0]]*4]:
        with pytest.raises(ValueError):m.coverage(bad,ends)

def test_profile_not_repartitioned(monkeypatch):
    m=module();p={'slices':[['-1/2048','1/2048']]};monkeypatch.setattr(m,'ORIGINAL_INPUT',lambda mode:p)
    assert m.input_for('profile')is p and len(p['slices'])==1

def test_only_fixed_parent_and_new_domain_partition(monkeypatch):
    m=module();p={'coverage_endpoints':['17/64','35/128'],'selected_artifact':{'global_chart':146},'chart':{'untouched':'native'}};monkeypatch.setattr(m,'ORIGINAL_INPUT',lambda mode:p)
    x=m.input_for('target');assert x['chart']=={'untouched':'native'}and len(x['slices'])==4
    p['selected_artifact']['global_chart']=145
    with pytest.raises(ValueError):m.input_for('target')

def test_wrapper_restores_on_failure(monkeypatch,tmp_path):
    m=module();old=m.base.sources,m.base.input_for,m.base.single_coverage
    def run(*args):
        assert m.base.sources is m.sources and m.base.input_for is m.input_for and m.base.single_coverage is m.coverage
        raise OSError('save')
    monkeypatch.setattr(m.base,'execute',run)
    with pytest.raises(OSError):m.execute(tmp_path,'profile')
    assert (m.base.sources,m.base.input_for,m.base.single_coverage)==old
