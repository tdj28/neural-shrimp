import importlib.util
from pathlib import Path
import pytest
from fractions import Fraction
p=Path(__file__).resolve().parents[1]/'tools/certify_s2_period18_earlier_quarter_stability.py';s=importlib.util.spec_from_file_location('earlier_quarter',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_exact_four_quarters_no_gaps_or_extra_domain():
 rows=m.partition(['25/128','27/128']);assert rows==[['53/256','27/128'],['13/64','53/256'],['51/256','13/64'],['25/128','51/256']]
 assert all(Fraction(b)-Fraction(a)==Fraction(1,256)for a,b in rows)
 assert m.coverage(rows,m.base.ENDS)==m.base.ENDS
 for changed in (rows[:-1],list(reversed(rows)),rows+[rows[0]]):
  with pytest.raises(ValueError):m.coverage(changed,m.base.ENDS)
 with pytest.raises(ValueError):m.partition(['24/128','27/128'])

def test_profile_has_no_failed_target_access(monkeypatch):
 marker=object();monkeypatch.setattr(m,'ORIGINAL_INPUT',lambda mode:marker);monkeypatch.setattr(m.io,'read',lambda _:pytest.fail('target read'))
 assert m.input_for('profile')is marker

def test_nested_overrides_restore_and_original_failed_root_is_stable(monkeypatch):
 saved=m.base.sources,m.base.input_for,m.base.single_coverage,m.base.RAW;actual=m.ORIGINAL_RAW
 def fail(*_):
  assert m.base.RAW==m.RAW and m.ORIGINAL_RAW==actual and m.base.single_coverage is m.coverage
  raise OSError('durable failed attempt')
 monkeypatch.setattr(m.base,'execute',fail)
 with pytest.raises(OSError):m.execute(m.RAW/'attempt001-profile/run','profile')
 assert (m.base.sources,m.base.input_for,m.base.single_coverage,m.base.RAW)==saved
 with pytest.raises(ValueError):m.execute(m.RAW/'wrong/run','profile')
