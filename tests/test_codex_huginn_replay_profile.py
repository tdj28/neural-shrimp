"""Pure admission controls; the full-size profile main is not executed here."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / 'codex/huginn-independent-replay'
spec = importlib.util.spec_from_file_location('huginn_profile_guard', DIRECTORY/'profile.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def plan():
    return json.loads((DIRECTORY/'synthetic-profile-001-plan.json').read_text())


def test_exact_released_source_admitted():
    p.check_plan(plan(), DIRECTORY)


@pytest.mark.parametrize('key,value', [('synthetic_only', False), ('target_access_authorized', True),
                                      ('schema', 'target-analysis')])
def test_scope_refused(key, value):
    x = plan(); x[key] = value
    with pytest.raises(ValueError): p.check_plan(x, DIRECTORY)


@pytest.mark.parametrize('value', [True, 1.0, 2])
def test_strict_thread_envelope(value):
    x = plan(); x['envelope']['cpu_threads'] = value
    with pytest.raises(ValueError): p.check_plan(x, DIRECTORY)


def test_drift_missing_source_and_counts_refused():
    for field in ['hash', 'inventory', 'count']:
        x = plan()
        if field == 'hash': x['sources']['statistics.py'] = '0'*64
        elif field == 'inventory': x['sources'].pop('profile.py')
        else: x['counts']['A_bootstrap'] = 20
        with pytest.raises(ValueError): p.check_plan(x, DIRECTORY)


def test_full_synthetic_fixture_shape_and_no_target_paths():
    spectra, a, b = p.fixtures()
    assert len(spectra) == 64 and len(a) == 256 and len(b) == 800
    assert {r['index'] for r in a} == set(range(64))
    assert all(len(r['native_residual_rows']) == 32 for r in a)
    assert all(len(r['by_steps']) == 9 for r in a)
    assert {r['index'] for r in b} == set(range(200))
    assert all(type(r['correct']) is bool and 0 <= r['new_tokens'] <= 256 for r in b)
