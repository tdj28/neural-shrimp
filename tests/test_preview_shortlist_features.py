"""Pre-preview access and numerical contracts; no research model fitting."""
import builtins
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import preview_shortlist_features as features


def synthetic():
    rng = np.random.default_rng(7)
    certificate = dict(period=37., transverse_spectral_radius=.2,
                       orbit_activations=rng.uniform(-1, 1, (2701, 3)))
    return dict(alpha=1.065, gamma=10.3), certificate, rng.normal(size=(7, 3, 2)), rng.normal(size=(7, 3))


def test_schema_is_exactly_pre_preview_and_excludes_safe_redundant_field():
    from hopfield_information_features import BASE_FEATURE_NAMES
    assert features.SHARED_FEATURE_NAMES == BASE_FEATURE_NAMES[:37]
    assert len(features.SHARED_FEATURE_NAMES) == len(set(features.SHARED_FEATURE_NAMES)) == 37
    assert not any('preview' in name for name in features.SHARED_FEATURE_NAMES)
    assert 'local_projected_mse' not in features.SHARED_FEATURE_NAMES
    assert len(features.GEOMETRY_FEATURE_NAMES) == 6


def test_pure_builder_reads_only_whitelisted_inputs():
    anchor, certificate, s, v = synthetic()

    class AllowedOnly(dict):
        def __getitem__(self, key):
            if key not in self.allowed:
                raise AssertionError(f'Forbidden feature input: {key}')
            return super().__getitem__(key)

    a = AllowedOnly(anchor, preview_score=12345, target_score=67890, eligibility=False)
    a.allowed = {'alpha', 'gamma'}
    c = AllowedOnly(certificate, preview_metrics=object(), target_metrics=object())
    c.allowed = {'period', 'transverse_spectral_radius', 'orbit_activations'}
    expected = features.shared_features(anchor, 675, [1., 0.], .001, certificate, s, v)
    actual = features.shared_features(a, 675, [1., 0.], .001, c, s, v)
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(TypeError):
        features.shared_features(a, 675, [1., 0.], .001, c, s, v, preview_metrics={})


def test_certificate_endpoint_is_excluded_and_phase_is_fixed():
    a, c, s, v = synthetic()
    expected = features.shared_features(a, 675, [1., 0.], .001, c, s, v)
    changed = copy.deepcopy(c)
    changed['orbit_activations'][-1] = 1e6
    np.testing.assert_array_equal(features.shared_features(a, 675, [1., 0.], .001, changed, s, v), expected)
    np.testing.assert_array_equal(expected[28:31], c['orbit_activations'][675])
    np.testing.assert_array_equal(expected[31:34], c['orbit_activations'][:-1].std(axis=0))


def test_projection_and_zero_velocity_have_independent_linear_response_meaning():
    a, c, s, v = synthetic()
    d = np.array([.6, .8]); delta = .001*d
    result = features.shared_features(a, 675, d, .001, c, s, v)
    response = s.reshape(-1, 2) @ delta
    flat_v = v.ravel()
    projection = float(flat_v @ response / (flat_v @ flat_v))
    assert result[34] == pytest.approx(np.sqrt(np.mean((response - projection*flat_v)**2)), abs=1e-18)
    assert result[35] == pytest.approx(np.sqrt(np.mean(response**2)), abs=1e-18)
    assert result[36] == pytest.approx(projection/c['period'], abs=1e-18)
    zero = features.shared_features(a, 675, d, .001, c, s, np.zeros_like(v))
    assert zero[34] == zero[35] and zero[36] == 0


def test_direct_response_preserves_small_difference_of_large_columns():
    a, c, _, _ = synthetic()
    s = np.empty((7, 3, 2)); s[..., 0] = 1e6; s[..., 1] = 1e6 + 1e-4
    d = np.array([1., -1.])/np.sqrt(2)
    result = features.shared_features(a, 675, d, .001, c, s, np.zeros((7, 3)))
    columns = s.reshape(-1, 2)
    delta = .001*d
    response = np.sum(columns.astype(np.longdouble)*delta.astype(np.longdouble), axis=1)
    reference = float(np.sqrt(np.mean(response**2)))
    gram_value = float(np.sqrt(max(0., delta @ (columns.T @ columns / len(columns)) @ delta)))
    assert result[34] > 0
    assert abs(result[34] - reference) < 2e-13
    assert abs(gram_value - reference) > 100 * abs(result[34] - reference)


@pytest.mark.parametrize('change', ['direction', 'phase', 'radius', 'certificate', 'sensitivity', 'velocity'])
def test_incomplete_or_nonfinite_feature_inputs_are_rejected(change):
    a, c, s, v = synthetic(); phase=675; radius=.001; d=[1., 0.]
    if change == 'direction': d=[float('nan'), 0.]
    elif change == 'phase': phase=True
    elif change == 'radius': radius=0.
    elif change == 'certificate': c['orbit_activations']=c['orbit_activations'][:-1]
    elif change == 'sensitivity': s[0, 0, 0]=np.nan
    else: v=v[:-1]
    with pytest.raises(ValueError):
        features.shared_features(a, phase, d, radius, c, s, v)


def test_crop_boundary_and_physical_origin_identity():
    a, c, s, v = synthetic(); a['alpha']=1.08 - .001
    at_edge = features.shared_features(a, 675, [1., 0.], .001, c, s, v)
    assert at_edge[26] == 1 and at_edge[27] == 0
    beyond = features.shared_features(a, 675, [1., 0.], .004, c, s, v)
    assert beyond[26] == 0 and beyond[27] < 0
    assert features.physical_group(1.065, 10.3) == features.physical_group(np.float64(1.065), np.float64(10.3))
    assert features.physical_group(1.065, 10.3) != features.physical_group(np.nextafter(1.065, np.inf), 10.3)


@pytest.fixture(scope='module')
def actual_bundle():
    return features.load_features()


def test_real_complete_census_names_groups_and_original_direction_parity(actual_bundle):
    bundle = actual_bundle
    assert bundle['shared'].shape == (3456, 37) and bundle['geometry'].shape == (3456, 6)
    assert len({r['group'] for r in bundle['rows']}) == 24
    assert len(bundle['source_bindings']) == 281
    assert len({(r['group'], r['phase'], r['radius']) for r in bundle['rows']}) == 192
    assert not bundle['shared'].flags.writeable and not bundle['geometry'].flags.writeable
    old = {r['id']: r for role in ('development', 'test') for r in json.loads(
        (features.ROOT/features.OLD_REL/f'{role}-features.json').read_text())}
    assert {r['id'] for r in bundle['rows']} == set(old)
    for row in bundle['rows']:
        np.testing.assert_array_equal(row['direction'], old[row['id']]['direction'])
    # This historical parity checker is separate from the label-free builder.
    shared = np.array([old[r['id']]['base'][:37] for r in bundle['rows']])
    geometry = np.array([old[r['id']]['geometry'] for r in bundle['rows']])
    np.testing.assert_allclose(bundle['shared'], shared, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(bundle['geometry'], geometry, rtol=1e-12, atol=1e-14)


def test_builder_file_access_is_limited_to_bound_pre_preview_inputs(monkeypatch, actual_bundle):
    # Derive the allowlist from the frozen dataset specification, not from the
    # loader's own claimed bindings: an accidental reader cannot authorize itself.
    relative = {features.OLD_REL/name for name in
                ('receipt.json', 'manifest.json', 'development-baselines.json', 'test-baselines.json')}
    relative.update(Path(f'evidence/EXP-004/attempt-001/focus/row-{i:03d}.npz') for i in range(181))
    relative.update(features.OLD_REL/f'sensitivity/{family}-{i:02d}-p{phase}.{extension}'
                    for family in ('development', 'information') for i in range(12)
                    for phase in (675, 1350) for extension in ('json', 'npz'))
    allowed = {(features.ROOT/p).resolve() for p in relative}
    assert len(allowed) == 281
    assert {(features.ROOT/b['path']).resolve() for b in actual_bundle['source_bindings']} == allowed
    observed = set()
    path_open, builtin_open = Path.open, builtins.open

    def check(path):
        if isinstance(path, (str, bytes, Path)):
            p = Path(path).resolve()
            if p not in allowed:
                raise AssertionError(f'Unapproved feature file access: {p}')
            observed.add(p)

    def checked_path(self, *args, **kwargs):
        check(self)
        return path_open(self, *args, **kwargs)

    def checked_builtin(file, *args, **kwargs):
        check(file)
        return builtin_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', checked_path)
    monkeypatch.setattr(builtins, 'open', checked_builtin)
    rebuilt = features.load_features()
    assert observed == allowed
    np.testing.assert_array_equal(rebuilt['shared'], actual_bundle['shared'])
    np.testing.assert_array_equal(rebuilt['geometry'], actual_bundle['geometry'])


def test_origin_aliases_are_grouped_across_experiments():
    old = json.loads((features.ROOT/'evidence/EXP-007/attempt-001/manifest.json').read_text())
    new = json.loads((features.ROOT/features.OLD_REL/'manifest.json').read_text())
    aliases = {a['id']: a for a in old['anchors']}
    overlap = [a for a in new['anchors'] if a['role'] == 'development']
    for a in overlap:
        prior = aliases[a['original_origin_id']]
        assert features.physical_group(a['alpha'], a['gamma']) == features.physical_group(prior['alpha'], prior['gamma'])
    assert len(overlap) == 12
