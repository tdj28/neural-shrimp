"""Synthetic pre-preview contracts and old-metadata-only selection replay.

No fresh trajectory, fitted model or research label is generated/read here.
The supplied arrays below are constructed algebraically, not by integration.
"""
import builtins
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import fresh_preview_features as fresh
import preview_shortlist_features as old
from hopfield_diagnostics import hopfield_rhs
from hopfield_information_design import build_atlas_geometry


def synthetic_grid():
    period = np.zeros((181, 181), dtype=int)
    period[24:151, 20:166] = 9
    return dict(alpha=np.linspace(1.056, 1.08, 181),
        gamma=np.linspace(10.14, 10.55, 181), period=period)


def prior_manifests(grid):
    def anchors(prefix, start, count):
        return [dict(id=f'{prefix}-{i:02d}', alpha=float(grid['alpha'][2+i]),
            gamma=float(grid['gamma'][start])) for i in range(count)]
    six = anchors('drift', 2, 12)
    seven = anchors('holdout', 5, 12)
    eight = copy.deepcopy(seven) + anchors('information', 8, 12)
    return {e: dict(experiment_id=e, anchors=a) for e, a in
        [('EXP-006', six), ('EXP-007', seven), ('EXP-008', eight)]}


@pytest.fixture(scope='module')
def selected():
    grid = synthetic_grid()
    return grid, prior_manifests(grid), fresh.select_origins(grid, prior_manifests(grid))


@pytest.fixture(scope='module')
def supplied(selected):
    grid, _, selection = selected
    anchors = selection['anchors']
    atlas = build_atlas_geometry(grid, fresh.CORE_INDEX)
    certificates, sensitivities = {}, {}
    for index, anchor in enumerate(anchors):
        period = 17. + index/10
        phase_grid = np.linspace(0., 2*np.pi, 2701)
        orbit = np.column_stack((np.sin(phase_grid), .2*np.cos(phase_grid),
            np.sin(2*phase_grid)/3)) + index*.001
        certificate = dict(alpha=anchor['alpha'], gamma=anchor['gamma'],
            verified=True, attracting=True, primitive=True, peak_count=9, expected_peak_count=9,
            gates=dict.fromkeys(fresh.CERTIFICATE_GATES, True), period=period,
            transverse_spectral_radius=.25, orbit_times=np.linspace(0., period, 2701),
            orbit_states=orbit, orbit_activations=np.tanh(orbit))
        certificates[anchor['id']] = certificate
        for phase in fresh.PHASES:
            times = fresh.SENSITIVITY_TIMES.copy()
            # Deliberately simple algebraic arrays, not a claimed ODE solution.
            states = np.broadcast_to(orbit[phase], (3001, 3)).copy()
            state_sensitivity = np.empty((3001, 3, 2))
            state_sensitivity[:, :, 0] = times[:, None]*np.array([.01, .02, -.01])
            state_sensitivity[:, :, 1] = times[:, None]*np.array([-.02, .01, .005])
            derivative = 1.-np.tanh(states)**2
            velocity = derivative*np.asarray([hopfield_rhs(t, x, anchor['alpha'], anchor['gamma'])
                for t, x in zip(times, states, strict=True)])
            sensitivities[f'{anchor["id"]}-p{phase}'] = dict(times=times, states=states,
                state_sensitivity=state_sensitivity,
                activation_sensitivity=derivative[:, :, None]*state_sensitivity,
                activation_velocity=velocity, tail_direction=np.array([1., 0.]),
                solver=dict(method='DOP853', rtol=1e-10, atol=1e-12, max_step=.1))
    return anchors, atlas, certificates, sensitivities


@pytest.fixture(scope='module')
def bundle(supplied):
    return fresh.build_fresh_features(*supplied)


def test_selector_keeps_exact_alias_registry_and_is_order_independent(selected):
    grid, manifests, result = selected
    assert result['experiment_id'] == 'EXP-013'
    assert len(result['anchors']) == len(result['blocks']) == 24
    assert len(result['exclusions']) == 36 and result['alias_count'] == 48
    assert sorted(len(a['aliases']) for a in result['exclusions']) == [1]*24 + [2]*12
    changed = copy.deepcopy(manifests)
    for manifest in changed.values():
        manifest['anchors'].reverse()
        manifest['future_metrics'] = {'must_not_be_read': object()}
    assert fresh.select_origins(grid, changed) == result
    assert all(a['group'] == old.physical_group(a['alpha'], a['gamma']) for a in result['anchors'])
    assert [a['id'] for a in result['anchors']] == [f'fresh-{i:02d}' for i in range(24)]


def test_selector_exclusion_crop_and_complete_leaf_membership_independently(selected):
    grid, _, result = selected
    atlas = build_atlas_geometry(grid, fresh.CORE_INDEX)
    prior = np.asarray([[a['alpha'], a['gamma']] for a in result['exclusions']])
    widths = np.array([atlas.alpha[-1]-atlas.alpha[0], atlas.gamma[-1]-atlas.gamma[0]])
    eligible = []
    for row, column in np.argwhere(atlas.component):
        point = np.array([atlas.alpha[column], atlas.gamma[row]])
        edge = min(point[0]-atlas.alpha[0], atlas.alpha[-1]-point[0],
            point[1]-atlas.gamma[0], atlas.gamma[-1]-point[1])
        distance = np.linalg.norm((prior-point)/widths, axis=1).min()
        if atlas.signed_margin[row, column] < edge and distance > .04:
            eligible.append([int(row), int(column)])
    assert result['eligible_pixels'] == sorted(eligible)
    assert sorted(cell for b in result['blocks'] for cell in b['members']) == sorted(eligible)
    for anchor, block in zip(result['anchors'], result['blocks'], strict=True):
        assert [anchor['row'], anchor['column']] in block['members']
        normalized = np.asarray([[(atlas.alpha[c]-atlas.alpha[0])/widths[0],
            (atlas.gamma[r]-atlas.gamma[0])/widths[1]] for r, c in block['members']])
        centroid = normalized.mean(axis=0)
        chosen = min(range(len(normalized)), key=lambda i:
            (float(np.sum((normalized[i]-centroid)**2)), *block['members'][i]))
        assert block['members'][chosen] == [anchor['row'], anchor['column']]


@pytest.mark.parametrize('mutation', ['missing_manifest', 'duplicate_alias', 'duplicate_physical', 'lost_overlap', 'nonfinite'])
def test_selector_rejects_incomplete_or_changed_prior_registry(selected, mutation):
    grid, manifests, _ = selected
    changed = copy.deepcopy(manifests)
    if mutation == 'missing_manifest':
        del changed['EXP-006']
    elif mutation == 'duplicate_alias':
        changed['EXP-006']['anchors'][1]['id'] = changed['EXP-006']['anchors'][0]['id']
    elif mutation == 'duplicate_physical':
        changed['EXP-006']['anchors'][1].update({k: changed['EXP-006']['anchors'][0][k] for k in ('alpha', 'gamma')})
    elif mutation == 'lost_overlap':
        changed['EXP-008']['anchors'][0]['alpha'] += .00001
    else:
        changed['EXP-007']['anchors'][0]['gamma'] = np.nan
    with pytest.raises(ValueError):
        fresh.select_origins(grid, changed)


def test_selector_has_no_origin_replacement_or_atlas_crop_fallback(selected):
    grid, manifests, _ = selected
    wrong = copy.deepcopy(grid)
    wrong['alpha'][-1] = 1.09
    with pytest.raises(ValueError, match='rectangle'):
        fresh.select_origins(wrong, manifests)
    sparse = copy.deepcopy(grid)
    sparse['period'][:] = 0
    sparse['period'][53, 60] = 9
    with pytest.raises(ValueError):
        fresh.select_origins(sparse, manifests)


def test_exact_case_and_metadata_only_sentinel_census(selected):
    census = fresh.build_census(selected[2]['anchors'])
    assert len(census['cases']) == len({r['id'] for r in census['cases']}) == 3456
    assert len(census['decision_contexts']) == 192
    assert len(census['sensitivity_contexts']) == len(census['shams']) == 48
    assert len(census['sentinel_ids']) == len(set(census['sentinel_ids'])) == 48
    by_id = {r['id']: r for r in census['cases']}
    sentinels = [by_id[identity] for identity in census['sentinel_ids']]
    assert {r['candidate'] for r in sentinels} == set(range(18))
    assert all(sum(r['radius'] == radius for r in sentinels) == 12 for radius in fresh.RADII)
    for i in range(24):
        for p, phase in enumerate((675, 1350)):
            expected = f'fresh-{i:02d}-p{phase}-r{fresh.RADII[(i+2*p)%4]:g}-c{(i+9*p)%18:02d}'
            assert expected in census['sentinel_ids']
    for context in census['decision_contexts']:
        assert [by_id[i]['candidate'] for i in context['candidate_ids']] == list(range(18))
    assert all('direction' not in row for row in census['cases'])
    assert census['counts']['total_short_traces'] == 10608


@pytest.mark.parametrize('mutation', ['missing', 'reordered', 'physical_alias', 'group_alias', 'pixel_alias', 'nonfinite'])
def test_census_rejects_aliases_subsets_and_changed_identities(selected, mutation):
    anchors = copy.deepcopy(selected[2]['anchors'])
    if mutation == 'missing':
        anchors.pop()
    elif mutation == 'reordered':
        anchors.reverse()
    elif mutation == 'physical_alias':
        anchors[1].update({k: anchors[0][k] for k in ('alpha', 'gamma', 'group')})
    elif mutation == 'group_alias':
        anchors[1]['group'] = anchors[0]['group']
    elif mutation == 'pixel_alias':
        anchors[1].update({k: anchors[0][k] for k in ('row', 'column')})
    else:
        anchors[0]['alpha'] = True
    with pytest.raises(ValueError):
        fresh.build_census(anchors)


def test_exact_historical_metadata_selection_reproduces_draft_without_reading_labels():
    manifests = {e: json.loads((ROOT / f'evidence/{e}/attempt-001/manifest.json').read_text())
        for e in ('EXP-006', 'EXP-007', 'EXP-008')}
    rows = []
    for path in sorted((ROOT / 'evidence/EXP-004/attempt-001/focus').glob('row-*.npz')):
        with np.load(path, allow_pickle=False) as row:
            rows.append({key: row[key].copy() for key in ('alpha', 'gamma', 'period')})
    grid = {key: np.asarray([row[key] for row in rows]) for key in ('alpha', 'gamma', 'period')}
    result = fresh.select_origins(grid, manifests)
    expected = [(28,34),(46,35),(52,32),(35,52),(48,42),(53,58),(66,38),(66,54),
        (67,59),(100,32),(77,56),(88,52),(32,77),(45,67),(47,82),(57,80),
        (81,66),(63,82),(55,101),(57,106),(52,112),(62,130),(73,144),(82,165)]
    assert [(a['row'], a['column']) for a in result['anchors']] == expected
    audit = result['selection_audit']
    assert (audit['core_component_cells'], audit['crop_eligible_cells'], audit['eligible_cells']) == (4824, 4714, 1703)
    assert audit['block_sizes'] == [72,67,75,73,67,69,65,86,69,71,77,70,74,73,67,66,69,71,69,68,68,76,67,74]
    assert audit['selected_minimum_development_distance'] == 0.04006168083848473
    assert audit['selected_minimum_pairwise_distance'] == 0.028327886186623477


def test_bundle_schema_rows_direction_duplicates_and_original_feature_formulas(supplied, bundle):
    anchors, atlas, certificates, sensitivities = supplied
    assert bundle['schema_version'] == old.SCHEMA_VERSION
    assert bundle['shared_names'] == list(old.SHARED_FEATURE_NAMES)
    assert bundle['geometry_names'] == list(old.GEOMETRY_FEATURE_NAMES)
    assert bundle['shared'].shape == (3456, 37) and bundle['geometry'].shape == (3456, 6)
    assert not bundle['shared'].flags.writeable and not bundle['geometry'].flags.writeable
    census = fresh.build_census(anchors)
    assert [r['id'] for r in bundle['rows']] == [r['id'] for r in census['cases']]
    for index in (0, 16, 17, 18, 143, 144, 2000, 3455):
        row = bundle['rows'][index]
        anchor = anchors[int(row['anchor'].split('-')[1])]
        sensitivity = sensitivities[f'{anchor["id"]}-p{row["phase"]}']
        mask = sensitivity['times'] > 75
        expected = old.shared_features(anchor, row['phase'], row['direction'], row['radius'],
            certificates[anchor['id']], sensitivity['activation_sensitivity'][mask],
            sensitivity['activation_velocity'][mask])
        np.testing.assert_array_equal(bundle['shared'][index], expected)
        np.testing.assert_array_equal(bundle['geometry'][index], old.geometry_features(
            atlas, anchor, row['direction'], row['radius'])['values'])
    assert bundle['rows'][0]['id'] != bundle['rows'][16]['id']
    np.testing.assert_array_equal(bundle['shared'][0], bundle['shared'][16])
    assert bundle['rows'][16]['direction'] == [1., 0.]
    assert bundle['rows'][17]['direction'] == [-1., -0.]
    assert set(bundle['rows'][0]) == {'id', 'role', 'anchor', 'group', 'phase', 'radius', 'candidate', 'direction'}
    # Censored out-of-crop destinations remain in the complete feature census.
    assert np.any(bundle['geometry'][:, 4] == 0)
    assert np.all(bundle['geometry'][bundle['geometry'][:, 4] == 0, 5] == 1)


def test_pure_bundle_has_no_file_label_or_optional_outcome_field_access(supplied, bundle, monkeypatch):
    anchors, atlas, certificates, sensitivities = supplied

    class AllowedOnly(dict):
        def __getitem__(self, key):
            if key not in self.allowed:
                raise AssertionError(f'Forbidden input read: {key}')
            return super().__getitem__(key)

    def limited(values):
        result = AllowedOnly(values, preview_labels=object(), targets=object(), scores=object())
        result.allowed = set(values)
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError('Pure feature adapter attempted file access')

    monkeypatch.setattr(builtins, 'open', forbidden)
    monkeypatch.setattr(Path, 'open', forbidden)
    monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(old, 'load_features', forbidden)
    observed = fresh.build_fresh_features([limited(a) for a in anchors], atlas,
        {key: limited(value) for key, value in certificates.items()},
        {key: limited(value) for key, value in sensitivities.items()})
    np.testing.assert_array_equal(observed['shared'], bundle['shared'])
    np.testing.assert_array_equal(observed['geometry'], bundle['geometry'])
    with pytest.raises(TypeError):
        fresh.build_fresh_features(anchors, atlas, certificates, sensitivities, preview_labels={})


@pytest.mark.parametrize('kind', ['missing_certificate', 'extra_certificate', 'missing_sensitivity', 'extra_sensitivity'])
def test_full_required_input_census(supplied, kind):
    anchors, atlas, certificates, sensitivities = supplied
    certificates, sensitivities = dict(certificates), dict(sensitivities)
    if kind == 'missing_certificate':
        certificates.pop('fresh-00')
    elif kind == 'extra_certificate':
        certificates['substitute'] = certificates['fresh-00']
    elif kind == 'missing_sensitivity':
        sensitivities.pop('fresh-00-p675')
    else:
        sensitivities['substitute'] = sensitivities['fresh-00-p675']
    with pytest.raises(ValueError, match='All and only'):
        fresh.build_fresh_features(anchors, atlas, certificates, sensitivities)


@pytest.mark.parametrize('kind', ['failed_gate', 'wrong_origin', 'missing_endpoint', 'wrong_times',
    'wrong_activations', 'failed_certificate', 'nonfinite', 'float32'])
def test_reference_representation_and_identity_are_required(supplied, kind):
    anchors, atlas, certificates, sensitivities = supplied
    certificates = dict(certificates)
    c = copy.deepcopy(certificates['fresh-00'])
    certificates['fresh-00'] = c
    if kind == 'failed_gate': c['gates']['primitive'] = False
    elif kind == 'wrong_origin': c['alpha'] += .001
    elif kind == 'missing_endpoint': c['orbit_states'] = c['orbit_states'][:-1]
    elif kind == 'wrong_times': c['orbit_times'][1] += 1e-10
    elif kind == 'wrong_activations': c['orbit_activations'][675, 0] += 1e-8
    elif kind == 'failed_certificate': c['verified'] = False
    elif kind == 'nonfinite': c['orbit_states'][10, 1] = np.nan
    else: c['orbit_states'] = c['orbit_states'].astype(np.float32)
    with pytest.raises(ValueError):
        fresh.build_fresh_features(anchors, atlas, certificates, sensitivities)


@pytest.mark.parametrize('kind', ['shifted_times', 'truncated', 'wrong_phase', 'nonzero_initial',
    'wrong_activation_sensitivity', 'wrong_velocity', 'nonfinite', 'wrong_solver', 'nonunit_direction', 'float32'])
def test_sensitivity_input_conventions_cannot_be_changed_or_aliased(supplied, kind):
    anchors, atlas, certificates, sensitivities = supplied
    sensitivities = dict(sensitivities)
    s = copy.deepcopy(sensitivities['fresh-00-p675'])
    sensitivities['fresh-00-p675'] = s
    if kind == 'shifted_times': s['times'][1] += 1e-12
    elif kind == 'truncated': s['activation_sensitivity'] = s['activation_sensitivity'][:-1]
    elif kind == 'wrong_phase': s['states'][0] = certificates['fresh-00']['orbit_states'][1350]
    elif kind == 'nonzero_initial': s['state_sensitivity'][0, 0, 0] = .1
    elif kind == 'wrong_activation_sensitivity': s['activation_sensitivity'][751, 0, 0] += .1
    elif kind == 'wrong_velocity': s['activation_velocity'][751, 0] += .1
    elif kind == 'nonfinite': s['state_sensitivity'][15, 0, 0] = np.inf
    elif kind == 'wrong_solver': s['solver']['rtol'] = 1e-8
    elif kind == 'nonunit_direction': s['tail_direction'] *= 2
    else: s['states'] = s['states'].astype(np.float32)
    with pytest.raises(ValueError):
        fresh.build_fresh_features(anchors, atlas, certificates, sensitivities)


def test_tail_strictly_excludes_75_and_local_sign_is_preserved(supplied, bundle):
    anchors, atlas, certificates, sensitivities = supplied
    changed = dict(sensitivities)
    key = 'fresh-00-p675'
    s = copy.deepcopy(changed[key]); changed[key] = s
    # Very large pre-tail values cannot alter the unchanged tail feature formula.
    s['state_sensitivity'][1:751] *= 1000
    derivative = 1.-np.tanh(s['states'])**2
    s['activation_sensitivity'] = derivative[:, :, None]*s['state_sensitivity']
    s['tail_direction'] = np.array([-.6, -.8])
    observed = fresh.build_fresh_features(anchors, atlas, certificates, changed)
    np.testing.assert_array_equal(observed['shared'][:16], bundle['shared'][:16])
    assert observed['rows'][16]['direction'] == [-.6, -.8]
    assert observed['rows'][17]['direction'] == [.6, .8]
    assert sensitivities[key]['tail_direction'].tolist() == [1., 0.]


def test_pure_outputs_do_not_mutate_or_share_matrix_storage(supplied, bundle):
    anchors, atlas, certificates, sensitivities = supplied
    assert certificates['fresh-00']['orbit_states'].flags.writeable
    assert sensitivities['fresh-00-p675']['times'].flags.writeable
    for values in (bundle['shared'], bundle['geometry']):
        assert values.flags.owndata
        assert not np.shares_memory(values, certificates['fresh-00']['orbit_states'])
        with pytest.raises(ValueError):
            values[0, 0] = 0.
