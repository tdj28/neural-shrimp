"""EXP-011 pre-preview inputs: no perturbed previews, target labels, or fits.

The loader reads only bound origin metadata, baseline certificates, saved
unperturbed sensitivities, and the original atlas. The pure builder cannot
accept preview/target metrics. Historical feature parity belongs in tests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from hopfield_information_design import build_atlas_geometry, geometry_features, GEOMETRY_FEATURE_NAMES
from hopfield_probe_selector import direction_candidates

ROOT = Path(__file__).resolve().parents[1]
OLD_REL = Path('evidence/EXP-008/attempt-001')
OLD_RECEIPT_SHA = 'ddb22c78a6b5f11a023088fd90f3403d25a2797e00624f01c316171ce35706fb'
SCHEMA_VERSION = 'EXP-011-pre-preview-37-direct-response-v1'
FEATURE_SOURCE_PATHS = ['tools/preview_shortlist_features.py',
    'tools/hopfield_information_design.py', 'tools/hopfield_drift_design.py',
    'tools/hopfield_probe_selector.py', 'tools/hopfield_drift_metrics.py']
SOURCE_PATHS = FEATURE_SOURCE_PATHS
SHARED_FEATURE_NAMES = (
    'alpha', 'gamma', 'destination_alpha', 'destination_gamma',
    'alpha_squared', 'alpha_gamma', 'gamma_squared',
    'phase_sin', 'phase_cos', 'period', 'log_rho', 'log_rho_per_time',
    'radius', 'log_radius', 'radius_squared', 'direction_alpha', 'direction_gamma',
    'delta_alpha', 'delta_gamma', 'delta_alpha_squared', 'delta_cross', 'delta_gamma_squared',
    'alpha_delta_alpha', 'alpha_delta_gamma', 'gamma_delta_alpha', 'gamma_delta_gamma',
    'rectangle_contains_destination', 'signed_crop_clearance',
    'initial_u1', 'initial_u2', 'initial_u3', 'orbit_std_u1', 'orbit_std_u2', 'orbit_std_u3',
    'local_projected_rmse', 'local_unprojected_rmse', 'local_phase_cycles')


def physical_group(alpha, gamma):
    if not np.isfinite([alpha, gamma]).all():
        raise ValueError('Physical origin coordinates must be finite')
    return f'hopfield-da-silva-rech-2022:{float(alpha).hex()}:{float(gamma).hex()}'


def shared_features(anchor, phase, direction, radius, certificate, sensitivity, velocity):
    """Exactly 37 whitelisted values; S/v are saved unperturbed tail arrays.

    Direct directional norms replace the old Gram-quadratic evaluation. This
    changes numerical arithmetic only; it does not reintegrate or regenerate
    the saved activation sensitivity and velocity arrays.
    """
    a, g = float(anchor['alpha']), float(anchor['gamma'])
    d = np.asarray(direction, dtype=float)
    if (not np.isfinite([a, g]).all() or d.shape != (2,) or not np.isfinite(d).all()
            or not np.isclose(np.linalg.norm(d), 1., rtol=0, atol=1e-12)):
        raise ValueError('Finite origin and unit direction required')
    if (type(phase) is not int or phase not in (675, 1350)
            or isinstance(radius, (bool, np.bool_)) or radius not in (.00025, .001, .004, .016)):
        raise ValueError('Unplanned phase or radius')
    period, rho = float(certificate['period']), float(certificate['transverse_spectral_radius'])
    orbit = np.asarray(certificate['orbit_activations'], float)
    if (not np.isfinite([period, rho]).all() or period <= 0 or not 0 < rho < 1
            or orbit.shape != (2701, 3) or not np.isfinite(orbit).all()):
        raise ValueError('Complete finite attracting reference certificate required')
    orbit = orbit[:-1]
    s, v = np.asarray(sensitivity, float), np.asarray(velocity, float)
    if (s.ndim != 3 or s.shape[1:] != (3, 2) or not len(s) or v.shape != s.shape[:2]
            or not np.isfinite(s).all() or not np.isfinite(v).all()):
        raise ValueError('Finite unperturbed sensitivity and velocity required')
    columns, phase_vector = s.reshape(-1, 2), v.ravel()
    norm = float(phase_vector @ phase_vector)
    coefficients = phase_vector @ columns / norm if norm > 0 else np.zeros(2)
    projected_columns = columns - phase_vector[:, None] * coefficients
    delta = float(radius) * d
    projected = float(np.sqrt(np.mean((projected_columns @ delta)**2)))
    unprojected = float(np.sqrt(np.mean((columns @ delta)**2)))
    phase_cycles = float(coefficients @ delta / period)
    dx, dy = delta
    da, dg = a + dx, g + dy
    clearance = min(da - 1.056, 1.08 - da, dg - 10.14, 10.55 - dg)
    values = [a, g, da, dg, a*a, a*g, g*g,
        np.sin(2*np.pi*phase/2700), np.cos(2*np.pi*phase/2700), period, np.log(rho), np.log(rho)/period,
        radius, np.log(radius), radius*radius, *d, dx, dy, dx*dx, dx*dy, dy*dy,
        a*dx, a*dy, g*dx, g*dy, float(clearance >= 0), clearance,
        *orbit[phase], *orbit.std(axis=0), projected, unprojected, phase_cycles]
    result = np.asarray(values, float)
    if result.shape != (37,) or not np.isfinite(result).all():
        raise ValueError('Nonfinite or changed pre-preview feature schema')
    return result


def load_features(root=ROOT):
    """Return bound, label-free arrays and metadata for all 3,456 old entries."""
    started = time.perf_counter()
    root = Path(root).resolve()
    bindings = {}

    def bound_path(relative, expected=None):
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError('Feature input escaped the source root')
        data = path.read_bytes()
        item = dict(path=str(relative), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        if expected is not None and (item['sha256'], item['bytes']) != (expected['sha256'], expected['bytes']):
            raise ValueError(f'Feature input binding differs: {relative}')
        bindings[str(relative)] = item
        return path

    receipt_path = bound_path(OLD_REL / 'receipt.json')
    if bindings[str(OLD_REL / 'receipt.json')]['sha256'] != OLD_RECEIPT_SHA:
        raise ValueError('Original EXP-008 receipt differs')
    receipt = json.loads(receipt_path.read_text())
    old_index = {i['path']: i for i in receipt['artifacts']}

    def old_path(name):
        return bound_path(OLD_REL / name, old_index[name])

    plan = json.loads(old_path('manifest.json').read_text())
    anchors = sorted(plan['anchors'], key=lambda a: (a['role'], a['id']))
    if (len(anchors) != 24 or len({physical_group(a['alpha'], a['gamma']) for a in anchors}) != 24
            or plan['phases'] != [675, 1350] or plan['radii'] != [.00025, .001, .004, .016]
            or plan['candidate_count'] != 18):
        raise ValueError('Original physical-origin/context census differs')
    atlas_started = time.perf_counter()
    atlas_items = sorted((i for i in plan['source_artifacts']
                          if i['path'].startswith('evidence/EXP-004/attempt-001/focus/row-')
                          and i['path'].endswith('.npz')), key=lambda i: i['path'])
    if not atlas_items:
        raise ValueError('Missing bound old atlas rows')
    grid_rows = []
    for item in atlas_items:
        with np.load(bound_path(Path(item['path']), item), allow_pickle=False) as raw:
            grid_rows.append({k: raw[k].copy() for k in ('alpha', 'gamma', 'period')})
    grid = {k: np.asarray([row[k] for row in grid_rows]) for k in ('alpha', 'gamma', 'period')}
    atlas = build_atlas_geometry(grid, plan['selection']['core_index'])
    atlas_seconds = time.perf_counter() - atlas_started
    baselines = {role: json.loads(old_path(f'{role}-baselines.json').read_text())
                 for role in ('development', 'test')}
    rows, shared, geometry = [], [], []
    sensitivity_old_seconds = []
    for original in anchors:
        anchor = {k: original[k] for k in ('alpha', 'gamma', 'row', 'column')}
        cert = baselines[original['role']][original['id']]['certificate']
        certificate = {k: cert[k] for k in ('period', 'transverse_spectral_radius', 'orbit_activations')}
        group = physical_group(anchor['alpha'], anchor['gamma'])
        for phase in plan['phases']:
            stem = f'sensitivity/{original["id"]}-p{phase}'
            metadata = json.loads(old_path(stem + '.json').read_text())
            sensitivity_old_seconds.append(float(metadata['wall_seconds']))
            with np.load(old_path(stem + '.npz'), allow_pickle=False) as raw:
                times = raw['times'].copy()
                if not np.array_equal(times, np.linspace(0, 300, 3001)):
                    raise ValueError('Unperturbed sensitivity time convention differs')
                mask = times > 75
                sensitivity, velocity = raw['activation_sensitivity'][mask], raw['activation_velocity'][mask]
                local = raw['tail_direction'].copy()
            directions = direction_candidates('uniform_16') + direction_candidates('local_signed', local_direction=local)
            for radius in plan['radii']:
                for candidate, action in enumerate(directions):
                    direction = action['direction']
                    shared.append(shared_features(anchor, phase, direction, radius, certificate, sensitivity, velocity))
                    geometry.append(geometry_features(atlas, anchor, direction, radius)['values'])
                    rows.append(dict(id=f'{original["id"]}-p{phase}-r{radius:g}-c{candidate:02d}',
                        role=original['role'], anchor=original['id'], group=group, phase=phase,
                        radius=radius, candidate=candidate, direction=direction))
    if len(rows) != 3456 or len({r['id'] for r in rows}) != 3456:
        raise ValueError('Incomplete or duplicate feature entries')
    shared, geometry = np.asarray(shared), np.asarray(geometry)
    if shared.shape != (3456, 37) or geometry.shape != (3456, 6) or not np.isfinite(geometry).all():
        raise ValueError('Feature census or values differ')
    shared.setflags(write=False); geometry.setflags(write=False)
    return dict(schema_version=SCHEMA_VERSION, shared_names=list(SHARED_FEATURE_NAMES),
        geometry_names=list(GEOMETRY_FEATURE_NAMES), rows=rows, shared=shared, geometry=geometry,
        source_bindings=[bindings[k] for k in sorted(bindings)],
        timings=dict(total_saved_input_feature_seconds=time.perf_counter() - started,
                     old_atlas_read_and_geometry_seconds=atlas_seconds,
                     original_sensitivity_call_seconds=sensitivity_old_seconds,
                     scope='Current saved-input construction only; original sensitivity times are from the old host. '
                           'No new integration, atlas acquisition, preview acquisition, or net speedup measurement.'))
