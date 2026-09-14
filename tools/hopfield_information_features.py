"""EXP008 fixed prediction inputs; no target outcome enters this module."""
from __future__ import annotations
import numpy as np

BASE_FEATURE_NAMES = (
    'alpha', 'gamma', 'destination_alpha', 'destination_gamma',
    'alpha_squared', 'alpha_gamma', 'gamma_squared',
    'phase_sin', 'phase_cos', 'period', 'log_rho', 'log_rho_per_time',
    'radius', 'log_radius', 'radius_squared', 'direction_alpha', 'direction_gamma',
    'delta_alpha', 'delta_gamma', 'delta_alpha_squared', 'delta_cross', 'delta_gamma_squared',
    'alpha_delta_alpha', 'alpha_delta_gamma', 'gamma_delta_alpha', 'gamma_delta_gamma',
    'rectangle_contains_destination', 'signed_crop_clearance',
    'initial_u1', 'initial_u2', 'initial_u3', 'orbit_std_u1', 'orbit_std_u2', 'orbit_std_u3',
    'local_projected_rmse', 'local_unprojected_rmse', 'local_phase_cycles',
    'preview_rmse', 'preview_synchronous_rmse', 'preview_phase_sin', 'preview_phase_cos',
    'preview_u1_rmse', 'preview_u2_rmse', 'preview_u3_rmse',
    'preview_first_window_rmse', 'preview_last_window_rmse', 'preview_window_change',
    'preview_rmse_squared', 'local_projected_mse', 'preview_times_local', 'preview_times_radius')


def base_features(anchor, phase, direction, radius, certificate, gram, preview_metrics,
                  *, crop=(1.056, 1.08, 10.14, 10.55)):
    """Explicit nonlinear columns shared by every learned selector.

    Orbit diagnostics and the 300-unit *unperturbed* variational solve are
    available to both methods. Perturbed trajectories contribute only their
    separately collected 0--75 preview. The crop rectangle is shared metadata.
    This fixed representation does not establish optimality over all learners.
    """
    a, g = float(anchor['alpha']), float(anchor['gamma'])
    direction = np.asarray(direction, float)
    if direction.shape != (2,) or not np.isclose(np.linalg.norm(direction), 1, atol=1e-12, rtol=0):
        raise ValueError('Expected one unit direction in native weight coordinates')
    if phase not in (675, 1350) or radius not in (.00025, .001, .004, .016):
        raise ValueError('Unplanned phase or radius')
    dx, dy = radius * direction
    da, dg = a + dx, g + dy
    low_a, high_a, low_g, high_g = crop
    clearance = min(da-low_a, high_a-da, dg-low_g, high_g-dg)
    T = float(certificate['period'])
    rho = float(certificate['transverse_spectral_radius'])
    if not (T > 0 and 0 < rho < 1):
        raise ValueError('A finite positive attracting transverse radius is required')
    orbit = np.asarray(certificate['orbit_activations'], float)[:-1]
    if orbit.shape != (2700, 3):
        raise ValueError('Expected the certified endpoint-excluded reference orbit')
    delta = np.array([dx, dy])
    projected = float(np.sqrt(max(0., delta @ np.asarray(gram['gram']) @ delta)))
    unprojected = float(np.sqrt(max(0., delta @ np.asarray(gram['unprojected_gram']) @ delta)))
    local_phase = float(np.asarray(gram['phase_coefficients']) @ delta / T)
    preview = preview_metrics
    score = float(preview['aligned_rms']['aggregate'])
    shift = float(preview['phase_offset_cycles'])
    first = float(preview['windows'][0]['aligned_rms']['aggregate'])
    last = float(preview['windows'][-1]['aligned_rms']['aggregate'])
    values = [a, g, da, dg, a*a, a*g, g*g,
              np.sin(2*np.pi*phase/2700), np.cos(2*np.pi*phase/2700), T, np.log(rho), np.log(rho)/T,
              radius, np.log(radius), radius*radius, *direction,
              dx, dy, dx*dx, dx*dy, dy*dy, a*dx, a*dy, g*dx, g*dy,
              float(clearance >= 0), clearance,
              *orbit[phase], *orbit.std(axis=0), projected, unprojected, local_phase,
              score, preview['synchronous_rms']['aggregate'], np.sin(2*np.pi*shift), np.cos(2*np.pi*shift),
              *preview['aligned_rms']['per_component'], first, last, last-first,
              score*score, projected*projected, score*projected, score*radius]
    result = np.asarray(values, float)
    if result.shape != (len(BASE_FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError('Missing or nonfinite fixed prediction input')
    return result
