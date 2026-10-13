"""Task7 fold-only numerical worker; no passage execution or certificate claim.

The pure orchestration API is ready for a later measured, externally admitted
launcher. This preparation release deliberately has no target-execution CLI.
It reuses the unchanged independent Operations and augmented Newton solver.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PARAMETER_FD_STEPS = (7e-6, 2e-6, 7e-7)
STATE_FD_STEPS = (3e-4, 1e-4, 3e-5)
TOLERANCES = dict(augmented=1e-11, closure=1e-10, primitive=1e-6,
    critical=1e-8, imaginary=1e-10, complementary_limit=.95,
    isolation=1e-3, eigen_residual=1e-8, dense_jvp_relative=1e-10,
    composed_fd_relative=1e-5, composed_fd_floor=1e-7,
    composed_fd_reduction=.3, coefficient_floor=1e-10, coefficient_fd_relative=1e-3)


def dependencies():
    # Loading these files constructs no model and imports no producer analysis.
    for path in (ROOT, Path(__file__).resolve().parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import resource_pilot
    import assets
    return resource_pilot.setup_module(), assets


def target():
    setup, assets = dependencies()
    return setup.Target('hostile', 2, 'log10T', 'alpha', assets.ALPHA,
                        assets.PARAMETER, (-.05, -.043))


def qualify_fold(model, seed, *, save, check, chunk=16, max_newton=6, max_backtracks=2):
    """Refine and qualify G=F²; all full outputs survive in stage callbacks.

    `check` is the caller's resource/deadline guard, never a scientific gate.
    The outer launcher must interrupt long native operations when necessary.
    The `alpha` key in the legacy solver stores the active scalar log10T here.
    """
    check()  # Must precede all model access, including the initial state.
    import numpy as np
    setup, assets = dependencies()
    chosen = setup.Target('hostile', 2, 'log10T', 'alpha', assets.ALPHA,
                          assets.PARAMETER, (-.05, -.043))
    if any(type(v) is not int or v < lo for v, lo in
           ((chunk, 1), (max_newton, 0), (max_backtracks, 0))):
        raise ValueError('integer chunk/Newton/backtrack bounds required')
    x0 = setup.array(setup.tensor(seed))
    ops = setup.Operations(model, chosen, chunk=chunk, save=save, check=check)
    result = dict(schema='codex-task7-hostile-fold-v1', passed=False,
        certificate_passed=False, evidence='rung2 numerical fold cross-check',
        target=asdict(chosen), active_parameter_units='log10T',
        return_map='G=F^2', clock=dict(forward_steps_per_return=2,
            passage_counts='No passages evaluated; future counts use G iterates.'),
        tolerances=dict(TOLERANCES), parameter_fd_steps=PARAMETER_FD_STEPS,
        state_fd_steps=STATE_FD_STEPS,
        scalar_solver_key='solver.alpha is log10T, not the fixed steering alpha',
        gates={}, passage_trajectories_evaluated=False)

    def finish(stage):
        result.update(stage=stage, operation_counts=dict(ops.counts))
        ops.emit('fold-result', result)
        return result

    ops.emit('initial', dict(seed=x0, target=asdict(chosen), chunk=chunk,
        max_newton=max_newton, max_backtracks=max_backtracks,
        parameter_fd_steps=PARAMETER_FD_STEPS, state_fd_steps=STATE_FD_STEPS))
    def period_trial(state, parameter):
        # The unchanged solver returns bookkeeping only when it finishes. These
        # immediate records retain rejected/partial trial data if it is stopped.
        ops.emit('corrector-trial-start', dict(state=np.asarray(state).copy(), parameter=parameter))
        try:
            returned = ops.period_map(state, parameter)
        except BaseException as error:
            ops.emit('corrector-trial-error', dict(state=np.asarray(state).copy(),
                parameter=parameter, error=repr(error), solver_history_complete=False))
            raise
        ops.emit('corrector-trial-value', dict(state=np.asarray(state).copy(),
            parameter=parameter, returned=returned, solver_history_complete=False))
        return returned

    solver = setup.fold_refine(period_trial, ops.dense, ops.parameter_derivative,
        ops.critical_grad, x0, chosen.initial_parameter,
        tol=TOLERANCES['augmented'], max_newton=max_newton,
        max_backtracks=max_backtracks)
    result['solver'] = solver
    result['solver_history_complete'] = True
    ops.emit('corrector', solver)
    if solver['converged'] is not True:
        return finish('fold-corrector-failed')
    x, p, matrix = solver['state'], solver['alpha'], solver['matrix']
    result.update(x_c=x, parameter_c=p, alpha=chosen.fixed_value, log10T=p, matrix=matrix)
    try:
        spectrum = setup.classify_fold(matrix, critical_tol=TOLERANCES['critical'],
            imaginary_tol=TOLERANCES['imaginary'],
            complementary_limit=TOLERANCES['complementary_limit'],
            isolation=TOLERANCES['isolation'], eigen_tol=TOLERANCES['eigen_residual'])
    except ValueError as error:
        result['gate_failure'] = str(error)
        return finish('crossing-classification-failed')
    result['spectrum'] = spectrum  # Full right/left eigenvectors and spectrum.
    points = np.stack(setup.cycle_points(lambda y: ops.step(y, p), x, chosen.period))
    diagnostics = setup.orbit_diagnostics(lambda y: ops.step(y, p),
        lambda y, v: ops.jvp(y, v, p), x, chosen.period, matrix=matrix,
        closure_tol=TOLERANCES['closure'], separation_tol=TOLERANCES['primitive'],
        eigen_residual_tol=TOLERANCES['eigen_residual'])
    diagnostics['raw_strict_radius_below_one'] = diagnostics.pop('spectral_contraction_passed')
    diagnostics['numerically_attracting'] = None  # Neutral +1 at a fold.
    phase_returns = [float(np.linalg.norm(ops.period_map(y, p)-y) /
                           max(np.linalg.norm(y), 1e-30)) for y in points[:-1]]
    gates = result['gates']
    gates.update(closure=diagnostics['closure_passed'],
        primitive=diagnostics['primitive_separation_passed'],
        eigenpairs=diagnostics['eigenpairs_passed'],
        all_phase_closure=bool(max(phase_returns) < TOLERANCES['closure']))
    result.update(orbitpoints=points, orbit_diagnostics=diagnostics, phase_returns=phase_returns)
    ops.emit('orbit-gates', dict(gates=gates, diagnostics=diagnostics,
                               points=points, phase_returns=phase_returns, spectrum=spectrum))
    if not all(gates.values()):
        return finish('orbit-gates-failed')
    r, l = spectrum['right'], spectrum['left']

    def action(y, direction):
        ops.guard(p, 'composed_state_jvp')
        alpha, temperature, cache = ops.fixed(p)
        value = setup.array(model.jvp(setup.tensor(y), setup.tensor(direction),
            alpha, temperature, period=chosen.period, cache=cache))
        check()
        return value

    probe = np.arange(1, len(x)+1, dtype=np.float64)
    probe /= np.linalg.norm(probe)
    composed = []
    scale = max(float(np.linalg.norm(x)), 1.)
    for label, direction in (('critical', r), ('deterministic', probe)):
        dense = matrix @ direction
        jvp = action(x, direction)
        dense_error = float(np.linalg.norm(jvp-dense)/max(np.linalg.norm(dense), 1e-30))
        ladder = []
        for relative_h in STATE_FD_STEPS:
            h = relative_h*scale
            fd = (ops.period_map(x+h*direction, p)-ops.period_map(x-h*direction, p))/(2*h)
            ladder.append(dict(relative_h=relative_h, h=h, derivative=fd,
                relative_error=float(np.linalg.norm(fd-dense)/max(np.linalg.norm(dense), 1e-30))))
        fine, coarse = ladder[-1]['relative_error'], ladder[0]['relative_error']
        row_pass = bool(dense_error < TOLERANCES['dense_jvp_relative'] and
            fine < TOLERANCES['composed_fd_relative'] and
            (fine < TOLERANCES['composed_fd_floor'] or fine < TOLERANCES['composed_fd_reduction']*coarse))
        composed.append(dict(direction_label=label, direction=direction.copy(),
            dense_action=dense, jvp=jvp, dense_jvp_relative_error=dense_error,
            finite_difference=ladder, passed=row_pass))
        ops.emit('composed-precision', composed[-1])
    critical_jvp = composed[0]['jvp']
    critical_estimate = float(r @ critical_jvp)  # ||r||=1.
    critical_error = abs(critical_estimate-float(spectrum['critical_multiplier'].real))
    gates.update(composed_precision=all(row['passed'] for row in composed),
        critical_multiplier_jvp=bool(critical_error < TOLERANCES['eigen_residual']))
    result.update(composed_precision=composed, critical_jvp_multiplier=critical_estimate,
                  critical_jvp_multiplier_discrepancy=critical_error)
    if not all(gates.values()):
        return finish('composed-precision-failed')

    # Classification and composed precision precede normal-form coefficients.
    gp, hrr = ops.parameter_derivative(x, p), ops.second(x, r, p)
    raw_a, raw_b = float(l @ gp), float(.5*l @ hrr)
    parameter_rows, state_rows = [], []
    for h in PARAMETER_FD_STEPS:
        fd = (ops.period_map(x, p+h)-ops.period_map(x, p-h))/(2*h)
        parameter_rows.append(dict(h=h, a=float(l @ fd), derivative=fd,
            vector_relative_error=float(np.linalg.norm(fd-gp)/max(np.linalg.norm(gp), 1e-30))))
        ops.emit('parameter-fd', parameter_rows[-1])
    for relative_h in STATE_FD_STEPS:
        h = relative_h*scale
        second_fd = (action(x+h*r, r)-action(x-h*r, r))/(2*h)
        state_rows.append(dict(relative_h=relative_h, h=h,
            b=float(.5*l @ second_fd), derivative=second_fd,
            vector_relative_error=float(np.linalg.norm(second_fd-hrr)/max(np.linalg.norm(hrr), 1e-30))))
        ops.emit('state-fd', state_rows[-1])
    floor, rtol = TOLERANCES['coefficient_floor'], TOLERANCES['coefficient_fd_relative']
    gates.update(nonzero_a=bool(np.isfinite(raw_a) and abs(raw_a)>floor),
        nonzero_b=bool(np.isfinite(raw_b) and abs(raw_b)>floor),
        parameter_ad_fd=setup.fd_agrees(parameter_rows, 'a', raw_a, rtol, floor),
        state_ad_fd=setup.fd_agrees(state_rows, 'b', raw_b, rtol, floor))
    result.update(parameter_derivative=gp, directional_second=hrr,
        parameter_fd=parameter_rows, state_fd=state_rows, raw_a=raw_a, raw_b=raw_b,
        raw_r=r, raw_l=l)
    if not np.isfinite([raw_a, raw_b, raw_a*raw_b]).all():
        return finish('nonfinite-coefficients')
    orientation = float(np.sign(raw_b))
    ab = raw_a*raw_b
    result.update(r=orientation*r, l=orientation*l, a=orientation*raw_a,
        b=abs(raw_b), ab=ab, absolute_ab=abs(ab),
        asymptotic_inverse_passage_coefficient=float(np.sqrt(abs(ab))/np.pi),
        ghost_parameter_direction=float(np.sign(ab)), orientation=orientation,
        U=.1/abs(raw_b) if raw_b else None,
        coordinate_convention='u=l dot (x-x_c); ||r||_2=1; l dot r=1; b>0; '
            'p=log10T; p-p_c=sign(ab)*epsilon with epsilon>0; one G iterate=two F steps',
        prediction_scope='Local normal form only. No natural reinjection or passage outcome was measured.')
    result['passed'] = bool(all(gates.values()))
    return finish('qualified' if result['passed'] else 'coefficient-gates-failed')


def validate_skeleton(plan):
    """Preparation artifact only; cannot silently become a launch authorization."""
    if type(plan) is not dict or plan.get('schema') != 'codex-task7-fold-preparation-v1':
        raise ValueError('Task7 fold preparation schema required')
    if plan.get('execution_status') != 'not_admitted_pending_measured_profile':
        raise ValueError('preparation release has no executable run-plan schema')
    if any(plan.get(key) is not None for key in ('profile_receipt', 'profile_review')):
        raise ValueError('measured profile admission requires separate reviewed integration')
    resources = plan.get('resources', {})
    if resources.get('wall_seconds') is not None:
        raise ValueError('wall cap must await measured profile and capacity review')
    return plan
