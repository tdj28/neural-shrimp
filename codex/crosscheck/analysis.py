"""Independent Task2 analyses using an already supplied float64 map object.

This module never loads a model or opens a reference artifact. The runner owns
input hashes, run cards, budgets and persistence. ``save(stage,payload)`` receives
incremental outputs (including NumPy arrays); ``check()`` runs before model
operations. Only independently authored local numerical helpers are imported.
"""
from __future__ import annotations

from collections import Counter
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .numerics import cycle_points, fold_normal_form, fold_refine, newton_cycle, orbit_diagnostics, tangent_product


def _tensor(value):
    result = torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=torch.float64, device="cpu")
    if result.ndim != 1 or not torch.isfinite(result).all():
        raise ValueError("finite one-dimensional state/mode required")
    return result


def _numpy(value):
    return value.detach().cpu().numpy().copy()


def _relative(delta, reference):
    scale = float(np.linalg.norm(reference))
    return float(np.linalg.norm(delta) / scale) if scale else float(np.linalg.norm(delta))


class _Operations:
    def __init__(self, model, log_temperature, period, chunk_size, save, check):
        self.model, self.t, self.period, self.chunk = model, float(log_temperature), int(period), int(chunk_size)
        if self.period < 1 or self.chunk < 1:
            raise ValueError("positive period and Jacobian chunk required")
        self.save = save if save is not None else lambda stage, payload: None
        self.check = check if check is not None else lambda: None
        self.counts, self.stages = Counter(), Counter()
        self.start = time.monotonic()
        self.fixed_alpha, self.cache = None, None

    def emit(self, stage, payload):
        self.check()
        self.stages[stage] += 1
        name = f"{stage}-{self.stages[stage]:03d}"
        record = dict(payload)
        record.update(elapsed_seconds=time.monotonic()-self.start, operation_counts=dict(self.counts))
        self.save(name, record)

    def fixed_cache(self, alpha):
        alpha = float(alpha)
        if self.fixed_alpha != alpha or self.cache is None:
            self.check()
            self.counts["prefix_cache"] += 1
            self.cache = self.model.prefix_cache(alpha)
            self.fixed_alpha = alpha
        return self.cache

    def step(self, x, alpha):
        cache = self.fixed_cache(alpha)
        self.check()
        self.counts["one_step"] += 1
        return _numpy(self.model.forward(_tensor(x), float(alpha), self.t, cache=cache))

    def jvp(self, x, v, alpha):
        cache = self.fixed_cache(alpha)
        self.check()
        self.counts["one_step_jvp"] += 1
        return _numpy(self.model.jvp(_tensor(x), _tensor(v), float(alpha), self.t, period=1, cache=cache))

    def period_map(self, x, alpha):
        cache = self.fixed_cache(alpha)
        self.check()
        self.counts["period_map"] += 1
        return _numpy(self.model.period_map(_tensor(x), float(alpha), self.t, self.period, cache=cache))

    def dense(self, x, alpha):
        cache = self.fixed_cache(alpha)
        self.emit("monodromy-start", dict(state=np.asarray(x).copy(), alpha=float(alpha), period=self.period))
        self.check()
        self.counts["dense_monodromy"] += 1
        self.counts["dense_tangent_rows"] += len(x)
        matrix = _numpy(self.model.jacobian(_tensor(x), float(alpha), self.t,
                                           period=self.period, chunk_size=self.chunk, cache=cache))
        if matrix.shape != (len(x), len(x)) or not np.isfinite(matrix).all():
            raise ValueError("dense monodromy must be a finite matrix matching the full state dimension")
        self.emit("monodromy", dict(state=np.asarray(x).copy(), alpha=float(alpha), matrix=matrix))
        return matrix

    def parameter_derivative(self, x, alpha):
        # No external prefix cache here: its alpha-dependence must be differentiated.
        self.check()
        self.counts["parameter_jvp"] += 1
        xt = _tensor(x)
        at = torch.tensor(float(alpha), dtype=torch.float64)
        derivative = torch.func.jvp(lambda a: self.model.period_map(xt, a, self.t, self.period),
                                    (at,), (torch.ones_like(at),))[1]
        result = _numpy(derivative)
        self.emit("parameter-derivative", dict(state=np.asarray(x).copy(), alpha=float(alpha), derivative=result))
        return result

    def critical_grad(self, x, alpha, left, right):
        self.check()
        self.counts["critical_gradient"] += 1
        xt, lt, rt = _tensor(x), _tensor(left), _tensor(right)
        at = torch.tensor(float(alpha), dtype=torch.float64)
        def eigenvalue_with_fixed_modes(state, a):
            action = torch.func.jvp(lambda y: self.model.period_map(y, a, self.t, self.period),
                                    (state,), (rt,))[1]
            return torch.dot(lt, action)
        gx, ga = torch.func.grad(eigenvalue_with_fixed_modes, argnums=(0, 1))(xt, at)
        result = (_numpy(gx), float(ga.detach()))
        self.emit("critical-gradient", dict(state=np.asarray(x).copy(), alpha=float(alpha),
                                            left=np.asarray(left).copy(), right=np.asarray(right).copy(),
                                            state_gradient=result[0], alpha_gradient=result[1]))
        return result

    def second(self, x, right, alpha):
        self.check()
        self.counts["nested_directional_jvp"] += 1
        xt, rt = _tensor(x), _tensor(right)
        # Keep the full composed graph, including changes in intermediate states.
        def first(state):
            return torch.func.jvp(lambda y: self.model.period_map(y, float(alpha), self.t, self.period),
                                  (state,), (rt,))[1]
        result = _numpy(torch.func.jvp(first, (xt,), (rt,))[1])
        self.emit("directional-second", dict(state=np.asarray(x).copy(), alpha=float(alpha),
                                             right=np.asarray(right).copy(), derivative=result))
        return result

    def summary(self):
        return dict(operation_counts=dict(self.counts), elapsed_seconds=time.monotonic()-self.start,
                    count_scope="Public analysis callback counts, not internal layer/native operations.")


def _reference_eigenvalues(reference):
    if reference is None or "multipliers" not in reference:
        return None
    values = reference["multipliers"]
    return np.array([complex(float(v[0]), float(v[1])) for v in values])


def _spectral_comparison(values, reference):
    expected = _reference_eigenvalues(reference)
    if expected is None:
        return []
    if len(expected) > len(values):
        raise ValueError("reference supplies more eigenvalues than the state dimension")
    rows, cols = linear_sum_assignment(np.abs(expected[:, None] - values[None, :]))
    return [dict(reference_index=int(i), independent_index=int(j),
                 reference=expected[i], independent=values[j],
                 absolute_difference=float(abs(values[j]-expected[i])),
                 relative_difference=float(abs(values[j]-expected[i])/abs(expected[i]))
                 if abs(expected[i]) else None) for i,j in zip(rows,cols)]


def orbit_crosscheck(model, seed, *, alpha, log_temperature, period, chunk_size=16,
                     reference=None, save=None, check=None, newton_options=None):
    """Refine one saved phase; inspect full return, spectrum and every phase.

    ``reference`` is optional decoded orbit metadata containing ``multipliers``,
    ``spectral_radius`` and/or ``newton_residuals``. It is comparison data only;
    it never controls solver convergence or the derivative calculations.
    """
    seed = _numpy(_tensor(seed))
    ops = _Operations(model, log_temperature, period, chunk_size, save, check)
    step, jvp = lambda x: ops.step(x, alpha), lambda x,v: ops.jvp(x,v,alpha)
    original_points = cycle_points(step, seed, period)
    old_residual = _relative(original_points[-1]-seed, seed)
    ops.emit("initial-return", dict(state=seed.copy(), alpha=float(alpha),
                                    log_temperature=float(log_temperature), period=period,
                                    relative_residual=old_residual, points=np.stack(original_points)))
    initial = seed.copy()
    if period == 2:
        displacement = np.sin(np.arange(1, seed.size+1, dtype=np.float64))
        displacement *= 1e-8 / np.linalg.norm(displacement)
        initial += displacement
    ops.emit("newton-initial", dict(reference_seed=seed.copy(), initial_state=initial.copy(),
                                    perturbation_norm=float(np.linalg.norm(initial-seed))))
    solver = newton_cycle(step, jvp, initial, period, **(newton_options or {}))
    ops.emit("newton", solver)
    if not solver["converged"]:
        return dict(passed=False, stage="newton", solver=solver, initial_relative_residual=old_residual,
                    **ops.summary())
    state = solver["state"]
    matrix = ops.dense(state, alpha)
    diagnostics = orbit_diagnostics(step, jvp, state, period, matrix=matrix)
    ops.emit("spectrum", diagnostics)
    points = cycle_points(step, state, period)
    phase_closures = []
    for phase, x in enumerate(points[:-1]):
        returned = ops.period_map(x, alpha)
        row = dict(phase=phase, relative_closure=_relative(returned-x, x))
        phase_closures.append(row)
        ops.emit("phase-closure", row)
    separations = np.array([_relative(x-seed, seed) for x in points[:-1]])
    aligned_phase = int(np.argmin(separations))
    # This route propagates a tangent sequentially through stored phase points;
    # model.jacobian above differentiated the entire composition directly.
    direction = np.linspace(-1., 1., state.size)
    if not np.linalg.norm(direction):
        direction = np.ones(state.size)
    direction /= np.linalg.norm(direction)
    product_action = tangent_product(jvp, points, direction)
    composed_action = matrix @ direction
    action_error = _relative(product_action-composed_action, composed_action)
    # For a very contracting orbit, use absolute AND relative diagnostics.
    action_absolute = float(np.linalg.norm(product_action-composed_action))
    action_passed = bool(action_absolute < 1e-10 or action_error < 1e-8)
    comparisons = _spectral_comparison(diagnostics["eigenvalues"], reference)
    result = dict(passed=bool(diagnostics["numerically_attracting"] and action_passed
                             and all(r["relative_closure"] < 1e-10 for r in phase_closures)),
                  state=state, points=np.stack(points), matrix=matrix, solver=solver, initial_state=initial,
                  diagnostics=diagnostics, initial_relative_residual=old_residual,
                  seed_shift_relative=_relative(state-seed,seed), seed_shift_absolute=float(np.linalg.norm(state-seed)),
                  phase_aligned_seed_shift_relative=float(separations[aligned_phase]), aligned_phase=aligned_phase,
                  phase_closures=phase_closures, spectral_comparison=comparisons,
                  tangent_comparison=dict(direction=direction, phase_product=product_action,
                                          composed=composed_action, absolute_error=action_absolute,
                                          relative_error=action_error, passed=action_passed),
                  **ops.summary())
    if reference is not None:
        result["reference_summary"] = {k:reference[k] for k in ("spectral_radius", "newton_residuals") if k in reference}
        if "spectral_radius" in reference:
            result["spectral_radius_difference"] = diagnostics["spectral_radius"]-float(reference["spectral_radius"])
    ops.emit("orbit-result", result)
    return result


def fold_crosscheck(model, saved_state, saved_right, *, alpha, log_temperature=-.115,
                    period=2, chunk_size=16, reference=None, save=None, check=None,
                    fold_options=None, normal_form_options=None):
    """Correct a deliberately displaced fold, then compare AD and FD coefficients.

    The perturbations are fixed: +1e-7 times the normalized saved right mode in
    state and +1e-9 in alpha. Normal form signs align with the saved right mode.
    Coefficients from nested AD and every difference-table entry are retained.
    """
    seed, orientation = _numpy(_tensor(saved_state)), _numpy(_tensor(saved_right))
    if seed.shape != orientation.shape or np.linalg.norm(orientation) == 0:
        raise ValueError("full nonzero right mode matching fold state required")
    orientation /= np.linalg.norm(orientation)
    ops = _Operations(model, log_temperature, period, chunk_size, save, check)
    initial = seed + 1e-7*orientation
    initial_alpha = float(alpha)+1e-9
    ops.emit("fold-initial", dict(saved_state=seed, saved_right=orientation, saved_alpha=float(alpha),
                                   initial_state=initial, initial_alpha=initial_alpha,
                                   perturbation_norm=1e-7, alpha_perturbation=1e-9))
    solver = fold_refine(ops.period_map, ops.dense, ops.parameter_derivative, ops.critical_grad,
                         initial, initial_alpha, **(fold_options or {}))
    ops.emit("fold-corrector", solver)
    if not solver["converged"]:
        return dict(passed=False, stage="fold-corrector", solver=solver, **ops.summary())
    state, corrected_alpha = solver["state"], solver["alpha"]
    fixed_step = lambda x: ops.step(x, corrected_alpha)
    fixed_jvp = lambda x,v: ops.jvp(x,v,corrected_alpha)
    critical_orbit = orbit_diagnostics(fixed_step, fixed_jvp, state, period, matrix=solver["matrix"])
    # A roundoff-side comparison of the critical +1 multiplier cannot classify
    # attraction at a fold. Preserve it as raw data, not a dynamical conclusion.
    critical_orbit["raw_strict_radius_below_one"] = critical_orbit["spectral_contraction_passed"]
    critical_orbit["spectral_contraction_passed"] = None
    critical_orbit["numerically_attracting"] = None
    critical_orbit["classification"] = "fold: neutral critical +1 mode; inspect complementary spectrum"
    critical_points = cycle_points(fixed_step, state, period)
    phase_closures = [_relative(ops.period_map(x,corrected_alpha)-x,x) for x in critical_points[:-1]]
    complementary_radius = solver["history"][-1]["other_spectral_radius"]
    prerequisites = dict(closure=critical_orbit["closure_passed"],
                         primitive=critical_orbit["primitive_separation_passed"],
                         eigenpairs=critical_orbit["eigenpairs_passed"],
                         all_phase_closure=all(v<1e-10 for v in phase_closures),
                         complementary_attraction=complementary_radius<1)
    ops.emit("fold-orbit-prerequisites", dict(gates=prerequisites, diagnostics=critical_orbit,
                                             points=np.stack(critical_points),
                                             phase_closures=phase_closures,
                                             complementary_radius=complementary_radius))
    if not all(prerequisites.values()):
        return dict(passed=False, stage="fold-orbit-prerequisites", solver=solver,
                    prerequisite_gates=prerequisites, diagnostics=critical_orbit,
                    phase_closures=phase_closures, **ops.summary())
    nf = fold_normal_form(ops.step, ops.jvp, state, corrected_alpha, period,
                          matrix=solver["matrix"], reference_right=orientation,
                          parameter_derivative=ops.parameter_derivative,
                          directional_second=ops.second, **(normal_form_options or {}))
    ops.emit("normal-form", nf)
    exact_a, exact_b = nf["exact_callback_a"], nf["exact_callback_b"]
    if exact_a is None or exact_b is None:
        raise ValueError("normal form requires both independent AD coefficient paths")
    ad = dict(a=exact_a, b=exact_b, ab=exact_a*exact_b)
    discrepancy = dict(a_fd_minus_ad=nf["a"]-exact_a, b_fd_minus_ad=nf["b"]-exact_b,
                        ab_fd_minus_ad=nf["ab"]-ad["ab"])
    reference_comparison = {}
    if reference is not None:
        for key in ("a", "b", "ab"):
            if key in reference:
                expected = float(reference[key])
                reference_comparison[key] = dict(reference=expected, independent_ad=ad[key],
                                                  ad_difference=ad[key]-expected,
                                                  ad_relative_difference=abs(ad[key]-expected)/abs(expected) if expected else None,
                                                  independent_fd=nf[key], fd_difference=nf[key]-expected)
        if "b_steps" in reference:
            by_step = {float(row["h_relative"]):row["b"] for row in reference["b_steps"]}
            reference_comparison["b_steps"] = [dict(row, reference=by_step.get(row["h_relative"]),
                difference=row["b"]-by_step[row["h_relative"]] if row["h_relative"] in by_step else None)
                for row in nf["b_steps"]]
    # Accuracy of derivatives remains quantitative, not declared by dtype.
    a_agrees = abs(nf["a"]-exact_a) <= 1e-3*max(abs(exact_a),1e-12)
    b_agrees = abs(nf["b"]-exact_b) <= 1e-3*max(abs(exact_b),1e-12)
    result = dict(passed=bool(solver["converged"] and a_agrees and b_agrees and ad["ab"] > 0
                             and nf["second_modulus"] < 1),
                  solver=solver, normal_form=nf, ad_coefficients=ad,
                  prerequisite_gates=prerequisites, phase_closures=phase_closures,
                  points=np.stack(critical_points),
                  derivative_discrepancy=discrepancy,
                  derivative_gates=dict(parameter_agreement=bool(a_agrees), second_agreement=bool(b_agrees)),
                  reference_comparison=reference_comparison,
                  alpha_difference=corrected_alpha-float(alpha),
                  state_shift_absolute=float(np.linalg.norm(state-seed)),
                  state_shift_relative=_relative(state-seed,seed), **ops.summary())
    ops.emit("fold-result", result)
    return result
