"""Independent numerical orbit/fold checks over user-supplied map callbacks.

No model implementation or project research tool is imported. These are
floating-point computations, never interval certificates. Tangent batches are
rows; dense monodromy columns retain the usual mathematical convention.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def _vector(value, n=None):
    v = np.asarray(value, dtype=np.float64)
    if v.ndim != 1 or (n is not None and v.size != n) or not np.isfinite(v).all():
        raise ValueError("finite one-dimensional vector of the expected size required")
    return v


def _scale(x, scale):
    s = float(np.linalg.norm(x)) if scale is None else float(scale)
    if not np.isfinite(s) or s < 0:
        raise ValueError("finite positive residual scale required")
    return s if s > 0 else 1.0


def cycle_points(step, state, period):
    """Return x_0,...,x_p, including the independently evaluated return."""
    if not isinstance(period, (int, np.integer)) or period < 1:
        raise ValueError("positive integer period required")
    x = _vector(state).copy()
    points = [x]
    for _ in range(period):
        x = _vector(step(x), x.size).copy()
        points.append(x)
    return points


def tangent_product(jvp, points, vector):
    """Chronological DF(x_(p-1)) ... DF(x_0) v, not the reversed product."""
    v = _vector(vector, points[0].size).copy()
    for x in points[:-1]:
        v = _vector(jvp(x, v), v.size)
    return v


def _solve(operator, rhs, rtol, restart, maxiter, preconditioner=None):
    iterations = []
    solution, info = gmres(operator, rhs, rtol=rtol, atol=0.0,
                           restart=restart, maxiter=maxiter, M=preconditioner,
                           callback=lambda r: iterations.append(float(r)),
                           callback_type="pr_norm")
    absolute = float(np.linalg.norm(operator @ solution - rhs))
    relative = absolute / max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    # Check actual residual: Krylov status/preconditioned history is insufficient.
    passed = bool(info == 0 and relative <= max(5 * rtol, 1e-12))
    return solution, dict(info=int(info), iterations=len(iterations),
                          relative_residual=relative, passed=passed,
                          preconditioned_residuals=iterations)


def newton_cycle(step, jvp, state, period, *, tol=1e-11, max_newton=12,
                 linear_rtol=1e-10, restart=30, max_linear_iterations=4,
                 residual_scale=None, max_backtracks=10):
    """Damped Newton-Krylov for F^p(x)-x, with explicit failure outcomes."""
    x = _vector(state).copy()
    history = []
    reason = "newton_limit"
    for iteration in range(max_newton + 1):
        points = cycle_points(step, x, period)
        residual = points[-1] - x
        norm = float(np.linalg.norm(residual))
        relative = norm / _scale(x, residual_scale)
        row = dict(iteration=iteration, relative_residual=relative,
                   absolute_residual=norm)
        history.append(row)
        if relative < tol:
            reason = "residual_passed"
            break
        if iteration == max_newton:
            break
        operator = LinearOperator((x.size, x.size), dtype=np.float64,
                                  matvec=lambda v: tangent_product(jvp, points, v) - v)
        delta, linear = _solve(operator, -residual, linear_rtol, restart,
                               max_linear_iterations)
        row["linear"] = linear
        if not linear["passed"]:
            reason = "linear_solve_failed"
            break
        for backtrack in range(max_backtracks + 1):
            damping = 2.0 ** -backtrack
            candidate = x + damping * delta
            trial = cycle_points(step, candidate, period)[-1] - candidate
            if np.linalg.norm(trial) <= (1 - 1e-4 * damping) * norm:
                x = candidate
                row["damping"] = damping
                break
        else:
            reason = "line_search_failed"
            break
    return dict(state=x, converged=reason == "residual_passed", reason=reason,
                relative_residual=history[-1]["relative_residual"], history=history)


def monodromy(step, jvp, state, period, *, jvp_rows=None, chunk=16):
    """Dense composed Jacobian; optional jvp_rows(x,V) uses row tangents."""
    if chunk < 1:
        raise ValueError("positive chunk size required")
    points = cycle_points(step, state, period)
    n = points[0].size
    matrix = np.empty((n, n), dtype=np.float64)
    for first in range(0, n, chunk):
        last = min(first + chunk, n)
        rows = np.eye(n, dtype=np.float64)[first:last]
        for x in points[:-1]:
            if jvp_rows is None:
                rows = np.vstack([_vector(jvp(x, v), n) for v in rows])
            else:
                rows = np.asarray(jvp_rows(x, rows), dtype=np.float64)
                if rows.shape != (last - first, n) or not np.isfinite(rows).all():
                    raise ValueError("invalid batched JVP shape or nonfinite values")
        matrix[:, first:last] = rows.T
    return matrix


def spectrum(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("finite square monodromy required")
    values, vectors = np.linalg.eig(matrix)
    order = np.argsort(-np.abs(values))
    values, vectors = values[order], vectors[:, order]
    residuals = np.linalg.norm(matrix @ vectors - vectors * values[None, :], axis=0)
    residuals /= np.maximum(np.linalg.norm(vectors, axis=0), np.finfo(float).tiny)
    return dict(eigenvalues=values, eigenvectors=vectors,
                absolute_eigenpair_residuals=residuals,
                spectral_radius=float(np.max(np.abs(values))))


def orbit_diagnostics(step, jvp, state, period, *, matrix=None, jvp_rows=None,
                      chunk=16, closure_tol=1e-11, separation_tol=1e-6,
                      eigen_residual_tol=1e-8, residual_scale=None):
    points = cycle_points(step, state, period)
    scale = _scale(points[0], residual_scale)
    returns = np.array([np.linalg.norm(p - points[0]) / scale for p in points[1:]])
    divisors = [d for d in range(1, period) if period % d == 0]
    if matrix is None:
        matrix = monodromy(step, jvp, state, period, jvp_rows=jvp_rows, chunk=chunk)
    eig = spectrum(matrix)
    eig_pass = bool(np.max(eig["absolute_eigenpair_residuals"]) < eigen_residual_tol)
    closure_pass = bool(returns[-1] < closure_tol)
    primitive_pass = bool(all(returns[d - 1] > separation_tol for d in divisors))
    spectral_pass = bool(eig_pass and eig["spectral_radius"] < 1)
    return dict(period=period, relative_returns=returns,
                proper_divisor_returns={d: float(returns[d - 1]) for d in divisors},
                closure_passed=closure_pass, primitive_separation_passed=primitive_pass,
                eigenpairs_passed=eig_pass, spectral_contraction_passed=spectral_pass,
                numerically_attracting=closure_pass and primitive_pass and spectral_pass,
                **eig)


def composed_parameter_derivative(step, state, alpha, period, h):
    """Centered alpha derivative; step(x,alpha) must include prefix dependence."""
    if not h > 0:
        raise ValueError("positive parameter difference required")
    plus = cycle_points(lambda x: step(x, alpha + h), state, period)[-1]
    minus = cycle_points(lambda x: step(x, alpha - h), state, period)[-1]
    return (plus - minus) / (2 * h)


def fold_normal_form(step, jvp, state, alpha, period, *, matrix=None,
                     reference_right=None, parameter_steps=(1e-5, 3e-6, 1e-6),
                     state_steps=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4),
                     parameter_derivative=None, directional_second=None,
                     critical_tol=1e-7, isolation=1e-3, eigen_residual_tol=1e-8,
                     closure_tol=1e-11, residual_scale=None):
    """Fold coefficient tables, signs tied to reference_right if supplied.

    The returned a/b average the last two entries; full tables are mandatory
    outputs. This deliberately avoids selecting a best-looking plateau after
    outcomes. Caller must assess convergence and report all discrepancies.
    """
    x = _vector(state)
    fixed_step = lambda y: step(y, alpha)
    fixed_jvp = lambda y, v: jvp(y, v, alpha)
    points = cycle_points(fixed_step, x, period)
    closure = float(np.linalg.norm(points[-1] - x) / _scale(x, residual_scale))
    if closure >= closure_tol:
        raise ValueError("fold coefficients require a converged periodic orbit")
    if matrix is None:
        matrix = monodromy(fixed_step, fixed_jvp, x, period)
    eigen = spectrum(matrix)
    values, vectors = eigen["eigenvalues"], eigen["eigenvectors"]
    k = int(np.argmin(np.abs(values - 1)))
    if abs(values[k] - 1) >= critical_tol or abs(values[k].imag) >= critical_tol:
        raise ValueError("fold coefficients require an isolated real +1 multiplier")
    others = np.delete(values, k)
    if others.size and np.min(np.abs(others - 1)) <= isolation:
        raise ValueError("critical +1 multiplier is not isolated")
    left_values, left_vectors = np.linalg.eig(np.asarray(matrix).T)
    j = int(np.argmin(np.abs(left_values - values[k])))
    r = vectors[:, k].real.copy()
    r /= np.linalg.norm(r)
    if reference_right is not None and np.dot(r, _vector(reference_right, x.size)) < 0:
        r *= -1
    l = left_vectors[:, j].real.copy()
    pairing = float(l @ r)
    if abs(pairing) < 1e-12:
        raise ValueError("critical left/right eigenvectors have unresolved pairing")
    l /= pairing
    eigen_error = max(np.linalg.norm(matrix @ r - r), np.linalg.norm(matrix.T @ l - l) / np.linalg.norm(l))
    if eigen_error >= eigen_residual_tol:
        raise ValueError("critical eigenvector residual exceeds tolerance")
    if not parameter_steps or not state_steps or min(parameter_steps) <= 0 or min(state_steps) <= 0:
        raise ValueError("positive nonempty derivative step tables required")
    atable = [dict(step=float(h), a=float(l @ composed_parameter_derivative(step, x, alpha, period, h)))
              for h in parameter_steps]
    btable = []
    for h in state_steps:
        delta = float(h) * _scale(x, None)
        xp, xm = x + delta * r, x - delta * r
        plus = tangent_product(fixed_jvp, cycle_points(fixed_step, xp, period), r)
        minus = tangent_product(fixed_jvp, cycle_points(fixed_step, xm, period), r)
        btable.append(dict(h_relative=float(h), b=float(l @ (plus - minus) / (4 * delta))))
    a = float(np.mean([row["a"] for row in atable[-2:]]))
    b = float(np.mean([row["b"] for row in btable[-2:]]))
    exact_a = None if parameter_derivative is None else float(l @ _vector(parameter_derivative(x, alpha), x.size))
    exact_b = None if directional_second is None else float(l @ _vector(directional_second(x, r, alpha), x.size) / 2)
    return dict(a=a, b=b, ab=a * b, a_steps=atable, b_steps=btable,
                exact_callback_a=exact_a, exact_callback_b=exact_b,
                right=r, left=l, closure=closure, critical_multiplier=values[k],
                second_modulus=float(np.max(np.abs(others))) if others.size else 0.0,
                eigen_residual=float(eigen_error), left_right_pairing=float(l @ r),
                left_right_cosine=float(abs(l @ r) / np.linalg.norm(l)))


def refine_fold(step, jvp, state, alpha, period, right, *, tol=1e-11,
                max_newton=8, linear_rtol=1e-7, restart=30,
                max_linear_iterations=6, difference_step=1e-5,
                parameter_step=1e-6, residual_scale=None, max_backtracks=10):
    """Moore-Spence extended Newton using matrix-free directional differences.

    Unknowns are (x,alpha,r); equations are G-x, (DG-I)r and (r.r-1)/2.
    Finite-difference derivatives are numerical approximations. This generic
    fallback must be profiled before a large model; it has no hidden retries.
    """
    x = _vector(state).copy()
    r = _vector(right, x.size).copy()
    if np.linalg.norm(r) == 0 or difference_step <= 0 or parameter_step <= 0:
        raise ValueError("nonzero mode and positive difference steps required")
    r /= np.linalg.norm(r)
    n = x.size
    z = np.concatenate([x, [float(alpha)], r])
    scale = _scale(x, residual_scale)
    history = []

    def equations(candidate):
        xx, aa, rr = candidate[:n], candidate[n], candidate[n + 1:]
        points = cycle_points(lambda y: step(y, aa), xx, period)
        q = tangent_product(lambda y, v: jvp(y, v, aa), points, rr) - rr
        return np.concatenate([(points[-1] - xx) / scale, q, [(rr @ rr - 1) / 2]])

    reason = "newton_limit"
    for iteration in range(max_newton + 1):
        residual = equations(z)
        norm = float(np.linalg.norm(residual))
        row = dict(iteration=iteration, residual=norm,
                   orbit_residual=float(np.linalg.norm(residual[:n])),
                   eigen_residual=float(np.linalg.norm(residual[n:2*n])),
                   normalization_residual=float(abs(residual[-1])))
        history.append(row)
        if norm < tol:
            reason = "extended_residual_passed"
            break
        if iteration == max_newton:
            break
        xx, aa, rr = z[:n], z[n], z[n + 1:]
        fixed_step = lambda y: step(y, aa)
        fixed_jvp = lambda y, v: jvp(y, v, aa)
        points = cycle_points(fixed_step, xx, period)
        ga = composed_parameter_derivative(step, xx, aa, period, parameter_step)

        def matvec(v):
            dx, da, dr = v[:n], v[n], v[n + 1:]
            first = (tangent_product(fixed_jvp, points, dx) - dx + ga * da) / scale
            direction_norm = float(np.hypot(np.linalg.norm(dx) / scale, da))
            if direction_norm:
                h = difference_step / direction_norm
                def mode_action(sign):
                    xh, ah = xx + sign * h * dx, aa + sign * h * da
                    ph = cycle_points(lambda y: step(y, ah), xh, period)
                    return tangent_product(lambda y, w: jvp(y, w, ah), ph, rr)
                change = (mode_action(1) - mode_action(-1)) / (2 * h)
            else:
                change = np.zeros(n)
            second = tangent_product(fixed_jvp, points, dr) - dr + change
            return np.concatenate([first, second, [rr @ dr]])

        operator = LinearOperator((2*n+1, 2*n+1), matvec=matvec, dtype=np.float64)
        delta, linear = _solve(operator, -residual, linear_rtol, restart,
                               max_linear_iterations)
        row["linear"] = linear
        if not linear["passed"]:
            reason = "linear_solve_failed"
            break
        for backtrack in range(max_backtracks + 1):
            damping = 2.0 ** -backtrack
            trial = z + damping * delta
            if np.linalg.norm(equations(trial)) <= (1 - 1e-4 * damping) * norm:
                z = trial
                row["damping"] = damping
                break
        else:
            reason = "line_search_failed"
            break
    return dict(state=z[:n], alpha=float(z[n]), right=z[n+1:],
                converged=reason == "extended_residual_passed", reason=reason,
                history=history, residual=history[-1]["residual"])


def quadratic_passage(u0, a, b, epsilon, exit_value, *, max_iter=100000):
    if not all(np.isfinite([u0, a, b, epsilon, exit_value])) or min(a, b, epsilon) <= 0:
        raise ValueError("positive-oriented finite bottleneck parameters required")
    u = float(u0)
    count = None
    for k in range(1, max_iter + 1):
        u += a * epsilon + b * u * u
        if u > exit_value:
            count = k
            break
    scale = np.sqrt(b / (a * epsilon))
    continuous = (np.arctan(exit_value * scale) - np.arctan(u0 * scale)) / np.sqrt(a*b*epsilon)
    return dict(discrete=count, continuous=float(continuous))


def fold_refine(period_map, monodromy_func, param_derivative, critical_grad,
                state, alpha, *, tol=1e-11, max_newton=8,
                critical_search_radius=0.25, isolation=1e-3,
                imaginary_tol=1e-10, eigen_residual_tol=1e-8,
                residual_scale=None, max_backtracks=5):
    """Spectral augmented Newton on (G-x, lambda_near_plus_one-1).

    Callbacks use (state,alpha). critical_grad(state,alpha,left,right)
    returns (gradient_with_respect_to_state, alpha_derivative) of l^T DG r
    with the supplied modes held fixed. Prefix dependence belongs to these
    callbacks. Dense monodromy is recomputed at every evaluated trial; callers
    must include this cost in their profile. No producer bordered solver used.
    """
    from scipy.linalg import solve

    x = _vector(state).copy()
    alpha = float(alpha)
    scale = _scale(x, residual_scale)
    history = []

    def evaluate(y, a):
        returned = _vector(period_map(y, a), y.size)
        matrix = np.asarray(monodromy_func(y, a), dtype=np.float64)
        if matrix.shape != (y.size, y.size) or not np.isfinite(matrix).all():
            raise ValueError("invalid fold monodromy")
        values, vectors = np.linalg.eig(matrix)
        k = int(np.argmin(np.abs(values - 1)))
        value = values[k]
        others = np.delete(values, k)
        if abs(value - 1) > critical_search_radius or abs(value.imag) > imaginary_tol:
            raise ValueError("fold corrector has no admissible real +1 mode")
        if others.size and np.min(np.abs(others - value)) < isolation:
            raise ValueError("fold corrector critical mode is not isolated")
        r = vectors[:, k].real.copy()
        r /= np.linalg.norm(r)
        lv, lvec = np.linalg.eig(matrix.T)
        j = int(np.argmin(np.abs(lv - value)))
        l = lvec[:, j].real.copy()
        if abs(l @ r) < 1e-12:
            raise ValueError("fold corrector left/right pairing unresolved")
        l /= l @ r
        eig_error = max(np.linalg.norm(matrix @ r - value.real*r),
                        np.linalg.norm(matrix.T @ l - value.real*l) / np.linalg.norm(l))
        if eig_error >= eigen_residual_tol:
            raise ValueError("fold corrector eigenpair residual too large")
        residual = np.concatenate([(returned - y) / scale, [value.real - 1]])
        return dict(matrix=matrix, left=l, right=r, multiplier=float(value.real),
                    other_spectral_radius=float(np.max(np.abs(others))) if others.size else 0.,
                    residual=residual, merit=float(np.linalg.norm(residual)),
                    orbit_residual=float(np.linalg.norm(residual[:-1])),
                    eigen_residual=float(eig_error))

    current = evaluate(x, alpha)
    reason = "newton_limit"
    for iteration in range(max_newton + 1):
        row = {k: current[k] for k in ("merit", "orbit_residual", "multiplier",
                                      "eigen_residual", "other_spectral_radius")}
        row.update(iteration=iteration, alpha=alpha)
        history.append(row)
        if current["merit"] < tol:
            reason = "augmented_residual_passed"
            break
        if iteration == max_newton:
            break
        ga = _vector(param_derivative(x, alpha), x.size)
        gx, galpha = critical_grad(x, alpha, current["left"], current["right"])
        gx = _vector(gx, x.size)
        galpha = float(galpha)
        if not np.isfinite(galpha):
            raise ValueError("nonfinite critical parameter derivative")
        jacobian = np.empty((x.size + 1, x.size + 1))
        jacobian[:-1, :-1] = (current["matrix"] - np.eye(x.size)) / scale
        jacobian[:-1, -1] = ga / scale
        jacobian[-1, :-1], jacobian[-1, -1] = gx, galpha
        try:
            delta = solve(jacobian, -current["residual"], assume_a="gen", check_finite=True)
        except np.linalg.LinAlgError:
            reason = "singular_augmented_jacobian"
            break
        linear_residual = float(np.linalg.norm(jacobian @ delta + current["residual"]) /
                                max(current["merit"], np.finfo(float).tiny))
        row["linear_relative_residual"] = linear_residual
        if not np.isfinite(delta).all() or linear_residual > 1e-8:
            reason = "linear_solve_failed"
            break
        for backtrack in range(max_backtracks + 1):
            damping = 2.0 ** -backtrack
            candidate_x, candidate_alpha = x + damping * delta[:-1], alpha + damping * delta[-1]
            try:
                candidate = evaluate(candidate_x, candidate_alpha)
            except ValueError as exc:
                row.setdefault("rejected_trials", []).append(dict(damping=damping, reason=str(exc)))
                continue
            if candidate["merit"] <= (1 - 1e-4*damping) * current["merit"]:
                x, alpha, current = candidate_x, float(candidate_alpha), candidate
                row["damping"] = damping
                break
        else:
            reason = "line_search_failed"
            break
    return dict(state=x, alpha=alpha, converged=reason == "augmented_residual_passed",
                reason=reason, matrix=current["matrix"], left=current["left"],
                right=current["right"], multiplier=current["multiplier"],
                residual=current["merit"], relative_residual=current["orbit_residual"],
                history=history)
