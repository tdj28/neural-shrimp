"""Isolated trace_branch copy with every Newton attempt and accepted state callbacks.
Numerical algorithm unchanged from producer; not a new continuation method.
"""
import numpy as np
from bifurcation_curves import refine_multipliers

def trace_branch(family, x0, theta0, free=0, direction=1, steps=60, ds=0.005, ds_min=1e-6, ds_max=0.02, tol=1e-10,
                 max_newton=8, beyond=0.02, bounds=None, on_point=None, k=8, refine_k=0, on_attempt=None, on_state=None, on_matrix=None):
    """One-parameter pseudo-arclength continuation of the period-p orbit through x0 in theta[free], the other
    parameter fixed, with all multipliers (dense monodromy) at every point.

    Coordinates are scaled, u = x / |x0|. A turn of theta[free] along the branch marks a fold; its estimate is the
    extremum of a quadratic through the last three points (in arclength). Stops `beyond` past the first turn,
    after `steps` points, when theta[free] leaves `bounds` or when the step falls below ds_min. Records use the
    keys of explore_latent_loop.pseudo_arclength ('lambda' is theta[free]); returns (records, stop_reason,
    states). With refine_k > 0 every point also records the two leading multipliers from a refine_k-step Arnoldi
    factorization of family.jvp_x ('multipliers_refined', 'spectral_radius_refined', 'refine_residuals'), which
    classify_crossings then uses."""
    n = family.n
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    s = float(np.linalg.norm(x0)) or 1.0
    theta0 = np.asarray(theta0, dtype=np.float64)
    I = np.eye(n)

    def theta_of(lam):
        th = theta0.copy()
        th[free] = lam
        return th

    def residual(z):
        x = z[:n] * s
        r = family.residual(x, theta_of(z[n]))
        return r / s, float(np.linalg.norm(r) / np.linalg.norm(x))

    def jacobian(z):
        x, th = z[:n] * s, theta_of(z[n])
        M = family.monodromy(x, th)
        J = np.zeros((n, n + 1))
        J[:, :n] = M - I
        J[:, n] = family.param_jacobian(x, th, which=(free,))[:, 0] / s
        return J, M

    def correct(z0, row, rhs_fn, J0):
        """Chord Newton: the Jacobian J0 of the last accepted point stays fixed and only residuals are evaluated
        (the dense monodromy dominates the cost); the full Jacobian is then computed once at the converged point,
        for the tangent, the multipliers and the next chord."""
        z = z0.copy()
        A = np.vstack([J0, row[None]])
        for it in range(max_newton + 1):
            G, rel = residual(z)
            if on_attempt is not None:
                on_attempt({"newton_iteration":it,"relative_residual":rel,"theta":theta_of(z[n]).tolist(),"state":(z[:n]*s).tolist()})
            if rel < tol:
                J, M = jacobian(z)
                return z, J, M, rel, it
            if it == max_newton:
                break
            z = z + np.linalg.solve(A, -np.concatenate([G, [rhs_fn(z)]]))
            # a chord step can throw the parameter far outside the bounds (an overflowing 10**log10T crashed one
            # temperature branch): treat it as a failed correction, so the step is retried smaller
            if not np.all(np.isfinite(z)) or (bounds is not None and not
                                               (bounds[0] - (bounds[1] - bounds[0]) <= z[n]
                                                <= bounds[1] + (bounds[1] - bounds[0]))):
                return None, None, None, float("inf"), it + 1
        return None, None, None, rel, max_newton

    def tangent(J, previous):
        y = np.linalg.solve(np.vstack([J, previous[None]]), np.concatenate([np.zeros(n), [1.0]]))
        return y / np.linalg.norm(y)

    records, states = [], []

    def record(z, rel, its, ds_used, step, t, turned, M):
        if on_matrix is not None:
            on_matrix(len(records), M.copy())
        vals = np.linalg.eigvals(M)
        vals = vals[np.argsort(-np.abs(vals))]
        real = [float(v.real) for v in vals[:k] if abs(v.imag) < 1e-9 * max(1.0, abs(v))]
        rec = {"index": len(records), "lambda": float(z[n]),
               "arclength": (records[-1]["arclength"] + step) if records else 0.0, "residual": rel, "newton": its,
               "ds": ds_used, "tangent_lambda": float(t[n]), "turned_since_previous": bool(turned),
               "multipliers_converged": True, "spectral_radius": float(np.abs(vals[0])),
               "real_multiplier_nearest_one": min(real, key=lambda m: abs(m - 1.0)) if real else None,
               "multipliers": [[float(v.real), float(v.imag)] for v in vals[:k]]}
        if refine_k:
            x, th = z[:n] * s, theta_of(z[n])
            ritz = refine_multipliers(lambda u: family.jvp_x(x, th, u), M, m=refine_k, count=2)
            rec["multipliers_refined"] = [[v.real, v.imag] for v, _ in ritz]
            rec["spectral_radius_refined"] = abs(ritz[0][0])
            rec["refine_residuals"] = [r for _, r in ritz]
        records.append(rec)
        states.append(z[:n] * s)
        if on_state is not None:
            on_state(len(records)-1, states[-1].copy())
        if on_point is not None:
            on_point(rec)
        return rec

    e_lam = np.zeros(n + 1)
    e_lam[n] = 1.0
    z = np.concatenate([x0 / s, [theta0[free]]])
    J, _ = jacobian(z)
    z, J, M, rel, its = correct(z, e_lam, lambda y: 0.0, J)
    if z is None:
        return records, f"no convergence at the start (residual {rel:.3g})", states
    t = tangent(J, e_lam * float(np.sign(direction)))
    record(z, rel, its, 0.0, 0.0, t, False, M)
    turn, reason = None, f"{steps} points"
    while len(records) < steps:
        zp = z + ds * t
        zn, Jn, Mn, rel, its = correct(zp, t, lambda y: float(t @ (y - zp)), J)
        tn = None if zn is None else tangent(Jn, t)
        if zn is None or float(tn @ t) < 0.5:                  # not converged, or jumped onto another branch
            ds *= 0.5
            if ds < ds_min:
                reason = f"step below {ds_min} (last residual {rel:.3g})"
                break
            continue
        turned = tn[n] * t[n] < 0
        step = float(np.linalg.norm(zn - z))
        z, t, J, M = zn, tn, Jn, Mn
        rec = record(z, rel, its, ds, step, t, turned, M)
        if turned and turn is None:
            lams = np.array([r["lambda"] for r in records[-3:]])
            turn = float(lams[-2])
            if len(records) >= 3:
                sig = np.array([r["arclength"] for r in records[-3:]])
                a, b, c = np.polyfit(sig - sig[0], lams, 2)
                if a != 0:
                    turn = float(c - b * b / (4 * a))
            rec["turn_lambda_estimate"] = turn
        if bounds is not None and not bounds[0] <= z[n] <= bounds[1]:
            reason = "parameter out of bounds"
            break
        if turn is not None and abs(z[n] - turn) >= beyond:
            reason = f"{beyond} past the first turn"
            break
        if its <= max_newton // 2:                         # chord Newton converges linearly: judge by half
            ds = min(ds * 1.5, ds_max)
        elif its >= max_newton - 1:
            ds *= 0.7
    return records, reason, states
