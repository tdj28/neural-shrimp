"""Two-parameter continuation of fold and flip curves of periodic orbits (minimally augmented systems).

A period-p orbit of x -> f(x; theta), theta in R^2, is a fixed point of the p-fold map Phi. It has a fold
(saddle-node) where Phi(x) = x and A = D_x Phi - I is singular, and a flip (period doubling) where A = D_x Phi + I
is singular. The singularity is measured by the scalar g of the bordered system

    [[A, b], [c^T, 0]] [v; g] = [0; 1],

which vanishes exactly when A is singular, b and c being approximate left and right null vectors of A (updated
along the curve). Its gradient needs only the second derivative of Phi contracted with the null vectors,
g_z = -w^T (d_z D_x Phi) v with w from the transposed system: one reverse-mode pass through a forward-mode
Jacobian-vector product. The curve {Phi(x) = x, g = 0} in (x, theta) is traced by pseudo-arclength. It stays
smooth through a cusp, where its projection onto the parameter plane reverses direction; such reversals and the
other multipliers reaching the unit circle (codimension-two points) are recorded along the curve.

The family is given by phi(x, theta) (torch, differentiable in both arguments). Dense monodromy matrices are
used, so the state may have up to a few thousand components; a family can supply faster versions of the
monodromy, the parameter derivatives or the gradient (see Family).
"""

import numpy as np


class Family:
    """A two-parameter family of p-fold maps phi(x, theta) -> x' with the derivatives the continuation needs.

    The defaults use torch.func on phi; override monodromy (dense D_x Phi), param_jacobian (D_theta Phi) or
    grad_s for speed. All methods take and return numpy arrays (x flattened, theta of length 2); `shape`,
    `dtype` and `device` describe the torch tensors phi expects. `context` wraps every derivative evaluation
    (e.g. a math attention kernel for forward-mode AD).
    """

    def __init__(self, phi, shape, dtype, device="cpu", context=None):
        import torch
        from contextlib import nullcontext

        self.torch, self.phi = torch, phi
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.n = int(np.prod(self.shape))
        self.context = context or nullcontext

    def to_x(self, x):
        return self.torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(self.shape), dtype=self.dtype,
                                    device=self.device)

    def to_theta(self, theta):
        return self.torch.as_tensor(np.asarray(theta, dtype=np.float64), dtype=self.dtype, device=self.device)

    @staticmethod
    def to_np(t):
        return t.detach().double().cpu().numpy().ravel()

    def residual(self, x, theta):
        with self.torch.no_grad():
            return self.to_np(self.phi(self.to_x(x), self.to_theta(theta))) - np.asarray(x).ravel()

    def monodromy(self, x, theta):
        torch, th = self.torch, self.to_theta(theta)
        with self.context():
            J = torch.func.jacfwd(lambda y: self.phi(y, th))(self.to_x(x))
        return J.detach().double().cpu().numpy().reshape(self.n, self.n)

    def param_jacobian(self, x, theta, which=(0, 1)):
        torch, X, th = self.torch, self.to_x(x), self.to_theta(theta)
        cols = []
        with self.context():
            for i in which:
                e = torch.zeros_like(th)
                e[i] = 1.0
                _, jv = torch.func.jvp(lambda t: self.phi(X, t), (th,), (e,))
                cols.append(self.to_np(jv))
        return np.column_stack(cols)

    def grad_s(self, x, theta, v, w):
        """Gradient in (x, theta) of s = w^T D_x Phi(x; theta) v at fixed v and w."""
        torch = self.torch
        V, W = self.to_x(v), self.to_x(w)

        def s(y, t):
            _, jv = torch.func.jvp(lambda u: self.phi(u, t), (y,), (V,))
            return (W * jv).sum()

        with self.context():
            gx, gt = torch.func.grad(s, argnums=(0, 1))(self.to_x(x), self.to_theta(theta))
        return self.to_np(gx), self.to_np(gt)


def bordered_solve(A, b, c):
    """v, g, w of the bordered system [[A, b], [c^T, 0]] [v; g] = [0; 1] and its transpose (w normalized by
    b^T w = 1)."""
    n = A.shape[0]
    B = np.zeros((n + 1, n + 1))
    B[:n, :n], B[:n, n], B[n, :n] = A, b, c
    rhs = np.zeros(n + 1)
    rhs[n] = 1.0
    vg = np.linalg.solve(B, rhs)
    wh = np.linalg.solve(B.T, rhs)
    return vg[:n], float(vg[n]), wh[:n]


def initial_borders(M, kind):
    """Left and right singular vectors of A = M -/+ I for its smallest singular value."""
    A = M + (1.0 if kind == "flip" else -1.0) * np.eye(M.shape[0])
    U, S, Vt = np.linalg.svd(A)
    return U[:, -1].copy(), Vt[-1].copy(), float(S[-1])


def multiplier_summary(vals, kind, k=8):
    """The k leading multipliers and the codimension-two indicators along a fold or flip curve."""
    vals = vals[np.argsort(-np.abs(vals))]
    target = 1.0 if kind == "fold" else -1.0
    i = int(np.argmin(np.abs(vals - target)))
    critical = vals[i]
    rest = np.delete(vals, i)
    real = rest[np.abs(rest.imag) <= 1e-9 * np.maximum(1.0, np.abs(rest))].real
    cplx = rest[rest.imag > 1e-9 * np.maximum(1.0, np.abs(rest))]
    return {"multipliers": [[float(v.real), float(v.imag)] for v in vals[:k]],
            "critical_multiplier": [float(critical.real), float(critical.imag)],
            "other_nearest_plus_one": float(real[np.argmin(np.abs(real - 1.0))]) if len(real) else None,
            "other_nearest_minus_one": float(real[np.argmin(np.abs(real + 1.0))]) if len(real) else None,
            "largest_complex_modulus": float(np.abs(cplx).max()) if len(cplx) else None,
            "other_spectral_radius": float(np.abs(rest).max()) if len(rest) else None}


def trace_curve(family, x0, theta0, kind, fix=1, direction=1, steps=100, ds=0.01, ds_min=1e-6, ds_max=0.05,
                tol=1e-10, gtol=1e-10, max_newton=8, bounds=None, on_point=None, k=8):
    """Trace the fold or flip curve of the period-p orbit near (x0, theta0) in the parameter plane.

    First the bifurcation point is located with theta[fix] held at theta0[fix] (Newton on the minimally augmented
    system from x0, which should be close to the orbit near its fold or flip). Then pseudo-arclength in the scaled
    coordinates (x / |x0|, theta) follows the curve in `direction` (sign of the initial d theta[1 - fix]).
    Every accepted point records theta, residuals, the multipliers and the parameter tangent; `cusp_candidate`
    marks a reversal of the parameter tangent between consecutive points (the projection of the curve has a cusp
    there). Stops after `steps` points, when theta leaves `bounds` ((lo0, hi0), (lo1, hi1)) or when the step falls
    below ds_min. Returns (records, stop_reason, final states).
    """
    if kind not in ("fold", "flip"):
        raise ValueError(f"kind must be fold or flip, not {kind}")
    sign = 1.0 if kind == "flip" else -1.0
    n = family.n
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    s = float(np.linalg.norm(x0)) or 1.0
    I = np.eye(n)

    def system(z, b, c):
        x, th = z[:n] * s, z[n:]
        r = family.residual(x, th)
        M = family.monodromy(x, th)
        v, g, w = bordered_solve(M + sign * I, b, c)
        gx, gt = family.grad_s(x, th, v, w)
        J = np.zeros((n + 1, n + 2))
        J[:n, :n] = M - I
        J[:n, n:] = family.param_jacobian(x, th) / s
        J[n, :n] = -gx * s
        J[n, n:] = -gt
        return np.concatenate([r / s, [g]]), J, M, v, w, float(np.linalg.norm(r) / np.linalg.norm(x))

    def correct(z0, b, c, row, rhs_fn):
        """Newton on the augmented system plus one scalar equation (row, rhs_fn). Returns the converged point with
        the Jacobian and monodromy evaluated there (reused for the tangent and the multipliers)."""
        z = z0.copy()
        for it in range(max_newton + 1):
            G, J, M, v, w, rel = system(z, b, c)
            if rel < tol and abs(G[n]) < gtol:
                return z, J, M, v, w, rel, abs(G[n]), it
            if it == max_newton:
                break
            dz = np.linalg.solve(np.vstack([J, row[None]]), -np.concatenate([G, [rhs_fn(z)]]))
            z = z + dz
        return None, None, None, None, None, rel, abs(G[n]), max_newton

    def tangent(J, previous):
        """Unit null vector of the (n+1) x (n+2) Jacobian, oriented along `previous`."""
        y = np.linalg.solve(np.vstack([J, previous[None]]), np.concatenate([np.zeros(n + 1), [1.0]]))
        return y / np.linalg.norm(y)

    b, c, smin = initial_borders(family.monodromy(x0, theta0), kind)
    z = np.concatenate([x0 / s, np.asarray(theta0, dtype=np.float64)])
    e_fix = np.zeros(n + 2)
    e_fix[n + fix] = 1.0
    zf, J, M, v, w, rel, gabs, its = correct(z, b, c, e_fix, lambda y: 0.0)
    if zf is None:
        return [], f"no {kind} point at fixed theta[{fix}] = {theta0[fix]} (residual {rel:.3g}, |g| {gabs:.3g})", []
    z = zf
    b, c = w / np.linalg.norm(w), v / np.linalg.norm(v)
    e_free = np.zeros(n + 2)
    e_free[n + 1 - fix] = float(np.sign(direction))
    t = tangent(J, e_free)
    records, states = [], []

    def record(z, rel, gabs, its, ds_used, step, t, t_prev, M):
        vals = np.linalg.eigvals(M)
        tp = t[n:] / max(np.linalg.norm(t[n:]), 1e-300)
        rec = {"index": len(records), "theta": [float(z[n]), float(z[n + 1])], "residual": rel, "g": gabs,
               "newton": its, "ds": ds_used,
               "arclength": (records[-1]["arclength"] + step) if records else 0.0,
               "parameter_tangent": [float(t[n]), float(t[n + 1])],
               "parameter_speed": float(np.linalg.norm(t[n:])),
               "cusp_candidate": bool(t_prev is not None and float(t[n:] @ t_prev[n:]) < 0)}
        rec.update(multiplier_summary(vals, kind, k))
        records.append(rec)
        states.append(z[:n] * s)
        if on_point is not None:
            on_point(rec)
        return tp

    record(z, rel, gabs, its, 0.0, 0.0, t, None, M)
    reason = f"{steps} points"
    while len(records) < steps:
        zp = z + ds * t
        zn, J, M, v, w, rel, gabs, its = correct(zp, b, c, t, lambda y: float(t @ (y - zp)))
        if zn is not None:
            bn, cn = w / np.linalg.norm(w), v / np.linalg.norm(v)
            tn = tangent(J, t)
        if zn is None or float(tn @ t) < 0.5:              # not converged, or jumped onto another branch
            ds *= 0.5
            if ds < ds_min:
                reason = f"step below {ds_min} (last residual {rel:.3g}, |g| {gabs:.3g})"
                break
            continue
        step = float(np.linalg.norm(zn - z))
        t_prev = t
        z, t, b, c = zn, tn, bn, cn
        record(z, rel, gabs, its, ds, step, t, t_prev, M)
        if bounds is not None and not all(lo <= z[n + i] <= hi for i, (lo, hi) in enumerate(bounds)):
            reason = "parameters out of bounds"
            break
        if its <= 2:
            ds = min(ds * 1.5, ds_max)
        elif its >= max_newton - 1:
            ds *= 0.7
    return records, reason, states


def classify_crossings(records):
    """Unit-circle crossings between consecutive points of a branch: the side where the spectral radius exceeds
    1 supplies the crossing multiplier (its largest-modulus one), recorded with modulus, argument (turns) and the
    next-largest modulus; typed as a complex pair (Neimark-Sacker), real +1 (fold) or real -1 (flip)."""
    crossings = []
    for a, b in zip(records, records[1:]):
        if a["spectral_radius"] is None or b["spectral_radius"] is None:
            continue
        if (a["spectral_radius"] - 1) * (b["spectral_radius"] - 1) < 0:
            outer = b if b["spectral_radius"] > 1 else a
            vals = [complex(re, im) for re, im in outer["multipliers"]]
            lead = max(vals, key=abs)
            kind = ("complex pair (Neimark-Sacker)" if abs(lead.imag) > 1e-9 * max(1.0, abs(lead)) else
                    "real +1 (fold)" if lead.real > 0 else "real -1 (flip)")
            others = sorted((abs(z) for z in vals if z != lead and z != lead.conjugate()), reverse=True)
            crossings.append({"between_log10_T": [a["lambda"], b["lambda"]], "between_indices": [a["index"], b["index"]],
                              "type": kind,
                              "crossing_multiplier": [lead.real, lead.imag, abs(lead), float(np.angle(lead) / (2 * np.pi))],
                              "next_largest_modulus": others[0] if others else None,
                              "multipliers_converged": bool(a.get("multipliers_converged", True) and
                                                            b.get("multipliers_converged", True))})
    return crossings


def trace_branch(family, x0, theta0, free=0, direction=1, steps=60, ds=0.005, ds_min=1e-6, ds_max=0.02, tol=1e-10,
                 max_newton=8, beyond=0.02, bounds=None, on_point=None, k=8):
    """One-parameter pseudo-arclength continuation of the period-p orbit through x0 in theta[free], the other
    parameter fixed, with all multipliers (dense monodromy) at every point.

    Coordinates are scaled, u = x / |x0|. A turn of theta[free] along the branch marks a fold; its estimate is the
    extremum of a quadratic through the last three points (in arclength). Stops `beyond` past the first turn,
    after `steps` points, when theta[free] leaves `bounds` or when the step falls below ds_min. Records use the
    keys of explore_latent_loop.pseudo_arclength ('lambda' is theta[free]); returns (records, stop_reason,
    states)."""
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
            if rel < tol:
                J, M = jacobian(z)
                return z, J, M, rel, it
            if it == max_newton:
                break
            z = z + np.linalg.solve(A, -np.concatenate([G, [rhs_fn(z)]]))
        return None, None, None, rel, max_newton

    def tangent(J, previous):
        y = np.linalg.solve(np.vstack([J, previous[None]]), np.concatenate([np.zeros(n), [1.0]]))
        return y / np.linalg.norm(y)

    records, states = [], []

    def record(z, rel, its, ds_used, step, t, turned, M):
        vals = np.linalg.eigvals(M)
        vals = vals[np.argsort(-np.abs(vals))]
        real = [float(v.real) for v in vals[:k] if abs(v.imag) < 1e-9 * max(1.0, abs(v))]
        rec = {"index": len(records), "lambda": float(z[n]),
               "arclength": (records[-1]["arclength"] + step) if records else 0.0, "residual": rel, "newton": its,
               "ds": ds_used, "tangent_lambda": float(t[n]), "turned_since_previous": bool(turned),
               "multipliers_converged": True, "spectral_radius": float(np.abs(vals[0])),
               "real_multiplier_nearest_one": min(real, key=lambda m: abs(m - 1.0)) if real else None,
               "multipliers": [[float(v.real), float(v.imag)] for v in vals[:k]]}
        records.append(rec)
        states.append(z[:n] * s)
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
