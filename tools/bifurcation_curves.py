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

    def jvp_x(self, x, theta, v):
        """D_x Phi(x; theta) v (one forward-mode pass)."""
        torch, th = self.torch, self.to_theta(theta)
        with self.context():
            _, jv = torch.func.jvp(lambda u: self.phi(u, th), (self.to_x(x),), (self.to_x(v),))
        return self.to_np(jv)

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


def refine_multipliers(matvec, M, m=4, count=2):
    """Leading multipliers from an m-step Arnoldi factorization of the Jacobian-vector product `matvec`, started
    from the leading eigenvectors (real and imaginary parts, `count` of them) of a dense monodromy M computed in
    lower precision. When the other multipliers are far smaller, as on the soft-token orbits (about 1e-6), the
    Ritz values are accurate to the precision of matvec while M's own eigenvalues carry its errors (float32
    monodromies of the P8 orbit are off by 3 %). Returns [(value, residual norm)] sorted by modulus."""
    n = M.shape[0]
    vals, vecs = np.linalg.eig(M)
    v0 = np.zeros(n)
    for i in np.argsort(-np.abs(vals))[:count]:
        v0 += vecs[:, i].real + vecs[:, i].imag
    if not np.linalg.norm(v0):
        v0 = np.random.default_rng(0).normal(size=n)
    Q = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    Q[:, 0] = v0 / np.linalg.norm(v0)
    k = m
    for j in range(m):
        w = np.asarray(matvec(Q[:, j]), dtype=np.float64)
        scale = np.linalg.norm(w)
        for _ in range(2):                                   # Gram-Schmidt, repeated once for orthogonality
            h = Q[:, :j + 1].T @ w
            H[:j + 1, j] += h
            w = w - Q[:, :j + 1] @ h
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] <= 1e-13 * max(scale, 1e-300):        # invariant subspace found
            k = j + 1
            break
        Q[:, j + 1] = w / H[j + 1, j]
    theta, S = np.linalg.eig(H[:k, :k])
    res = np.abs(H[k, k - 1]) * np.abs(S[k - 1, :]) / np.linalg.norm(S, axis=0)
    order = np.argsort(-np.abs(theta))
    return [(complex(theta[i]), float(res[i])) for i in order[:count]]


def fold_normal_form(family, x, theta, param=1, param_scale=1.0, steps=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4)):
    """Normal form u' = u + a eps + b u^2 of a fold of the map family.phi at (x, theta): r and l are the right and
    left eigenvectors of the multiplier nearest +1 (|r| = 1, l^T r = 1), a = l^T d Phi / d p with
    p = theta[param] / param_scale (eps is measured in p), and b = l^T D^2 Phi[r, r] / 2 from central differences
    of JVPs at x +- h |x| r for each relative step h. The stable-branch multipliers satisfy
    (1 - mu)^2 / (4 |p - p_c|) -> a b, and passage through the bottleneck at eps > 0 takes about
    pi / sqrt(a b eps) iterates. Returns a dict with the coefficients, the b plateau and the multiplier."""
    x = np.asarray(x, dtype=np.float64).ravel()
    M = family.monodromy(x, theta)
    vals, R = np.linalg.eig(M)
    i = int(np.argmin(np.abs(vals - 1)))
    valsL, L = np.linalg.eig(M.T)
    j = int(np.argmin(np.abs(valsL - vals[i])))
    r, l = np.real(R[:, i]), np.real(L[:, j])
    r = r / np.linalg.norm(r)
    l = l / float(l @ r)
    a = float(l @ family.param_jacobian(x, theta, which=(param,))[:, 0]) * param_scale
    scale = float(np.linalg.norm(x)) or 1.0                   # steps relative to |x| (absolute at the origin)
    table = []
    for h in steps:
        step = h * scale
        d2 = (family.jvp_x(x + step * r, theta, r) - family.jvp_x(x - step * r, theta, r)) / (2 * step)
        table.append({"h_relative": h, "b": 0.5 * float(l @ d2)})
    bs = np.array([t["b"] for t in table])
    k = int(np.argmin(np.abs(np.diff(bs)))) if len(bs) > 1 else 0   # flattest adjacent pair: the plateau
    b = float(0.5 * (bs[k] + bs[k + 1])) if len(bs) > 1 else float(bs[0])
    others = np.delete(vals, i)
    return {"multiplier": [float(vals[i].real), float(vals[i].imag)],
            "second_modulus": float(np.abs(others).max()) if len(others) else None,
            "a": a, "b": b, "ab": a * b, "b_steps": table, "b_plateau_pair": [steps[k], steps[k + 1]]
            if len(bs) > 1 else [steps[0]], "left_right_cosine": float(abs(l @ r) / np.linalg.norm(l)),
            "r": r, "l": l}


def fold_cubic(family, x, theta, steps=(1e-2, 3e-3, 1e-3, 3e-4)):
    """Cubic coefficient c of the fold normal form u' = u + a eps + b u^2 + c u^3 at (x, theta), the nondegeneracy
    coefficient of a cusp (where b = 0). With r, l the right and left eigenvectors of the multiplier nearest +1
    (|r| = 1, l^T r = 1) and B, C the second and third derivatives of the p-fold map, the quadratic centre-manifold
    term h2 solves (I - M) h2 = B(r, r) - 2 b r with l^T h2 = 0 (bordered system), and
    c = l^T C(r, r, r) / 6 + l^T B(r, h2) / 2. Unlike a and b, c keeps its sign when r and l are reversed. B(r, r),
    C(r, r, r) and B(r, h2) are central differences of JVPs at x +- s r, s = h |x|, for each relative step h; the
    returned c is the mean of the flattest adjacent pair of steps (the plateau), with the full table."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    M = family.monodromy(x, theta)
    vals, R = np.linalg.eig(M)
    i = int(np.argmin(np.abs(vals - 1)))
    valsL, L = np.linalg.eig(M.T)
    j = int(np.argmin(np.abs(valsL - vals[i])))
    r, l = np.real(R[:, i]), np.real(L[:, j])
    r = r / np.linalg.norm(r)
    l = l / float(l @ r)
    bordered = np.block([[np.eye(n) - M, r[:, None]], [l[None, :], np.zeros((1, 1))]])
    scale = float(np.linalg.norm(x)) or 1.0
    j0 = family.jvp_x(x, theta, r)
    table = []
    for h in steps:
        s = h * scale
        jp, jm = family.jvp_x(x + s * r, theta, r), family.jvp_x(x - s * r, theta, r)
        brr = (jp - jm) / (2 * s)
        crrr = (jp - 2 * j0 + jm) / s ** 2
        b = 0.5 * float(l @ brr)
        sol = np.linalg.solve(bordered, np.concatenate([brr - 2 * b * r, [0.0]]))
        h2 = sol[:n]
        brh = (family.jvp_x(x + s * r, theta, h2) - family.jvp_x(x - s * r, theta, h2)) / (2 * s)
        direct, centre = float(l @ crrr) / 6, 0.5 * float(l @ brh)
        table.append({"h_relative": h, "b": b, "c": direct + centre, "c_direct": direct,
                      "c_centre_manifold": centre, "h2_norm": float(np.linalg.norm(h2)),
                      "bordered_sigma": float(sol[n])})
    cs = np.array([t["c"] for t in table])
    k = int(np.argmin(np.abs(np.diff(cs)))) if len(cs) > 1 else 0
    c = float(0.5 * (cs[k] + cs[k + 1])) if len(cs) > 1 else float(cs[0])
    others = np.delete(vals, i)
    return {"multiplier": [float(vals[i].real), float(vals[i].imag)],
            "second_modulus": float(np.abs(others).max()) if len(others) else None,
            "c": c, "c_steps": table, "c_plateau_pair": [steps[k], steps[k + 1]] if len(cs) > 1 else [steps[0]],
            "c_plateau_relative_spread": float(abs(cs[k] - cs[k + 1]) / max(abs(c), 1e-300)) if len(cs) > 1
            else None}


def passage_count(u0, a, b, eps, u_out, max_iter=100000):
    """Iterates of the quadratic normal form u' = u + a eps + b u^2 from u0 until u > u_out (None if not within
    max_iter), the discrete prediction for passage through the bottleneck."""
    u = float(u0)
    for n in range(1, max_iter + 1):
        u = u + a * eps + b * u * u
        if u > u_out:
            return n
    return None


def passage_arctan(u_in, u_out, a, b, eps):
    """Continuous-time passage time between u_in and u_out for du/dn = a eps + b u^2 (a, b, eps > 0)."""
    s = np.sqrt(b / (a * eps))
    return float((np.arctan(u_out * s) - np.arctan(u_in * s)) / np.sqrt(a * b * eps))


def trace_curve(family, x0, theta0, kind, fix=1, direction=1, steps=100, ds=0.01, ds_min=1e-6, ds_max=0.05,
                tol=1e-10, gtol=1e-10, max_newton=8, bounds=None, on_point=None, k=8, chord=True, gmres_tol=1e-10,
                gmres_maxiter=4, refine_k=0):
    """Trace the fold or flip curve of the period-p orbit near (x0, theta0) in the parameter plane.

    First the bifurcation point is located with theta[fix] held at theta0[fix] (Newton on the minimally augmented
    system from x0, which should be close to the orbit near its fold or flip). Then pseudo-arclength in the scaled
    coordinates (x / |x0|, theta) follows the curve in `direction` (sign of the initial d theta[1 - fix]).
    Every accepted point records theta, residuals, the multipliers and the parameter tangent; `cusp_candidate`
    marks a reversal of the parameter tangent between consecutive points (the projection of the curve has a cusp
    there). Stops after `steps` points, when theta leaves `bounds` ((lo0, hi0), (lo1, hi1)) or when the step falls
    below ds_min. Returns (records, stop_reason, final states).

    With chord=True the corrector keeps the extended Jacobian of the last accepted point and evaluates only the
    residual and g at each iterate, g by GMRES on the bordered system with Jacobian-vector products, preconditioned
    by the last point's factorized bordered matrix; the dense monodromy is computed once per accepted point.
    Set gmres_tol above the relative residual the JVPs can reach (with float32 derivatives of the soft-token map,
    4.3e-6 at the P8 fold): a tolerance below it is never met, so every GMRES call runs all gmres_maxiter restart
    cycles of up to 30 products. The g it returns is still converged (identical to 10 digits at rtol 1e-8 and
    1e-10 there); gmres_maxiter=1 caps the cost of a call at 31 products. Each record counts the JVPs spent
    since the previous point ('jvps'), the GMRES calls that stopped short of gmres_tol ('gmres_unconverged') and
    the rejected steps ('rejected'). With refine_k > 0 every point also records the critical multiplier from a
    refine_k-step Arnoldi factorization of family.jvp_x ('critical_multiplier_refined', with its Ritz residual):
    use it when the dense monodromy is computed in lower precision than the JVPs.
    """
    if kind not in ("fold", "flip"):
        raise ValueError(f"kind must be fold or flip, not {kind}")
    sign = 1.0 if kind == "flip" else -1.0
    n = family.n
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    s = float(np.linalg.norm(x0)) or 1.0
    I = np.eye(n)
    cost = {"jvps": 0, "gmres_unconverged": 0, "rejected": 0}

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
        the Jacobian and monodromy evaluated there (reused for the tangent and the multipliers), |g| and the dense
        |g|. With chord=True, g itself comes from the JVPs (g_free), as in the continuation, so that a monodromy
        computed in lower precision enters only the Newton matrix."""
        z = z0.copy()
        for it in range(max_newton + 1):
            G, J, M, v, w, rel = system(z, b, c)
            gd = abs(G[n])
            if chord:
                G[n] = g_free(z, b, c, factor(M, b, c))
            if rel < tol and abs(G[n]) < gtol:
                return z, J, M, v, w, rel, abs(G[n]), it, gd
            if it == max_newton:
                break
            dz = np.linalg.solve(np.vstack([J, row[None]]), -np.concatenate([G, [rhs_fn(z)]]))
            z = z + dz
        return None, None, None, None, None, rel, abs(G[n]), max_newton, gd

    def tangent(J, previous):
        """Unit null vector of the (n+1) x (n+2) Jacobian, oriented along `previous`."""
        y = np.linalg.solve(np.vstack([J, previous[None]]), np.concatenate([np.zeros(n + 1), [1.0]]))
        return y / np.linalg.norm(y)

    def factor(M, b, c):
        from scipy.linalg import lu_factor
        B = np.zeros((n + 1, n + 1))
        B[:n, :n], B[:n, n], B[n, :n] = M + sign * I, b, c
        return lu_factor(B)

    def g_free(z, b, c, lu):
        """g at z from the bordered system solved by GMRES with JVPs (preconditioner: last point's LU)."""
        from scipy.linalg import lu_solve
        from scipy.sparse.linalg import LinearOperator, gmres

        x, th = z[:n] * s, z[n:]

        def mv(y):
            cost["jvps"] += 1
            y = np.asarray(y).ravel()
            Av = family.jvp_x(x, th, y[:n]) + sign * y[:n]
            return np.concatenate([Av + y[n] * b, [c @ y[:n]]])

        rhs = np.zeros(n + 1)
        rhs[n] = 1.0
        y, info = gmres(LinearOperator((n + 1, n + 1), matvec=mv, dtype=np.float64), rhs,
                        M=LinearOperator((n + 1, n + 1), matvec=lambda r: lu_solve(lu, r), dtype=np.float64),
                        rtol=gmres_tol, restart=30, maxiter=gmres_maxiter)
        cost["gmres_unconverged"] += int(info != 0)
        return float(y[n])

    def correct_chord(z0, row, rhs_fn, J0, b, c, lu):
        z = z0.copy()
        A = np.vstack([J0, row[None]])
        gabs = np.inf
        for it in range(max_newton + 1):
            x = z[:n] * s
            r = family.residual(x, z[n:])
            rel = float(np.linalg.norm(r) / np.linalg.norm(x))
            g = g_free(z, b, c, lu)
            gabs = abs(g)
            if rel < tol and gabs < gtol:
                G, J, M, v, w, _ = system(z, b, c)                # dense, once, at the converged point
                return z, J, M, v, w, rel, gabs, it, abs(G[n])
            if it == max_newton:
                break
            z = z + np.linalg.solve(A, -np.concatenate([r / s, [g], [rhs_fn(z)]]))
        return None, None, None, None, None, rel, gabs, max_newton, None

    b, c, smin = initial_borders(family.monodromy(x0, theta0), kind)
    z = np.concatenate([x0 / s, np.asarray(theta0, dtype=np.float64)])
    e_fix = np.zeros(n + 2)
    e_fix[n + fix] = 1.0
    zf, J, M, v, w, rel, gabs, its, gd = correct(z, b, c, e_fix, lambda y: 0.0)
    if zf is None:
        return [], f"no {kind} point at fixed theta[{fix}] = {theta0[fix]} (residual {rel:.3g}, |g| {gabs:.3g})", []
    z = zf
    e_free = np.zeros(n + 2)
    e_free[n + 1 - fix] = float(np.sign(direction))
    t = tangent(J, e_free)

    def rebase(J, M, v, w):
        """New borders from the null vectors at the accepted point; the g row of the chord Jacobian rescales with
        them (g' = g / (|v| |w|) to first order on the curve), and the bordered matrix is refactored."""
        nv, nw = np.linalg.norm(v), np.linalg.norm(w)
        J = J.copy()
        J[n] /= nv * nw
        b, c = w / nw, v / nv
        return J, b, c, (factor(M, b, c) if chord else None)

    J, b, c, lu = rebase(J, M, v, w)
    records, states = [], []

    def record(z, rel, gabs, its, ds_used, step, t, t_prev, M, gd):
        vals = np.linalg.eigvals(M)
        tp = t[n:] / max(np.linalg.norm(t[n:]), 1e-300)
        rec = {"index": len(records), "theta": [float(z[n]), float(z[n + 1])], "residual": rel, "g": gabs, "g_dense": gd,
               "newton": its, "ds": ds_used,
               "arclength": (records[-1]["arclength"] + step) if records else 0.0,
               "parameter_tangent": [float(t[n]), float(t[n + 1])],
               "parameter_speed": float(np.linalg.norm(t[n:])),
               "cusp_candidate": bool(t_prev is not None and float(t[n:] @ t_prev[n:]) < 0)}
        rec.update(cost)
        cost.update(jvps=0, gmres_unconverged=0, rejected=0)
        rec.update(multiplier_summary(vals, kind, k))
        if refine_k:
            x, th = z[:n] * s, z[n:]
            ritz = refine_multipliers(lambda u: family.jvp_x(x, th, u), M, m=refine_k, count=2)
            target = 1.0 if kind == "fold" else -1.0
            mu, res = min(ritz, key=lambda r: abs(r[0] - target))
            rec["critical_multiplier_refined"] = [mu.real, mu.imag]
            rec["refine_residual"] = res
        records.append(rec)
        states.append(z[:n] * s)
        if on_point is not None:
            on_point(rec)
        return tp

    record(z, rel, gabs, its, 0.0, 0.0, t, None, M, gd)
    reason = f"{steps} points"
    while len(records) < steps:
        zp = z + ds * t
        if chord:
            zn, Jn, Mn, v, w, rel, gabs, its, gd = correct_chord(zp, t, lambda y: float(t @ (y - zp)), J, b, c, lu)
        else:
            zn, Jn, Mn, v, w, rel, gabs, its, gd = correct(zp, b, c, t, lambda y: float(t @ (y - zp)))
        tn = None if zn is None else tangent(Jn, t)
        if zn is None or float(tn @ t) < 0.5:              # not converged, or jumped onto another branch
            cost["rejected"] += 1
            ds *= 0.5
            if ds < ds_min:
                reason = f"step below {ds_min} (last residual {rel:.3g}, |g| {gabs:.3g})"
                break
            continue
        step = float(np.linalg.norm(zn - z))
        t_prev = t
        z, t, M = zn, tn, Mn
        J, b, c, lu = rebase(Jn, Mn, v, w)
        record(z, rel, gabs, its, ds, step, t, t_prev, M, gd)
        if bounds is not None and not all(lo <= z[n + i] <= hi for i, (lo, hi) in enumerate(bounds)):
            reason = "parameters out of bounds"
            break
        if its <= (max_newton // 2 if chord else 2):
            ds = min(ds * 1.5, ds_max)
        elif its >= max_newton - 1:
            ds *= 0.7
    return records, reason, states


def classify_crossings(records):
    """Unit-circle crossings between consecutive points of a branch: the side where the spectral radius exceeds
    1 supplies the crossing multiplier (its largest-modulus one), recorded with modulus, argument (turns) and the
    next-largest modulus; typed as a complex pair (Neimark-Sacker), real +1 (fold) or real -1 (flip). Uses the
    refined (Arnoldi) multipliers when every record has them."""
    refined = bool(records) and all("spectral_radius_refined" in r for r in records)
    rho_key, mult_key = (("spectral_radius_refined", "multipliers_refined") if refined else
                         ("spectral_radius", "multipliers"))
    crossings = []
    for a, b in zip(records, records[1:]):
        if a[rho_key] is None or b[rho_key] is None:
            continue
        if (a[rho_key] - 1) * (b[rho_key] - 1) < 0:
            outer = b if b[rho_key] > 1 else a
            vals = [complex(re, im) for re, im in outer[mult_key]]
            lead = max(vals, key=abs)
            kind = ("complex pair (Neimark-Sacker)" if abs(lead.imag) > 1e-9 * max(1.0, abs(lead)) else
                    "real +1 (fold)" if lead.real > 0 else "real -1 (flip)")
            others = sorted((abs(z) for z in vals if z != lead and z != lead.conjugate()), reverse=True)
            crossings.append({"between_log10_T": [a["lambda"], b["lambda"]], "between_indices": [a["index"], b["index"]],
                              "type": kind,
                              "crossing_multiplier": [lead.real, lead.imag, abs(lead), float(np.angle(lead) / (2 * np.pi))],
                              "next_largest_modulus": others[0] if others else None,
                              "multipliers_converged": bool(a.get("multipliers_converged", True) and
                                                            b.get("multipliers_converged", True)),
                              "multipliers_source": "refined" if refined else "dense"})
    return crossings


def trace_branch(family, x0, theta0, free=0, direction=1, steps=60, ds=0.005, ds_min=1e-6, ds_max=0.02, tol=1e-10,
                 max_newton=8, beyond=0.02, bounds=None, on_point=None, k=8, refine_k=0):
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
