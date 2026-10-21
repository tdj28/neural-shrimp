"""Explicit memory intervention on the original soft-token feedback map.

beta=0 is exactly the original map; positive beta retains the previous state.
Negative beta is mathematical extrapolation, not a retention fraction.
"""
import numpy as np


class MemoryOrbit:
    def __init__(self, base, beta):
        if not np.isfinite(beta) or abs(beta) > .05:
            raise ValueError("memory pilot requires finite |beta| <= .05")
        if base.shape[0] != 1:
            raise ValueError("only the declared one-token state is supported")
        self.base, self.beta = base, float(beta)
        for name in ("torch", "shape", "n", "start"):
            setattr(self, name, getattr(base, name))

    def f(self, x):
        y = self.base.f(x)
        return y if self.beta == 0 else (1-self.beta)*y+self.beta*x

    def jvp_rows(self, x, v):
        jv = self.base.jvp_rows(x, v)
        return jv if self.beta == 0 else (1-self.beta)*jv+self.beta*v

    def jacobian(self, x):
        j = self.base.jacobian(x)
        return j if self.beta == 0 else (1-self.beta)*j+self.beta*np.eye(self.n)

    def probabilities(self, x):
        return self.base.probabilities(x)

    def points(self, x, period):
        rows = [x]
        for _ in range(period):
            rows.append(self.f(rows[-1]))
        return rows

    def iterate(self, x, period):
        return self.points(x, period)[-1]

    def return_distances(self, x, period):
        return [float((p-x).norm()/x.norm()) for p in self.points(x, period)[1:]]


def qualified_ritz(apply, q, *, max_passes=6, tolerance=1e-6):
    """Matrix-free spectrum with relative residual and isolated-pair gates.

    The final eigenvectors are independently reapplied, not qualified just by
    their small projected residual. The third Ritz value is a separation
    diagnostic, conditional on the converged block; baseline dense replay is
    required separately by the experiment driver.
    """
    q = np.linalg.qr(np.asarray(q, float))[0]
    if not np.isfinite(q).all() or q.shape[1] < 4:
        raise ValueError("at least four subspace directions required")
    history = []
    for iteration in range(max_passes):
        w = apply(q)
        if w.shape!=q.shape or not np.isfinite(w).all():
            raise ValueError('nonfinite or mismatched matrix-free operator action')
        vals, small = np.linalg.eig(q.T@w)
        order = np.argsort(-abs(vals))
        vals, small = vals[order], small[:, order]
        if not np.isfinite(vals).all() or min(abs(vals[:2]))==0:
            raise ValueError('nonfinite or zero leading pair; requires separate classification')
        v = q@small
        residual = np.linalg.norm(w@small-v*vals, axis=0)/np.maximum(abs(vals)*np.linalg.norm(v, axis=0), 1e-300)
        history.append({"iteration":iteration, "multipliers":[[float(z.real),float(z.imag)] for z in vals],
                        "relative_residuals":residual.tolist()})
        if iteration >= 1 and max(residual[:2]) < tolerance:
            if np.max(abs(vals[:2].imag)) > 1e-10*max(abs(vals[:2])):
                raise ValueError("tracked real pair became complex")
            if abs(vals[1]) <= 100*abs(vals[2]):
                raise ValueError("tracked pair lost separation from third Ritz value")
            vectors = v[:, :2].real
            direct = apply(vectors)
            errors = np.linalg.norm(direct-vectors*vals[:2].real, axis=0)/np.maximum(abs(vals[:2])*np.linalg.norm(vectors, axis=0),1e-300)
            if max(errors) < tolerance:
                return {"values":vals,"vectors":vectors,"direct":direct,"relative_residuals":errors,
                        "q":q,"w":w,"history":history}
        q = np.linalg.qr(w)[0]
    raise ValueError(f"matrix-free pair did not converge: {history}")
