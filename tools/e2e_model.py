"""Run E2E: the soft-token map of Qwen2.5-0.5B-Instruct (philosophy direction at layer 12, fixed prefix and opener of
every map) bound to the model-free numerics of tools/e2e_numerics.py.

The map is SoftLoop/SoftOrbit of tools/explore_llm_soft_loop.py in float64 on the CPU. In float64, Engine casts the
model to float64 and applies keep_float64_norms (RMSNorm and rotary tables in float64), so this is the patched
float64 map of every float64 result in the paper. Only the window K = 1 is used: a state is one embedding x (896).

Cell(alpha, log10 T)
  F(X)             one step for a batch of states (B, d) (numpy in, numpy out)
  Fp(x, p)         the p-fold map of one state
  probs/logits(x)  next-token distribution and logits at a state (batched)
  apply_M_fd       M V for the p-fold map by central differences of batched steps (Newton-Picard, continuation)
  apply_M_ad       M V by forward-mode JVPs through the orbit's p steps (reported multipliers)
  solve            Newton-Picard orbit with exact float64 residual; returns the orbit and its AD multipliers
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e2e_numerics as en  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
OUT = DATA / "explore" / "e2e"
VECTORS = "llm-phase-map/vectors-l12"
CONCEPT = "philosophy"
TOKENS = {"is": 374, "it": 432, "nn": 271, "The": 785, "existence": 13885, "question": 3405, ".": 13,
          "This": 1096, "there": 1052, "a": 264}


class Model:
    def __init__(self, threads=8):
        import explore_llm_attractors as ela
        import explore_llm_phase_map as elpm
        import explore_llm_soft_loop as esl

        self.esl = esl
        self.loop = esl.SoftLoop("qwen0.5b", 12, threads, "open", ela.OPENER, 1, "float64", "cpu")
        if self.loop.state_dtype != self.loop.torch.float64 or self.loop.model.dtype != self.loop.torch.float64:
            raise SystemExit("e2e needs the float64 map")
        self.torch = self.loop.torch
        self.v = np.asarray(elpm.load_vectors(VECTORS)[CONCEPT])
        self.E = self.loop.E.cpu().numpy()                    # (V, d) float64
        self.tok = self.loop.sm.tok
        self.d = self.E.shape[1]

    def decode(self, i):
        return self.tok.decode([int(i)])

    def cell(self, alpha, log_t):
        return Cell(self, alpha, log_t)


class Cell:
    def __init__(self, model, alpha, log_t):
        self.m, self.torch = model, model.torch
        self.alpha, self.log_t = float(alpha), float(log_t)
        self.T = 10.0 ** self.log_t
        self.orb = model.esl.SoftOrbit(model.loop, self.alpha * model.v, self.T)
        self.start = self.orb.start[0, 0].cpu().numpy().copy()     # the opener's embedding (the fixed start)
        self.d = model.d

    def _t(self, X):
        X = np.asarray(X, dtype=np.float64).reshape(-1, 1, self.d)
        return self.torch.as_tensor(X, dtype=self.torch.float64)

    def F(self, X):
        with self.torch.no_grad():
            return self.orb.f(self._t(X))[:, -1].cpu().numpy()

    def Fp(self, x, p):
        X = np.asarray(x, dtype=np.float64).reshape(-1, self.d)
        for _ in range(p):
            X = self.F(X)
        return X[0] if np.ndim(x) == 1 else X

    def orbit_points(self, x, p):
        pts = [np.asarray(x, dtype=np.float64)]
        for _ in range(p - 1):
            pts.append(self.F(pts[-1])[0])
        return pts

    def logits(self, X):
        X = self._t(X)
        self.orb._use(X.shape[0])
        with self.torch.no_grad():
            return self.m.loop.logits(X).cpu().numpy()

    def probs(self, X):
        z = self.logits(X) / self.T
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def apply_M_fd(self, x, p, V, h_rel=1e-6):
        """M V for the p-fold map at x by central differences, all 2k perturbed states iterated in one batch."""
        V = np.asarray(V, dtype=np.float64)
        k = V.shape[1]
        nrm = np.linalg.norm(V, axis=0)
        nrm[nrm == 0] = 1.0
        h = h_rel * np.linalg.norm(x) / nrm                  # the same relative step along every column
        X = np.concatenate([x[None] + h[:, None] * V.T, x[None] - h[:, None] * V.T])
        Y = self.Fp(X, p)
        return ((Y[:k] - Y[k:]) / (2 * h[:, None])).T

    def apply_M_ad(self, pts, V):
        """M V by forward-mode JVPs through the orbit points pts (x_0 .. x_{p-1}); V (d, k)."""
        torch = self.torch
        W = torch.as_tensor(np.ascontiguousarray(V.T).reshape(-1, 1, self.d), dtype=torch.float64)
        for x in pts:
            W = self.orb.jvp_rows(self._t(x), W)
        return W[:, -1].detach().cpu().numpy().T

    def multipliers(self, x, p, k=6, iters=12, Q0=None, ad=True, tol=1e-9):
        pts = self.orbit_points(x, p)
        apply = (lambda V: self.apply_M_ad(pts, V)) if ad else (lambda V: self.apply_M_fd(x, p, V))
        return en.subspace_iteration(apply, self.d, k, iters=iters, Q0=Q0, tol=tol)

    def solve(self, x0, p, k=6, Q0=None, tol=1e-12, fd_iters=8, ad_iters=10, maxit=40, ad=True, cycles=8, inner=4):
        """Newton-Picard for F^p(x) = x from x0. The dominant subspace and the coupling products come from central
        differences at the current iterate and are refreshed every `inner` iterations (the linear model drifts when
        x0 is far from the orbit); multipliers at the converged orbit from forward-mode JVPs (ad) or differences.
        Fails loudly (converged False) when tol is not reached: the caller decides; nothing is substituted."""
        x, Q, hist, ok = np.asarray(x0, dtype=np.float64).copy(), Q0, [], False
        vals_fd = None
        for c in range(cycles):
            xc = x.copy()
            Q, H, vals_fd, _, _ = en.subspace_iteration(lambda V: self.apply_M_fd(xc, p, V), self.d, k,
                                                        iters=fd_iters if c == 0 and Q0 is None else 3, Q0=Q,
                                                        tol=1e-7)
            V, Hm = en.dominant_block(Q, H, max(1, k // 2))
            x, h, ok = en.newton_picard(lambda y: self.Fp(y, p), lambda W: self.apply_M_fd(xc, p, W), xc, V, Hm,
                                        tol=tol, maxit=inner if c < cycles - 1 else maxit)
            hist += h if not hist else h[1:]
            if ok or not np.all(np.isfinite(x)):
                break
        info = {"converged": bool(ok), "residuals": hist, "fd_multipliers": [[float(z.real), float(z.imag)]
                                                                            for z in vals_fd[:4]]}
        if not ok:
            return x, info, Q
        # multipliers at the converged orbit: forward-mode JVPs (ad) or central differences of batched steps
        Q2, H2, vals, conv, _ = self.multipliers(x, p, k=k, iters=ad_iters if ad else 4, Q0=Q, ad=ad)
        info["multipliers"] = [[float(z.real), float(z.imag)] for z in vals]
        info["multipliers_converged"] = bool(conv)
        info["multipliers_source"] = "forward-mode JVP" if ad else "central differences (h = 1e-6 |x|)"
        info["crossing_side"] = en.classify(vals)
        return x, info, Q2

    def text(self, x, p):
        """Leading token, its probability and the runner-up at each point of a p-orbit from x."""
        pts = self.orbit_points(x, p)
        P = self.probs(np.array(pts))
        out = []
        for row in P:
            i = np.argsort(-row)[:3]
            out.append([[self.m.decode(j), float(row[j])] for j in i])
        return out
