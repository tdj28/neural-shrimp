"""Exploratory: soft-token decoding as a smooth map, scanned for periodic windows and chaos.

Instead of sampling a token, the loop feeds back the probability-weighted input
embedding x_new = E^T softmax(z / T), where z are the next-token logits after a
fixed prefix plus the last K fed-back embeddings (a sliding window at fixed
positions, so the map is autonomous). As T -> 0 the map becomes greedy decoding
with a K-token window; at finite T it is a smooth map of R^(K d), so periodic
orbits, multipliers, Lyapunov exponents and bifurcation curves are defined in
the usual sense. An additive steering vector alpha * v on one decoder layer's
output gives a second continuous parameter.

Subcommands
  scan   Grid in (log10 T, alpha): iterate all cells in one batch, then measure
         the period of the attractor (state recurrence) and the largest Lyapunov
         exponent (twin trajectory, renormalized every step).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402

DATA = ela.DATA


class SoftLoop:
    def __init__(self, model, layer, threads, prompt, opener, window, dtype="float64", device="cpu"):
        self.sm = elpm.SteeredModel(model, layer, threads, dtype)
        self.torch = self.sm.torch
        torch = self.torch
        self.device = torch.device(device)
        self.model = self.sm.model.to(self.device)
        self.dtype = self.model.dtype
        self.state_dtype = torch.float64 if dtype == "float64" else torch.float32
        self.E = self.model.get_input_embeddings().weight.detach().to(self.state_dtype)      # (V, d)
        messages = [{"role": "system", "content": ela.PROMPTS[prompt]}, {"role": "user", "content": opener}]
        text = self.sm.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.prefix_ids = torch.tensor([self.sm.tok(text, add_special_tokens=False)["input_ids"]], device=self.device)
        self.L0 = self.prefix_ids.shape[1]
        self.K = int(window)
        self.cache = None
        self.steer_rows = None

    def prepare(self, steer_rows):
        """Prefix key-value cache for every row, computed with that row's steering."""
        from transformers import DynamicCache

        torch = self.torch
        B = len(steer_rows)
        self.steer_rows = torch.as_tensor(np.asarray(steer_rows), dtype=self.dtype, device=self.device)[:, None, :]
        self.cache = DynamicCache()
        self.sm.steer = self.steer_rows
        try:
            with torch.no_grad():
                # logits_to_keep=1: the prefix logits are not needed (all of them cost B x L0 x vocab floats)
                self.model(input_ids=self.prefix_ids.repeat(B, 1), past_key_values=self.cache, use_cache=True,
                           logits_to_keep=1)
        finally:
            self.sm.steer = None
        with torch.no_grad():
            start = self.model.get_input_embeddings()(self.prefix_ids[:, -1:])[0, 0].to(self.state_dtype)
        return start.repeat(B, self.K, 1).clone()                              # initial window (B, K, d)

    def logits(self, X):
        torch = self.torch
        self.sm.steer = self.steer_rows
        try:
            with torch.no_grad():
                z = self.model(inputs_embeds=X.to(self.dtype), past_key_values=self.cache, use_cache=True,
                               logits_to_keep=1).logits[:, -1, :].to(self.state_dtype)
        finally:
            self.sm.steer = None
            self.cache.crop(self.L0)                                           # keep only the prefix (argument = length kept)
        return z

    def step(self, X, temps):
        """One soft-decoding step: X (B, K, d) -> shifted window with the new soft embedding appended."""
        torch = self.torch
        z = self.logits(X)
        p = torch.softmax(z / torch.as_tensor(temps, dtype=self.state_dtype, device=self.device)[:, None], dim=-1)
        return torch.cat([X[:, 1:, :], (p @ self.E)[:, None, :]], dim=1), p


class SoftOrbit:
    """Periodic orbits of the soft-decoding map at one (temperature, steering) cell: Newton's method on the
    p-fold map with dense central-difference Jacobians (2n rows per Jacobian in chunked batches over a prefix
    cache computed once and repeated), and the orbit's multipliers from the product of those Jacobians."""

    def __init__(self, loop, steer, temperature, h_rel=1e-5, chunk=512, jacobian_method="ad"):
        self.loop, self.torch = loop, loop.torch
        self.T = float(temperature)
        self.start = loop.prepare(np.asarray(steer)[None])        # one-row prefix cache and initial window
        self.cache1, self.steer1 = loop.cache, loop.steer_rows
        self.shape = tuple(self.start.shape[1:])
        self.n = int(np.prod(self.shape))
        self.h_rel, self.chunk = h_rel, int(chunk)
        self.jacobian_method = jacobian_method
        self.caches = {1: self.cache1}                          # prefix caches repeated to each batch size used

    def _use(self, rows):
        import copy

        if rows not in self.caches:
            c = copy.deepcopy(self.cache1)
            c.batch_repeat_interleave(rows)
            self.caches[rows] = c
        self.loop.cache, self.loop.steer_rows = self.caches[rows], self.steer1.repeat(rows, 1, 1)

    def f(self, X):
        """One soft-decoding step for a batch of windows X (B, K, d) at this cell."""
        self._use(X.shape[0])
        Y, _ = self.loop.step(X, np.full(X.shape[0], self.T))
        return Y

    def probabilities(self, X):
        self._use(1)
        _, p = self.loop.step(X, np.array([self.T]))
        return p[0]

    def jvp_rows(self, X, V):
        """Exact Jacobian-vector products J(X) v for tangent rows V (m, K, d) at X (1, K, d): forward-mode AD
        through the model (math attention kernel), on a shallow copy of the repeated prefix cache so that the
        stored prefix tensors are never modified."""
        import copy

        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch, loop = self.torch, self.loop
        m = V.shape[0]
        self._use(m)
        base = self.caches[m]
        cache = copy.copy(base)
        cache.layers = [copy.copy(layer) for layer in base.layers]
        steer = self.steer1.repeat(m, 1, 1)

        def new_embedding(inp):
            loop.sm.steer = steer
            try:
                z = loop.model(inputs_embeds=inp.to(loop.dtype), past_key_values=cache, use_cache=True,
                               logits_to_keep=1).logits[:, -1, :].to(loop.state_dtype)
            finally:
                loop.sm.steer = None
            return torch.softmax(z / self.T, dim=-1) @ loop.E

        with sdpa_kernel(SDPBackend.MATH):
            _, jv = torch.func.jvp(new_embedding, (X.repeat(m, 1, 1),), (V,))
        return torch.cat([V[:, 1:, :], jv[:, None, :]], dim=1)          # the window shifts

    def jacobian(self, X, method=None):
        """Dense Jacobian of one step at X (1, K, d): forward-mode AD (exact) or central differences with
        step h_rel * RMS(X)."""
        torch = self.torch
        method = method or self.jacobian_method
        eye = torch.eye(self.n, dtype=X.dtype, device=X.device).reshape((self.n,) + self.shape)
        h = self.h_rel * float(X.norm()) / np.sqrt(self.n)
        cols = []
        for s in range(0, self.n, self.chunk):
            e = eye[s:s + self.chunk]
            m = e.shape[0]
            if method == "ad":
                cols.append(self.jvp_rows(X, e).reshape(m, -1))
            else:
                Y = self.f(torch.cat([X + h * e, X - h * e]))
                cols.append(((Y[:m] - Y[m:]) / (2 * h)).reshape(m, -1))
        return torch.cat(cols).T.double().cpu().numpy()                 # J[:, i] = d f / d x_i

    def iterate(self, X, steps):
        for _ in range(steps):
            X = self.f(X)
        return X

    def return_distances(self, X, max_period):
        """Relative distances |f^q(X) - X| / |X| for q = 1..max_period."""
        hist = [X]
        for _ in range(max_period):
            hist.append(self.f(hist[-1]))
        return [float((hist[q] - X).norm() / X.norm()) for q in range(1, max_period + 1)]

    def detect_period(self, X, max_period, tol):
        dist = self.return_distances(X, max_period)
        return next((q for q, d in enumerate(dist, 1) if d < tol), 0)

    def newton(self, X, period, tol=1e-11, max_newton=8):
        """Newton for f^p(X) = X; returns (X, relative residual history, multipliers, orbit points)."""
        torch = self.torch
        history = []
        for it in range(max_newton + 1):
            pts = [X]
            for _ in range(period):
                pts.append(self.f(pts[-1]))
            r = (pts[period] - X).reshape(-1).double().cpu().numpy()
            rel = float(np.linalg.norm(r) / float(X.norm()))
            history.append(rel)
            Js = [self.jacobian(x) for x in pts[:period]]
            M = np.eye(self.n)
            for J in Js:
                M = J @ M
            if rel < tol or it == max_newton:
                break
            delta = np.linalg.solve(M - np.eye(self.n), -r)
            X = X + torch.as_tensor(delta.reshape((1,) + self.shape), dtype=X.dtype, device=X.device)
        vals = np.linalg.eigvals(M)
        return X, history, vals[np.argsort(-np.abs(vals))], pts[:period]


    def points(self, X, period):
        pts = [X]
        for _ in range(period):
            pts.append(self.f(pts[-1]))
        return pts

    def subspace_multipliers(self, X, period, k=8, iters=40, seed=0):
        """Leading multipliers of the p-fold map at X by block subspace iteration with batched forward-mode
        JVPs (k tangent rows per model call) and a final Rayleigh-Ritz step. Unlike Arnoldi with a relative
        tolerance, it does not stall when every multiplier is tiny (super-stable orbits). Returns (multipliers,
        converged), converged meaning the moduli of the leading half of the block changed by less than 1e-6
        (relative to the largest) over the last 5 iterations; the trailing Ritz values converge last and are
        returned only for information."""
        torch = self.torch
        pts = self.points(X, period)[:period]

        def apply(V):
            for x in pts:
                V = self.jvp_rows(x, V)
            return V

        rng = np.random.default_rng(seed)
        Q = np.linalg.qr(rng.normal(size=(self.n, k)))[0]
        history = []
        for it in range(iters):
            W = apply(torch.as_tensor(Q.T.reshape((k,) + self.shape), dtype=X.dtype, device=X.device))
            W = W.reshape(k, -1).double().cpu().numpy().T                  # columns M q_i
            H = Q.T @ W
            vals = np.linalg.eigvals(H)
            history.append(np.sort(np.abs(vals))[::-1][: max(2, k // 2)])
            Q = np.linalg.qr(W)[0]
        ok = len(history) > 5 and float(np.abs(history[-1] - history[-6]).max()) <= 1e-6 * max(history[-1][0], 1e-300)
        return vals[np.argsort(-np.abs(vals))], ok

    def phi(self, period):
        """The p-fold soft map as a differentiable function of (window X (1, K, d), log10 T) for continuation in
        temperature (the prefix cache does not depend on T); every model call uses a shallow copy of the
        one-row prefix cache, so forward-mode tangents never reach the stored prefix tensors."""
        import copy

        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch, loop = self.torch, self.loop

        def step(X, temperature):
            cache = copy.copy(self.cache1)
            cache.layers = [copy.copy(layer) for layer in self.cache1.layers]
            loop.sm.steer = self.steer1
            try:
                with sdpa_kernel(SDPBackend.MATH):
                    z = loop.model(inputs_embeds=X.to(loop.dtype), past_key_values=cache, use_cache=True,
                                   logits_to_keep=1).logits[:, -1, :].to(loop.state_dtype)
            finally:
                loop.sm.steer = None
            p = torch.softmax(z / temperature, dim=-1)
            return torch.cat([X[:, 1:, :], (p @ loop.E)[:, None, :]], dim=1)

        def phi(X, log_t):
            temperature = 10.0 ** log_t
            for _ in range(period):
                X = step(X, temperature)
            return X

        return phi


def orbit_report(orb, X, history, vals, pts, k=8, tol=1e-9):
    tok = orb.loop.sm.tok
    seps = [float((p - pts[0]).norm() / pts[0].norm()) for p in pts[1:]]
    tokens = []
    for x in pts:
        p = orb.probabilities(x).double().cpu().numpy()
        top = int(p.argmax())
        tokens.append({"token": tok.decode([top]), "probability": float(p[top]),
                       "entropy": float(-(p * np.log(np.maximum(p, 1e-300))).sum())})
    return {"newton_residuals": history, "converged": history[-1] < tol,
            "spectral_radius": float(np.abs(vals[0])),
            "multipliers": [[float(v.real), float(v.imag), float(abs(v)), float(np.angle(v) / (2 * np.pi))]
                            for v in vals[:k]],
            "min_separation_from_first_point": min(seps) if seps else None, "points": tokens}


def orbit(args):
    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                    args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    orb = SoftOrbit(loop, args.alpha * v, 10.0 ** args.log_temperature, args.h_rel, args.chunk, args.jacobian)
    tic = time.perf_counter()
    X = orb.iterate(orb.start, args.transient)
    returns = orb.return_distances(X, args.max_period)
    period = args.period or next((q for q, d in enumerate(returns, 1) if d < args.detect_tol), 0)
    if period == 0:
        raise SystemExit(f"no period up to {args.max_period} at tolerance {args.detect_tol}; nothing to verify "
                         f"(smallest return distance {min(returns):.3g} at q = {int(np.argmin(returns)) + 1})")
    X, history, vals, pts = orb.newton(X, period, args.tol, args.max_newton)
    rec = {"args": vars(args), "period": period, "seconds": round(time.perf_counter() - tic, 1),
           "return_distances_after_transient": returns}
    rec.update(orbit_report(orb, X, history, vals, pts, tol=args.tol))
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "orbit.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    np.save(out / "orbit-state.npy", X.double().cpu().numpy())
    print(json.dumps({k: v for k, v in rec.items() if k != "args"}, ensure_ascii=False))


def tline(args):
    """Continuation in log10 T, at fixed steering, of the periodic orbit reached at (log10 T, alpha): pseudo-
    arclength through folds with multipliers at every point (explore_latent_loop.pseudo_arclength), and the
    classification of every unit-circle crossing found along the branch."""
    import explore_latent_loop as ell

    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                    args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    orb = SoftOrbit(loop, args.alpha * v, 10.0 ** args.log_temperature, chunk=args.chunk)
    X = orb.iterate(orb.start, args.transient)
    period = args.period or orb.detect_period(X, args.max_period, args.detect_tol)
    if period == 0:
        raise SystemExit(f"no period up to {args.max_period} at tolerance {args.detect_tol}; nothing to continue")
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    states = []

    def on_point(rec, x):
        states.append(x.double().cpu().numpy()[0])
        print(json.dumps({k: v for k, v in rec.items() if k != "multipliers"} | {"leading": rec["multipliers"][:3]}),
              flush=True)

    def multipliers(x, log_t):
        orb.T = 10.0 ** log_t
        return orb.subspace_multipliers(x, period, k=args.k, iters=args.subspace_iter)

    records, reason = ell.pseudo_arclength(
        orb.phi(period), X, args.log_temperature, direction=args.direction, steps=args.steps, ds=args.ds,
        ds_min=args.ds_min, ds_max=args.ds_max, tol=args.tol, max_newton=args.max_newton, gmres_tol=args.gmres_tol,
        restart=args.restart, gmres_iter=args.gmres_iter, k=args.k, beyond=args.beyond,
        lam_bounds=tuple(args.bounds), on_point=on_point, multipliers_fn=multipliers)
    crossings = []
    for a, b in zip(records, records[1:]):
        if a["spectral_radius"] is None or b["spectral_radius"] is None:
            continue
        if (a["spectral_radius"] - 1) * (b["spectral_radius"] - 1) < 0:
            outer = b if b["spectral_radius"] > 1 else a          # the side where the crossing multiplier is outside
            vals = [complex(re, im) for re, im in outer["multipliers"]]
            lead = max(vals, key=abs)
            kind = ("complex pair (Neimark-Sacker)" if abs(lead.imag) > 1e-9 else
                    "real +1 (fold)" if lead.real > 0 else "real -1 (flip)")
            others = sorted((abs(z) for z in vals if z != lead and z != lead.conjugate()), reverse=True)
            crossings.append({"between_log10_T": [a["lambda"], b["lambda"]], "type": kind,
                              "crossing_multiplier": [lead.real, lead.imag, abs(lead), float(np.angle(lead) / (2 * np.pi))],
                              "next_largest_modulus": others[0] if others else None,
                              "multipliers_converged": bool(a["multipliers_converged"] and b["multipliers_converged"])})
    np.save(out / "states.npy", np.array(states))
    rec = {"args": vars(args), "period": period, "stop": reason, "crossings": crossings, "points": records}
    (out / "tline.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"period": period, "stop": reason, "points": len(records), "crossings": crossings}))


def precision(args):
    """Effective precision of the float32 map: at cells (log10 T, alpha), iterate the float64 map (norms patched)
    to its attractor, then compare one step and q steps of the float32 map (on --device) with the float64 map from
    the same states. Also reports the float64 attractor's return distances, so a period read at tolerance tol in
    float32 can be checked against the float32 error level."""
    v = elpm.load_vectors(args.vectors)[args.concept]
    hi = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, "float64",
                  args.device64)
    lo = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, "float32",
                  args.device)
    torch = hi.torch
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for log_t, alpha in args.cells:
        oh = SoftOrbit(hi, alpha * v, 10.0 ** log_t)
        ol = SoftOrbit(lo, alpha * v, 10.0 ** log_t)
        X = oh.iterate(oh.start, args.transient)
        returns = oh.return_distances(X, args.max_period)
        Xh, Xl, step_err = X, X.to(lo.state_dtype).to(lo.device), []
        for _ in range(args.steps):
            Xh, Xl = oh.f(Xh), ol.f(Xl.to(lo.state_dtype))
            step_err.append(float((Xl.double().to(Xh.device) - Xh).norm() / Xh.norm()))
        one = []
        Y = X
        for _ in range(args.steps):                         # one-step errors from float64 states
            Yh = oh.f(Y)
            Yl = ol.f(Y.to(lo.state_dtype).to(lo.device)).double().to(Yh.device)
            one.append(float((Yl - Yh).norm() / Yh.norm()))
            Y = Yh
        row = {"log10_temperature": log_t, "alpha": alpha, "float64_return_distances": returns,
               "float32_one_step_error": {"median": float(np.median(one)), "max": float(np.max(one))},
               "float32_trajectory_error": step_err, "state_norm": float(X.norm())}
        rows.append(row)
        print(json.dumps({k: (v if k != "float32_trajectory_error" else [v[0], v[-1]]) for k, v in row.items()}),
              flush=True)
    (out / "precision.json").write_text(json.dumps({"args": vars(args), "cells": rows}, indent=1))


def soft_family(loop, v, period, alpha_scale=10.0, chunk=512, reference=None, exact_jvp=False):
    """The p-fold soft map as a two-parameter family for bifurcation_curves, theta = (log10 T, alpha * alpha_scale).
    The steering acts on the prefix too, so phi recomputes the prefix cache inside the differentiable function
    (derivatives in alpha include the prefix); the dense monodromy reuses SoftOrbit's batched forward-mode
    Jacobians at fixed theta. With a `reference` SoftLoop (float64, norms patched) the residual Phi(x) - x is
    computed there and the dense derivatives in `loop`'s precision: Newton then converges to the reference
    precision. float32 derivatives of one step are accurate to about 5e-5, but those of the P8 orbit's 8-step map
    only to about 3 % (dense monodromy and JVPs alike), enough for a Newton matrix but not for multipliers or the
    test function g. exact_jvp=True takes the Jacobian-vector products (jvp_x: g by GMRES, refined multipliers)
    from the reference as well."""
    import copy

    import bifurcation_curves as bc
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import DynamicCache

    torch = loop.torch
    v = np.asarray(v)

    def make_phi(lp):
        v_t = torch.as_tensor(v, dtype=lp.dtype, device=lp.device)[None, None, :]

        def phi(X, theta):
            temperature = 10.0 ** theta[0]
            cache = DynamicCache()
            lp.sm.steer = (theta[1] / alpha_scale).to(lp.dtype) * v_t
            try:
                lp.model(input_ids=lp.prefix_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                for _ in range(period):
                    c = copy.copy(cache)                    # every step sees the prefix only (fixed positions)
                    c.layers = [copy.copy(layer) for layer in cache.layers]
                    z = lp.model(inputs_embeds=X.to(lp.dtype), past_key_values=c, use_cache=True,
                                 logits_to_keep=1).logits[:, -1, :].to(lp.state_dtype)
                    p = torch.softmax(z / temperature, dim=-1)
                    X = torch.cat([X[:, 1:, :], (p @ lp.E)[:, None, :]], dim=1)
            finally:
                lp.sm.steer = None
            return X

        return phi

    phi = make_phi(loop)
    if exact_jvp and reference is None:
        raise ValueError("exact_jvp needs a reference SoftLoop")
    exact = bc.Family(make_phi(reference), (1, loop.K, loop.E.shape[1]), reference.state_dtype, reference.device,
                      context=lambda: sdpa_kernel(SDPBackend.MATH)) if exact_jvp else None

    class Family(bc.Family):
        def __init__(self):
            super().__init__(phi, (1, loop.K, loop.E.shape[1]), loop.state_dtype, loop.device,
                             context=lambda: sdpa_kernel(SDPBackend.MATH))
            self.orbit, self.ref_orbit = None, None

        def soft_orbit(self, theta):
            key = tuple(float(t) for t in theta)
            if self.orbit is None or self.orbit[0] != key:
                self.orbit = None                           # release the old repeated caches first
                self.orbit = (key, SoftOrbit(loop, key[1] / alpha_scale * v, 10.0 ** key[0], chunk=chunk))
            return self.orbit[1]

        def residual(self, x, theta):
            if reference is None:
                X = self.soft_orbit(theta).iterate(self.to_x(x), period)
                return self.to_np(X) - np.asarray(x).ravel()
            key = tuple(float(t) for t in theta)
            if self.ref_orbit is None or self.ref_orbit[0] != key:
                self.ref_orbit = (key, SoftOrbit(reference, key[1] / alpha_scale * v, 10.0 ** key[0], chunk=chunk))
            ref = self.ref_orbit[1]
            X = torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(self.shape), dtype=reference.state_dtype,
                                device=reference.device)
            return self.to_np(ref.iterate(X, period)) - np.asarray(x).ravel()

        def monodromy(self, x, theta):
            orb = self.soft_orbit(theta)
            X, M = self.to_x(x), np.eye(self.n)
            for _ in range(period):
                M = orb.jacobian(X) @ M
                X = orb.f(X)
            return M

        def jvp_x(self, x, theta, v):
            return exact.jvp_x(x, theta, v) if exact is not None else super().jvp_x(x, theta, v)

    return Family()


def curve_start(tl, states, kind, index):
    """Starting point for curve: the tline point nearest the first turn (fold) or the first multiplier -1
    crossing (flip). kind 'auto' takes the first boundary found along the tline. Refuses a boundary whose
    crossing was classified otherwise (or not at all)."""
    pts = tl["points"]
    turns = [i for i, r in enumerate(pts) if r.get("turned_since_previous")]
    flips = [c for c in tl["crossings"] if c["type"].startswith("real -1")]
    if kind == "auto":
        lam0 = pts[0]["lambda"]
        cands = []
        if turns:
            cands.append((abs(pts[turns[0]]["lambda"] - lam0), "fold"))
        if flips:
            cands.append((min(abs(v - lam0) for v in flips[0]["between_log10_T"]), "flip"))
        if not cands:
            kinds = sorted({c["type"] for c in tl["crossings"]})
            raise SystemExit(f"no fold turn and no flip crossing along the tline (crossings: {kinds}); nothing to trace")
        kind = min(cands)[1]
    if kind == "fold":
        if not turns:
            raise SystemExit("no turn along the tline: no fold to start from")
        i = turns[0]
        est = pts[i].get("turn_lambda_estimate", pts[i]["lambda"])
        near = [c for c in tl["crossings"] if min(abs(v - est) for v in c["between_log10_T"]) < 0.02]
        if not any(c["type"].startswith("real +1") for c in near):
            raise SystemExit(f"the turn near log10 T {est:.5f} has no real +1 crossing recorded (crossings there: "
                             f"{[c['type'] for c in near]}); refusing to trace it as a fold")
        auto = min((i - 1, i), key=lambda j: abs(pts[j]["lambda"] - est))
    else:
        if not flips:
            raise SystemExit("no real -1 crossing along the tline: no flip to start from")
        a, b = flips[0]["between_log10_T"]
        cand = [j for j, r in enumerate(pts) if r["lambda"] in (a, b) and r["spectral_radius"] is not None]
        auto = min(cand, key=lambda j: abs(pts[j]["spectral_radius"] - 1))
    i = auto if index < 0 else index
    return kind, i, states[i].ravel(), float(pts[i]["lambda"])


def branch(args):
    """Dense continuation in log10 T, at fixed steering, of the periodic orbit reached at (log10 T, alpha):
    bifurcation_curves.trace_branch with all multipliers at every point (eigenvalues of the dense monodromy),
    residuals in the reference precision (float64 by default) and derivatives in --dtype. Writes branch.json in
    the format of tline.json (points, classified crossings), which curve can start from."""
    import bifurcation_curves as bc

    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                    args.device)
    ref = (SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window,
                    args.reference_dtype, args.device) if args.reference_dtype != "none" else None)
    v = elpm.load_vectors(args.vectors)[args.concept]
    top = ref or loop
    orb = SoftOrbit(top, args.alpha * v, 10.0 ** args.log_temperature)
    X = orb.iterate(orb.start, args.transient)
    returns = orb.return_distances(X, args.max_period)
    period = args.period or next((q for q, d in enumerate(returns, 1) if d < args.detect_tol), 0)
    if period == 0:
        raise SystemExit(f"no period up to {args.max_period} at tolerance {args.detect_tol} (smallest return "
                         f"distance {min(returns):.3g} at q = {int(np.argmin(returns)) + 1}); nothing to continue")
    fam = soft_family(loop, v, period, args.alpha_scale, args.chunk, reference=ref, exact_jvp=args.exact_jvp)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)

    progress = out / "progress.jsonl"                           # readable while the run is going
    progress.write_text("")

    def on_point(rec):
        line = json.dumps({k: v for k, v in rec.items() if k != "multipliers"} | {"leading": rec["multipliers"][:3]})
        print(line, flush=True)
        with progress.open("a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {line}\n")

    tic = time.perf_counter()
    free = 0 if args.free == "temperature" else 1
    sc = 1.0 if free == 0 else args.alpha_scale                # the free parameter as continued
    bounds = args.bounds or ([-1.5, 0.5] if free == 0 else [-0.2, 0.2])
    records, reason, states = bc.trace_branch(
        fam, fam.to_np(X), [args.log_temperature, args.alpha * args.alpha_scale], free=free,
        direction=args.direction, steps=args.steps, ds=args.ds, ds_min=args.ds_min, ds_max=args.ds_max, tol=args.tol,
        max_newton=args.max_newton, beyond=args.beyond * sc, bounds=(bounds[0] * sc, bounds[1] * sc),
        on_point=on_point, k=args.k, refine_k=args.refine_k)
    for r in records:                                          # 'lambda' in natural units (alpha unscaled)
        r["lambda"] /= sc
        if "turn_lambda_estimate" in r:
            r["turn_lambda_estimate"] /= sc
    crossings = bc.classify_crossings(records)
    np.save(out / "states.npy", np.array(states).reshape((len(states),) + tuple(X.shape[1:])))
    rec = {"args": vars(args), "free": args.free, "period": period, "return_distances_after_transient": returns,
           "stop": reason, "seconds": round(time.perf_counter() - tic, 1), "crossings": crossings,
           "points": records}
    (out / "branch.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"period": period, "stop": reason, "points": len(records), "crossings": crossings,
                      "seconds": rec["seconds"]}))


def curve(args):
    """Fold or flip curve of a period-p orbit in (log10 T, alpha), traced by two-parameter continuation
    (bifurcation_curves.trace_curve) from the boundary point a tline run found at fixed alpha."""
    import bifurcation_curves as bc

    src = DATA / "explore" / args.start
    tl = json.loads((src / ("branch.json" if (src / "branch.json").exists() else "tline.json")).read_text())
    states = np.load(src / "states.npy")
    kind, index, x0, lam0 = curve_start(tl, states, args.kind, args.start_index)
    ta = tl["args"]
    period = int(tl["period"])
    if tl.get("free", "temperature") == "temperature":          # located at fixed alpha
        log_t0, alpha0, fix = lam0, float(ta["alpha"]), 1
    else:                                                      # located at fixed log10 T
        log_t0, alpha0, fix = float(ta["log_temperature"]), lam0, 0
    loop = SoftLoop(ta["model"], ta["layer"], args.threads, ta["prompt"], ta["opener"], ta["window"], args.dtype,
                    args.device)
    ref = (SoftLoop(ta["model"], ta["layer"], args.threads, ta["prompt"], ta["opener"], ta["window"],
                    args.reference_dtype, args.device) if args.reference_dtype != "none" else None)
    v = elpm.load_vectors(ta["vectors"])[ta["concept"]]
    fam = soft_family(loop, v, period, args.alpha_scale, args.chunk, reference=ref, exact_jvp=args.exact_jvp)
    if args.gmres_tol is None:                                  # above the JVPs' floor (float32: 4.3e-6 at P8)
        args.gmres_tol = 1e-9 if args.exact_jvp else 1e-5
    sc = args.alpha_scale
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)

    def readable(rec):
        rec = dict(rec)
        rec["log10_temperature"], rec["alpha"] = rec["theta"][0], rec["theta"][1] / sc
        return rec

    progress = out / "progress.jsonl"                           # readable while the run is going
    progress.write_text("")

    def on_point(rec):
        r = readable(rec)
        keys = ("index", "log10_temperature", "alpha", "residual", "g", "g_dense", "newton", "jvps",
                "gmres_unconverged", "rejected", "critical_multiplier", "critical_multiplier_refined",
                "refine_residual", "other_spectral_radius", "parameter_speed", "cusp_candidate")
        line = json.dumps({k: r[k] for k in keys if k in r})
        print(line, flush=True)
        with progress.open("a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {line}\n")

    tic = time.perf_counter()
    bounds = ((args.log_temperature_bounds[0], args.log_temperature_bounds[1]),
              (args.alpha_bounds[0] * sc, args.alpha_bounds[1] * sc))
    records, reason, curve_states = bc.trace_curve(
        fam, x0, [log_t0, alpha0 * sc], kind, fix=fix, direction=args.direction, steps=args.steps, ds=args.ds,
        ds_min=args.ds_min, ds_max=args.ds_max, tol=args.tol, gtol=args.gtol, max_newton=args.max_newton,
        bounds=bounds, on_point=on_point, gmres_tol=args.gmres_tol, gmres_maxiter=args.gmres_maxiter,
        refine_k=args.refine_k)
    rec = {"args": vars(args), "kind": kind, "period": period, "start": {"tline_index": index,
           "log10_temperature": log_t0, "alpha": alpha0}, "stop": reason,
           "seconds": round(time.perf_counter() - tic, 1), "points": [readable(r) for r in records]}
    (out / "curve.json").write_text(json.dumps(rec, indent=1))
    np.save(out / "states.npy", np.array(curve_states))
    cusps = [(r["log10_temperature"], r["alpha"]) for r in rec["points"] if r["cusp_candidate"]]
    print(json.dumps({"kind": kind, "period": period, "points": len(records), "stop": reason,
                      "cusp_candidates": cusps, "seconds": rec["seconds"]}))


def scan(args):
    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                    args.device)
    torch = loop.torch
    v = elpm.load_vectors(args.vectors)[args.concept] if args.vectors else None
    log_t = np.linspace(*args.log_temperature, args.resolution[0])
    alphas = np.linspace(*args.alpha, args.resolution[1])
    cells = [(lt, a) for lt in log_t for a in alphas]
    B = len(cells)
    d = loop.E.shape[1]
    lyap = np.zeros(B)
    period = np.zeros(B, dtype=int)
    entropy = np.zeros(B)
    rng = np.random.default_rng(0)
    tic = time.perf_counter()
    chunk = args.chunk or B
    for start in range(0, B, chunk):
        idx = list(range(start, min(start + chunk, B)))
        n = len(idx)
        steer = np.zeros((n, d)) if v is None else np.array([cells[i][1] * v for i in idx])
        temps2 = np.tile(np.array([10.0 ** cells[i][0] for i in idx]), 2)
        XY = loop.prepare(np.concatenate([steer, steer]))     # rows 0..n-1: trajectory, n..2n-1: its twin
        for _ in range(args.transient):
            XY, _ = loop.step(XY, temps2)
        delta = torch.as_tensor(rng.normal(size=tuple(XY[:n].shape)), dtype=loop.state_dtype, device=loop.device)
        delta = delta / delta.flatten(1).norm(dim=1)[:, None, None] * args.eps
        XY = torch.cat([XY[:n], XY[:n] + delta])
        log_growth = np.zeros(n)
        history = []
        for k in range(args.samples):
            XY, p = loop.step(XY, temps2)
            X, Y = XY[:n], XY[n:]
            p = p[:n]
            diff = (Y - X)
            dist = diff.flatten(1).norm(dim=1).double().cpu().numpy()
            # a twin that collapses onto the trajectory (super-stable cycles, exact zero in float32) contributes a
            # floor of log(1e-30) per step; the floor also keeps the renormalization factor finite in float32
            dist = np.maximum(dist, args.eps * 1e-30)
            log_growth += np.log(dist / args.eps)
            Y = X + diff * torch.as_tensor(args.eps / dist, dtype=loop.state_dtype, device=loop.device)[:, None, None]
            XY = torch.cat([X, Y])
            if k >= args.samples - args.record:
                history.append(X[:, -1, :].double().cpu().numpy().copy())
        lyap[idx] = log_growth / args.samples
        hist = np.array(history)
        scale = np.maximum(1.0, np.linalg.norm(hist[-1], axis=1))
        per = np.zeros(n, dtype=int)
        for q in range(min(args.max_period, len(hist) - 1), 0, -1):
            close = np.linalg.norm(hist[-1 - q] - hist[-1], axis=1) < args.tol * scale
            per = np.where(close, q, per)
        period[idx] = per
        last_p = p.double().cpu().numpy()
        entropy[idx] = -(last_p * np.log(np.maximum(last_p, 1e-300))).sum(1)
        del p, XY, X, Y
        print(json.dumps({"done": idx[-1] + 1, "of": B, "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
    seconds = round(time.perf_counter() - tic, 1)
    shape = tuple(args.resolution)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scan.npz", log10_temperature=log_t, alpha=alphas, lyapunov=lyap.reshape(shape),
                        period=period.reshape(shape), entropy=entropy.reshape(shape))
    summary = {"args": vars(args), "seconds": seconds, "cells": B,
               "lyapunov_positive": int((lyap > args.chaos_level).sum()),
               "period_counts": {int(q): int((period == q).sum()) for q in np.unique(period)},
               "lyapunov_range": [float(lyap.min()), float(lyap.max())]}
    (out / "scan.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}))
    figure(log_t, alphas, lyap.reshape(shape), period.reshape(shape), entropy.reshape(shape), out / "scan.png", args)


def figure(log_t, alphas, lyap, period, entropy, path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    lim = max(float(np.nanmax(lyap)), 0.05)                 # clip: super-stable cells reach -700
    im0 = axes[0].pcolormesh(alphas, log_t, np.clip(lyap, -lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim,
                             shading="auto")
    axes[0].set_title("largest Lyapunov exponent per step (clipped)")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(alphas, log_t, np.where(period > 0, period, np.nan), cmap="tab20", vmin=0.5,
                             vmax=20.5, shading="auto")
    axes[1].set_title("period of the attractor (white: none up to the maximum)")
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].pcolormesh(alphas, log_t, entropy, cmap="viridis", shading="auto")
    axes[2].set_title("next-token entropy at the last step (nats)")
    fig.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel(f"steering along '{args.concept}'" if args.vectors else "alpha (no steering vector)")
        ax.set_ylabel("log10 temperature")
    fig.suptitle(f"soft-token decoding, {ela.MODELS[args.model]}, window {args.window}, layer {args.layer}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def curve_points(src):
    """(kind, points) of a curve run: curve.json when it is complete and newer than the run's progress file,
    otherwise the points written so far to progress.jsonl (kind from the sign of the critical multiplier)."""
    done, prog = src / "curve.json", src / "progress.jsonl"
    if done.exists() and (not prog.exists() or done.stat().st_mtime >= prog.stat().st_mtime):
        cv = json.loads(done.read_text())
        return cv["kind"], cv["points"]
    pts = [json.loads(line.split(" ", 1)[1]) for line in prog.read_text().splitlines() if line.strip()]
    if not pts:
        raise SystemExit(f"{src}: no curve points yet")
    return ("fold" if pts[0]["critical_multiplier"][0] > 0 else "flip"), pts


def render(args):
    """Period map of a finished scan with chaos marked: discrete colours for periods 1..max, black where the
    largest Lyapunov exponent exceeds --chaos-level, grey where no period was found below it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    src = DATA / "explore" / args.scan
    d = np.load(src / "scan.npz")
    lt, al, per, ly = d["log10_temperature"], d["alpha"], d["period"], d["lyapunov"]
    top = int(args.max_period)
    base = plt.get_cmap("tab20")(np.arange(20))
    colours = np.concatenate([np.delete(base, [14, 15], axis=0), plt.get_cmap("tab20b")(np.arange(20))])[:top]
    # (tab20's two greys are dropped: grey marks 'no period')
    img = np.full(per.shape + (4,), 0.0)
    for q in range(1, top + 1):
        img[per == q] = colours[(q - 1) % len(colours)]
    img[(per == 0) | (per > top)] = (0.75, 0.75, 0.75, 1.0)
    img[ly > args.chaos_level] = (0.0, 0.0, 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest",
              extent=(al[0], al[-1], lt[0], lt[-1]))
    handles = [matplotlib.patches.Patch(color=colours[(q - 1) % len(colours)], label=str(q))
               for q in range(1, top + 1) if np.any(per == q)]
    handles += [matplotlib.patches.Patch(color="black", label=f"Lyapunov > {args.chaos_level}"),
                matplotlib.patches.Patch(color=(0.75, 0.75, 0.75), label="no period, not chaotic")]
    styles = {"fold": dict(color="white", lw=1.6), "flip": dict(color="magenta", lw=1.6, ls="--")}
    for path in args.curves or []:                         # traced bifurcation curves over the brute-force map
        kind, pts = curve_points(DATA / "explore" / path)
        ax.plot([r["alpha"] for r in pts], [r["log10_temperature"] for r in pts], **styles[kind])
        cusps = [r for r in pts if r["cusp_candidate"]]
        ax.plot([r["alpha"] for r in cusps], [r["log10_temperature"] for r in cusps], "o", ms=6, mfc="none",
                mec="yellow")
    if args.curves:
        handles += [matplotlib.lines.Line2D([], [], label=f"{k} curve", **st) for k, st in styles.items()]
        ax.set_xlim(al[0], al[-1])
        ax.set_ylim(lt[0], lt[-1])
    if args.window:
        ax.set_xlim(*args.window[:2])
        ax.set_ylim(*args.window[2:])
    ax.legend(handles=handles, fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="period")
    ax.set_xlabel("steering alpha")
    ax.set_ylabel("log10 temperature")
    ax.set_title(f"soft-token decoding: {args.scan}")
    fig.tight_layout()
    fig.savefig(src / (args.name or "render.png"), dpi=150)
    counts = {int(q): int((per == q).sum()) for q in np.unique(per)}
    print(json.dumps({"cells": int(per.size), "chaotic": int((ly > args.chaos_level).sum()), "period_counts": counts}))


def readout(args):
    """Top tokens of every point of a saved periodic orbit: the soft token fed back at each step is a mixture,
    E^T p; list its leading components (token, probability) and entropy, step by step around the orbit. The state
    comes from an orbit run (orbit-state.npy) or a branch/curve run (states.npy at --index), with the cell's
    (log10 T, alpha) taken from that run unless given."""
    src = DATA / "explore" / args.run
    if (src / "orbit.json").exists():
        rec = json.loads((src / "orbit.json").read_text())
        X0 = np.load(src / "orbit-state.npy")
        log_t, alpha, period = rec["args"]["log_temperature"], rec["args"]["alpha"], rec["period"]
    else:
        name = next(n for n in ("branch.json", "tline.json", "curve.json") if (src / n).exists())
        rec = json.loads((src / name).read_text())
        X0 = np.load(src / "states.npy")[args.index]
        pt = rec["points"][args.index]
        if name == "curve.json":
            log_t, alpha = pt["log10_temperature"], pt["alpha"]
        elif rec.get("free", "temperature") == "temperature":
            log_t, alpha = pt["lambda"], rec["args"]["alpha"]
        else:
            log_t, alpha = rec["args"]["log_temperature"], pt["lambda"]
        period = rec["period"]
    log_t = args.log_temperature if args.log_temperature is not None else log_t
    alpha = args.alpha if args.alpha is not None else alpha
    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                    args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    orb = SoftOrbit(loop, alpha * v, 10.0 ** log_t)
    tok = loop.sm.tok
    X = loop.torch.as_tensor(np.asarray(X0).reshape((1,) + orb.shape), dtype=loop.state_dtype, device=loop.device)
    steps = []
    for i in range(period):
        p = orb.probabilities(X).double().cpu().numpy()
        top = np.argsort(-p)[: args.top]
        steps.append({"step": i, "entropy": float(-(p * np.log(np.maximum(p, 1e-300))).sum()),
                      "top": [[tok.decode([int(t)]), float(p[t])] for t in top]})
        X = orb.f(X)
    closure = float((X - loop.torch.as_tensor(np.asarray(X0).reshape((1,) + orb.shape), dtype=loop.state_dtype,
                                               device=loop.device)).norm())
    out = {"run": args.run, "index": args.index, "log10_temperature": log_t, "alpha": alpha, "period": period,
           "closure_after_one_period": closure, "steps": steps}
    print(json.dumps(out, ensure_ascii=False))


def walk(args):
    """Long trajectories of the soft map at several cells along alpha (one batch, fixed log10 T): after a
    transient, record every step's probabilities of a fixed token list, the entropy and the leading token.
    Because the map contracts all but one direction strongly, a scalar such as p(' it') two steps apart traces a
    one-dimensional return map; near a fold of a 2-cycle it shows the channel (tangency) and the reinjection.
    Writes walk.npz and walk.json; the figure plots p_(n+2) against p_n of the first token for every cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loop = SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                    args.device)
    torch, tok = loop.torch, loop.sm.tok
    v = elpm.load_vectors(args.vectors)[args.concept]
    alphas = np.array(args.alphas, dtype=float)
    ids = [tok.encode(t, add_special_tokens=False) for t in args.tokens]
    if any(len(i) != 1 for i in ids):
        raise SystemExit(f"each --tokens entry must be one token: {dict(zip(args.tokens, ids))}")
    ids = [i[0] for i in ids]
    X = loop.prepare(alphas[:, None] * v[None])
    temps = np.full(len(alphas), 10.0 ** args.log_temperature)
    probs, ent, lead = [], [], []
    with torch.no_grad():
        for i in range(args.transient + args.steps):
            X, p = loop.step(X, temps)
            if i >= args.transient:
                probs.append(p[:, ids].double().cpu().numpy())
                ent.append((-(p * torch.log(p.clamp_min(1e-300))).sum(-1)).double().cpu().numpy())
                lead.append(p.argmax(-1).cpu().numpy())
    probs, ent, lead = np.array(probs), np.array(ent), np.array(lead)          # (steps, B, tokens), (steps, B)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "walk.npz", alphas=alphas, tokens=np.array(args.tokens), token_ids=np.array(ids),
                        probabilities=probs, entropy=ent, leading=lead)
    summary = []
    for b, a in enumerate(alphas):
        top, counts = np.unique(lead[:, b], return_counts=True)
        order = np.argsort(-counts)[:6]
        summary.append({"alpha": float(a), "mean_entropy": float(ent[:, b].mean()),
                        "leading_tokens": [[tok.decode([int(top[j])]), int(counts[j])] for j in order]})
    (out / "walk.json").write_text(json.dumps({"args": vars(args), "cells": summary}, indent=1, ensure_ascii=False))
    cols = min(len(alphas), 4)
    rows_ = int(np.ceil(len(alphas) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.2 * cols, 3.2 * rows_), squeeze=False)
    for b, a in enumerate(alphas):
        ax = axes[b // cols][b % cols]
        q = probs[:, b, 0]
        ax.plot(q[:-2], q[2:], ".", ms=1)
        ax.plot([0, 1], [0, 1], "k-", lw=0.5)
        ax.set_title(f"alpha {a:.5f}", fontsize=8)
        ax.set_xlabel(f"p_n({args.tokens[0]!r})", fontsize=7)
        ax.set_ylabel(f"p_(n+2)({args.tokens[0]!r})", fontsize=7)
    fig.suptitle(f"soft map at log10 T {args.log_temperature}: two-step return map", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "walk.png", dpi=150)
    for row in summary:
        print(json.dumps(row, ensure_ascii=False))


def branch_plot(args):
    """Leading multipliers along branch or tline runs against log10 T: modulus of the largest (log scale) and the
    real parts of the real multipliers, with turns and classified crossings marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for path in args.runs:
        src = DATA / "explore" / path
        rec = json.loads((src / ("branch.json" if (src / "branch.json").exists() else "tline.json")).read_text())
        pts = rec["points"]
        lam = np.array([r["lambda"] for r in pts])
        rho = np.array([r["spectral_radius"] if r["spectral_radius"] is not None else np.nan for r in pts])
        lead = np.array([r["multipliers"][0][0] if r["multipliers"] else np.nan for r in pts])
        line, = ax1.semilogy(lam, rho, ".-", ms=3, label=f"{path} (period {rec['period']})")
        ax2.plot(lam, lead, ".-", ms=3, color=line.get_color())
        for r in pts:
            if "turn_lambda_estimate" in r:
                for ax in (ax1, ax2):
                    ax.axvline(r["turn_lambda_estimate"], color=line.get_color(), ls=":", lw=1)
        for c in rec.get("crossings", []):
            ax2.annotate(c["type"].split(" (")[0], (np.mean(c["between_log10_T"]), c["crossing_multiplier"][0]),
                         fontsize=7, color=line.get_color())
    ax1.axhline(1.0, color="k", lw=0.8)
    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.axhline(-1.0, color="k", lw=0.8)
    ax2.axhline(0.0, color="grey", lw=0.5)
    ax1.set_ylabel("spectral radius of the monodromy")
    ax2.set_ylabel("largest-modulus multiplier (real part)")
    ax2.set_xlabel("log10 temperature")
    ax1.legend(fontsize=7)
    fig.tight_layout()
    out = DATA / "explore" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(json.dumps({"figure": str(out)}))


def windows(args):
    """Candidate windows in a finished scan: connected regions (4-neighbour) of one period p >= 2, ranked by
    how much of their border touches chaotic cells (periodic islands embedded in chaos)."""
    from scipy import ndimage

    src = DATA / "explore" / args.scan
    d = np.load(src / "scan.npz")
    lt, al, per, ly = d["log10_temperature"], d["alpha"], d["period"], d["lyapunov"]
    chaos = ly > args.chaos_level
    found = []
    for q in np.unique(per):
        if q < args.min_period:
            continue
        lab, n = ndimage.label(per == q)
        for k in range(1, n + 1):
            region = lab == k
            area = int(region.sum())
            if area < args.min_area:
                continue
            border = ndimage.binary_dilation(region) & ~region
            touching = float(chaos[border].mean()) if border.any() else 0.0
            rows, cols = np.nonzero(region)
            i, j = int(np.round(rows.mean())), int(np.round(cols.mean()))
            if not region[i, j]:                                  # centroid outside a curved region
                i, j = int(rows[len(rows) // 2]), int(cols[len(cols) // 2])
            found.append({"period": int(q), "cells": area, "border_cells": int(border.sum()),
                          "border_chaotic_fraction": round(touching, 3),
                          "inside_point": {"log10_temperature": float(lt[i]), "alpha": float(al[j])},
                          "log10_temperature_range": [float(lt[rows.min()]), float(lt[rows.max()])],
                          "alpha_range": [float(al[cols.min()]), float(al[cols.max()])]})
    found.sort(key=lambda w: (-w["border_chaotic_fraction"], -w["cells"]))
    (src / "windows.json").write_text(json.dumps(found, indent=1))
    for w in found[: args.show]:
        print(json.dumps(w))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sc = sub.add_parser("scan")
    sc.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    sc.add_argument("--layer", type=int, default=12)
    sc.add_argument("--threads", type=int, default=8)
    sc.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    sc.add_argument("--device", default="cpu")
    sc.add_argument("--chunk", type=int, default=0, help="cells per batch (0: all at once)")
    sc.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    sc.add_argument("--opener", default=ela.OPENER)
    sc.add_argument("--window", type=int, default=1)
    sc.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    sc.add_argument("--concept", default="philosophy", choices=sorted(elpm.CONCEPTS))
    sc.add_argument("--log-temperature", type=float, nargs=2, default=[-1.0, 0.3])
    sc.add_argument("--alpha", type=float, nargs=2, default=[-0.12, 0.12])
    sc.add_argument("--resolution", type=int, nargs=2, default=[16, 16], help="rows (temperature) and columns (alpha)")
    sc.add_argument("--transient", type=int, default=300)
    sc.add_argument("--samples", type=int, default=300)
    sc.add_argument("--record", type=int, default=80)
    sc.add_argument("--max-period", type=int, default=32)
    sc.add_argument("--tol", type=float, default=1e-6)
    sc.add_argument("--eps", type=float, default=1e-6)
    sc.add_argument("--chaos-level", type=float, default=1e-3)
    sc.add_argument("--out", required=True)
    ob = sub.add_parser("orbit")
    ob.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    ob.add_argument("--layer", type=int, default=12)
    ob.add_argument("--threads", type=int, default=8)
    ob.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ob.add_argument("--device", default="cpu")
    ob.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    ob.add_argument("--opener", default=ela.OPENER)
    ob.add_argument("--window", type=int, default=1)
    ob.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ob.add_argument("--concept", default="philosophy", choices=sorted(elpm.CONCEPTS))
    ob.add_argument("--log-temperature", type=float, required=True)
    ob.add_argument("--alpha", type=float, required=True)
    ob.add_argument("--period", type=int, default=0, help="0: detect from the attractor reached")
    ob.add_argument("--transient", type=int, default=600)
    ob.add_argument("--max-period", type=int, default=32)
    ob.add_argument("--detect-tol", type=float, default=1e-6)
    ob.add_argument("--tol", type=float, default=1e-11)
    ob.add_argument("--max-newton", type=int, default=8)
    ob.add_argument("--jacobian", default="ad", choices=("ad", "fd"), help="forward-mode AD or central differences")
    ob.add_argument("--h-rel", type=float, default=1e-5)
    ob.add_argument("--chunk", type=int, default=512)
    ob.add_argument("--out", required=True)
    tl = sub.add_parser("tline")
    tl.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    tl.add_argument("--layer", type=int, default=12)
    tl.add_argument("--threads", type=int, default=8)
    tl.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    tl.add_argument("--device", default="cpu")
    tl.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    tl.add_argument("--opener", default=ela.OPENER)
    tl.add_argument("--window", type=int, default=1)
    tl.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    tl.add_argument("--concept", default="philosophy", choices=sorted(elpm.CONCEPTS))
    tl.add_argument("--log-temperature", type=float, required=True)
    tl.add_argument("--alpha", type=float, required=True)
    tl.add_argument("--period", type=int, default=0, help="0: detect from the attractor reached")
    tl.add_argument("--transient", type=int, default=600)
    tl.add_argument("--max-period", type=int, default=32)
    tl.add_argument("--detect-tol", type=float, default=1e-6)
    tl.add_argument("--direction", type=int, default=1, choices=(-1, 1))
    tl.add_argument("--steps", type=int, default=40)
    tl.add_argument("--ds", type=float, default=0.01)
    tl.add_argument("--ds-min", type=float, default=1e-6)
    tl.add_argument("--ds-max", type=float, default=0.05)
    tl.add_argument("--beyond", type=float, default=0.02, help="stop this far in log10 T past the first turn")
    tl.add_argument("--bounds", type=float, nargs=2, default=[-1.5, 0.5], help="log10 T range")
    tl.add_argument("--tol", type=float, default=1e-10)
    tl.add_argument("--max-newton", type=int, default=8)
    tl.add_argument("--gmres-tol", type=float, default=1e-9)
    tl.add_argument("--restart", type=int, default=60)
    tl.add_argument("--gmres-iter", type=int, default=3)
    tl.add_argument("--k", type=int, default=8, help="block size of the subspace iteration for multipliers")
    tl.add_argument("--subspace-iter", type=int, default=40)
    tl.add_argument("--chunk", type=int, default=512)
    tl.add_argument("--out", required=True)
    rd = sub.add_parser("render")
    rd.add_argument("--scan", required=True, help="explore-relative directory of a finished scan")
    rd.add_argument("--max-period", type=int, default=16)
    rd.add_argument("--chaos-level", type=float, default=0.005)
    rd.add_argument("--curves", nargs="*", help="explore-relative directories of curve runs to draw on the map")
    rd.add_argument("--name", default="", help="output file name inside the scan directory (default render.png)")
    rd.add_argument("--window", type=float, nargs=4, default=None, metavar=("ALPHA0", "ALPHA1", "LOGT0", "LOGT1"),
                    help="zoom to this part of the map")
    ro = sub.add_parser("readout")
    ro.add_argument("--run", required=True, help="explore-relative orbit, branch, tline or curve run directory")
    ro.add_argument("--index", type=int, default=0, help="point of a branch/tline/curve run")
    ro.add_argument("--log-temperature", type=float, default=None)
    ro.add_argument("--alpha", type=float, default=None)
    ro.add_argument("--top", type=int, default=5)
    ro.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    ro.add_argument("--layer", type=int, default=12)
    ro.add_argument("--threads", type=int, default=8)
    ro.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ro.add_argument("--device", default="cpu")
    ro.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    ro.add_argument("--opener", default=ela.OPENER)
    ro.add_argument("--window", type=int, default=1)
    ro.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ro.add_argument("--concept", default="philosophy", choices=sorted(elpm.CONCEPTS))
    bp = sub.add_parser("branch-plot")
    bp.add_argument("--runs", nargs="+", required=True, help="explore-relative branch or tline run directories")
    bp.add_argument("--out", required=True, help="explore-relative output png")
    wi = sub.add_parser("windows")
    wi.add_argument("--scan", required=True, help="explore-relative directory of a finished scan")
    wi.add_argument("--chaos-level", type=float, default=0.005)
    wi.add_argument("--min-period", type=int, default=2)
    wi.add_argument("--min-area", type=int, default=4)
    wi.add_argument("--show", type=int, default=15)
    pr = sub.add_parser("precision")
    pr.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    pr.add_argument("--layer", type=int, default=12)
    pr.add_argument("--threads", type=int, default=8)
    pr.add_argument("--device", default="cpu", help="device of the float32 map")
    pr.add_argument("--device64", default="cpu", help="device of the float64 reference map")
    pr.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    pr.add_argument("--opener", default=ela.OPENER)
    pr.add_argument("--window", type=int, default=1)
    pr.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    pr.add_argument("--concept", default="philosophy", choices=sorted(elpm.CONCEPTS))
    pr.add_argument("--cells", type=float, nargs=2, action="append", required=True, metavar=("LOG10_T", "ALPHA"))
    pr.add_argument("--transient", type=int, default=600)
    pr.add_argument("--max-period", type=int, default=32)
    pr.add_argument("--steps", type=int, default=32)
    pr.add_argument("--out", required=True)
    bb = sub.add_parser("branch")
    bb.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    bb.add_argument("--layer", type=int, default=12)
    bb.add_argument("--threads", type=int, default=8)
    bb.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    bb.add_argument("--reference-dtype", default="float64", choices=("none", "float64"))
    bb.add_argument("--device", default="cpu")
    bb.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    bb.add_argument("--opener", default=ela.OPENER)
    bb.add_argument("--window", type=int, default=1)
    bb.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    bb.add_argument("--concept", default="philosophy", choices=sorted(elpm.CONCEPTS))
    bb.add_argument("--log-temperature", type=float, required=True)
    bb.add_argument("--alpha", type=float, required=True)
    bb.add_argument("--period", type=int, default=0, help="0: detect from the attractor reached")
    bb.add_argument("--transient", type=int, default=600)
    bb.add_argument("--max-period", type=int, default=32)
    bb.add_argument("--detect-tol", type=float, default=1e-9)
    bb.add_argument("--alpha-scale", type=float, default=10.0)
    bb.add_argument("--direction", type=int, default=1, choices=(-1, 1))
    bb.add_argument("--steps", type=int, default=60)
    bb.add_argument("--ds", type=float, default=0.004)
    bb.add_argument("--ds-min", type=float, default=1e-7)
    bb.add_argument("--ds-max", type=float, default=0.01)
    bb.add_argument("--beyond", type=float, default=0.01, help="stop this far past the first turn (free parameter)")
    bb.add_argument("--free", default="temperature", choices=("temperature", "alpha"),
                    help="the continued parameter (the other stays fixed)")
    bb.add_argument("--bounds", type=float, nargs=2, default=None,
                    help="range of the free parameter (default log10 T -1.5..0.5, alpha -0.2..0.2)")
    bb.add_argument("--tol", type=float, default=1e-10)
    bb.add_argument("--max-newton", type=int, default=10)
    bb.add_argument("--k", type=int, default=12, help="multipliers recorded per point")
    bb.add_argument("--exact-jvp", action=argparse.BooleanOptionalAction, default=True,
                    help="refined multipliers from float64 JVPs of the reference")
    bb.add_argument("--refine-k", type=int, default=4,
                    help="Arnoldi steps for refined leading multipliers (0: dense float32 multipliers only)")
    bb.add_argument("--chunk", type=int, default=512)
    bb.add_argument("--out", required=True)
    wk = sub.add_parser("walk")
    wk.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    wk.add_argument("--layer", type=int, default=12)
    wk.add_argument("--threads", type=int, default=8)
    wk.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    wk.add_argument("--device", default="cpu")
    wk.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    wk.add_argument("--opener", default=ela.OPENER)
    wk.add_argument("--window", type=int, default=1)
    wk.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    wk.add_argument("--concept", default="philosophy", choices=sorted(elpm.CONCEPTS))
    wk.add_argument("--log-temperature", type=float, required=True)
    wk.add_argument("--alphas", type=float, nargs="+", required=True)
    wk.add_argument("--transient", type=int, default=300)
    wk.add_argument("--steps", type=int, default=4000)
    wk.add_argument("--tokens", nargs="+", default=[" it", " a", " there", " is", ".", " This"])
    wk.add_argument("--out", required=True)
    cv = sub.add_parser("curve")
    cv.add_argument("--start", required=True, help="explore-relative directory of a finished branch or tline run")
    cv.add_argument("--reference-dtype", default="float64", choices=("none", "float64"))
    cv.add_argument("--kind", default="auto", choices=("auto", "fold", "flip"))
    cv.add_argument("--start-index", type=int, default=-1, help="tline point to start from (-1: nearest the boundary)")
    cv.add_argument("--threads", type=int, default=8)
    cv.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    cv.add_argument("--device", default="cpu")
    cv.add_argument("--alpha-scale", type=float, default=10.0, help="alpha is continued as alpha * scale")
    cv.add_argument("--direction", type=int, default=1, choices=(-1, 1),
                    help="sign of the initial change of the parameter that varied along the start run")
    cv.add_argument("--steps", type=int, default=60)
    cv.add_argument("--ds", type=float, default=0.005)
    cv.add_argument("--ds-min", type=float, default=1e-6)
    cv.add_argument("--ds-max", type=float, default=0.02)
    cv.add_argument("--tol", type=float, default=1e-10)
    cv.add_argument("--gtol", type=float, default=1e-4, help="float32 monodromies make g accurate to about 1e-5")
    cv.add_argument("--max-newton", type=int, default=8)
    cv.add_argument("--exact-jvp", action=argparse.BooleanOptionalAction, default=True,
                    help="JVPs (g by GMRES, refined multipliers) from the float64 reference; float32 derivatives of "
                         "the P8 map are off by 3 %%")
    cv.add_argument("--refine-k", type=int, default=4, help="Arnoldi steps for the refined critical multiplier")
    cv.add_argument("--gmres-tol", type=float, default=None,
                    help="relative GMRES tolerance for g (default 1e-9 with exact JVPs, else 1e-5: float32 JVPs "
                         "reach about 4e-6 at the P8 fold)")
    cv.add_argument("--gmres-maxiter", type=int, default=1, help="GMRES restart cycles (30 products each) per g")
    cv.add_argument("--log-temperature-bounds", type=float, nargs=2, default=[-1.5, 0.5])
    cv.add_argument("--alpha-bounds", type=float, nargs=2, default=[-0.2, 0.2])
    cv.add_argument("--chunk", type=int, default=512)
    cv.add_argument("--out", required=True)
    args = parser.parse_args()
    {"scan": scan, "orbit": orbit, "tline": tline, "branch": branch, "curve": curve, "precision": precision,
     "walk": walk,
     "render": render, "windows": windows, "branch-plot": branch_plot, "readout": readout}[args.command](args)


if __name__ == "__main__":
    main()
