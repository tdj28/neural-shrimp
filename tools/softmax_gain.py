"""The soft-token map's derivative factorises through the next-token covariance (paper, Section "Why the loop can be
chaotic").

For F(x) = E^T softmax(z(x)/T) with tied embeddings (output matrix = input embedding E, z = E h(x), h the final
normalised hidden state), the Jacobian is DF(x) = (1/T) Sigma_p J_h(x), where Sigma_p = E^T (diag p - p p^T) E is the
covariance of the token embeddings under p = softmax(z/T). Sigma_p has rank below the number of tokens that carry
the probability, so steps with a sharp distribution are nearly rank-deficient.

identity  checks the factorisation against forward-mode derivatives of the full map at a state on the attractor of
          one cell: relative error of (1/T) Sigma_p J_h u against DF u for random directions u.
convergence  the identity check (same state, same random directions) together with central finite differences
          (F(x + h u) - F(x - h u)) / (2h) along unit directions u for steps h from 1e-1 to 1e-10: their relative
          error against the forward-mode derivative shows h^2 convergence down to the rounding floor, against which
          the factorisation's agreement is read (figure with --fig)
orbit     at q consecutive states on the attractor of a period-q cell: per step the entropy, the leading
          probabilities, the leading eigenvalues of Sigma_p and the leading singular values of the dense DF (from
          forward-mode derivatives); for the cycle the multipliers and singular values of the monodromy DF_q ... DF_1,
          the closure error, and the products prod sigma_1 and prod sigma_1 sigma_2, which bound |mu_1| and
          |mu_1 mu_2|; the dense step Jacobians and states are saved (orbit-jacobians.npz).

Usage: python tools/softmax_gain.py identity --vectors VEC --concept C --alpha A --log-temperature LT --out OUT
       python tools/softmax_gain.py convergence --vectors VEC --concept C --alpha A --log-temperature LT --out OUT
                                        [--directions 8] [--fig PNG]
       python tools/softmax_gain.py orbit --vectors VEC --concept C --alpha A --log-temperature LT --period Q --out OUT
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def attractor(loop, v, alpha, temp, transient, q):
    """States x_0..x_q on the attractor (x_q = F^q(x_0)) after `transient` steps from the fixed start."""
    X = loop.prepare((alpha * v)[None])
    for _ in range(transient):
        X, _ = loop.step(X, np.array([temp]))
    states = [X[0, -1].clone()]
    for _ in range(q):
        X, _ = loop.step(X, np.array([temp]))
        states.append(X[0, -1].clone())
    return states


class Maps:
    """F and h for a batch of B copies of one cell (prefix cache and steering repeated B times)."""

    def __init__(self, loop, v, alpha, temp, B):
        self.loop, self.temp, self.B = loop, temp, B
        loop.prepare(np.repeat((alpha * v)[None], B, axis=0))
        # prefix keys and values; restored by reference after every pass, so that tensors created inside a
        # forward-mode transform never stay in the cache
        self.prefix = [(layer.keys, layer.values) for layer in loop.cache.layers]

    def hidden(self, X):
        loop = self.loop
        loop.sm.steer = loop.steer_rows
        try:
            out = loop.model.model(inputs_embeds=X[:, None, :].to(loop.dtype), past_key_values=loop.cache,
                                   use_cache=True)
        finally:
            loop.sm.steer = None
            for layer, (k, v) in zip(loop.cache.layers, self.prefix):
                layer.keys, layer.values = k, v
        return out.last_hidden_state[:, -1].to(loop.state_dtype)

    def F(self, X):
        loop = self.loop
        z = loop.model.lm_head(self.hidden(X).to(loop.dtype)).to(loop.state_dtype)
        return loop.torch.softmax(z / self.temp, dim=-1) @ loop.E

    def p(self, x):
        loop = self.loop
        with loop.torch.no_grad():
            z = loop.model.lm_head(self.hidden(x.repeat(self.B, 1)).to(loop.dtype)).to(loop.state_dtype)[0]
        return loop.torch.softmax(z / self.temp, dim=-1)

    def jacobian(self, x):
        """Dense DF(x) (d x d) from batched forward-mode derivatives, B columns per pass."""
        torch = self.loop.torch
        d = x.shape[0]
        cols = []
        eye = torch.eye(d, dtype=x.dtype, device=x.device)
        for s in range(0, d, self.B):
            U = eye[s:s + self.B]
            n = U.shape[0]
            if n < self.B:
                U = torch.cat([U, torch.zeros(self.B - n, d, dtype=x.dtype, device=x.device)])
            _, dF = torch.func.jvp(self.F, (x.repeat(self.B, 1),), (U,))
            cols.append(dF[:n])
        return torch.cat(cols).T                                           # column j = DF e_j


def covariance(loop, p):
    """Sigma_p = E^T (diag p - p p^T) E, the covariance of the token embeddings under p."""
    E = loop.E
    m = p @ E
    return (E * p[:, None]).T @ E - loop.torch.outer(m, m)


def entropy(loop, p):
    torch = loop.torch
    return float(-(p * torch.log(torch.clamp(p, min=1e-300))).sum())


def main():
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("identity", "convergence", "orbit"))
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--log-temperature", type=float, required=True)
    ap.add_argument("--period", type=int, default=1)
    ap.add_argument("--transient", type=int, default=1000)
    ap.add_argument("--directions", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default=None, help="convergence mode: write the figure here")
    args = ap.parse_args()
    if args.dtype != "float64":
        raise SystemExit("float64 only: the check compares derivatives to rounding level")
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, args.dtype, args.device)
    torch = loop.torch
    cfg = loop.model.config
    if not getattr(cfg, "tie_word_embeddings", False) or \
            loop.model.lm_head.weight.data_ptr() != loop.model.get_input_embeddings().weight.data_ptr():
        raise SystemExit("the factorisation through Sigma_p needs tied embeddings (output matrix = input embedding)")
    if loop.hybrid:
        raise SystemExit("hybrid (linear-attention) models keep recurrent prefix states; not supported here")
    if getattr(loop.model.lm_head, "bias", None) is not None:
        raise SystemExit("output layer has a bias; the factorisation assumes z = E h")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    v = elpm.load_vectors(args.vectors)[args.concept]
    temp = 10.0 ** args.log_temperature
    q = args.period if args.mode == "orbit" else 1
    states = attractor(loop, v, args.alpha, temp, args.transient, q)
    result = {"args": vars(args), "temperature": temp}
    with sdpa_kernel(SDPBackend.MATH):
        if args.mode in ("identity", "convergence"):
            maps = Maps(loop, v, args.alpha, temp, 1)
            x = states[0]
            p = maps.p(x)
            S = covariance(loop, p)
            rng = np.random.default_rng(0)
            errs = []
            for _ in range(args.directions):
                u = torch.as_tensor(rng.normal(size=x.shape[0]), dtype=x.dtype, device=x.device)[None]
                _, dF = torch.func.jvp(maps.F, (x[None],), (u,))
                _, dh = torch.func.jvp(maps.hidden, (x[None],), (u,))
                pred = (S @ dh[0]) / temp
                errs.append(float((pred - dF[0]).norm() / dF[0].norm()))
            result.update({"entropy_nats": entropy(loop, p), "relative_errors": errs, "max_relative_error": max(errs)})
            if args.mode == "convergence":
                steps_h = np.logspace(-1, -10, 19)
                rng = np.random.default_rng(0)                  # the same directions as the identity check
                fd = []
                with torch.no_grad():
                    for _ in range(args.directions):
                        u = torch.as_tensor(rng.normal(size=x.shape[0]), dtype=x.dtype, device=x.device)
                        u = u / u.norm()
                        _, dF = torch.func.jvp(maps.F, (x[None],), (u[None],))
                        row = []
                        for h in steps_h:
                            diff = (maps.F((x + h * u)[None]) - maps.F((x - h * u)[None]))[0] / (2 * h)
                            row.append(float((diff - dF[0]).norm() / dF[0].norm()))
                        fd.append(row)
                result.update({"fd_steps": steps_h.tolist(), "fd_relative_errors": fd,
                               "state_norm": float(x.norm())})
                if args.fig:
                    from plot_derivative_check import plot_convergence
                    plot_convergence(result, args.fig)
        else:
            maps = Maps(loop, v, args.alpha, temp, args.batch)
            closure = float((states[q] - states[0]).norm() / states[0].norm())
            steps, M, jacobians = [], None, []
            for k in range(q):
                x = states[k]
                p = maps.p(x)
                S = covariance(loop, p)
                ev = torch.linalg.eigvalsh(S).flip(0)[:4]
                D = maps.jacobian(x)
                sv = torch.linalg.svdvals(D)[:4]
                top = torch.topk(p, 3)
                steps.append({"token": loop.sm.tok.decode([int(top.indices[0])]), "entropy_nats": entropy(loop, p),
                              "top3_p": [float(t) for t in top.values],
                              "sigma_p_eigenvalues": [float(e) for e in ev],
                              "df_singular_values": [float(s) for s in sv]})
                M = D if M is None else D @ M
                jacobians.append(D.cpu().numpy())
                print(json.dumps({"step": k, **steps[-1]}), flush=True)
            mu = torch.linalg.eigvals(M)
            mu = mu[torch.argsort(mu.abs(), descending=True)][:4]
            msv = torch.linalg.svdvals(M)[:4]
            s1 = np.array([s["df_singular_values"][0] for s in steps])
            s2 = np.array([s["df_singular_values"][1] for s in steps])
            result.update({"closure_relative": closure, "steps": steps,
                           "multipliers": [[float(m.real), float(m.imag)] for m in mu],
                           "multiplier_moduli": [float(m.abs()) for m in mu],
                           "monodromy_singular_values": [float(x) for x in msv],
                           "prod_sigma1": float(np.prod(s1)), "prod_sigma1_sigma2": float(np.prod(s1 * s2)),
                           "prod_sigma2_over_sigma1": float(np.prod(s2 / s1))})
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "orbit":
        np.savez_compressed(out / "orbit-jacobians.npz", jacobians=np.array(jacobians),
                            states=np.array([x.cpu().numpy() for x in states]))
    (out / f"softmax-gain-{args.mode}.json").write_text(json.dumps(result, indent=1))
    summary = {k: v for k, v in result.items() if k not in ("steps", "args")}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
