"""Exploratory CIFAR-10 (subset A): period-2 orbit and spectrum in exact reduced coordinates.

The first-layer weights split into an input-span part (512 x 32 coordinates)
and a complement that only decays (see ``neural_shrimp.cifar_reduced``). In the
16,746 active coordinates the two-step Jacobian is dense but small enough for
direct linear algebra, which removes the near-neutral complement directions
that stalled earlier Krylov solves.

Subcommands:
  profile  Exactness of the reduction on a saved state (one step in full and in
           reduced coordinates) and timings of the dense blocks.
  check-mode  At a saved reduced orbit point: dense two-step Jacobian, the
           eigenvalue nearest a shift (shift-invert Arnoldi), verification of
           M v = mu v with matrix-free Jacobian-vector products, and the
           eigenvector's composition by parameter block (first-layer rows along
           the data span, its near-null column, biases, second layer).
  orbit    Dense Newton for the period-2 orbit at each rate of a list, starting
           from a saved state and continuing from each solution; the complex
           multipliers nearest the unit circle at every rate (full spectrum or
           shift-invert Arnoldi); optional bisection of a unit-circle crossing.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


def setup(lam, threads):
    import torch

    torch.set_num_threads(threads)
    import explore_cifar as ec
    from neural_shrimp.cifar_reduced import reduce_problem

    problem, model = ec.subset_a_problem(lam)
    reduced, q = reduce_problem(problem)
    return problem, model, reduced, q


def profile(args):
    import torch
    from neural_shrimp.cifar_reduced import dense_hessian, join_state, split_state

    problem, model, reduced, q = setup(args.lam, args.threads)
    state = np.load(args.state)
    theta_r, complement = split_state(state, model, q)
    tic = time.perf_counter()
    full_next = problem.update(torch.as_tensor(state), args.eta).numpy()
    t_full = time.perf_counter() - tic
    tic = time.perf_counter()
    reduced_next = reduced.update(torch.as_tensor(theta_r), args.eta).numpy()
    t_red = time.perf_counter() - tic
    joined = join_state(reduced_next, (1 - args.eta * args.lam) * complement, model, q)
    record = {"full_parameters": int(state.size), "reduced_parameters": int(theta_r.size),
              "basis_columns": int(q.shape[1]), "complement_norm": float(np.linalg.norm(complement)),
              "step_mismatch_max": float(np.max(np.abs(joined - full_next))),
              "step_norm": float(np.linalg.norm(full_next - state)),
              "seconds_full_step": t_full, "seconds_reduced_step": t_red}
    tic = time.perf_counter()
    h = dense_hessian(reduced, theta_r, chunk=args.chunk)
    record["seconds_dense_hessian"] = time.perf_counter() - tic
    tic = time.perf_counter()
    m = h @ h
    record["seconds_dense_matmul"] = time.perf_counter() - tic
    from scipy.linalg import lu_factor

    tic = time.perf_counter()
    lu_factor(m, overwrite_a=True, check_finite=False)
    record["seconds_dense_lu"] = time.perf_counter() - tic
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


def critical_pairs(values, count=6):
    """Complex multipliers (Im > 0) sorted by distance of their modulus from 1."""
    values = np.asarray(values)
    upper = values[values.imag > 1e-9]
    order = np.argsort(np.abs(np.abs(upper) - 1.0))
    return [{"re": float(upper[i].real), "im": float(upper[i].imag), "modulus": float(abs(upper[i])),
             "turns": float(np.angle(upper[i]) / (2 * np.pi))} for i in order[:count]]


def spectrum(m, previous, mode):
    """Eigenvalues of m: all of them, or the few nearest a shift at the previous critical pair."""
    if mode == "full" or previous is None:
        tic = time.perf_counter()
        values = np.linalg.eigvals(m)
        return values, {"mode": "full", "seconds": time.perf_counter() - tic}
    from scipy.sparse.linalg import eigs

    sigma = complex(previous["re"], previous["im"])
    tic = time.perf_counter()
    values = eigs(m.astype(np.complex128), k=8, sigma=sigma, return_eigenvectors=False)
    return values, {"mode": "shift-invert", "sigma": [sigma.real, sigma.imag],
                    "seconds": time.perf_counter() - tic}


def orbit(args):
    import torch
    from neural_shrimp.cifar_reduced import newton_period2, split_state

    problem, model, reduced, q = setup(args.lam, args.threads)
    state = np.load(args.state)
    theta_r, complement = split_state(state, model, q)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    previous = None
    if args.pre_steps:
        from neural_shrimp.cifar_reduced import descend

        tic = time.perf_counter()
        before = theta_r.copy()
        theta_r = descend(reduced, theta_r, args.etas[0], args.pre_steps)
        t = torch.as_tensor(theta_r)
        pre_residual = float((reduced.update(reduced.update(t, args.etas[0]), args.etas[0]) - t).norm() / t.norm())
        print(json.dumps({"pre_steps": args.pre_steps, "seconds": round(time.perf_counter() - tic, 1),
                          "moved": float(np.linalg.norm(theta_r - before)), "relative_residual": pre_residual}))
    for k, eta in enumerate(args.etas):
        tic = time.perf_counter()
        steps_taken = []
        theta_r, m, history = newton_period2(reduced, theta_r, eta, tol=args.tol, max_iter=args.max_newton,
                                             chunk=args.chunk, emit=steps_taken.append)
        seconds_newton = time.perf_counter() - tic
        t = torch.as_tensor(theta_r)
        separation = float((reduced.update(t, eta) - t).norm() / t.norm())
        mode = "full" if (k == 0 or args.spectrum == "full") else "shift"
        values, info = spectrum(m, previous, mode)
        pairs = critical_pairs(values)
        real_top = values[np.abs(values.imag) <= 1e-9]
        row = {"eta": eta, "newton": history, "newton_events": steps_taken, "seconds_newton": seconds_newton,
               "relative_separation_F": separation, "critical_pairs": pairs, "spectrum": info,
               "largest_real": float(real_top.real.max()) if len(real_top) else None,
               "smallest_real": float(real_top.real.min()) if len(real_top) else None,
               "spectral_radius": float(np.max(np.abs(values)))}
        if mode == "full":
            np.save(out / f"eigenvalues-eta{eta:.8f}.npy", values)
        np.save(out / f"orbit-eta{eta:.8f}.npy", theta_r)
        rows.append(row)
        previous = pairs[0] if pairs else previous
        print(json.dumps({"eta": eta, "residual": history[-1], "separation": separation,
                          "critical": pairs[:2], "spectral_radius": row["spectral_radius"],
                          "seconds": round(seconds_newton + info["seconds"], 1)}))
        (out / "orbit.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
        if history[-1] > args.tol * 100 or separation < 1e-8:
            print(json.dumps({"stop": "Newton did not converge or the orbit collapsed to a fixed point"}))
            break


def check_mode(args):
    import torch
    from scipy.sparse.linalg import eigs
    from neural_shrimp.cifar_reduced import two_step

    problem, model, reduced, q = setup(args.lam, args.threads)
    theta = np.load(args.orbit)
    t2, t1, m = two_step(reduced, theta, args.eta, chunk=args.chunk)
    vals, vecs = eigs(m, k=args.k, sigma=args.sigma, which="LM")
    order = np.argsort(-np.abs(vals))
    rows = []
    r, h = q.shape[1], model.width
    t = torch.as_tensor(theta)
    for i in order:
        mu, v = vals[i], vecs[:, i]
        v = v / np.linalg.norm(v)
        mv = reduced.jvp_two_step(t, torch.as_tensor(np.real(v)), args.eta).numpy() + 1j * \
            reduced.jvp_two_step(t, torch.as_tensor(np.imag(v)), args.eta).numpy()
        resid = float(np.linalg.norm(mv - mu * v))
        w = np.abs(v) ** 2
        b = w[: r * h].reshape(r, h)
        rows.append({"mu": [float(mu.real), float(mu.imag)], "matrix_free_residual": resid,
                     "share_first_layer_span": float(b[:-1].sum()), "share_first_layer_last_column": float(b[-1].sum()),
                     "share_hidden_bias": float(w[r * h: r * h + h].sum()),
                     "share_second_layer": float(w[r * h + h:].sum()),
                     "top_hidden_units": np.argsort(-b.sum(axis=0))[:4].tolist()})
        print(json.dumps({k: (round(x, 7) if isinstance(x, float) else x) for k, x in rows[-1].items()}))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "check-mode.json").write_text(json.dumps({"args": vars(args), "modes": rows}, indent=1))


def minimum(args):
    """Critical point of the regularized loss near a state: descent, dense Newton on the gradient, spectrum."""
    import torch
    from scipy.linalg import lu_factor, lu_solve
    from neural_shrimp.cifar_reduced import dense_hessian, descend, split_state

    problem, model, reduced, q = setup(args.lam, args.threads)
    if args.reduced_state:
        theta = np.load(args.reduced_state)
    else:
        theta, _ = split_state(np.load(args.state), model, q)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = {"args": vars(args), "descent": None, "newton": []}

    def loss_grad(th):
        t = torch.as_tensor(th)
        return float(reduced.loss(t)), reduced.grad(t).numpy()

    if args.descent_steps:
        tic = time.perf_counter()
        theta = descend(reduced, theta, args.descent_eta, args.descent_steps)
        loss, g = loss_grad(theta)
        record["descent"] = {"steps": args.descent_steps, "eta": args.descent_eta, "loss": loss,
                             "grad_norm": float(np.linalg.norm(g)), "seconds": round(time.perf_counter() - tic, 1)}
        print(json.dumps(record["descent"]))
    for it in range(args.max_newton):
        loss, g = loss_grad(theta)
        gn = float(np.linalg.norm(g))
        if gn < args.grad_tol:
            break
        tic = time.perf_counter()
        h = dense_hessian(reduced, theta, args.chunk)
        step = lu_solve(lu_factor(h, overwrite_a=True, check_finite=False), -g)
        del h
        alpha = 1.0
        for _ in range(12):                                # backtracking on the gradient norm
            trial = theta + alpha * step
            trial_loss, gt = loss_grad(trial)
            if np.all(np.isfinite(gt)) and np.linalg.norm(gt) < gn:
                break
            alpha *= 0.5
        theta = trial
        row = {"iteration": it, "grad_norm_before": gn, "step_fraction": alpha,
               "grad_norm_after": float(np.linalg.norm(gt)), "loss": trial_loss,
               "seconds": round(time.perf_counter() - tic, 1)}
        record["newton"].append(row)
        print(json.dumps(row))
        (out / "minimum.json").write_text(json.dumps(record, indent=1))
    tic = time.perf_counter()
    loss, g = loss_grad(theta)
    h = dense_hessian(reduced, theta, args.chunk)
    values, vectors = np.linalg.eigh(h)
    del h
    negative = values < 0
    record["final"] = {"loss": loss, "data_loss": loss - 0.5 * args.lam * float(theta @ theta),
                       "grad_norm": float(np.linalg.norm(g)), "norm": float(np.linalg.norm(theta)),
                       "negative_count": int(negative.sum()), "smallest": values[:8].tolist(),
                       "largest": values[-8:].tolist(), "eta_flip": float(2.0 / values[-1]),
                       "spectrum_seconds": round(time.perf_counter() - tic, 1)}
    np.save(out / "critical-point.npy", theta)
    np.save(out / "hessian-eigenvalues.npy", values)
    np.save(out / "top-eigenvectors.npy", vectors[:, -4:])
    if negative.any():
        np.save(out / "negative-eigenvectors.npy", vectors[:, :min(16, int(negative.sum()))])
    print(json.dumps(record["final"]))
    (out / "minimum.json").write_text(json.dumps(record, indent=1))


def escape_plateau(args):
    """Leave a plateau: kick along the most negative Hessian eigenvector (both signs), descend, keep the lower loss."""
    import torch
    from neural_shrimp.cifar_reduced import dense_hessian, descend

    problem, model, reduced, q = setup(args.lam, args.threads)
    start = Path(args.start)
    theta = np.load(start / "critical-point.npy")
    direction = np.load(start / "negative-eigenvectors.npy")[:, 0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = {"args": vars(args), "rounds": []}

    def loss_grad(th):
        t = torch.as_tensor(th)
        return float(reduced.loss(t)), reduced.grad(t).numpy()

    for rnd in range(args.rounds):
        tic = time.perf_counter()
        trials = []
        for sign in (1, -1):
            trial = descend(reduced, theta + sign * args.kick * direction, args.descent_eta, args.descent_steps)
            loss, g = loss_grad(trial)
            trials.append({"sign": sign, "loss": loss, "grad_norm": float(np.linalg.norm(g)), "state": trial})
        best = min(trials, key=lambda t: t["loss"])
        theta = best["state"]
        h = dense_hessian(reduced, theta, args.chunk)
        values, vectors = np.linalg.eigh(h)
        del h
        negative = values < 0
        row = {"round": rnd, "chosen_sign": best["sign"], "losses": [t["loss"] for t in trials],
               "grad_norms": [t["grad_norm"] for t in trials],
               "data_loss": best["loss"] - 0.5 * args.lam * float(theta @ theta), "norm": float(np.linalg.norm(theta)),
               "negative_count": int(negative.sum()), "smallest": values[:6].tolist(), "largest": float(values[-1]),
               "eta_flip": float(2.0 / values[-1]), "seconds": round(time.perf_counter() - tic, 1)}
        record["rounds"].append(row)
        print(json.dumps(row))
        np.save(out / f"state-round{rnd}.npy", theta)
        np.save(out / f"eigenvalues-round{rnd}.npy", values)
        (out / "escape.json").write_text(json.dumps(record, indent=1))
        if not negative.any():
            np.save(out / "top-eigenvectors.npy", vectors[:, -4:])
            break
        direction = vectors[:, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cm = sub.add_parser("check-mode")
    cm.add_argument("--lam", type=float, default=1e-2)
    cm.add_argument("--threads", type=int, default=12)
    cm.add_argument("--chunk", type=int, default=256)
    cm.add_argument("--orbit", required=True)
    cm.add_argument("--eta", type=float, required=True)
    cm.add_argument("--sigma", type=float, default=1.0002)
    cm.add_argument("--k", type=int, default=6)
    cm.add_argument("--out", required=True)
    for name in ("profile", "orbit"):
        p = sub.add_parser(name)
        p.add_argument("--lam", type=float, default=1e-2)
        p.add_argument("--state", default=str(DATA / "explore/cifar/sweep/state-lam0.01-eta0.00827753.npy"))
        p.add_argument("--threads", type=int, default=12)
        p.add_argument("--chunk", type=int, default=256)
        p.add_argument("--out", required=True)
        if name == "profile":
            p.add_argument("--eta", type=float, default=0.00827753)
        else:
            p.add_argument("--etas", type=float, nargs="+", required=True)
            p.add_argument("--tol", type=float, default=1e-12)
            p.add_argument("--max-newton", type=int, default=10)
            p.add_argument("--spectrum", choices=("full", "shift"), default="shift")
            p.add_argument("--pre-steps", type=int, default=0,
                           help="gradient-descent steps in reduced coordinates before the first Newton")
    mn = sub.add_parser("minimum")
    mn.add_argument("--lam", type=float, required=True)
    mn.add_argument("--state", default=str(DATA / "explore/cifar/sweep/state-lam0.01-eta0.00938931.npy"))
    mn.add_argument("--reduced-state", default=None, help="start from a saved reduced state instead")
    mn.add_argument("--descent-eta", type=float, default=0.006)
    mn.add_argument("--descent-steps", type=int, default=400000)
    mn.add_argument("--max-newton", type=int, default=12)
    mn.add_argument("--grad-tol", type=float, default=1e-11)
    mn.add_argument("--threads", type=int, default=12)
    mn.add_argument("--chunk", type=int, default=256)
    mn.add_argument("--out", required=True)
    ep = sub.add_parser("escape-plateau")
    ep.add_argument("--lam", type=float, required=True)
    ep.add_argument("--start", required=True, help="folder of a minimum run (critical-point, negative eigenvectors)")
    ep.add_argument("--kick", type=float, default=0.05)
    ep.add_argument("--descent-eta", type=float, default=0.006)
    ep.add_argument("--descent-steps", type=int, default=400000)
    ep.add_argument("--rounds", type=int, default=6)
    ep.add_argument("--threads", type=int, default=12)
    ep.add_argument("--chunk", type=int, default=256)
    ep.add_argument("--out", required=True)
    args = parser.parse_args()
    {"profile": profile, "orbit": orbit, "check-mode": check_mode, "minimum": minimum,
     "escape-plateau": escape_plateau}[args.command](args)


if __name__ == "__main__":
    main()
