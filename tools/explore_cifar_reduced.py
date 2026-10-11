"""Exploratory CIFAR-10 (subset A): period-2 orbit and spectrum in exact reduced coordinates.

The first-layer weights split into an input-span part (512 x 32 coordinates)
and a complement that only decays (see ``neural_shrimp.cifar_reduced``). In the
16,746 active coordinates the two-step Jacobian is dense but small enough for
direct linear algebra, which removes the near-neutral complement directions
that stalled earlier Krylov solves.

Subcommands:
  profile  Exactness of the reduction on a saved state (one step in full and in
           reduced coordinates) and timings of the dense blocks.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
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
    args = parser.parse_args()
    {"profile": profile, "orbit": orbit}[args.command](args)


if __name__ == "__main__":
    main()
