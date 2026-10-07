"""Exploratory bounded CIFAR-10 pilot (float64 full-batch gradient descent, CPU).

Subcommands:
  data    download CIFAR-10 (torchvision) into DATA/datasets/cifar10, record the
          archive SHA-256, draw disjoint class-balanced subsets A and B of 512
          images (seed 28201) and seal B (index list and hash only)
  wd      on subset A, compare weight decays: sub-edge gradient descent from the
          default initialization, gradient norm, parameter drift and sharpness
  sweep   single learning rate sweep from the initialization on subset A:
          classify each run's tail (fixed point, period 2, other, divergent)
  orbit   certify a period-2 orbit (Newton-GMRES), compute two-step multipliers
          (Arnoldi), and continue it in the learning rate until a multiplier
          leaves the unit circle; type the crossing
  circle  just past a Neimark-Sacker crossing: long run from the orbit plus a
          small kick, rotation from principal-plane projections, loss spectrum

Only subset A is ever loaded as pixels. Outputs under DATA/explore/cifar/.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp import cifar  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))
OUT = DATA / "explore" / "cifar"
SUBSET_SEED = 28201
INIT_SEED = 28202


def subset_paths():
    return OUT / "subsets" / "subset-A.json", OUT / "subsets" / "subset-B-sealed.json"


def cmd_data(args):
    import torchvision
    root = DATA / "datasets" / "cifar10"
    root.mkdir(parents=True, exist_ok=True)
    torchvision.datasets.CIFAR10(root=str(root), train=True, download=True)
    archive = root / "cifar-10-python.tar.gz"
    labels = cifar.load_cifar_train(root)[1]
    a, b = cifar.balanced_subsets(labels, SUBSET_SEED, size=512)
    path_a, path_b = subset_paths()
    hash_a = cifar.seal_indices(a, path_a, "development subset A (CIFAR-10 train indices)")
    hash_b = cifar.seal_indices(b, path_b, "held-out subset B: sealed, not loaded or used in exploration")
    manifest = {"archive": str(archive), "archive_sha256": cifar.sha256_file(archive),
                "torchvision": torchvision.__version__, "subset_seed": SUBSET_SEED, "size": 512,
                "subset_A_sha256_int64": hash_a, "subset_B_sha256_int64": hash_b,
                "class_counts_A": np.bincount(labels[a], minlength=10).tolist()}
    (OUT / "subsets" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest))


def subset_a_problem(lam, width=32):
    path_a, _ = subset_paths()
    record = json.loads(path_a.read_text())
    indices = np.asarray(record["indices"])
    if cifar.index_hash(indices) != record["sha256_int64"]:
        raise ValueError("subset A index hash mismatch")
    pixels, labels = cifar.load_cifar_train(DATA / "datasets" / "cifar10")
    x = cifar.standardized(pixels[indices])
    y = cifar.one_hot(labels[indices])
    model = cifar.MLP(width=width)
    return cifar.Problem(model, x, y, lam), model


def sharpness(problem, theta, iterations=60, seed=0):
    """Largest Hessian eigenvalue by power iteration on Hessian-vector products."""
    v = torch.as_tensor(np.random.default_rng(seed).normal(size=theta.numel()))
    v = v / v.norm()
    value = 0.0
    for _ in range(iterations):
        w = problem.hvp(theta, v)
        value = float(v @ w)
        v = w / w.norm()
    return value


def cmd_wd(args):
    torch.set_num_threads(args.threads)
    rows = []
    for lam in args.lams:
        problem, model = subset_a_problem(lam)
        theta = torch.as_tensor(model.init(INIT_SEED))
        s0 = sharpness(problem, theta)
        eta = args.eta_fraction * 2.0 / s0
        started = time.perf_counter()
        trace = []
        for t in range(args.steps):
            value, gradient = problem.loss_and_grad(theta)
            theta = theta - eta * gradient
            if t % args.every == 0 or t == args.steps - 1:
                trace.append({"step": t, "loss": float(value), "grad_norm": float(gradient.norm()),
                              "theta_norm": float(theta.norm())})
        seconds = (time.perf_counter() - started) / args.steps
        s_end = sharpness(problem, theta)
        rows.append({"lam": lam, "sharpness_init": s0, "eta": eta, "sharpness_end": s_end,
                     "edge_eta_end": 2.0 / s_end, "seconds_per_step": seconds, "trace": trace})
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "trace"}), flush=True)
        print("   last:", json.dumps(trace[-1]), flush=True)
    (OUT / "wd").mkdir(parents=True, exist_ok=True)
    (OUT / "wd" / "wd-check.json").write_text(json.dumps(rows, indent=1))


def sweep_one(task):
    lam, eta, steps, tail, threads, output_ratio = task
    torch.set_num_threads(threads)
    problem, model = subset_a_problem(lam)
    rate = rates(model, eta, output_ratio)
    theta = torch.as_tensor(model.init(INIT_SEED))
    losses = np.empty(steps)
    states = []
    started = time.perf_counter()
    for t in range(steps):
        value, gradient = problem.loss_and_grad(theta)
        losses[t] = float(value)
        theta = theta - rate * gradient
        if not torch.isfinite(theta).all() or float(theta.norm()) > 1e8:
            return {"eta": eta, "status": "divergent", "diverged_at": t, "losses_tail": losses[max(0, t - 64):t + 1].tolist()}
        if t >= steps - tail:
            states.append(theta.numpy().copy())
    seconds = (time.perf_counter() - started) / steps
    s = np.asarray(states)
    norm = float(np.linalg.norm(s[-1]))
    d1 = float(np.linalg.norm(s[-1] - s[-2]) / norm)
    d2 = float(np.linalg.norm(s[-1] - s[-3]) / norm)
    d2_series = np.linalg.norm(s[2:] - s[:-2], axis=1) / norm
    if d1 < 1e-9:
        status = "fixed"
    elif d2 < 1e-9:
        status = "period2"
    else:
        status = "other"
    out = {"eta": eta, "status": status, "d1": d1, "d2": d2, "d2_tail_max": float(d2_series.max()),
           "d2_tail_first": float(d2_series[0]), "loss_last": float(losses[-1]),
           "loss_tail_std": float(losses[-tail:].std()), "theta_norm": norm, "seconds_per_step": seconds}
    if status in ("period2", "other"):
        tag = "" if output_ratio == 1.0 else f"-ratio{output_ratio:g}"
        path = OUT / "sweep" / f"state-lam{lam:g}{tag}-eta{eta:.6g}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, s[-1])
        np.save(path.with_name(path.stem + "-losses.npy"), losses[-min(steps, 20000):])
        out["state"] = str(path)
    return out


def cmd_sweep(args):
    etas = np.geomspace(args.eta[0], args.eta[1], args.count)
    tasks = [(args.lam, float(e), args.steps, args.tail, args.threads, args.output_ratio) for e in etas]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(sweep_one, tasks))
    (OUT / "sweep").mkdir(parents=True, exist_ok=True)
    tag = "" if args.output_ratio == 1.0 else f"-ratio{args.output_ratio:g}"
    name = OUT / "sweep" / f"sweep-lam{args.lam:g}{tag}-{args.eta[0]:.4g}-{args.eta[1]:.4g}.json"
    name.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
    for r in rows:
        print(json.dumps({k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()
                          if k not in ("state", "losses_tail")}), flush=True)


def rates(model, eta, output_ratio=1.0):
    """Scalar rate, or per-parameter rates with the output layer at output_ratio x eta."""
    if output_ratio == 1.0:
        return eta
    hidden = model.sizes[0] + model.sizes[1]
    vector = torch.full((model.n_parameters,), float(eta), dtype=torch.float64)
    vector[hidden:] = float(eta) * float(output_ratio)
    return vector


def certify(problem, theta, eta, tol=1e-11):
    theta, history = cifar.newton_period2(problem, theta, eta, tol=tol)
    t1 = problem.update(theta, eta)
    separation = float((t1 - theta).norm() / theta.norm())
    return theta, history, separation


def classify(multipliers):
    """Dominant multiplier over both Arnoldi searches and its kind."""
    candidates = [complex(*v) for mode in multipliers["modes"].values() for v in mode]
    top = max(candidates, key=abs)
    kind = "complex" if abs(top.imag) > 1e-9 else ("plus" if top.real > 0 else "minus")
    return top, kind


def cmd_orbit(args):
    torch.set_num_threads(args.threads)
    problem, model = subset_a_problem(args.lam)
    folder = OUT / "orbit"
    folder.mkdir(parents=True, exist_ok=True)
    theta = torch.as_tensor(np.load(args.state))
    log_eta = np.log10(args.eta)
    rows, previous = [], []
    started = time.perf_counter()
    stable = None
    for _ in range(args.max_steps):
        eta = rates(model, 10.0 ** log_eta, args.output_ratio)
        if len(previous) == 2:  # secant predictor in log eta
            theta = previous[-1] + (previous[-1] - previous[-2])
        theta, history, separation = certify(problem, theta, eta)
        residual = history[-1]["relative_residual"]
        multipliers = cifar.two_step_multipliers(problem, theta.numpy(), eta, k=args.k)
        top, kind = classify(multipliers)
        row = {"log10_eta": log_eta, "relative_residual": residual, "newton_iterations": len(history),
               "separation": separation, "top": [top.real, top.imag], "top_modulus": abs(top),
               "top_kind": kind, "dominant_complex": multipliers["dominant_complex"],
               "dominant_complex_modulus": multipliers["dominant_complex_modulus"],
               "theta_over_2pi": multipliers["dominant_complex_theta_over_2pi"],
               "seconds": time.perf_counter() - started}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if residual > 1e-8 or separation < 1e-7:
            row["stop"] = "corrector failed or orbit collapsed to a fixed point"
            break
        if abs(top) >= 1.0:
            row["stop"] = "multiplier outside the unit circle"
            break
        stable = (log_eta, theta.clone(), multipliers)
        np.save(folder / f"orbit-lam{args.lam:g}-last-stable.npy", theta.numpy())
        previous = (previous + [theta.clone()])[-2:]
        log_eta += args.step
    crossing = None
    if stable is not None and rows and rows[-1].get("stop") == "multiplier outside the unit circle":
        lo, hi = stable[0], rows[-1]["log10_eta"]
        theta_lo = stable[1]
        while hi - lo > args.bisect_tol:
            mid = 0.5 * (lo + hi)
            eta_mid = rates(model, 10.0 ** mid, args.output_ratio)
            theta_mid, history, separation = certify(problem, theta_lo, eta_mid)
            multipliers = cifar.two_step_multipliers(problem, theta_mid.numpy(), eta_mid, k=args.k)
            top, kind = classify(multipliers)
            if abs(top) < 1.0:
                lo, theta_lo, stable = mid, theta_mid, (mid, theta_mid, multipliers)
            else:
                hi = mid
        top, kind = classify(stable[2])
        crossing = {"log10_eta_last_stable": lo, "log10_eta_first_unstable": hi, "multiplier": [top.real, top.imag],
                    "modulus": abs(top), "type": {"complex": "neimark_sacker", "minus": "flip", "plus": "fold"}[kind],
                    "theta_over_2pi": abs(np.angle(top)) / (2 * np.pi)}
        np.save(folder / f"orbit-lam{args.lam:g}-at-crossing.npy", theta_lo.numpy())
        print(json.dumps({"crossing": crossing}), flush=True)
    (folder / f"orbit-lam{args.lam:g}-from{args.eta:.6g}.json").write_text(
        json.dumps({"args": vars(args), "rows": rows, "crossing": crossing}, indent=1))


def cmd_circle(args):
    """Long runs just past a crossing: rotation, loss spectrum and largest exponent."""
    from scipy.sparse.linalg import LinearOperator, eigs
    torch.set_num_threads(args.threads)
    problem, model = subset_a_problem(args.lam)
    folder = OUT / "circle"
    folder.mkdir(parents=True, exist_ok=True)
    base = torch.as_tensor(np.load(args.state))
    n = base.numel()
    eta0 = rates(model, 10.0 ** args.log10_eta_crossing, args.output_ratio)
    t1 = problem.update(base, eta0)

    def matvec(v):
        out = problem.jvp_two_step(base, cifar.as_tensor(np.real(v)), eta0, theta_next=t1).numpy()
        if np.iscomplexobj(v):
            out = out + 1j * problem.jvp_two_step(base, cifar.as_tensor(np.imag(v)), eta0, theta_next=t1).numpy()
        return out

    operator = LinearOperator((n, n), dtype=np.float64, matvec=matvec)
    values, vectors = eigs(operator, k=4, which="LI", tol=1e-10)
    index = int(np.argmax(np.abs(values) * (values.imag > 0)))
    q = vectors[:, index]
    plane = np.linalg.qr(np.stack([q.real, q.imag], 1))[0]
    rng = np.random.default_rng(28203)
    random_dirs = np.linalg.qr(rng.normal(size=(n, 64)))[0]
    rows = []
    for offset in args.offsets:
        eta = rates(model, 10.0 ** (args.log10_eta_crossing + offset), args.output_ratio)
        theta = base + args.kick * torch.as_tensor(plane[:, 0])
        losses = np.empty(args.steps)
        proj_plane, proj_random = [], []
        tangent = torch.as_tensor(rng.normal(size=n))
        tangent /= tangent.norm()
        log_growth, growth_steps = 0.0, 0
        started = time.perf_counter()
        finite = True
        for t in range(args.steps):
            value, gradient = problem.loss_and_grad(theta)
            losses[t] = float(value)
            if t >= args.steps - args.ftle_steps:
                tangent = problem.jvp_update(theta, tangent, eta)
                norm = float(tangent.norm())
                log_growth += np.log(norm)
                growth_steps += 1
                tangent /= norm
            theta = theta - eta * gradient
            if not torch.isfinite(theta).all():
                finite = False
                break
            if t >= args.transient and (t - args.transient) % 2 == 0:
                x = theta.numpy()
                proj_plane.append(plane.T @ x)
                proj_random.append(random_dirs.T @ x)
        seconds = (time.perf_counter() - started) / args.steps
        row = {"offset_log10": offset, "log10_eta": args.log10_eta_crossing + offset,
               "seconds_per_step": seconds, "finite": finite}
        if finite and len(proj_plane) > 16:
            pp, pr = np.asarray(proj_plane), np.asarray(proj_random)
            rho_plane, share_plane = cifar.rotation_from_samples(pp)
            rho_random, share_random = cifar.rotation_from_samples(pr)
            centred = pp - pp.mean(0)
            even = losses[args.transient::2]
            spectrum = np.abs(np.fft.rfft(even - even.mean())) ** 2
            freqs = np.fft.rfftfreq(len(even))
            row.update({"rotation_plane": rho_plane, "plane_share": share_plane,
                        "rotation_random64": rho_random, "random_share": share_random,
                        "amplitude_in_plane": float(np.sqrt((centred ** 2).sum(1)).mean()),
                        "loss_spectrum_peak": float(freqs[1 + int(np.argmax(spectrum[1:]))]),
                        "loss_tail_std": float(losses[args.transient:].std()),
                        "ftle": log_growth / max(growth_steps, 1)})
            np.save(folder / f"proj-offset{offset:g}.npy", pp)
            np.save(folder / f"losses-offset{offset:g}.npy", losses)
        rows.append(row)
        print(json.dumps(row), flush=True)
    birth = abs(np.angle(values[index])) / (2 * np.pi)
    (folder / "circle.json").write_text(json.dumps({"args": vars(args), "birth_multiplier":
                                                    [values[index].real, values[index].imag],
                                                    "birth_theta_over_2pi": birth, "rows": rows}, indent=1))
    print(json.dumps({"birth_theta_over_2pi": birth}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data")
    wd = sub.add_parser("wd")
    wd.add_argument("--lams", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2])
    wd.add_argument("--steps", type=int, default=20000)
    wd.add_argument("--every", type=int, default=1000)
    wd.add_argument("--eta-fraction", type=float, default=0.5, help="eta as a fraction of 2/sharpness at init")
    wd.add_argument("--threads", type=int, default=16)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--lam", type=float, required=True)
    sweep.add_argument("--eta", type=float, nargs=2, required=True)
    sweep.add_argument("--count", type=int, default=16)
    sweep.add_argument("--steps", type=int, default=40000)
    sweep.add_argument("--tail", type=int, default=64)
    sweep.add_argument("--workers", type=int, default=4)
    sweep.add_argument("--threads", type=int, default=4)
    sweep.add_argument("--output-ratio", type=float, default=1.0, help="output-layer rate / hidden-layer rate")
    orbit = sub.add_parser("orbit")
    orbit.add_argument("--output-ratio", type=float, default=1.0)
    orbit.add_argument("--lam", type=float, required=True)
    orbit.add_argument("--eta", type=float, required=True)
    orbit.add_argument("--state", required=True)
    orbit.add_argument("--step", type=float, default=0.01, help="continuation step in log10 eta")
    orbit.add_argument("--max-steps", type=int, default=60)
    orbit.add_argument("--bisect-tol", type=float, default=1e-5)
    orbit.add_argument("--k", type=int, default=6)
    orbit.add_argument("--threads", type=int, default=16)
    circle = sub.add_parser("circle")
    circle.add_argument("--output-ratio", type=float, default=1.0)
    circle.add_argument("--lam", type=float, required=True)
    circle.add_argument("--state", required=True, help="orbit point at the crossing")
    circle.add_argument("--log10-eta-crossing", type=float, required=True)
    circle.add_argument("--offsets", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    circle.add_argument("--kick", type=float, default=1e-4)
    circle.add_argument("--steps", type=int, default=60000)
    circle.add_argument("--transient", type=int, default=30000)
    circle.add_argument("--ftle-steps", type=int, default=4000)
    circle.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"data": cmd_data, "wd": cmd_wd, "sweep": cmd_sweep, "orbit": cmd_orbit, "circle": cmd_circle}[args.command](args)


if __name__ == "__main__":
    main()
