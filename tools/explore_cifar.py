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
    lam, eta, steps, tail, threads = task
    torch.set_num_threads(threads)
    problem, model = subset_a_problem(lam)
    theta = torch.as_tensor(model.init(INIT_SEED))
    losses = np.empty(steps)
    states = []
    started = time.perf_counter()
    for t in range(steps):
        value, gradient = problem.loss_and_grad(theta)
        losses[t] = float(value)
        theta = theta - eta * gradient
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
        path = OUT / "sweep" / f"state-lam{lam:g}-eta{eta:.6g}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, s[-1])
        np.save(path.with_name(path.stem + "-losses.npy"), losses[-min(steps, 20000):])
        out["state"] = str(path)
    return out


def cmd_sweep(args):
    etas = np.geomspace(args.eta[0], args.eta[1], args.count)
    tasks = [(args.lam, float(e), args.steps, args.tail, args.threads) for e in etas]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(sweep_one, tasks))
    (OUT / "sweep").mkdir(parents=True, exist_ok=True)
    name = OUT / "sweep" / f"sweep-lam{args.lam:g}-{args.eta[0]:.4g}-{args.eta[1]:.4g}.json"
    name.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
    for r in rows:
        print(json.dumps({k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()
                          if k not in ("state", "losses_tail")}), flush=True)


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
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"data": cmd_data, "wd": cmd_wd, "sweep": cmd_sweep}[args.command](args)


if __name__ == "__main__":
    main()
