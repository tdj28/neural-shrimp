"""Float64 throughput of batched full-batch gradient descent in PyTorch (Phase 0).

Times the planned bounded real-data workload without any dataset: a tanh MLP
3072-32-10 (CIFAR-10 input size), MSE loss with weight decay, full-batch on N
synthetic inputs (seeded), with K learning-rate cells trained simultaneously
(stacked weights, batched matmuls), plus one matrix-free Hessian-vector product
per cell (torch.func), the unit of Arnoldi multiplier estimates. Prints and
writes a JSON receipt with device, versions and seconds per step.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch


def make(k, n, d_in, width, d_out, generator, device):
    x = torch.randn(n, d_in, generator=generator, dtype=torch.float64).to(device)
    y = torch.randn(n, d_out, generator=generator, dtype=torch.float64).to(device)
    w1 = (torch.randn(k, d_in, width, generator=generator, dtype=torch.float64) / d_in ** 0.5).to(device)
    w2 = (torch.randn(k, width, d_out, generator=generator, dtype=torch.float64) / width ** 0.5).to(device)
    return x, y, w1, w2


def loss_fn(w1, w2, x, y, ridge):
    hidden = torch.tanh(x @ w1)
    out = hidden @ w2
    return 0.5 * ((out - y) ** 2).sum(-1).mean() + 0.5 * ridge * ((w1 ** 2).sum() + (w2 ** 2).sum())


def timed(fn, repeats, device):
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[1, 32, 128, 256])
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(990001)
    rows = []
    for k in args.cells:
        x, y, w1, w2 = make(k, args.samples, 3072, args.width, 10, generator, device)
        rates = torch.logspace(-1, 0.5, k, dtype=torch.float64, device=device)
        ridge = 1e-3

        def batched_loss(a, b):
            return torch.vmap(lambda u, v: loss_fn(u, v, x, y, ridge))(a, b).sum()

        def step():
            nonlocal w1, w2
            g1, g2 = torch.func.grad(batched_loss, argnums=(0, 1))(w1, w2)
            w1 = w1 - rates[:, None, None] * g1
            w2 = w2 - rates[:, None, None] * g2

        v1, v2 = torch.randn_like(w1), torch.randn_like(w2)

        def hvp():
            grad = torch.func.grad(batched_loss, argnums=(0, 1))
            return torch.func.jvp(grad, (w1, w2), (v1, v2))[1]

        step_seconds = timed(step, args.repeats, device)
        hvp_seconds = timed(hvp, max(2, args.repeats // 4), device)
        rows.append({"cells": k, "seconds_per_step": step_seconds, "seconds_per_hvp": hvp_seconds,
                     "cell_steps_per_second": k / step_seconds})
        print(json.dumps(rows[-1]), flush=True)
    receipt = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
               "capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
               "torch": torch.__version__, "cuda": torch.version.cuda, "samples": args.samples,
               "width": args.width, "dtype": "float64", "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
