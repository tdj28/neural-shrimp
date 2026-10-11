"""Real-data networks for exploratory generality checks (UCI digits, 8 x 8).

The digits file is the copy bundled with scikit-learn (1,797 images, 64
pixels in 0..16, class label last), stored read-only under the external data
root and verified by SHA-256 on load. A task draws a fixed training subset,
standardizes its pixels on that subset (constant pixels keep scale one) and
uses +-1 targets. Networks are :class:`neural_shrimp.gdmaps.TwoLayerNet`
instances in standard parameterization (``in_scale = out_scale = 1``, biases)
with PyTorch-default uniform initialization ``U(-1/sqrt(fan_in),
1/sqrt(fan_in))`` for weights and biases, so a single global learning rate is
the equal-rates diagonal of the per-layer rate plane.
"""

import gzip
import hashlib
import os
from pathlib import Path

import numpy as np

from .gdmaps import TwoLayerNet

DIGITS_SHA256 = "09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22"
TASKS = ("halves", "parity")
RESERVED_SEEDS = range(29000, 30000)


def data_root():
    return Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))


def load_digits_array(path=None):
    """Return (pixels (1797, 64) float64, labels (1797,) int) after a hash check."""
    path = Path(path) if path is not None else data_root() / "external" / "uci-digits" / "digits.csv.gz"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DIGITS_SHA256:
        raise ValueError(f"digits file hash mismatch: {digest}")
    table = np.loadtxt(gzip.open(path, "rt"), delimiter=",")
    return table[:, :-1].astype(np.float64), table[:, -1].astype(int)


def _check_seed(seed):
    if int(seed) in RESERVED_SEEDS:
        raise ValueError("seeds 29000-29999 are reserved for confirmatory tests")


def digits_task(seed, n, task="halves", pixels=None, labels=None):
    """Fixed training subset with standardized pixels and +-1 targets."""
    _check_seed(seed)
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if pixels is None or labels is None:
        pixels, labels = load_digits_array()
    if not 1 <= n <= len(labels):
        raise ValueError("n must be between 1 and the dataset size")
    stream = np.random.SeedSequence(int(seed)).spawn(2)[0]
    index = np.sort(np.random.default_rng(stream).choice(len(labels), size=n, replace=False))
    x = pixels[index].astype(np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    x = (x - mean) / std
    chosen = labels[index]
    y = np.where(chosen >= 5, 1.0, -1.0) if task == "halves" else np.where(chosen % 2 == 1, 1.0, -1.0)
    return x, y


def torch_default_init(net, seed):
    """PyTorch nn.Linear default initialization for a TwoLayerNet."""
    _check_seed(seed)
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(2)[1])
    hidden_bound, output_bound = 1.0 / np.sqrt(net.d), 1.0 / np.sqrt(net.width)
    parts = [rng.uniform(-hidden_bound, hidden_bound, size=net.d * net.width)]
    if net.bias:
        parts.append(rng.uniform(-hidden_bound, hidden_bound, size=net.width))
    parts.append(rng.uniform(-output_bound, output_bound, size=net.width))
    if net.bias:
        parts.append(rng.uniform(-output_bound, output_bound, size=1))
    return np.concatenate(parts)


def make_digits(seed, width=8, n=1024, task="halves", ridge=0.0, loss_scale=1.0, pixels=None, labels=None):
    """Standard-parameterization tanh network on a digits subset.

    Returns ``(net, x, y, theta0)``. ``loss_scale=1`` is the mean squared error
    of PyTorch's ``MSELoss``; ``ridge`` is weight decay on all parameters.
    """
    x, y = digits_task(seed, n, task=task, pixels=pixels, labels=labels)
    net = TwoLayerNet(d=x.shape[1], width=width, activation="tanh", in_scale=1.0, out_scale=1.0,
                      bias=True, loss_scale=loss_scale, ridge=ridge)
    return net, x, y, torch_default_init(net, seed)
