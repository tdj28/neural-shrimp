import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _tool(name):
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    return importlib.import_module(name)


def test_rotation_of_points_on_an_embedded_circle():
    tool = _tool("explore_realdata")
    rho = 0.2137
    k = np.arange(600)
    plane = np.column_stack((np.cos(2 * np.pi * rho * k), 0.7 * np.sin(2 * np.pi * rho * k)))
    embed = np.random.default_rng(0).normal(size=(2, 32))
    points = plane @ embed + 3.0
    estimate, forward, share = tool.rotation(points)
    assert abs(estimate - rho) < 1e-3
    assert forward == 1.0 and share > 0.999


def test_polished_minimum_reaches_a_critical_point():
    tool = _tool("explore_realdata")
    from neural_shrimp.gdmaps import TwoLayerNet

    rng = np.random.default_rng(4)
    net = TwoLayerNet(d=3, width=2, activation="tanh", loss_scale=1.0, ridge=1e-2)
    x, y = rng.normal(size=(20, 3)), rng.normal(size=20)
    theta, grad_norm = tool.polished_minimum(net, x, y, 0.3 * rng.normal(size=net.n_parameters))
    assert grad_norm < 1e-10
    assert np.linalg.eigvalsh(net.hessian(theta, x, y))[0] > 0


def test_spectrum_peaks_find_sideband_frequencies():
    tool = _tool("explore_realdata")
    t = np.arange(4096)
    signal = np.cos(np.pi * t) + 0.3 * np.cos(2 * np.pi * 0.11 * t) + 0.1 * np.cos(2 * np.pi * 0.39 * t)
    peaks = tool.spectrum_peaks(signal, count=3)
    freqs = sorted(round(f, 2) for f, _ in peaks)
    assert freqs == [0.11, 0.39, 0.5]
    assert tool.spectrum_peaks(np.ones(64)) == []
