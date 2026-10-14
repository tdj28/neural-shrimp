import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import plot_zoom70 as pz


def band(root, k, lt, alpha, norm=2.0):
    d = root / f"b{k}"
    (d / "scan").mkdir(parents=True)
    (d / "vectors").mkdir()
    shape = (len(lt), len(alpha))
    np.savez(d / "scan" / "scan.npz", log10_temperature=lt, alpha=alpha, lyapunov=np.full(shape, 0.1 * k),
             period=np.zeros(shape, int), entropy=np.zeros(shape), family=np.full(shape, k))
    v = np.zeros(5)
    v[0] = norm
    np.savez(d / "vectors" / "vectors.npz", **{"hostility-vs-kindness": v})
    return d


def test_stitch_joins_contiguous_bands(tmp_path):
    alpha = np.linspace(0, 1, 3)
    lt = np.linspace(0.0, 0.7, 8)
    dirs = [band(tmp_path, k, lt[2 * k:2 * k + 2], alpha) for k in range(4)]
    m = pz.stitch(dirs)
    assert m["lyapunov"].shape == (8, 3)
    assert np.allclose(m["log10_temperature"], lt)
    assert m["family"][7, 0] == 3


@pytest.mark.parametrize("bad", ["gap", "alpha", "norm"])
def test_stitch_refuses_mismatched_bands(tmp_path, bad):
    alpha = np.linspace(0, 1, 3)
    lt = np.linspace(0.0, 0.7, 8)
    dirs = []
    for k in range(4):
        rows = lt[2 * k:2 * k + 2] + (0.05 if bad == "gap" and k == 3 else 0.0)
        a = alpha + (0.1 if bad == "alpha" and k == 2 else 0.0)
        dirs.append(band(tmp_path, k, rows, a, norm=2.5 if bad == "norm" and k == 1 else 2.0))
    with pytest.raises(SystemExit):
        pz.stitch(dirs)
