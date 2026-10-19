"""plot_p6_shrimp.map_image: a crop of a map scaled over the whole scan keeps the full map's colors cell by cell."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import plot_p6_shrimp as p6  # noqa: E402


def fake_scan(root, name="s"):
    rng = np.random.default_rng(0)
    al, lt = np.linspace(0.0, 1.0, 12), np.linspace(-1.0, 0.0, 9)
    ly = rng.normal(0.0, 0.3, (9, 12))
    per = np.where(ly > 0.1, 0, 6).astype(float)              # chaotic cells have no period
    per[0, 0], ly[0, 0] = 5.0, 0.4                            # a detected period wins over a positive exponent
    folder = root / "explore" / name
    folder.mkdir(parents=True)
    np.savez(folder / "scan.npz", alpha=al, log10_temperature=lt, lyapunov=ly, period=per)
    return al, lt, ly, per


def test_crop_with_norm_full_keeps_the_full_maps_colors(monkeypatch, tmp_path):
    monkeypatch.setattr(p6, "DATA", tmp_path)
    al, lt, ly, per = fake_scan(tmp_path)
    full, ext, _, chaos, norm = p6.map_image("s")
    window = (al[3], al[8], lt[2], lt[6])
    crop, cext, cly, cchaos, cnorm = p6.map_image("s", window, norm_full=True)
    assert crop.shape == (5, 6, 4) and np.allclose(crop, full[2:7, 3:9])
    assert cnorm.vmax == norm.vmax and not chaos[0, 0]
    own = p6.map_image("s", window)[0]                        # scaled over the window: generally different colors
    assert own.shape == crop.shape
