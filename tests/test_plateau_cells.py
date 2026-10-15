import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import plateau_cells as pc  # noqa: E402


def toy():
    lt, al = np.linspace(0.1, 0.2, 4), np.linspace(0.0, 0.3, 5)
    L = np.full((4, 5), -0.5)
    P = np.ones((4, 5), int)
    L[0, :2], P[0, :2] = 0.3, 0           # two chaotic cells
    L[1, 4], P[1, 4] = -0.1, 0            # one unresolved cell
    return L, P, lt, al


def test_classes_and_shortfall():
    L, P, lt, al = toy()
    cells, short = pc.draw(L, P, lt, al, (3, 4, 2), seed=1)
    labels = [c["map_label"] for c in cells]
    assert labels.count("chaotic") == 2 and short["chaotic"] == 1       # too few: all taken, shortfall recorded
    assert labels.count("periodic") == 4 and short["periodic"] == 0
    assert labels.count("unresolved") == 1 and short["unresolved"] == 1
    for c in cells:                                                     # every cell carries its own map label
        r, k = int(np.argmin(abs(lt - c["log10T"]))), int(np.argmin(abs(al - c["alpha"])))
        chaotic = L[r, k] > 1e-3 and P[r, k] <= 0
        assert c["map_label"] == ("chaotic" if chaotic else "periodic" if P[r, k] > 0 else "unresolved")


def test_seeded_draw_is_reproducible():
    L, P, lt, al = toy()
    a, _ = pc.draw(L, P, lt, al, (1, 3, 1), seed=7)
    b, _ = pc.draw(L, P, lt, al, (1, 3, 1), seed=7)
    assert a == b
