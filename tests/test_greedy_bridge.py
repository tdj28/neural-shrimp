"""tools/greedy_bridge.py: cycles and basins of a functional graph; final repetition loops."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import greedy_bridge as gb  # noqa: E402


def test_cycles_and_basins():
    nxt = np.array([1, 2, 0, 0, 3, 5, 5, 6])          # 0->1->2->0 cycle; 3,4 flow in; 5 fixed point; 6,7 flow to 5
    cycles, owner = gb.cycles_of(nxt)
    canon = sorted(gb.canonical(c) for c in cycles)
    assert canon == [(0, 1, 2), (5,)]
    basin = np.bincount(owner)
    sizes = sorted(basin.tolist())
    assert sizes == [3, 5]                              # {5,6,7} and {0,1,2,3,4}


def test_final_loop():
    assert gb.final_loop([9, 8, 1, 2, 1, 2, 1, 2]) == [1, 2]
    assert gb.final_loop([1, 2, 3, 4, 5]) is None
    assert gb.canonical([2, 1]) == (1, 2)
