"""The pair comparison of plot_dialogue counts entrained and chaos-made pairs against the diagonal agents."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import plot_dialogue as pdl  # noqa: E402


def test_pair_comparison_counts():
    # 3 x 3 grid; diagonal classes: chaotic, chaotic, periodic
    classes = [["visible chaos", "periodic", "visible chaos"],      # alpha_B index 0
               ["periodic", "hidden chaos", "periodic"],            # alpha_B index 1
               ["visible chaos", "visible chaos", "periodic"]]      # alpha_B index 2
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append({"alpha_a": 0.1 * j, "alpha_b": 0.1 * i, "class": classes[i][j], "period": 2})
    aa, ab, idx = pdl.grid(cells)
    counts, marks = pdl.pair_stats(cells, idx)
    # off-diagonal pairs with both agents chaotic: (0,1) and (1,0): both periodic -> 2 entrained
    assert counts["both_chaotic_agents"] == 2 and counts["entrained"] == 2
    # no pair has two periodic agents except diagonal-only agent 2 with itself
    assert counts["both_periodic_agents"] == 0 and counts["made_chaotic"] == 0
    assert counts["off_diagonal"] == 6 and counts["diagonal"] == {"chaotic": 2, "periodic": 1, "other": 0}
