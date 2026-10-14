import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import st72_summary as s


def cell(first, final, rates=(0.1, 0.1, 0.1)):
    return {"log10_temperature": 0.2, "alpha": 0.0, "first_above_dmax": list(first), "final_distance": list(final),
            "growth_rates": list(rates), "entropy_mean": 5.0}


def test_two_of_three_twins_must_pass():
    assert s.score_cell(cell([50, 60, None], [0.5, 0.4, 1e-6]))["grows"]
    assert not s.score_cell(cell([50, None, None], [0.5, 1e-6, 1e-6]))["grows"]


def test_bursts_are_flagged_but_still_count():
    r = s.score_cell(cell([70, 72, 75], [2e-5, 3e-5, 0.2]))
    assert r["grows"] and r["bursts"] == 2


def test_summary_counts_and_max_rate():
    out = s.summarize([cell([50, 60, 70], [0.1] * 3, (0.8, 0.7, 0.75)), cell([None] * 3, [1e-9] * 3, (0.0, 0.0, 0.0))])
    assert out["growing"] == 1 and out["n"] == 2 and out["max_rate"] == 0.8
