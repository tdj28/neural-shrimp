import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prefix_pair_score as pps


def e(c1):
    return {"C1": c1, "M": 0.0}


def test_less_holds_with_three_of_four():
    t = {"qwen72b": {2: e(0), 3: e(1), 4: e(5), 5: e(2)}, "qwen14b": {2: e(14), 3: e(9), 4: e(5), 5: e(15)}}
    s = pps.score(t, "qwen72b", "qwen14b", "less")
    assert s["successes"] == 3 and s["holds"] and s["per_prefix"][4] is False


def test_more_and_missing_counts_against():
    t = {"olmo2-32b": {2: e(5), 3: e(6), 4: None, 5: e(1)}, "olmo2-7b": {2: e(1), 3: e(2), 4: e(0), 5: e(1)}}
    s = pps.score(t, "olmo2-32b", "olmo2-7b", "more")
    assert s["successes"] == 2 and not s["holds"] and s["missing"] == [4]


def test_stage1_counts_both_offsets():
    cell = lambda lt, l, period=0, lead=0: {"log10_temperature": lt, "lyapunov": l, "period": period, "lead_period": lead}
    r = pps.stage1([cell(0.1, [0.2, 0.3]), cell(0.2, [0.2, -0.1]), cell(0.3, [-0.5, -0.5], period=2)])
    assert r["C1"] == 1 and r["chaotic_log10T"] == [0.1] and abs(r["M"] - 0.2) < 1e-12
