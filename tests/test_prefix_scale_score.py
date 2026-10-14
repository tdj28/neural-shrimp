import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prefix_scale_score as pss


def e(c1, m, c2=0):
    return {"C1": c1, "C2": c2, "M": m}


def table(pairs):
    t = {"qwen7b": {}, "qwen14b": {}}
    for i, (a, b) in pairs.items():
        t["qwen7b"][i], t["qwen14b"][i] = a, b
    return t


def test_holds_with_six_of_eight():
    pairs = {i: (e(5, 0.3), e(1, 0.1)) for i in range(2, 8)}
    pairs.update({8: (e(0, -0.1), e(0, -0.2)), 9: (e(1, 0.2), e(2, 0.25))})
    s = pss.score(table(pairs))
    assert s["PS1"] == "holds" and s["PS1_successes"] == 6
    assert s["PS2"] == "holds" and s["PS2_successes"] == 7


def test_violation_fails_ps1():
    pairs = {i: (e(5, 0.3), e(1, 0.1)) for i in range(2, 9)}
    pairs[9] = (e(1, 0.1), e(4, 0.3))          # 14B exceeds 7B by more than 2
    assert pss.score(table(pairs))["PS1"] == "fails"


def test_ties_are_not_successes_and_missing_is_undetermined():
    pairs = {i: (e(0, -0.2), e(0, -0.3)) for i in range(2, 10)}
    s = pss.score(table(pairs))
    assert s["PS1"] == "fails" and s["PS1_successes"] == 0
    pairs = {i: (e(3, 0.3), e(1, 0.1)) for i in range(2, 8)}
    pairs.update({8: (None, e(0, 0.0)), 9: (e(1, 0.1), None)})
    s = pss.score(table(pairs))
    assert s["held_out_missing"] == 2 and s["PS1"] == "undetermined" and s["PS2"] == "holds"


def test_endpoints_counts_both_stages():
    cell = lambda lt, l, period=0, lead=0: {"log10_temperature": lt, "lyapunov": l, "period": period, "lead_period": lead}
    coarse = [cell(0.0, [0.1, 0.12]), cell(0.1, [0.1, -0.01]), cell(0.2, [-0.3, -0.3], period=4)]
    fine = [cell(0.05, [0.2, 0.15], lead=7), cell(0.06, [0.0005, 0.002])]
    r = pss.endpoints(coarse, fine)
    assert r["C1"] == 1 and r["C2"] == 1 and abs(r["M"] - 0.15) < 1e-12
    assert pss.endpoints(None, fine) is None
