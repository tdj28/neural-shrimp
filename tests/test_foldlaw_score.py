import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import foldlaw_score as fs  # noqa: E402


def synthetic_line(C=0.11, delta=-0.6, q=4, tc=-0.0912, spacing=1e-5, laps=range(3, 9)):
    """Base period-4 cells below the fold; above it windows of period 4x at d = C / (x + delta)^2, each followed
    (away from the fold) by its period doubling and then chaos; a far, wider window of period 12 as a decoy."""
    ts = np.round(np.arange(tc - 0.0013, tc + 0.035, spacing), 9)
    cells = []
    edges = {q * x: tc + C / (x + delta) ** 2 for x in laps}
    for t in ts:
        period = 0
        if t < tc:
            period = q
        for p, e in edges.items():
            width = 0.15 * (e - tc)
            if e <= t < e + width:
                period = p
            elif e + width <= t < e + 1.4 * width:
                period = 2 * p
        if tc + 0.030 <= t < tc + 0.033:
            period = 12
        cells.append({"log10_temperature": float(t), "period": period, "class": "periodic" if period else "visible chaos"})
    return cells, edges


def test_recovers_prefactor_offset_and_fan():
    cells, edges = synthetic_line()
    nf = {"a": -7.0, "b": -21.0, "source": {"log10_temperature": -0.0912, "alpha": 0.0586}}
    res = fs.score(cells, nf, 4)
    assert res["side"] == 1.0 and res["fan"] == [32, 28, 24, 20, 16, 12]
    assert res["law_holds"] and res["law_holds_bounded"] and res["windows_meeting_law"] == 6
    full = res["fits"][-1]
    assert full["offset_physical"] and abs(full["bounded"]["C"] / full["C"] - 1) < 1e-3
    assert abs(full["C"] - 0.11) < 0.11 * 0.02 and abs(full["delta"] + 0.6) < 0.05 and full["spread"] < 0.03
    assert abs(full["passage_fraction"] - np.sqrt(0.11 * 147.0) / np.pi) < 0.02


def test_doublings_and_single_cells_are_not_windows():
    cells, _ = synthetic_line()
    periods = {w["period"] for w in fs.windows(cells, -0.0912, 4)[0]}
    assert not periods & {24 * 2, 28 * 2, 32 * 2, 64}
    cells[-1] = dict(cells[-1], period=36, **{"class": "periodic"})
    assert 36 not in {w["period"] for w in fs.windows(cells, -0.0912, 4)[0]}


def test_another_power_needs_an_unphysical_offset():
    tc = -0.0912                          # d = 0.004 / x instead of C / (x + delta)^2
    bent = []
    for x in range(3, 9):
        e = tc + 0.004 / x
        bent += [{"log10_temperature": e + i * 1e-5, "period": 4 * x, "class": "periodic"} for i in range(3)]
        bent += [{"log10_temperature": e + 3e-5, "period": 0, "class": "visible chaos"}]
    bent += [{"log10_temperature": tc - 1e-4, "period": 4, "class": "periodic"}]
    res = fs.score(bent, {"a": -7.0, "b": -21.0, "source": {"log10_temperature": tc, "alpha": 0.0586}}, 4)
    assert res["fan"] == [32, 28, 24, 20, 16, 12]
    full = res["fits"][-1]
    assert full["spread"] < 0.10 and full["delta"] > 1 and not full["offset_physical"]
    assert full["bounded"]["spread"] > 0.10 and not res["law_holds_bounded"]


def test_windows_outside_the_progression_do_not_end_the_fan():
    cells, _ = synthetic_line()
    tc = -0.0912
    e = tc + 0.11 / (5 - 0.6) ** 2 + 0.0004            # a period-21 window between the period-20 and period-16 windows
    cells = [dict(c, period=21, **{"class": "periodic"}) if e <= c["log10_temperature"] < e + 5e-5 else c for c in cells]
    res = fs.score(cells, {"a": -7.0, "b": -21.0, "source": {"log10_temperature": tc, "alpha": 0.0586}}, 4)
    assert 21 in {w["period"] for w in res["windows"]} and res["fan"] == [32, 28, 24, 20, 16, 12]
