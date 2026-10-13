import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import p1_select_dose as sd  # noqa: E402


def test_choose_closest_with_lower_dose_on_ties():
    assert sd.choose({0.04: 0.1, 0.06: 0.45, 0.08: 0.8}) == 0.06
    assert sd.choose({0.04: 0.4, 0.06: 0.6}) == 0.04            # tie at distance 0.1: the lower dose
    assert sd.choose({0.04: 0.0, 0.06: 0.0}, target=0.5) == 0.04
