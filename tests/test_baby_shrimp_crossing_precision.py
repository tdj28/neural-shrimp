import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from baby_shrimp_crossing_precision import select_phase


def test_every_phase_required_and_all_failed_is_explicit():
    rows = [{"phase": p, "primitive": {"closed": True, "no_lower_period": True},
             "precision": {"passed": False}} for p in (0, 6, 12)]
    with pytest.raises(ValueError, match="all three"):
        select_phase(rows[:1])
    with pytest.raises(ValueError, match="no cyclic phase"):
        select_phase(rows)
    rows[1]["precision"]["passed"] = True
    assert select_phase(rows) == 6
    assert rows[0]["precision"]["passed"] is False
