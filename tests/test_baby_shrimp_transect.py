import sys
import json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import baby_shrimp_transect as transect


def test_fixed_transect_rejects_horizon_or_axis_changes():
    path=Path(__file__).resolve().parents[1]/"docs/baby-shrimp-zoom/jones-transect-plan.json"
    plan=json.loads(path.read_text());transect.validate(plan)
    plan["cells"][32]["log10_temperature"]+=1e-5
    with pytest.raises(ValueError,match="temperature"):transect.validate(plan)
    plan=json.loads(path.read_text());plan["record"]=300
    with pytest.raises(ValueError,match="bounds"):transect.validate(plan)
