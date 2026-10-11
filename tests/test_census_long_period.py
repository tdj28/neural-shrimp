import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import certify_census_long_period as C

def test_selection_requires_agreement_and_unique_maximum():
 rows=[dict(map_id='x',component=i,period=i+2)for i in range(61)]
 s={'gate':{'qualifying':rows}};a={'audit_gate':{'qualifying':rows}}
 assert C.selected_indices(s,a,rows)==[60]
 with pytest.raises(ValueError):C.selected_indices(s,{'audit_gate':{'qualifying':rows[:-1]}},rows)

def test_measured_forecast_counts_seven_full58_returns():
 assert C.profile_forecast([.01,.02,.03],58)==2*7*58*.03+30
 with pytest.raises(ValueError):C.profile_forecast([0,.02,.03],58)
