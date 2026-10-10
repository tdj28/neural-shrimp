import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import scout_native_unit_pair as S

def test_measured_operation_forecast():
 f=S.forecast([.1]*12,100000,[1000]*12);assert f['admitted']and f['seconds']==92 and f['native_phase_call_cap']==12960
 assert not S.forecast([1.]*12,100000,[1000]*12)['admitted']
 assert not S.forecast([.1]*12,100000,[100000]*12)['admitted']
 for v in [[],[-1.]*12,[float('nan')]*12]:
  with pytest.raises(ValueError):S.forecast(v,100000,[1000]*12)

def test_all704_endpoint_comparison_exact_and_strict():
 rows=[{'mid':[1,0],'rad':[1,-10]}for _ in range(704)];seed=[1.]*704
 assert S.endpoint(seed,rows)['passed']
 seed[699]=1.+2**-10;assert not S.endpoint(seed,rows)['passed']
 with pytest.raises(ValueError):S.endpoint(seed[:-1],rows)

def test_all_native_rate_order_in_endpoint_packet():
 assert S.SETTINGS['max_points']==64 and S.SETTINGS['max_evaluations']==360
 assert S.SOURCES.count('src/neural_shrimp/unit_pair_scout.py')==1
