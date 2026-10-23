import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from huginn_continuation_pilot import classify
@pytest.mark.parametrize('period',[1,2,4,8,16,32,64])
def test_smallest_passing_lag(period):
    tail=np.tile(np.eye(period,dtype=np.float32),(256//period+1,1))[:256,None,:]+1
    labels,errors=classify(tail)
    assert labels.tolist()==[period]
    assert errors.shape==(64,1)
def test_every_row_and_every_position_gate():
    tail=np.ones((256,2,3));tail[-1,1,0]=2
    assert classify(tail)[0].tolist()==[1,0]
def test_irregular_is_unresolved_not_chaos():
    tail=np.random.default_rng(0).normal(size=(256,4,6))
    assert classify(tail)[0].tolist()==[0]*4
