import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prompt_control_attractors import recurrence_errors

def test_primitive_two_cycle():
    states=np.tile([[1.,2.],[3.,4.]],(128,1))
    errors=recurrence_errors(states)
    assert errors[0]>.1 and errors[1]==0

def test_drift_is_not_recurrence():
    states=np.arange(1.,257.)[:,None]
    assert min(recurrence_errors(states))>1e-3
