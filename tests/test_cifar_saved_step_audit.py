import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_cifar_saved_shooting_step import direction

def test_descent_without_newton_gate_is_kept_distinct():
    g=direction(np.ones(2)*10,np.array([1.,0.]),np.array([.1,0.]),np.array([-.3,0.]))
    assert g['passed'] and not g['old_newton_forcing_pass']

def test_no_merit_step_for_ascent():
    assert not direction(np.ones(2)*10,np.array([1.,0.]),np.array([.1,0.]),np.array([.3,0.]))['passed']

def test_unbounded_direction_rejected():
    assert not direction(np.ones(2),np.array([1.,0.]),np.array([2.,0.]),np.array([-.3,0.]))['passed']

def test_full_linear_overshoot_rejected():
    assert not direction(np.ones(2)*10,np.array([1.,0.]),np.array([.1,0.]),np.array([-3.,0.]))['passed']
