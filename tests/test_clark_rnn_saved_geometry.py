import importlib.util
from pathlib import Path
import numpy as np
S=importlib.util.spec_from_file_location('savedgeom',Path(__file__).parents[1]/'tools/plot_clark_rnn_p7_geometry.py')
M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_odd_phase_match_preserves_native_period():
    a=np.array([[1.,2.],[3.,4.],[5.,6.]])
    b=-np.roll(a,1,axis=0)
    assert M.cycle_match(a,b)==(0.,-1,2)
def test_near_parameter_does_not_force_state_identity():
    a=np.array([[1.,2.],[3.,4.],[5.,6.]])
    b=a+.3
    assert M.cycle_match(a,b)[0]>.04
