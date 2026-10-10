import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import explore_cifar_saved_shooting_merit as C

def test_merit_direction_does_not_relabel_failed_newton_gate():
 q=C.gate(np.array([10.,0.]),np.array([1.,0.]),np.array([.1,0.]),np.array([-.3,0.]))
 assert q['passed']and not q['old_newton_forcing_pass']
 assert not C.gate(np.ones(2),np.ones(2),np.ones(2),-np.ones(2))['passed']
 assert not C.gate(np.ones(2),np.ones(2),np.zeros(2),np.ones(2))['passed']

def test_native_replay_and_all_trials_fit_accounting():
 assert C.CAPS==dict(action=0,j2vp=0,hvp=2,direct_gradient=2+2*13)
