import sys
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp.cifar_stationary_merit import direction_gate,trial_gate

def test_inexact_direction_above_old_gate_can_decrease_true_gradient_merit():
 x=np.array([1.,2.]);g=x.copy();p=-.25*g;hp=p.copy();gate=direction_gate(x,g,p,hp)
 assert gate['passed']and not gate['old_forcing_pass']and gate['forcing']==.75
 assert gate['slope']<=gate['cauchy_descent_upper']+1e-14
 # Small enough damped direction respects fixed5% relative trust bound.
 t=trial_gate(x,x+.125*p,g+.125*hp,.125,gate);assert t['accepted']and t['relative_gradient']<1.01

def test_non_descent_and_outside_trust_cannot_accept():
 x=np.ones(3);g=x.copy();gate=direction_gate(x,g,g,g);assert not gate['passed']
 assert not trial_gate(x,x-g,np.zeros(3),1.,gate)['accepted']
 good=direction_gate(x,g,-g,-g);assert not trial_gate(x,np.zeros(3),np.zeros(3),1.,good)['accepted']
 with pytest.raises(ValueError):direction_gate(x,np.zeros(3),x,x)

def test_driver_caps_source_closure_and_terminal_rules():
 sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
 import explore_cifar_stationary_saved_merit as W
 import launch_cifar_stationary_saved_merit as L
 assert W.BASE==L.OUT and W.CAPS==dict(action=0,j2vp=0,hvp=1,direct_gradient=14)
 assert all((W.ROOT/n).is_file()for n in W.EXTRA)
 assert not L.scientific_complete({})
