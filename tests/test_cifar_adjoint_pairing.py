import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).parents[1]/'tools/qualify_cifar_adjoint.py';s=importlib.util.spec_from_file_location('pairing_qualification',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_noncommuting_forward_reverse_and_wrong_transpose():
 a=np.array([[1.,4.,0.],[0.,2.,1.],[2.,0.,3.]]);u,v=m.vectors(3,2)
 assert m.pairing(u,v,a@u,a.T@v)['passed']
 assert not m.pairing(u,v,a@u,a@v)['passed']
def test_degenerate_or_nonfinite_data_rejected():
 assert not m.pairing(np.ones(3),np.ones(3),np.zeros(3),np.zeros(3))['passed']
 with pytest.raises(ValueError):m.pairing(np.ones(3),np.ones(3),np.ones(3)*np.nan,np.ones(3))
def test_fixed_cost_gate_and_calls():
 assert m.cost([(.1,.2),(.2,.3)],.1)==11.4
 with pytest.raises(ValueError):m.cost([(4.,4.),(4.,4.)],1.)
 assert m.CAPS==dict(action=5,adjoint=5,j2vp=5,hvp=20,direct_gradient=2)
