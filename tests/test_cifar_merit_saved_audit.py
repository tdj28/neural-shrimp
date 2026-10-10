import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).parents[1]/'tools/audit_cifar_merit_saved.py';s=importlib.util.spec_from_file_location('saved_audit',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_fsum_cancellation_and_exact_synthetic_acceptance():
 assert m.dot([1e16,1.,-1e16],[1.,1.,1.])==1.
 x=np.array([1.,2.]);r=np.array([1.,0.]);ap=np.array([-.5,0.]);a=m.metrics(x,x+.1,x+r+ap,r,ap,1.)
 assert a['accepted_local_step']and a['ratio']==1. and a['merit']==.125
 assert not m.metrics(x,x+.1,x+2*r,r,ap,1.)['accepted_local_step']
 assert not m.metrics(x,x,x+r+ap,r,ap,1.)['accepted_local_step']
def test_prediction_must_use_actual_alpha():
 x=np.array([1.,2.]);r=np.array([1.,0.]);ap=np.array([-.5,0.]);a=m.metrics(x,x+.1,x+r+.25*ap,r,ap,.25)
 assert a['ratio']==1.and a['accepted_local_step']
