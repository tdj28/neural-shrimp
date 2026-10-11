import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'tools/reviews'))
import review_cifar_reorthogonalized as R


def test_physical_forcing_and_negative_material_gate():
 g=np.array([3.,4.]);old=-.2*g;actual=-.34*g
 o=R.comparison(g,old,actual,actual,0.,256)
 assert o['qualified']and o['true_forcing']==pytest.approx(.66)and o['baseline_forcing']==pytest.approx(.8)
 assert not o['material_improvement']and not o['strict_forcing_pass']


def test_partial_basis_and_predicted_only_are_rejected():
 g=np.ones(5);hp=-.9*g
 assert not R.comparison(g,-.2*g,hp,hp,0.,255)['qualified']
 assert not R.comparison(g,-.2*g,-.3*g,hp,0.,256)['qualified']
 assert not R.comparison(g,-.2*g,hp,hp,1e-5,256)['qualified']


def test_basis_assembly_and_numerical_rank():
 V=np.eye(4)[:,:3];W=np.diag([1.,2.,0.,3.])@V;c=np.array([1.,-.5,2.]);p,hp,gram,error,rank,s=R.assembly(V,W,c)
 assert np.array_equal(p,[1.,-.5,2.,0.])and np.array_equal(hp,[1.,-1.,0.,0.])
 assert error==0 and rank==2 and np.array_equal(gram,np.eye(3))


def test_nonfinite_or_mismatched_vectors_rejected():
 with pytest.raises(ValueError):R.comparison(np.ones(3),np.ones(2),np.ones(3),np.ones(3),0.,256)
 with pytest.raises(ValueError):R.assembly(np.eye(2),np.full((2,2),np.nan),np.ones(2))


def test_event_parser_rejects_orphan_and_state_hash_change():
 with pytest.raises(ValueError):R.events_bound([dict(kind='operation_returned',operation='hvp',index=1,seconds=.1)],{})
 inv={'s':dict(sha256='good',bytes=8),'v':dict(sha256='v',bytes=8),'r':dict(sha256='r',bytes=8)}
 e=[dict(kind='native_input',operation='hvp',index=1,state=dict(file='s',identity=dict(sha256='bad',bytes=8)),direction=dict(file='v',identity=inv['v'])),dict(kind='operation_started',operation='hvp',index=1),dict(kind='operation_returned',operation='hvp',index=1,seconds=.1),dict(kind='native_output',operation='hvp',index=1,result=dict(file='r',identity=inv['r']))]
 with pytest.raises(ValueError):R.events_bound(e,inv)
