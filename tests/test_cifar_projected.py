import numpy as np
import pytest
from neural_shrimp.cifar_projected import basis,proposal,qualify


def test_nonnormal_physical_subspace_and_fresh_action_authority():
    A=np.array([[2.,7,0,0],[0,3,1,0],[0,0,4,2],[0,0,0,5]])
    candidates=[np.array([1.,2,0,0]),np.array([0.,1,1,0]),
                np.array([0.,0,1,1]),np.array([1.,0,0,1])]
    U,rows=basis(candidates);r=np.array([1.,2,-1,3])
    p,pred,record=proposal(U,A@U,r)
    assert len(rows)==4 and record['dimension']==4
    out=qualify(r,pred,A@p)
    assert out['forcing_pass'] and out['action_linearity_pass']
    assert not qualify(r,pred,A@p+10)['action_linearity_pass']


def test_dependent_candidates_retained_and_unreachable_residual_rejected():
    e=np.array([1.,0,0]);U,rows=basis([e,2*e,np.zeros(3),-e])
    assert U.shape==(3,1) and [r['accepted']for r in rows]==[True,False,False,False]
    r=np.array([0.,1,0]);p,pred,_=proposal(U,U,r)
    assert not qualify(r,pred,p)['forcing_pass']
    with pytest.raises(ValueError):basis([e]*3)


def test_scale_invariance_and_geometry_only_rank_choice():
    e=np.eye(4);U,a=basis(list(e));V,b=basis([e[i]*10.**(i-2)for i in range(4)])
    assert np.allclose(U,V) and [x['accepted']for x in a]==[x['accepted']for x in b]
    with pytest.raises(ValueError):qualify(np.zeros(4),e[0],e[0])
