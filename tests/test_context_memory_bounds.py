import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.special import ndtr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from context_memory_bounds import bounds,batch,LABELS


def test_identical_distributions_prove_chance():
    b=bounds(np.zeros((4,3)),.001,LABELS[:,0])
    np.testing.assert_array_equal(b[:3],[.5,.5,.5])
    b=bounds(np.array([[-2.],[2.],[-2.],[2.]]),.1,LABELS[:,0])
    np.testing.assert_allclose(b[:2],[.5,.5])


def test_two_gaussian_case_has_exact_lower_and_upper():
    for gap in [.01,.1,1.,10.]:
        m=LABELS[:,0,None]*gap
        b=bounds(m,1.,LABELS[:,0])
        np.testing.assert_allclose(b[:3],[ndtr(gap)]*3,rtol=1e-14)


def test_non_affine_xor_is_still_recoverable():
    means=LABELS[:,:2].astype(float)
    b=bounds(means,.1,LABELS[:,2])
    assert b[0]>.999 and b[1]>.999 and b[2]==.5


def test_bounds_invariant_to_rotation_translation_permutation_scale():
    rng=np.random.default_rng(5)
    means=rng.normal(size=(4,3));q=np.linalg.qr(rng.normal(size=(3,3)))[0]
    a=bounds(means,.3,LABELS[:,0])
    b=bounds((means@q+10)[[3,1,0,2]]*7,2.1,LABELS[[3,1,0,2],0])
    np.testing.assert_allclose(a[:4],b[:4],atol=1e-13)


def test_random_instances_have_consistent_bounds():
    rng=np.random.default_rng(20260924)
    out=batch(rng.normal(size=(100,4,18)),1.)
    assert np.all(out[:,:,0]>=.5) and np.all(out[:,:,1]<=1)
    assert np.all(out[:,:,0]<=out[:,:,1]+1e-12)
    assert np.all(out[:,:,0]>=out[:,:,2]-1e-12)


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):bounds(np.zeros((4,3)),0,LABELS[:,0])
    with pytest.raises(ValueError):bounds(np.zeros((4,3)),1,np.ones(4))
