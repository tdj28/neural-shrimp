import sys
from pathlib import Path
import numpy as np
from scipy.special import ndtr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from context_robustness import success_counts,failed_contexts,score_counts


def test_exact_templates_recover_and_reversed_labels_fail():
    m=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],dtype=float)
    z=np.zeros((17,2))
    assert np.all(success_counts(m,m,.1,z)==4)
    assert np.all(success_counts(m,m[[2,3,0,1]],.1,z)==0)
    assert np.all(success_counts(m,np.tile(m,(2,1)),.1,z)==8)


def test_no_signal_is_chance_with_balanced_labels():
    z=np.random.default_rng(3).normal(size=(100,3))
    assert np.all(success_counts(np.zeros((4,3)),np.zeros((8,3)),.1,z)==4)


def test_projected_distances_match_literal_noisy_observations():
    rng=np.random.default_rng(6);m=rng.normal(size=(4,18));a=rng.normal(size=(8,18));z=rng.normal(size=(200,18))
    literal=a[None,:,None,:]+.3*z[:,None,None,:]-m[None,None,:,:]
    pred=np.argmin(np.sum(literal**2,axis=-1),axis=-1)
    labels=np.array([-1,-1,1,1]);truth=np.tile(labels,2)
    expected=np.sum(labels[pred]==truth,axis=1)
    np.testing.assert_array_equal(success_counts(m,a,.3,z),expected)


def test_two_gaussian_accuracy_matches_closed_form():
    m=np.array([[-1.],[-1.],[1.],[1.]])
    z=np.random.default_rng(19).normal(size=(100000,1))
    assert abs(success_counts(m,m,1.,z).mean()/4-ndtr(1))<.003


def test_precision_selection_and_count_normalization():
    a=np.zeros((2,2,5,4,61,3));b=a.copy();b[1,0,3,2,60,1]=.01
    assert failed_contexts(a,b,1e-4)[0]==[3]
    counts=np.zeros((2,3,128,8),dtype=np.uint8);counts[:,0]=4;counts[:,1:]=4
    context,accuracy,se=score_counts(counts)
    np.testing.assert_array_equal(accuracy,[[1,.5,.5],[1,.5,.5]])
    assert not se.any()
