import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_periodic_lyapunov import periodic_witness,positive_definite


def test_periodic_metric_proves_stable_non_normal_product():
    # Each phase may expand Euclidean norm. The cyclic product is strictly stable.
    blocks=[arb_mat([[2,3],[0,'1/2']]),arb_mat([['1/5',0],[0,'1/3']])]
    r=periodic_witness(blocks)
    assert r['passed'] and len(r['phases'])==2


def test_interval_variation_small_passes_large_fails():
    with ctx.workprec(256):
        assert periodic_witness([arb_mat([[arb('1/2','1/1000')]])])['passed']
        r=periodic_witness([arb_mat([[arb('1/2','3/4')]])])
        assert r['point_passed'] and not r['passed']


def test_unstable_point_does_not_get_interval_claim():
    r=periodic_witness([arb_mat([['11/10']])])
    assert not r['point_passed'] and not r['passed']
    assert 'interval_D' not in r['phases'][0]


def test_positive_definite_rejects_indefinite_and_singular():
    assert not positive_definite(arb_mat([[1,2],[2,1]]))['passed']
    assert not positive_definite(arb_mat([[1,0],[0,0]]))['passed']


def test_saved_point_metrics_precede_interval_interrupt():
    saved=[];ticks=0
    def tick():
        nonlocal ticks
        ticks+=1
        if ticks==5:raise TimeoutError('before interval positivity')
    with pytest.raises(TimeoutError):
        periodic_witness([arb_mat([['1/2']])],tick=tick,observe=lambda k,r:saved.append((k,r)))
    assert len(saved)==1 and saved[0][1]['point_passed']
