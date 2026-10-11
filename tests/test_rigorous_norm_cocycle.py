import pytest
import numpy as np
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_norm_cocycle import (advance,operator_upper,exact_frames,scalar_remainder_metric,framed_cocycle,frobenius_upper)
from neural_shrimp.rigorous_correlated_cocycle import polynomial_value
from neural_shrimp.rigorous_connection import restore_ball


def scalar(a,b=0,c=0):return [arb_mat([[a]]),arb_mat([[b]]),arb_mat([[c]])]


def test_operator_bound_is_strict_certificate_not_float_gate():
    with ctx.workprec(256):
        matrix=arb_mat([[1,2],[0,1]]);record,gamma=operator_upper(matrix)
        assert record['passed'] and gamma>2
        matrix=arb_mat([[arb(1,10)]])
        record,_=operator_upper(matrix)
        assert not record['passed']


def test_nonsymmetric_native_products_and_norm_errors():
    with ctx.workprec(256):
        a=[arb_mat([[1,2],[0,1]]),arb_mat([[0,1],[1,0]]),arb_mat([[1,0],[0,0]])]
        b=[arb_mat([[2,0],[3,1]]),arb_mat([[1,0],[0,-1]]),arb_mat([[0,0],[1,0]])]
        result,_=advance((b,arb('1/100'),arb(4)),(a,arb('1/50'),arb(3)))
        for t in (-1,0,1):
            # These diagonal errors have operator norms exactly their size.
            aa=polynomial_value(a,arb(t))+arb_mat([[arb('1/50'),0],[0,arb('1/50')]])
            bb=polynomial_value(b,arb(t))+arb_mat([[arb('1/100'),0],[0,arb('1/100')]])
            error=aa*bb-polynomial_value(result[0],arb(t))
            assert np.linalg.norm(np.array([[float(error[i,j].mid()) for j in range(2)] for i in range(2)]),2)<float(result[1])
            assert frobenius_upper(error)<result[1]*2
            # Actual operator <=F; a conservative direct squared-F comparison.
            assert max(abs(error[i,j]).upper() for i in range(2) for j in range(2))<result[1]
        assert result[0][0] != b[0]*a[0]


def test_fixed_cyclic_similarity_and_shared_variable_cancellation():
    with ctx.workprec(256):
        rows=[{'phase':0,'center_J':arb_mat([[2]]),'whole_Jprime':arb_mat([[1]])},
              {'phase':1,'center_J':arb_mat([['2/5']]),'whole_Jprime':arb_mat([['-1/5']])}]
        r=framed_cocycle(rows,[arb_mat([[3]]),arb_mat([[7]])],arb(1))
        assert r['passed'] and r['phases']==2
        assert restore_ball(r['remainder_norm_upper'])<arb('1e-60')


def test_quarter_scaling_keeps_center_uncertainty():
    with ctx.workprec(256):
        rows=[{'phase':0,'center_J':arb_mat([[arb('1/2','1/100')]]),'whole_Jprime':arb_mat([[0]])}]
        r1=framed_cocycle(rows,[arb_mat([[1]])],arb(1))
        r2=framed_cocycle(rows,[arb_mat([[1]])],arb('1/4'))
        assert restore_ball(r1['remainder_norm_upper'])==restore_ball(r2['remainder_norm_upper'])
        assert r1['passed'] and r2['passed']


def test_invalid_frames_and_order_fail_closed():
    with pytest.raises((ValueError,ZeroDivisionError)):exact_frames([arb_mat([[0]])])
    with pytest.raises(ValueError):exact_frames([arb_mat([[arb(1,'1/10')]])])
    with pytest.raises(ValueError):framed_cocycle([{'phase':1}],[arb_mat([[1]])],arb(1))


def test_neutral_and_remainder_destroying_gap_cannot_pass():
    for a,e in [('11/10',0),('1/2',1)]:
        r=scalar_remainder_metric(scalar(a),arb(e),arb(a))
        assert not r['passed']
