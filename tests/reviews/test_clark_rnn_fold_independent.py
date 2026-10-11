import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
from flint import ctx,arb
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'src')]
from review_clark_rnn_fold_math import Native,exact,strict_nested,endpoints
from fractions import Fraction
from neural_shrimp.rigorous_orbits import ball_json
from neural_shrimp.rigorous_folds import augmented
from neural_shrimp.rigorous_clark_rnn_fold import FoldClarkRNN
INPUT={'schema':'clark-rnn-fold-map-v1','j0':[['1/3','-1/2'],['2/5','1/4']],'gain':'3/2'}

def values(m):return np.array([[float(m[i,j].mid())for j in range(m.ncols())]for i in range(m.nrows())])

@pytest.mark.parametrize('period',[1,3,5])
def test_exp_native_augmented_matches_independent_centered_finite_differences(period):
    z=np.array([.1,-.2,.6,.7,-.4]);eps=1e-6
    with ctx.workprec(256):
        native=Native(INPUT)
        def run(x):return native.augmented([exact(str(v))for v in x],period)
        r,J,_=run(z)
        for k in range(len(z)):
            e=np.eye(len(z))[k]*eps;fd=(values(run(z+e)[0])-values(run(z-e)[0]))/(2*eps)
            np.testing.assert_allclose(values(J)[:,k],fd[:,0],rtol=3e-7,atol=3e-10)

@pytest.mark.parametrize('period',[1,3,5])
def test_independent_native_recurrence_agrees_with_shared_mixedjet(period):
    with ctx.workprec(256):
        z=[exact(v)for v in ['1/10','-1/5','3/5','7/10','-2/5']]
        r,J,data=Native(INPUT).augmented(z,period)
        old,j,olddata=augmented(FoldClarkRNN(INPUT),z,period)
        assert all(r[i,0].overlaps(old[i])for i in range(5))
        assert all(J[i,k].overlaps(j[i,k])for i in range(5)for k in range(5))
        assert all(data['M'][i,k].overlaps(olddata['monodromy'][i,k])for i in range(2)for k in range(2))

def test_delta_parameter_chain_contains_explicit_identity_term():
    with ctx.workprec(256):
        z=[exact(v)for v in ['0','0','1/2','1','0']]
        _,J,d=Native(INPUT).augmented(z,1)
        assert J[2,2].contains(exact('-1/2')) # (-I+gJ)q, first component
        lo,hi=endpoints(ball_json(J[3,2]));assert lo<=Fraction(3,5)<=hi
        assert all(x.is_zero()for x in d['quadratic'])

def test_interrupt_before_native_call():
    def stop():raise TimeoutError('control')
    with pytest.raises(TimeoutError):Native(INPUT).augmented([arb(0)]*5,1,stop)
