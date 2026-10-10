import numpy as np
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_folds import augmented
from neural_shrimp.rigorous_clark_rnn_fold import FoldClarkRNN,check_rnn_fold,relabel
from neural_shrimp.rigorous_orbits import exact,dyadic
from neural_shrimp.clark_rnn_boundaries import jets

MAP={'schema':'clark-rnn-fold-map-v1','j0':[['1/5','-7/10'],['3/10','2/5']],'gain':'17/10'}

def test_native_one_step_all_mixed_blocks_independent_analytic_formula():
    with ctx.workprec(256):
        m=FoldClarkRNN(MAP);x=[arb('3/10'),arb('-1/5')];q=[arb('2/3'),arb('-1/7')];d=arb('3/5')
        g,j,data=augmented(m,x+[d]+q,1)
        t=[v.tanh() for v in x];s=[1-v*v for v in t]
        for i in range(2):
            expected=-x[i]+m.g*sum((m.j[i][k]*t[k] for k in range(2)),arb(0))
            assert data['b'][i].overlaps(expected)
            for k in range(2):
                a=-d*int(i==k)+d*m.g*m.j[i][k]*s[k]
                assert j[i,k].overlaps(a) and j[2+i,3+k].overlaps(a)
                second=d*m.g*m.j[i][k]*(-2*t[k]*s[k])*q[k]
                assert j[2+i,k].overlaps(second)
            mixed=sum(((-int(i==k)+m.g*m.j[i][k]*s[k])*q[k] for k in range(2)),arb(0))
            assert j[2+i,2].overlaps(mixed)

def test_composed_mixed_jets_against_separate_native_second_jet_implementation():
    with ctx.workprec(256):
        m=FoldClarkRNN(MAP);x=np.array([.3,-.2]);q=np.array([.4,-.7]);d=.6
        g,j,a=augmented(m,[exact(dyadic(v)) for v in np.r_[x,d,q]],3)
        end,first,second,_,_=jets(x,np.array([[.2,-.7],[.3,.4]]),1.7,d,3)
        np.testing.assert_allclose([float(v) for v in a['b']],first[:,3],rtol=2e-14,atol=2e-15)
        expect=np.einsum('ijk,j->ik',second[:,:2,:],q)
        got=np.array([[float(j[2+i,k]) for k in range(3)] for i in range(2)])
        np.testing.assert_allclose(got,expect[:,[0,1,3]],rtol=2e-13,atol=2e-15)

def test_invalid_settings_and_leak_before_proof():
    inp={'schema':'clark-rnn-fold-input-v1','period':5,'map_inputs':MAP,'approximation':['0','0','1/2','1','0']}
    for extra in [{'precision':True},{'newton_steps':13},{'radius_exponent':0}]:
        with pytest.raises(ValueError):check_rnn_fold(inp,**extra)
    for d in ['0','1','-1']:
        with pytest.raises(ValueError):check_rnn_fold(dict(inp,approximation=['0','0',d,'1','0']))
    with pytest.raises(ValueError):FoldClarkRNN(dict(MAP,gain='0'))

def test_generic_engine_labels_are_copied_and_native_parameter_is_explicit():
    old={'gates':{'positive_hidden_rate':True},'hidden_rate_second_derivative':{'mid':[1,0],'rad':[0,0]}}
    new=relabel(old,2)
    assert new['gates']=={'positive_delta':True} and 'delta_second_derivative' in new
    assert 'positive_hidden_rate' in old['gates'] and 'hidden_rate_second_derivative' in old

def test_tick_interrupts_native_composition_without_swallowing():
    def fail():raise TimeoutError('test')
    with pytest.raises(TimeoutError):augmented(FoldClarkRNN(MAP),[arb(0),arb(0),arb('1/2'),arb(1),arb(0)],5,fail)
