import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_correlated_cocycle import (polynomial_value,multiply_models,
    total_jacobian_derivative,phase_model,cocycle,monodromy_metric,integer_power,correlated_slice)
from neural_shrimp.rigorous_orbits import vector_json


def scalar(a,b=0,c=0,e=0):
    return [arb_mat([[a]]),arb_mat([[b]]),arb_mat([[c]])],arb_mat([[arb(0,e)]])


def test_shared_variable_cancellation_proves_stability():
    with ctx.workprec(256):
        result=cocycle([scalar(2,1),scalar('2/5','-1/5')])
        assert result[0][1][0,0].contains(0) and result[0][1][0,0].rad()<arb('1e-70')
        assert result[0][2][0,0].contains(arb('-1/5'))
        assert monodromy_metric(result)['passed']
        # Independent phase intervals allow products above one.
        assert ((arb(2,1))*arb('2/5','1/5')).upper()>1


def test_noncommuting_order_and_every_remainder_cross_term():
    with ctx.workprec(256):
        a=([arb_mat([[1,2],[0,1]]),arb_mat([[0,1],[1,0]]),arb_mat([[1,0],[0,0]])],arb_mat([[arb(0,'1/100'),0],[0,arb(0,'1/200')]]))
        b=([arb_mat([[2,0],[3,1]]),arb_mat([[1,0],[0,-1]]),arb_mat([[0,0],[1,0]])],arb_mat([[arb(0,'1/50'),0],[0,arb(0,'1/100')]]))
        product=multiply_models(a,b)
        for t in ('-1','-1/2','0','1/2','1'):
            for sign in (-1,1):
                exact_a=polynomial_value(a[0],arb(t))+arb_mat([[sign*arb('1/100'),0],[0,sign*arb('1/200')]])
                exact_b=polynomial_value(b[0],arb(t))+arb_mat([[sign*arb('1/50'),0],[0,sign*arb('1/100')]])
                enclosed=polynomial_value(product[0],arb(t))+product[1]
                actual=exact_a*exact_b
                assert all(enclosed[i,j].contains(actual[i,j]) for i in range(2) for j in range(2))
        assert product[0][0] != b[0][0]*a[0][0]


def test_total_physical_j_derivative_includes_actual_rate_tangent():
    class RateMap:
        n=1
        def jet_step(self,x,h):return [x[0]-h*x[0]*x[0]]
    j,d=total_jacobian_derivative(RateMap(),[arb(2)],arb(3),[arb(5)],arb(7))
    assert j==arb_mat([[-11]]) and d==arb_mat([[-58]])


def test_linear_proposal_defect_is_retained_in_integral_remainder():
    with ctx.workprec(256):
        m=phase_model(arb_mat([[arb('1/2','1/10000')]]),arb_mat([[arb(2,'1/10')]]),arb('1/16'))
        for t in (-1,0,1):
            actual=arb('1/2')+arb('1/10000')+(arb('21/10')/16)*t
            assert (polynomial_value(m[0],arb(t))+m[1])[0,0].contains(actual)


def test_neutral_and_unstable_cannot_pass():
    for val in ('1','11/10'):
        try:r=monodromy_metric(scalar(val))
        except (ValueError,ZeroDivisionError):continue
        assert not r['passed']


def test_interruption_before_product_is_preserved():
    def stop():raise TimeoutError('synthetic')
    with pytest.raises(TimeoutError):cocycle([scalar('1/2')],tick=stop)


def test_integer_powers_are_finite_on_signed_parameter_intervals():
    for k in range(5):
        value=integer_power(arb(0,1),k)
        assert value.is_finite() and value.contains(1) and value.contains((-1)**k)
        if k and k%2==0:assert value.contains(0)


def test_integrated_fixed_slice_uses_true_state_and_rate_tangent():
    from test_rigorous_arclength_slices import StableFold,parent
    with ctx.workprec(256):
        x=arb('35/64','1/64');h=arb(1)+x*x
        result=correlated_slice(StableFold(),StableFold(),parent(),vector_json([x,h]),
            vector_json([arb(1),2*x]),'1/32','1/16')
        assert result['passed'] and result['exact_endpoints']==['1/32','1/16']
