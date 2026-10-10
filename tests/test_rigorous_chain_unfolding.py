import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_generalized_chain import augmented_chain
from neural_shrimp.rigorous_chain_unfolding import chain_coefficients
from neural_shrimp.rigorous_eigenpair_march import decode

class NormalForm:
    n=3
    def __init__(self,A=2,B=7,shear=False,rank=True,mix=False):self.A=A;self.B=B;self.shear=shear;self.rank=rank;self.mix=mix
    def cusp_step(self,x,a,b):
        u=[x[0]-x[2],x[1]-2*x[2],x[2]]if self.shear else x
        shift=3*(a-1)-2*(b-1)if self.mix else 0
        u=[u[0]-shift,u[1],u[2]]
        alpha=2*(a-1)+(b-1)if self.mix else a-1
        beta=(a-1)+3*(b-1)if self.mix else (b-1 if self.rank else a-1)
        v=[u[0]+u[1]+shift,u[1]+alpha+beta*u[1]+self.A*u[0]*u[0]+self.B*u[0]*u[1],u[2]*.5+3*u[0]*u[0]]
        return [v[0]+v[2],v[1]+2*v[2],v[2]]if self.shear else v

def box():return [arb(0)]*3+[arb(1)]*2+[arb(1),arb(0),arb(0)]+[arb(0),arb(1),arb(0)]

def calc(model,z=None):
    z=box()if z is None else z
    _,j,_=augmented_chain(model,z,1)
    return chain_coefficients(z,j)

@pytest.mark.parametrize('shear',[False,True])
def test_known_normal_form_and_complement_coupling(shear):
    with ctx.workprec(256):
        r=calc(NormalForm(shear=shear))
        assert decode(r['a']).contains(4)and decode(r['b']).contains(7)
        assert decode(r['s']).contains(12)and decode(r['unfolding_determinant']).contains(1)
        assert r['quadratic_nondegenerate']and r['resonant_nondegenerate']and r['unfolding_transverse']
        if shear:assert decode(r['p0'][2]).contains(-2)and decode(r['p1'][2]).contains(-1)

def test_degenerate_quadratic_coefficient_is_not_generic():
    r=calc(NormalForm(A=0));assert not r['quadratic_nondegenerate']and not r['unfolding_transverse']

def test_degenerate_resonant_and_parameter_coefficients_rejected():
    assert not calc(NormalForm(B=4))['resonant_nondegenerate']
    r=calc(NormalForm(rank=False));assert not r['unfolding_transverse']and decode(r['unfolding_determinant']).contains(0)

def test_outward_root_box_contains_exact_known_values():
    with ctx.workprec(256):
        z=[arb(v,arb((1,-80)))for v in box()];r=calc(NormalForm(shear=True),z)
        for k,value in [('a',4),('b',7),('s',12),('unfolding_determinant',1)]:assert decode(r[k]).contains(value)
        assert r['unfolding_transverse']

def test_wrong_chain_and_invalid_shapes_fail():
    z=box();z[-2]=arb(2)
    with pytest.raises(ValueError):calc(NormalForm(),z)
    _,j,_=augmented_chain(NormalForm(),box(),1)
    with pytest.raises(ValueError):chain_coefficients(box()[:-1],j)


def test_state_shift_and_mixed_physical_rates_have_known_rank_determinant():
    with ctx.workprec(256):
        r=calc(NormalForm(shear=True,mix=True))
        assert decode(r['unfolding_determinant']).contains(5)and r['unfolding_transverse']
        assert decode(r['a']).contains(4)and decode(r['b']).contains(7)
