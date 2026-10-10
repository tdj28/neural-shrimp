import pytest
from flint import arb, ctx
from neural_shrimp.rigorous_cusps import cusp_proof, augmented_cusp, spectral_exclusion
from neural_shrimp.rigorous_orbits import exact
from neural_shrimp.rigorous_connection import restore_ball

class CoupledCusp:
    n=2
    def __init__(self,kappa):
        self.k=exact(kappa)
    def cusp_step(self,state,a,b):
        x,y=state; den=1-self.k; B=exact('1/4')
        r1=(a-1)+(b-1)*x+B*y-(B/den)*x*x+3*x*y-x*x*x
        r2=(self.k-1)*y+x*x+2*(a-1)+exact('1/3')*(b-1)
        return [x+r1,y+r2]

@pytest.mark.parametrize('kappa,h,w,cubic,uf,det',[
 ('1/2','4','1/2','5',[['2','1/6'],['12','3']],'4'),
 ('2','-2','-1/4','-4',[['1/2','-1/12'],['-6','0']],'-1/2'),
])
@pytest.mark.parametrize('sign',[1,-1])
def test_coupled_nonnormal_cusp_complement_and_unfolding(kappa,h,w,cubic,uf,det,sign):
    with ctx.workprec(256):
        z=[arb(0),arb(0),arb(1),arb(1),arb(sign),arb(0),arb(0),exact(h)]
        r=cusp_proof(CoupledCusp(kappa),z,1,newton_steps=1)
        assert r['passed'],r['gates']
        assert restore_ball(r['cubic_coefficient']).contains(exact(cubic))
        assert restore_ball(r['unfolding_determinant']).contains(sign*exact(det))
        assert restore_ball(r['left'][0][0]).contains(sign)
        assert restore_ball(r['left'][0][1]).contains(sign*exact(w))
        for i in range(2):
            for j in range(2):
                assert restore_ball(r['unfolding'][i][j]).contains((sign if i==0 else 1)*exact(uf[i][j]))

class MissingUnfolding:
    n=1
    def cusp_step(self,x,a,b):return [x[0]+(a-1)+x[0]*x[0]*x[0]]

def test_missing_unfolding_cannot_prove_root():
    with ctx.workprec(256):
        with pytest.raises(ZeroDivisionError):
            cusp_proof(MissingUnfolding(),[arb(0),arb(1),arb(1),arb(1),arb(0)],1,newton_steps=1)

def test_whole_box_spectral_uncertainty_rejects_neutral_contact():
    from flint import arb_mat
    with ctx.workprec(256):
        m=arb_mat([[arb(0),arb(0)],[arb(0),arb('1/2','3/4')]])
        assert not spectral_exclusion(m)['passed']
