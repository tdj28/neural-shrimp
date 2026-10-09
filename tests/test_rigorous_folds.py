from flint import arb, arb_mat, ctx
import pytest
from neural_shrimp.rigorous_folds import MixedJet, FoldTanhMap, augmented, _proof, check_fold
from neural_shrimp.rigorous_orbits import BallTanhMap, exact


def synthetic_inputs(bias=True):
    return {'schema':'neural-shrimp-c1-input-v1','period':1,
        'network':{'activation':'tanh','d':2,'width':1,'bias':bias,
                   'in_scale':'3/2','out_scale':'2/3','loss_scale':'3/4','ridge':'1/20'},
        'x':[['1/3','-1/4'],['-2/5','1/2']],'y':['1/7','-2/9'],
        'lr_hidden':'1/8','lr_output':'1/9'}


def test_mixed_product_polynomial_exact():
    x = MixedJet.variable(arb(2),2,0,arb(3))
    y = MixedJet.variable(arb(5),2,1,arb(7))
    q = x*x*y
    assert q.f == 20 and q.d == 88
    assert list(q.g) == [20,4]
    assert list(q.m) == [58,12]


def test_tanh_mixed_chain():
    with ctx.workprec(200):
        x = MixedJet.variable(arb('1/3'),1,0,arb(2))
        q = (x*x).tanh()
        z=x.f*x.f;t=z.tanh();s=1-t*t
        assert q.d.overlaps(s*4*x.f)
        assert q.m[0].overlaps(4*s-16*x.f*x.f*t*s)


@pytest.mark.parametrize('bias',[False,True])
def test_network_map_jacobian_matches_independent_existing_formula(bias):
    with ctx.workprec(200):
        inputs=synthetic_inputs(bias)
        old=BallTanhMap(inputs);new=FoldTanhMap(inputs)
        theta=[arb(i+1)/17 for i in range(old.n)]
        v=[arb(i+2)/23 for i in range(old.n)]
        z=theta+[exact(inputs['lr_hidden'])]+v
        g,j,data=augmented(new,z,1)
        end,jac=old.step(theta)
        for i in range(old.n):
            assert (g[i]+theta[i]).overlaps(end[i])
            assert g[old.n+i].overlaps(sum(((jac[i,k]-int(i==k))*v[k] for k in range(old.n)),arb(0)))
            for k in range(old.n):
                assert j[i,k].overlaps(jac[i,k]-int(i==k))
                assert j[old.n+i,old.n+1+k].overlaps(jac[i,k]-int(i==k))
        # Hidden-rate partial is negative gradient, output rows are zero.
        for i in range(old.n):
            expected=(end[i]-theta[i])/exact(inputs['lr_hidden']) if i<old.hidden else arb(0)
            assert data['b'][i].overlaps(expected)


class PolynomialFold:
    n=2
    def __init__(self,degenerate=False,defective=False,unstable=False):
        self.degenerate,self.defective,self.unstable=degenerate,defective,unstable
    def jet_step(self,x,eta):
        if self.defective:
            return [x[0]+x[1]+eta-1,x[1]]
        term=x[0]*x[0]
        if self.degenerate:
            term=term*x[0]
        return [x[0]+eta-1+term,x[1]*(2 if self.unstable else arb(1)/2)]


def test_augmented_jacobian_polynomial_all_blocks():
    g,j,_=augmented(PolynomialFold(),[arb(2),arb(3),arb(5),arb(7),arb(11)],1)
    assert g == [8,arb(-3)/2,28,arb(-11)/2,arb(169)/2]
    assert j == arb_mat([[4,0,1,0,0],[0,arb(-1)/2,0,0,0],
                         [14,0,0,4,0],[0,0,0,0,arb(-1)/2],[0,0,0,7,11]])


def test_generic_fold_all_gates_and_left_orientation():
    with ctx.workprec(200):
        r=_proof(PolynomialFold(),[arb(0),arb(0),arb(1),arb(1),arb(0)],1,-60,0,16)
        assert r['passed']
        assert r['status']=='primitive_nondegenerate_fold_with_attracting_complement'
        assert r['complement_witness_power']==1


def test_other_unstable_mode_fails_only_complement():
    with ctx.workprec(200):
        r=_proof(PolynomialFold(unstable=True),[arb(0),arb(0),arb(1),arb(1),arb(0)],1,-60,0,16)
        assert not r['passed'] and r['gates']['augmented_root_unique']
        assert r['gates']['quadratic_nondegeneracy'] and not r['gates']['other_modes_attracting']


@pytest.mark.parametrize('kwargs',[{'degenerate':True},{'defective':True}])
def test_degenerate_or_defective_cannot_pass(kwargs):
    with ctx.workprec(200),pytest.raises(ZeroDivisionError):
        _proof(PolynomialFold(**kwargs),[arb(0),arb(0),arb(1),arb(1),arb(0)],1,-60,0,16)


def test_period_multiple_fails_primitivity():
    with ctx.workprec(200):
        r=_proof(PolynomialFold(),[arb(0),arb(0),arb(1),arb(1),arb(0)],2,-60,0,16)
        assert r['gates']['augmented_root_unique'] and not r['gates']['primitive_period']
        assert not r['passed']


def test_deadline_is_not_swallowed():
    def tick(): raise TimeoutError('synthetic deadline')
    with pytest.raises(TimeoutError):
        augmented(PolynomialFold(),[arb(0),arb(0),arb(1),arb(1),arb(0)],1,tick)


def test_invalid_public_settings_before_model():
    with pytest.raises(ValueError):
        check_fold({'schema':'neural-shrimp-fold-input-v1','period':18},precision=True)


def test_nonnormal_left_border_is_not_right_vector():
    class Map:
        n=2
        def jet_step(self,x,eta):
            p=x[0]-2*x[1]
            return [p+eta-1+p*p+x[1],x[1]*(arb(1)/2)]
    with ctx.workprec(200):
        r=_proof(Map(),[arb(0),arb(0),arb(1),arb(1),arb(0)],1,-60,0,16)
        assert r['passed']
        def restore(b):
            return arb(arb(tuple(b['mid'])),arb(tuple(b['rad'])))
        assert restore(r['left_vector'][0]).contains(1)
        assert restore(r['left_vector'][1]).contains(-2)
        assert restore(r['a']).contains(1) and restore(r['c']).contains(2)


def test_independent_high_precision_differences_check_mixed_columns():
    # Synthetic-only numerical test. Finite differences never enter a proof.
    with ctx.workprec(256):
        inp=synthetic_inputs(True);model=FoldTanhMap(inp);n=model.n
        x=[arb(i+1)/17 for i in range(n)];eta=exact(inp['lr_hidden']);v=[arb(i+2)/23 for i in range(n)]
        _,j,_=augmented(model,x+[eta]+v,2)
        eps=arb((1,-40));tol=arb((1,-65))
        for col in range(n+1):
            values=[]
            for sign in (-1,1):
                xx=list(x);old=BallTanhMap(inp)
                if col<n:xx[col]+=sign*eps
                else:old.rates=[eta+sign*eps]*old.hidden+old.rates[old.hidden:]
                _,m,_,_=old.iterate(xx,2)
                values.append([sum(((m[i,k]-int(i==k))*v[k] for k in range(n)),arb(0)) for i in range(n)])
            for i in range(n):
                assert abs((values[1][i]-values[0][i])/(2*eps)-j[n+i,col]) < tol
