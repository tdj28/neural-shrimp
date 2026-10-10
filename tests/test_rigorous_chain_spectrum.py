import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_chain_spectrum import chain_complement


def example(extra):
    # Nonnormal real similarity prevents a coordinate-diagonal shortcut.
    s=arb_mat([[1,2,3,0],[0,1,-1,2],[2,0,1,1],[1,-1,0,1]])
    j=arb_mat([[1,1,0,0],[0,1,0,0],[0,0,extra,0],[0,0,0,arb(1)/4]])
    m=s*j*s.inv();q=[s[i,0]for i in range(4)];r=[s[i,1]for i in range(4)]
    return m,q,r

@pytest.mark.parametrize('extra,passed,attracting',[(arb(1)/2,True,True),(arb(2),True,False),(arb(1),False,False),(arb(-1),False,False)])
def test_full_spectrum_separates_extra_neutral_and_unstable(extra,passed,attracting):
    with ctx.workprec(256):
        result=chain_complement(*example(extra))
        assert result['passed']==passed and result['complement_attracting']==attracting


def test_interval_nonnormal_frame_and_monodromy():
    with ctx.workprec(256):
        m,q,r=example(arb(1)/2);radius=arb((1,-80))
        m=arb_mat([[arb(m[i,j].mid(),radius)for j in range(4)]for i in range(4)])
        result=chain_complement(m,[arb(v.mid(),radius)for v in q],[arb(v.mid(),radius)for v in r])
        assert result['passed'] and result['complement_attracting']


def test_wrong_chain_or_singular_frame_cannot_qualify():
    with ctx.workprec(256):
        m,q,r=example(arb(1)/2)
        with pytest.raises(ValueError):chain_complement(m,q,[v*2 for v in r])
        with pytest.raises(ZeroDivisionError):chain_complement(m,q,q)
