import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_unit_pair_spectrum import native_complement,invariant_frame
from neural_shrimp.rigorous_unit_pair import entire_pair
from neural_shrimp.rigorous_orbits import eye,exact


def fixture(n=4,unstable=False,t='1/64'):
    vs=[];flat=[];blocks=[];c,e,d=entire_pair(exact(t))
    for k in range(2):
        V=eye(n);V[2,0]=exact(str(k+1)+'/8');V[2,1]=exact('-1/4');V[0,2]=exact('1/3');V[1,2]=exact(str(k+1)+'/16')
        vs.append(V);flat.extend([arb(0)]*n+[V[i,0]for i in range(n)]+[V[i,1]for i in range(n)])
    flat.extend([arb(1)]*2)
    for k in range(2):
        T=eye(n)
        if k==1:T[0,0]=c;T[0,1]=e;T[1,0]=-d;T[1,1]=c
        for j in range(2,n):
            T[j,j]=exact('2'if unstable and j==2 else'1/2');T[0,j]=exact('1/5');T[1,j]=exact('-1/7')
        blocks.append(vs[(k+1)%2]*T*vs[k].inv())
    return blocks,flat,exact(t)

@pytest.mark.parametrize('t',['0','1/64'])
def test_nonorthogonal_phase_frames_remove_critical_plane_before_product(t):
    with ctx.workprec(256):
        blocks,flat,tt=fixture(t=t);out=native_complement(blocks,flat,2,4,tt)
        assert out['passed']and out['complement_attracting']and len(out['spectral_exclusion']['discs'])==2
        assert len(out['frames'])==len(out['quotient_blocks'])==2

def test_unstable_complement_not_mislabelled_attracting():
    with ctx.workprec(256):
        b,z,t=fixture(unstable=True);r=native_complement(b,z,2,4,t)
        assert r['passed']and not r['complement_attracting']

def test_wrong_native_mode_state_derivative_or_phase_is_rejected():
    with ctx.workprec(256):
        b,z,t=fixture();z[4]+=1
        with pytest.raises(ValueError):native_complement(b,z,2,4,t)
        b,z,t=fixture();b[0][2,0]+=1
        with pytest.raises(ValueError):native_complement(b,z,2,4,t)
        b,z,t=fixture()
        with pytest.raises(ValueError):native_complement(list(reversed(b)),z,2,4,t)

def test_dependent_modes_do_not_supply_a_valid_frame():
    with pytest.raises((ZeroDivisionError,ValueError)):invariant_frame([arb(1),arb(0)],[arb(2),arb(0)])

def test_all11_complementary_modes_are_retained():
    with ctx.workprec(256):
        b,z,t=fixture(n=13);r=native_complement(b,z,2,13,t)
        assert r['complement_attracting']and len(r['spectral_exclusion']['discs'])==11
