import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.unit_flip_corner import gradients,block
from neural_shrimp import junction_audit as A

def fixture(n=6,degenerate=False):
    k=3*n+2;dim=4*n+3;t=arb(1)/16;a=t.sqrt();c=a.cos();e=a.sin()/a;d=t*e
    M=arb_mat(n,n)
    for i,row in enumerate([[c,e,0],[-d,c,0],[0,0,-1]]):
        for j,z in enumerate(row):M[i,j]=z
    for j in range(3,n):
        M[j,j]=arb(j+1)/(n+5)
        for i in range(3):M[i,j]=arb((i+1)*(j+1))/17
    box=[arb(0)for _ in range(dim)];box[n]=box[n+1]=arb(1);box[n+2]=arb(1);box[2*n+3]=arb(1);box[k+3]=arb(1);box[k]=t
    U=arb_mat([[arb((i+1)*(j+1))/19 for j in range(2)]for i in range(n)])
    J=arb_mat(dim,dim);Fp=(A.eye(n)-M)*U
    for i in range(n):
        for j in range(n):J[i,j]=M[i,j]-int(i==j)
        for p in range(2):J[i,n+p]=Fp[i,p]
    C=arb_mat(n,n);C[3,0]=arb(1)/7;C[1,4]=arb(1)/5
    Mx=arb_mat(n,n);Mx[0,0]=arb(1)/8
    for p,(sigma,tau)in enumerate([(2,5),(3,arb(15)/2 if degenerate else 7)]):
        actual=arb(p+1)/3*(C*M-M*C)
        for i in range(2):
            for j in range(2):actual[i,j]+=sigma*M[i,j]
        actual[2,2]+=tau
        partial=actual-Mx*U[0,p]
        for j,offset in enumerate((n,2*n,k)):
            for i in range(n):J[offset+i,0]=Mx[i,j];J[offset+i,n+p]=partial[i,j]
    return M,J,box,list(range(3,n))

def test_nonnormal_physical_total_derivatives_and_orientation():
    with ctx.workprec(256):
        z=gradients(*fixture());assert z['passed']
        G=z['physical_gradient']
        for i,row in enumerate([[2,3],[5,7]]):
            for j,x in enumerate(row):assert G[i,j].contains(x)and G[i,j].rad()<arb((1,-180))
        assert z['determinant'].contains(-1)and z['flip_derivative_along_unit_hidden'].contains(arb(1)/3)
        assert any(not z['left_critical'][i,j].contains(0)for i in range(3)for j in range(3,6))

def test_degenerate_transversality_stays_unproved():
    with ctx.workprec(256):assert not gradients(*fixture(degenerate=True))['passed']

def test_full13_profile_dimension():
    with ctx.workprec(256):
        z=gradients(*fixture(13));assert z['passed']and z['sylvester'].nrows()==30

def test_rejects_wrong_critical_block_and_frame():
    with ctx.workprec(256):
        M,J,b,c=fixture();M[0,0]+=1
        with pytest.raises(ValueError):gradients(M,J,b,c)
        M,J,b,c=fixture()
        with pytest.raises((ValueError,ZeroDivisionError)):gradients(M,J,b,[0,1,2])

def test_total_derivative_is_not_frozen_parameter_derivative():
    with ctx.workprec(256):
        M,J,b,c=fixture();right=gradients(M,J,b,c)['physical_gradient']
        for i in range(6):
            for p in range(2):J[i,6+p]=0
        wrong=gradients(M,J,b,c)['physical_gradient'];assert not right[0,0].overlaps(wrong[0,0])
