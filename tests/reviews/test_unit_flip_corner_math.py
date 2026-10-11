import importlib.util
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
p=Path(__file__).resolve().parents[2]/'tools/reviews/review_unit_flip_corner_math.py'
s=importlib.util.spec_from_file_location('independent_corner_math',p);A=importlib.util.module_from_spec(s);s.loader.exec_module(A)

def fixture(n=7,degenerate=False):
    t=arb(1)/16;a=t.sqrt();c=a.cos();s=a.sin();B=arb_mat(n,n)
    B[0,0]=B[1,1]=c;B[0,1]=s/a;B[1,0]=-a*s;B[2,2]=-1
    for i in range(3,n):B[i,i]=arb(i)/(n+2)
    S=A.eye(n)
    for i in range(n-1):S[i,i+1]=arb(1)/4
    Sinv=S.inv();M=S*B*Sinv;k=3*n+2;dim=4*n+3
    box=[arb(0)for _ in range(dim)];box[n]=box[n+1]=arb(1);box[k]=t
    for i in range(n):box[n+2+i]=S[i,0];box[2*n+2+i]=S[i,1];box[k+1+i]=S[i,2]
    T=arb_mat([[arb((i+2)*(p+1))/23 for p in range(2)]for i in range(n)])
    J=arb_mat(dim,dim);Fp=(A.eye(n)-M)*T;Mx=arb_mat(n,n);Mx[0,0]=arb(3)/7
    K=arb_mat(n,n);K[3,0]=arb(1)/5;K[1,3]=arb(1)/9
    for i in range(n):
        for p in range(2):J[i,n+p]=Fp[i,p]
    for p,(sig,tau)in enumerate([(2,5),(3,arb(15)/2 if degenerate else 7)]):
        D=arb_mat(n,n)
        for i in range(2):
            for j in range(2):D[i,j]=sig*B[i,j]
        # Arbitrary angular derivative has zero logarithmic modulus derivative.
        omega=arb(p+2)/11;D[0,0]-=omega*s;D[1,1]-=omega*s;D[0,1]+=omega*c/a;D[1,0]-=omega*a*c;D[2,2]=tau
        total=S*D*Sinv+arb(p+1)/13*(K*M-M*K);partial=total-Mx*T[0,p]
        for col,offset in enumerate((n,2*n,k)):
            v=arb_mat([[S[i,col]]for i in range(n)]);dx=Mx*v;dp=partial*v
            for i in range(n):J[offset+i,0]=dx[i,0];J[offset+i,n+p]=dp[i,0]
    return M,J,box

def test_complex_nonnormal_total_rate_derivatives():
    with ctx.workprec(384):
        z=A.physical_gradients(*fixture());assert z['passed']
        for i,row in enumerate([[2,3],[5,7]]):
            for j,v in enumerate(row):assert z['physical_gradient'][i,j].contains(v)and z['physical_gradient'][i,j].rad()<arb((1,-300))
        assert z['determinant'].contains(-1)and z['flip_derivative_along_unit_hidden'].contains(arb(1)/3)

def test_actual_state_motion_is_required():
    with ctx.workprec(256):
        m,j,b=fixture();correct=A.physical_gradients(m,j,b)['physical_gradient']
        for i in range(m.nrows()):
            for p in range(2):j[i,m.nrows()+p]=0
        wrong=A.physical_gradients(m,j,b)['physical_gradient'];assert not correct[0,0].overlaps(wrong[0,0])

def test_invalid_critical_mode_rejected():
    with ctx.workprec(256):
        m,j,b=fixture();b[m.nrows()+2]+=arb(1)/10
        with pytest.raises(ValueError):A.physical_gradients(m,j,b)

def test_full13_and_degenerate_determinant():
    with ctx.workprec(256):
        assert A.physical_gradients(*fixture(13))['complex_border'].nrows()==28
        assert not A.physical_gradients(*fixture(degenerate=True))['passed']

def test_border_and_inverse_reject_singular():
    with pytest.raises((ValueError,ZeroDivisionError)):A.inverse(arb_mat([[1,1],[1,1]]))
