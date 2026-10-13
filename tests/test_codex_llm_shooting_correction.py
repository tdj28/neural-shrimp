import sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx,fmpq,fmpq_mat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'codex/certify-llm'))
from shooting_correction import correct

@pytest.fixture(autouse=True)
def precision():
    old=ctx.prec;ctx.prec=256
    yield
    ctx.prec=old

def test_two_phase_linear_sign_and_phase_order():
    centers=[[arb(3)],[arb(-2)]]
    A=[arb_mat([[0]]),arb_mat([[arb(1)/2]])]
    values=[[arb(1)],[centers[1][0]/2-arb(1)/4]]
    result=correct(centers,values,A,arb_mat([[1]]))
    assert result['centers']==[[arb(1)/4],[arb(1)]]
    assert result['record']['certificate_passed'] is False

def test_four_phase_nonnormal_correction_matches_exact_dense_system():
    N=fmpq_mat([[0,fmpq(1,2)],[0,0]]);I=fmpq_mat([[1,0],[0,1]])
    A=[I,I,I,N];B=I+N;p=4;n=2
    centers=[[arb(i-2),arb(i+1)] for i in range(p)]
    constants=[[fmpq(i,2),fmpq(-i,4)] for i in range(p)]
    images=[]
    dense=fmpq_mat(p*n,p*n);rhs=fmpq_mat(p*n,1)
    for i in range(p):
        images.append([sum((arb(A[i][j,k])*centers[i][k] for k in range(n)),arb(0))+arb(constants[i][j]) for j in range(n)])
        for j in range(n):
            rhs[i*n+j,0]=-constants[i][j]
            for k in range(n):dense[i*n+j,i*n+k]=A[i][j,k]
            dense[i*n+j,((i+1)%p)*n+j]=-1
    solution=dense.inv()*rhs
    result=correct(centers,images,[arb_mat(a) for a in A],arb_mat(B))
    for i in range(p):
        for j in range(n):assert result['centers'][i][j]==arb(solution[i*n+j,0])

def test_uncertain_images_retained_but_new_centers_are_points():
    result=correct([[arb(0)],[arb(0)]],[[arb(1,(1,-100))],[arb(2,(1,-101))]],
                   [arb_mat([[0]]),arb_mat([[0]])],arb_mat([[1]]))
    assert all(v.is_exact() for phase in result['centers'] for v in phase)
    assert any(v['rad'][0]>0 for phase in result['record']['corrections'] for v in phase)

@pytest.mark.parametrize('bad', [arb('nan'),arb('inf'),arb(0,(1,-50)),0.0])
def test_nonpoint_or_nonarb_centers_refused(bad):
    with pytest.raises(ValueError):correct([[bad],[arb(0)]],[[arb(0)],[arb(0)]],
                    [arb_mat([[0]]),arb_mat([[0]])],arb_mat([[1]]))

def test_inexact_preconditioner_refused():
    with pytest.raises(ValueError):correct([[arb(0)],[arb(0)]],[[arb(0)],[arb(0)]],
                    [arb_mat([[0]]),arb_mat([[0]])],arb_mat([[arb(1,(1,-50))]]))

def test_missing_phase_refused():
    with pytest.raises(ValueError):correct([[arb(0)],[arb(0)]],[[arb(0)]],
                    [arb_mat([[0]]),arb_mat([[0]])],arb_mat([[1]]))

def test_nonfinite_image_refused():
    with pytest.raises(ValueError):correct([[arb(0)],[arb(0)]],[[arb('nan')],[arb(0)]],
                    [arb_mat([[0]]),arb_mat([[0]])],arb_mat([[1]]))
