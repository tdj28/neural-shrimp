import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/reviews'))
from flint import arb,arb_mat,ctx
import pytest
import review_correlated_norm_math as M
from neural_shrimp.rigorous_orbits import matrix_json,ball_json


def simple_positive(A,basis):
 # Exact identity bases in synthetic controls. No producer routines imported.
 assert basis==M.identity(A.nrows())
 margins=[A[i,i].lower()-sum((abs(A[i,j]).upper() for j in range(A.ncols()) if j!=i),arb(0)) for i in range(A.nrows())]
 return {'passed':all(v>0 for v in margins),'row_margins':[ball_json(v) for v in margins]}

def records(values):
 n=values[0].nrows();I=matrix_json(M.identity(n));zero=matrix_json(arb_mat(n,n))
 phases=[dict(phase=j,center_J=matrix_json(A),whole_total_Jprime=zero) for j,A in enumerate(values)]
 frames={'cyclic_last_index':0,'witnesses':[dict(phase=j,B=I) for j in range(len(values))]}
 norms=[dict(phase=j,operator_bound=dict(gamma=ball_json(arb(2)),positive=dict(basis=I))) for j in range(len(values))]
 metric=dict(P=I,P_positive=dict(basis=I),complete_D_positive=dict(basis=I))
 return phases,frames,norms,metric

def test_noncommuting_convolution_order():
 A=arb_mat([[1,2],[0,1]]);B=arb_mat([[1,0],[3,1]]);Z=arb_mat(2,2)
 got=M.convolution([A,B],[B,A])
 assert got[0]==A*B and got[1]==A*A+B*B and got[2]==B*A
 assert got[0]!=B*A

def test_full_remainder_covers_exact_product_at_fixed_t():
 A=arb_mat([[1,arb('1/4')],[0,arb('1/2')]]);B=arb_mat([[arb('1/2'),0],[arb('1/3'),1]]);Z=arb_mat(2,2)
 ca=[A,B/8,A/16];cb=[B,A/9,B/18]
 coeff,eps,gamma=M.bound_product((cb,arb('1/100'),arb(2)),(ca,arb('1/200'),arb(2)))
 for t in [arb('-1'),arb('-1/2'),arb(0),arb('1/2'),arb(1)]:
  pa=sum((v*t**k for k,v in enumerate(ca)),Z);pb=sum((v*t**k for k,v in enumerate(cb)),Z)
  full=(pa+M.identity(2)/200)*(pb+M.identity(2)/100)
  truncated=sum((v*t**k for k,v in enumerate(coeff)),Z)
  assert M.frobenius(full-truncated)<eps

def test_polynomial_ranges_cover_negative_and_positive_even_powers():
 A=arb_mat([[1]]);Z=arb_mat(1,1);bound=M.polynomial([Z,Z,A,Z,-A/2])
 for t in [arb('-1'),arb('-1/2'),arb(0),arb('1/2'),arb(1)]:assert bound[0,0].contains(t*t-t*t*t*t/2)

def test_stable_native_cycle_passes_without_producer_math():
 args=records([arb_mat([[arb('1/2')]]),arb_mat([[arb('1/3')]])])
 result=M.audit(*args,'1/128',positive=simple_positive)
 assert result['passed'] and result['precision']==384 and result['native_phases']==2

def test_unstable_native_cycle_fails_decrease():
 args=records([arb_mat([[arb('3/2')]])])
 with pytest.raises(ValueError,match='complete decrease'):M.audit(*args,'1/128',positive=simple_positive)

@pytest.mark.parametrize('defect',['order','cyclic','norm','frame','metric'])
def test_corrupted_prerequisites_fail_closed(defect):
 phases,frames,norms,metric=records([arb_mat([[arb('1/2')]])])
 if defect=='order':phases[0]['phase']=1
 if defect=='cyclic':frames['cyclic_last_index']=1
 if defect=='norm':norms[0]['operator_bound']['gamma']=ball_json(arb('1/4'))
 if defect=='frame':frames['witnesses'][0]['B']=matrix_json(arb_mat([[0]]))
 if defect=='metric':metric['P']=matrix_json(arb_mat([[-1]]))
 with pytest.raises((ValueError,ZeroDivisionError)):M.audit(phases,frames,norms,metric,'1/128',positive=simple_positive)
