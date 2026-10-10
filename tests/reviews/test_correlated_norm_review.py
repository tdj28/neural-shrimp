import sys
from pathlib import Path
from copy import deepcopy
import pytest
from flint import arb,arb_mat,ctx
sys.path.insert(0,str(Path(__file__).parents[2]/'tools/reviews'))
from neural_shrimp.rigorous_norm_cocycle import framed_cocycle
from neural_shrimp.rigorous_orbits import matrix_json,ball_json
from review_correlated_norm_math import audit,multiply,interval_polynomial

def fixture():
 with ctx.workprec(256):
  rows=[{'phase':0,'center_J':arb_mat([['.6','.12'],['-.08','.5']]),'whole_Jprime':arb_mat([[arb('.03','.0001'),'.02'],['.01','-.04']])},{'phase':1,'center_J':arb_mat([['.4','-.17'],['.09','.55']]),'whole_Jprime':arb_mat([['-.02','.03'],['-.04','.01']])}]
  events=[];r=framed_cocycle(rows,[arb_mat([[1,0],[0,1]])]*2,arb('1/32'),observe=lambda k,v:events.append((k,v)))
  assert r['passed']
  phases=[{'phase':v['phase'],'center_J':matrix_json(v['center_J']),'whole_total_Jprime':matrix_json(v['whole_Jprime'])} for v in rows]
  return phases,next(v for k,v in events if k=='frames'),[v for k,v in events if k=='norm_phase'],next(v for k,v in events if k=='norm_metric')

def test_independent_noncommuting_variable_product_passes():
 args=fixture();assert audit(*args,'1/32')['passed']

def test_all_product_degrees_and_order_retained():
 with ctx.workprec(384):
  a=[arb_mat([[1,2],[0,1]]),arb_mat([[0,1],[2,0]]),arb_mat([[1,0],[0,2]])];b=[arb_mat([[2,0],[3,1]]),arb_mat([[1,0],[0,-1]]),arb_mat([[0,1],[1,0]])]
  q=multiply(a,b)
  for t in [arb('-1'),arb('-.25'),arb('.375'),arb('1')]:
   left=sum((a[i]*(t**i) for i in range(3)),arb_mat(2,2));right=sum((b[i]*(t**i) for i in range(3)),arb_mat(2,2));poly=sum((q[i]*(t**i) for i in range(5)),arb_mat(2,2))
   assert all((poly-left*right)[i,j].contains(0) for i in range(2) for j in range(2))
   bound=interval_polynomial(q);assert all(bound[i,j].contains(poly[i,j]) for i in range(2) for j in range(2))

@pytest.mark.parametrize('kind',['gamma','metric','order','frame','unstable'])
def test_unsound_claims_fail_closed(kind):
 phases,frames,norms,metric=fixture()
 if kind=='gamma':norms[0]['operator_bound']['gamma']=ball_json(arb('1/100'))
 if kind=='metric':metric['P'][0][0]=ball_json(arb(-1))
 if kind=='order':phases.reverse()
 if kind=='frame':frames['witnesses'][0]['B']=matrix_json(arb_mat([[0,0],[0,0]]))
 if kind=='unstable':phases[0]['center_J']=matrix_json(arb_mat([[3,0],[0,3]]))
 with pytest.raises((ValueError,ZeroDivisionError)):audit(phases,frames,norms,metric,'1/32')
