import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[2]/'tools/reviews'))
import audit_refined_tangent_saved as M
from flint import arb,arb_mat,ctx

def test_actual_rate_cross_term_twice():
 with ctx.workprec(256):
  f,details=M.forcing([(arb_mat([[-1]]),arb_mat([[-6]]))],arb_mat([[4],[3]]),arb(1),1)
  assert f[0,0]==-48 and f[1,0]==0
  assert details[0][1][0,0]==-8

def test_output_group_has_no_hidden_rate_column():
 f,_=M.forcing([(arb_mat([[0,0],[0,0]]),arb_mat([[0,0],[0,0]]))],arb_mat([[2],[3],[4]]),arb(1),1)
 assert f[0,0]==-8 and f[1,0]==0

def test_norm_composition_noncommuting_and_full_error():
 a=[arb_mat([[1,2],[0,1]]),arb_mat([[0,1],[1,0]]),arb_mat(2,2)]
 b=[arb_mat([[1,0],[3,1]]),arb_mat(2,2),arb_mat(2,2)]
 cs,e,g=M.advance(a,arb('1/8'),arb(4),b,arb('1/4'),arb(5))
 assert cs[0][0,0]==7 and cs[0][1,1]==1 and e>0 and g==20
