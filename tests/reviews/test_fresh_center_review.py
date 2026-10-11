import ast,copy,sys
from pathlib import Path
from fractions import Fraction as F
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import review_fresh_center as A
Q=A.Q


def test_polynomial_exact_ranges_and_noncommuting_convolution():
 with ctx.workprec(480):
  x=arb_mat([[1,2],[3,4]]);y=arb_mat([[2,-1],[0,3]])
  p=Q.polynomial([x,y,x],F(-1,4),F(1,8))
  for t in [F(-1,4),F(0),F(1,8)]:
   e=x+y*Q.R.exact(t)+x*Q.R.exact(t*t)
   assert all(p[i,j].contains(e[i,j])for i in range(2)for j in range(2))
  with pytest.raises(ValueError):Q.polynomial([x],1,0)


def test_conditional_same_root_point_and_tangent():
 with ctx.workprec(480):
  C=[arb_mat([[1]]),arb_mat([[0]])];D=[arb_mat([[arb('0.25')]])];g=arb_mat([[-1]]);old=arb_mat([[arb(0,2)]]);w=[arb(1)]
  e,p=Q.root(D,[arb_mat([[1]])],w,old,old,0,0);true=arb(4)/3;assert e[0,0].contains(true)
  box,proof=Q.point(C,D,g,[arb(0)],[arb(0,2)],w,0);assert box[0].contains(true)
  v,tp=Q.tangent(C,D,[arb_mat([[-1]])],arb_mat([[0]]),[arb(0)],w,0,0);assert v[0].contains(true)
  with pytest.raises(ValueError):Q.defect([arb_mat([[1]])],w,0,0)
  with pytest.raises(ValueError):Q.original([Q.ball_json(arb(0,1))],[arb(0)],[arb(2)])


def test_native_all_state_rate_tangent_domains():
 z=Q.ball_json(arb(0));one=Q.ball_json(arb(1));I=Q.matrix_json(arb_mat([[int(i==j)for j in range(13)]for i in range(13)]));zero=Q.matrix_json(arb_mat(13,13));v=[arb(0)]*705
 inp=dict(kind='whole_tangent',phase=0,state=[z]*13,rates=[z]*2,state_tangent=[z]*13,rate_tangent=[z]*2);out=dict(kind='whole_tangent',phase=0,A=I,Aprime=zero)
 A.native(inp,out,v,v,v,'whole_tangent',0)
 for field in ['state','rates','state_tangent','rate_tangent']:
  bad=copy.deepcopy(inp);bad[field][0]=one
  with pytest.raises(ValueError):A.native(bad,out,v,v,v,'whole_tangent',0)


def test_all705_residual_rebuild_from36_recorded_returns(tmp_path,monkeypatch):
 import probe_native705_fresh_center as P
 class Linear:
  n=13
  def cusp_step(self,x,a,b):return[xi*arb('0.5')for xi in x]
 with ctx.workprec(480):
  monkeypatch.setattr(P,'OUT',tmp_path);w=P.Work();w.calls=216
  x=[arb(0)]*705;x[-3:]=[arb(1),arb(1),arb((1,-6))];x[13]=arb(1);x[27]=arb(1)
  normal=[arb(0)]*705;normal[-3]=arb(1);origin=[arb(0)]*705;parameter=arb(1)
  expected=w.residual(Linear(),x,normal,origin,parameter,'target2');point,value=A.residual(tmp_path,normal,origin,parameter)
  assert len(point)==705 and value.nrows()==705 and all(value[i,0].overlaps(expected[i,0])for i in range(705))
  f=tmp_path/'native/217-input.json';d=A.read(f);d['inputs'][0]['g'][0]=Q.ball_json(arb(0));import json;f.write_text(json.dumps(d))
  with pytest.raises(ValueError):A.residual(tmp_path,normal,origin,parameter)


def test_source_closure_and_no_native_entrypoints():
 for n in A.FILES:assert(ROOT/n).is_file()
 tree=ast.parse((ROOT/'tools/reviews/review_fresh_center.py').read_text())
 called={ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)}
 assert not any(x in called for x in ['B.target','B.synthetic','B.localize_polynomial_root','B.refine_polynomial_tangent','B.correlated_complement','model.cusp_step'])
 assert "400" in (ROOT/'tools/reviews/launch_fresh_center_review.py').read_text()
