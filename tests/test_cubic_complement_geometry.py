from fractions import Fraction as F
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.cubic_complement_geometry import predictor,tangent,residual_derivative,domains,cell,children,complete

def test_cubic_predictor_and_tangent_on_signed_domains():
 with ctx.workprec(256):
  c,s,a,b=[arb(3)],[arb(-2)],[arb(7)],[arb(11)]
  for d in [arb(0),arb(-1)/4,arb(1)/2,arb(0,arb(1)/4)]:
   p=predictor(c,s,a,b,d)[0];v=tangent(s,a,b,d)[0]
   for t in [arb(0),arb(-1)/4,arb(1)/4,arb(1)/2]:
    if not d.contains(t):continue
    assert p.contains(3-2*t+7*t*t/2+11*t*t*t)
    assert v.contains(-2+7*t+33*t*t)

def test_residual_derivative_retains_full_fourth_order():
 v=[arb_mat([[i],[i+2]])for i in range(5)];r=residual_derivative(v)
 assert len(r)==4 and all(r[k]==(k+1)*v[k+1]for k in range(4))
 with pytest.raises(ValueError):residual_derivative(v[:-1])

def ok(c):return dict(cell=c,passed=True,completed=True,native_calls=126,dimension=705,complement_dimension=11)

def test_exact_generic_cover_and_fail_closed_subdivision():
 d=domains('39/1024','55/1024',2);accepted=[ok(cell(d,1)),ok(cell(d,2))]
 assert complete(d,accepted)['endpoints']==['39/1024','55/1024']
 failed={**ok(cell(d,1)), 'passed':False};kids=children(d,failed['cell'],failed)
 assert complete(d,[*[ok(c)for c in kids],accepted[1]])['accepted_cells']==3
 for rows in [accepted[:1],accepted+[accepted[0]]]:
  with pytest.raises(ValueError):complete(d,rows)
 for k,v in [('native_calls',125),('completed',False),('passed',True)]:
  with pytest.raises(ValueError):children(d,failed['cell'],{**failed,k:v})
