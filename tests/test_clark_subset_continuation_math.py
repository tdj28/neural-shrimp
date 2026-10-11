import sys
from pathlib import Path
from fractions import Fraction as F
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import clark_subset_continuation_math as H

def fixture():
 C=[arb_mat([[1,0],[0,1]]),arb_mat(2,2)];J=[arb_mat([[1,-2],[0,1]]),arb_mat(2,2),arb_mat(2,2)];R=[arb_mat([[0],[arb(1)/100]])]+[arb_mat(2,1)]*3;r=[arb(1)]*2;z=[arb(0)]*2;data=H.T.reduce(C,J,R,r,arb(1)/16)
 return C,J,R,data,r,z,arb(0),z,z,z,arb(0),arb(1)/16

def test_nonnormal_full_error_restricts_and_retains_proposal_before_result():
 with ctx.workprec(384):
  f=fixture();events=[];answer=H.restrict(*f,observe=lambda k,v:events.append((k,v)))
  assert answer is not None and [x[0]for x in events]==['bounds','proposal','verification','reduced']
  whole,ends,data,proof,new=answer
  assert data['passed']and all(0<a<=b for a,b in zip(new,f[4]))and proof['known_join']['passed']
  assert proof['actual_error']==H.vector_json([arb(0,1)]*2)and proof['radii']!=proof['acquisition_radii']

def test_entire_known_box_and_scalar_not_just_midpoint():
 with ctx.workprec(384):
  f=list(fixture());f[5]=[arb(0),arb(0,arb(3)/4)];events=[]
  assert H.restrict(*f,observe=lambda k,v:events.append(k))is None
  assert events[:2]==['bounds','proposal']
  f=list(fixture());f[6]=arb(1)
  with pytest.raises(ValueError,match='known scalar'):H.restrict(*f)

def test_outside_subset_solver_and_already_passed_reject(monkeypatch):
 with ctx.workprec(384):
  f=list(fixture());f[3]=dict(f[3],passed=True)
  with pytest.raises(ValueError,match='completed original'):H.restrict(*f)
  f=fixture()
  def bad(A,beta,anchor,observe):
   p={'success':True,'x':[2,2]};observe(p);return dict(proposal=p,verification=H.S.verify(A,beta,anchor,[F(2)]*2))
  monkeypatch.setattr(H.S,'propose',bad);assert H.restrict(*f)is None

def test_exact_dyadic_rounding_and_signed_known_projection():
 with ctx.workprec(384):
  x=F(7,2**50);assert H.exact(H.dyadic(x))==x
  with pytest.raises(ValueError,match='dyadic'):H.dyadic(F(1,3))
  f=fixture();args=list(f[3:]);args[2]=[arb(2,arb(1)/8),arb(0)];args[4]=[arb(2),arb(0)]
  A,beta,anchor=H.envelopes(*args);assert F(1,8)<=anchor[0]<F(126,1000)
