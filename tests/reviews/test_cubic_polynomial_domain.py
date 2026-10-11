import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'src')]
from fractions import Fraction as F
from flint import arb,ctx
import pytest
import review_cubic_polynomial_domain as H

def a(q):return arb(F(q).numerator)/arb(F(q).denominator)

def test_signed_crosszero_and_zero_endpoint_exact_polynomials():
 with ctx.workprec(480):
  c,s,v,b=[list(map(a,row))for row in([3,-2,0],[2,-5,3],[-4,2,0],[3,-2,-5])]
  for lo,hi in [(F(-1,64),F(0)),(F(0),F(1,64)),(F(-1,64),F(1,32)),(F(-1,64),F(-1,64))]:
   p,t=H.polynomial_domain(c,s,v,b,lo,hi)
   for j in range(17):
    d=a(lo+(hi-lo)*j/16)
    for i in range(3):
     assert p[i].contains(c[i]+s[i]*d+v[i]*d*d/2+b[i]*d*d*d)
     assert t[i].contains(s[i]+v[i]*d+3*b[i]*d*d)

def test_exact_even_power_avoids_spurious_sign_on_negative_interval():
 with ctx.workprec(480):
  p,t=H.polynomial_domain([arb(0)],[arb(0)],[arb(2)],[arb(0)],F(-1,128),F(0))
  lo,hi=H.Q.M.interval(H.Q.ball_json(p[0]));assert lo>=-F(1,2**40)and hi>=F(1,128**2)
  naive=arb(a(F(-1,256)),a(F(1,256)));old=naive*naive
  assert H.Q.M.interval(H.Q.ball_json(old))[0]<lo

def test_actual_full_dimension_and_invalid_coefficients():
 with ctx.workprec(480):
  n=705;z=[arb(0)]*n;p,t=H.polynomial_domain(z,[arb(1)]*n,z,z,F(-1,256),F(1,256));assert len(p)==len(t)==705 and all(v==1 for v in t)
  with pytest.raises(ValueError):H.polynomial_domain([arb(0,1)],[arb(0)],[arb(0)],[arb(0)],0,1)
  with pytest.raises(ValueError):H.polynomial_domain(z,z,z,z,1,-1)
