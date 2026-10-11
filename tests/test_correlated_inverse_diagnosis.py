from flint import arb,arb_mat,ctx
import pytest
from fractions import Fraction as F
from neural_shrimp.correlated_inverse_diagnosis import interval_power,polynomial,weighted_norm

def test_exact_integer_ranges_across_zero_and_one_sided():
 with ctx.workprec(384):
  for a,b in [(F(-1,16),F(1,32)),(F(-1,16),F(-1,32)),(F(1,32),F(1,16))]:
   for k in range(6):
    z=interval_power(a,b,k)
    for t in [a,(a+b)/2,b]:assert z.contains(arb(t.numerator)**k/arb(t.denominator)**k)
    assert z.is_finite()
  even=interval_power('-1/16','1/16',2)
  assert even.contains(0) and even.contains(arb(1)/256)
  # Arb's fixed-width radius rounds outward, so an enclosure of [0,h²]
  # may have a tiny negative lower endpoint despite the exact scalar range.
  assert even.lower()>-arb(1)/2**30

def test_polynomial_correlation_and_weighted_defect():
 with ctx.workprec(192):
  D=[arb_mat([[0,arb(1)/32],[0,0]]),arb_mat([[1,0],[0,-1]])]
  q,rows=weighted_norm(polynomial(D,'-1/64','1/64'),[arb(1),arb(2)])
  assert q<arb(1)/8 and len(rows)==2
  with pytest.raises(ValueError):weighted_norm(D[0],[arb(0),arb(1)])
  with pytest.raises(ValueError):interval_power('-1','1',6)
