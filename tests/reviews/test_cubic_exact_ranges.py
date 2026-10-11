import sys
from pathlib import Path
from fractions import Fraction as F
from flint import arb,ctx
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import review_cubic_polynomial_domain as C


def test_exact_quadratic_vertex_and_endpoint_extrema():
    assert C.quadratic_range(1,-4,2,0,2)==(F(-1),F(1))
    assert C.quadratic_range(-1,4,-2,0,2)==(F(-1),F(1))
    assert C.quadratic_range(1,2,0,-1,3)==(F(-1),F(7))
    assert C.quadratic_range(1,0,2,2,2)==(F(9),F(9))
    with pytest.raises(ValueError):C.quadratic_range(0,0,0,1,0)


def test_monotone_cubic_has_exact_endpoint_range():
    p,d,m=C.exact_ranges(1,2,0,3,F(-1,64),F(1,32))
    assert m=='monotone_endpoints'
    assert p==(1+2*F(-1,64)+3*F(-1,64)**3,1+2*F(1,32)+3*F(1,32)**3)
    assert d[0]==2 and d[1]==2+9*F(1,32)**2


def test_nonmonotone_bernstein_hulls_cover_exact_cubic_values():
    for coeff in[(0,-3,0,1),(2,0,-6,1),(-1,1,4,-3)]:
        p,d,m=C.exact_ranges(*coeff,-2,2);assert m=='eight_bernstein_hulls'
        for j in range(257):
            x=F(-2)+F(4*j,256);c,s,a,b=map(F,coeff)
            assert p[0]<=c+s*x+a*x*x/2+b*x*x*x<=p[1]
            assert d[0]<=s+a*x+3*b*x*x<=d[1]
    assert C.exact_ranges(0,-3,0,1,-2,2)[0][0]<=-2
    assert C.exact_ranges(0,-3,0,1,-2,2)[0][1]>=2


def test_entire705_signed_dyadic_and_nondyadic_output_bounds():
    with ctx.workprec(480):
        c=[arb(0)]*705;s=[arb(1)]*705;a=[arb(-2)]*705;b=[arb(3)]*705
        p,dp,e=C.bounded_domain(c,s,a,b,F(-1,128),F(0));assert len(p)==len(dp)==len(e)==705
        for row,x,y in zip(e,p,dp):
            lo,hi=C.Q.M.interval(C.Q.ball_json(x));l,r=map(F,row['predictor_bounds']);assert lo<=l<=r<=hi
            lo,hi=C.Q.M.interval(C.Q.ball_json(y));l,r=map(F,row['derivative_bounds']);assert lo<=l<=r<=hi
        q=C.enclosing_ball(F(-1,3),F(2,7));lo,hi=C.Q.M.interval(C.Q.ball_json(q));assert lo<=F(-1,3)<F(2,7)<=hi
        with pytest.raises(ValueError):C.bounded_domain([arb(0,1)],[arb(0)],[arb(0)],[arb(0)],0,1)
