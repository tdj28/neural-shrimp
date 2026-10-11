"""Exact rational ranges for a fixed cubic and its quadratic derivative.

Derivative extrema are exact. A monotone cubic uses endpoint extrema; every
other cubic uses eight fixed Bernstein convex hulls. Acquired domain bounds are
never used to construct these enclosures.
"""
from fractions import Fraction as F
from flint import arb
import review_polynomial_center_math as Q


def quadratic_range(s,a,b,l,r):
    """Exact extrema of s+a*x+b*x*x on the closed rational interval."""
    s,a,b,l,r=map(F,(s,a,b,l,r));Q.need(l<=r,'ordered rational interval')
    def value(x):return s+a*x+b*x*x
    values=[value(l),value(r)]
    if b:
        vertex=-a/(2*b)
        if l<=vertex<=r:values.append(value(vertex))
    return min(values),max(values)


def exact_ranges(c,s,a,b,l,r):
    """c+s*x+(a/2)*x*x+b*x*x*x and its exact derivative."""
    c,s,a,b,l,r=map(F,(c,s,a,b,l,r));Q.need(l<=r,'ordered closed scalar set')
    def value(x):return c+s*x+a*x*x/2+b*x*x*x
    def derivative(x):return s+a*x+3*b*x*x
    qlo,qhi=quadratic_range(s,a,3*b,l,r)
    if qlo>=0 or qhi<=0:
        values=[value(l),value(r)];return (min(values),max(values)),(qlo,qhi),'monotone_endpoints'
    # For each interval [u,v], these are the exact degree3 Bernstein
    # coefficients of p(u+(v-u)t), t in[0,1]. Their convex hull encloses p.
    values=[]
    for k in range(8):
        u=l+(r-l)*k/8;v=l+(r-l)*(k+1)/8;h=v-u
        values.extend((value(u),value(u)+h*derivative(u)/3,value(v)-h*derivative(v)/3,value(v)))
    return (min(values),max(values)),(qlo,qhi),'eight_bernstein_hulls'


def enclosing_ball(l,r):
    l,r=F(l),F(r);Q.need(l<=r,'ordered exact bounds')
    value=arb(Q.R.exact((l+r)/2),Q.R.exact((r-l)/2))
    a,b=Q.M.interval(Q.ball_json(value))
    Q.need(value.is_finite()and a<=l<=r<=b,'outward ball contains both exact rational bounds')
    return value


def bounded_domain(c,s,a,b,left,right):
    n=len(c);Q.need(1<=n<=705 and len(s)==len(a)==len(b)==n,'complete cubic coefficient columns')
    Q.need(all(v.is_exact()and v.is_finite()for row in(c,s,a,b)for v in row),'fixed finite exact coefficients')
    coeff=[[Q.M.point(Q.ball_json(v))for v in row]for row in(c,s,a,b)]
    p=[];dp=[];evidence=[]
    for i in range(n):
        pr,dr,method=exact_ranges(*(row[i]for row in coeff),left,right)
        p.append(enclosing_ball(*pr));dp.append(enclosing_ball(*dr));evidence.append(dict(coordinate=i,method=method,predictor_bounds=list(map(str,pr)),derivative_bounds=list(map(str,dr))))
    return p,dp,evidence


def polynomial_domain(c,s,a,b,left,right):
    p,dp,_=bounded_domain(c,s,a,b,left,right);return p,dp


def candidates(c,s,a,b,left,right):
    """Compatibility diagnostic: exact-rational bounds and valid nested derivative."""
    p,dp,_=bounded_domain(c,s,a,b,left,right)
    d=arb(Q.R.exact((F(left)+F(right))/2),Q.R.exact((F(right)-F(left))/2))
    nested=[s[i]+a[i]*d+(3*b[i]*d)*d for i in range(len(c))]
    return p,dp,nested
