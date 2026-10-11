"""Independent exact-power enclosures for a fixed cubic predictor and derivative."""
from fractions import Fraction as F
from flint import arb_mat
import review_polynomial_center_math as Q


def polynomial_domain(c,s,a,b,left,right):
    n=len(c)
    Q.need(1<=n<=705 and len(s)==len(a)==len(b)==n,'full matched cubic predictor')
    Q.need(all(v.is_exact()and v.is_finite()for row in(c,s,a,b)for v in row),'fixed exact cubic coefficients')
    Q.need(F(left)<=F(right),'ordered exact scalar subset')
    columns=[arb_mat([[v]for v in row])for row in(c,s,a,b)]
    # The cubic convention is c+s*d+(a/2)*d^2+b*d^3. Apply
    # the independently implemented parity-aware monomial range, separately
    # to this exact polynomial and its exact derivative. No interval remainder
    # is differentiated; both statements enclose the same original scalar d.
    predictor=Q.polynomial([columns[0],columns[1],columns[2]/2,columns[3]],left,right)
    derivative=Q.polynomial([columns[1],columns[2],3*columns[3]],left,right)
    return [predictor[i,0]for i in range(n)],[derivative[i,0]for i in range(n)]
