"""Analytic squared-amplitude fold equations for the fixed-weight Clark map.

x±=m±sqrt(b)q, G=-F³. Even/odd paired equations extend through b=0
without divided differences or a small-amplitude exclusion. This source-only
helper establishes no exact root or branch identity by itself.
"""
from math import factorial
from fractions import Fraction
from flint import arb,arb_mat
from .rigorous_folds import MixedJet
from .rigorous_orbits import eye,midpoint_matrix,norm_inf_upper,ball_json,matrix_json,vector_json


def require(v,message):
 if not v:raise ValueError(message)


def rational(q):return arb(str(q))

def unary(x,value,first,second):
 return MixedJet(value,[first*v for v in x.g],first*x.d,[first*m+second*g*x.d for m,g in zip(x.m,x.g,strict=True)])


def reciprocal(x):
 require(not x.f.contains(0),'nonzero analytic pair denominator')
 return unary(x,1/x.f,-1/(x.f*x.f),2/(x.f*x.f*x.f))


def entire(z,offset,derivative=0):
 """C(z)=cosh(sqrt(z)), S(z)=sinh(sqrt(z))/sqrt(z), derivatives0..2."""
 require(offset in(0,1)and derivative in(0,1,2)and z.is_finite()and abs(z).upper()<=16,'bounded entire pair function')
 result=arb(0)
 for k in range(derivative,25):
  coefficient=rational(Fraction(factorial(k),factorial(k-derivative)*factorial(2*k+offset)))
  result+=coefficient*z**(k-derivative)
 # At k>=25, j<=2, |z|<=16, consecutive absolute terms have ratio
 # <=(26/24)*16/(51*52)<1/2, so twice the first omitted term bounds the tail.
 tail=2*rational(Fraction(factorial(25),factorial(25-derivative)*factorial(50+offset)))*abs(z).upper()**(25-derivative)
 return result+arb(0,tail)


def entire_jet(z,offset):return unary(z,*[entire(z.f,offset,j)for j in range(3)])


def tanh_pair(m,q,b):
 u=2*m;value=u.f.sinh();sinh=unary(u,value,u.f.cosh(),value);value=u.f.cosh();cosh=unary(u,value,u.f.sinh(),value)
 z=4*b*q*q;inverse=reciprocal(cosh+entire_jet(z,0))
 return sinh*inverse,2*q*entire_jet(z,1)*inverse


def paired_family(j,z,b,observe=lambda *_:None):
 """25 equations, z=(m12,leak,q12,gain), external squared amplitude b."""
 n=len(j);require(1<=n<=12 and len(z)==2*n+2 and all(len(row)==n for row in j),'bounded complete Clark pair')
 m0,d,q0,g=z[:n],z[n],z[n+1:-1],z[-1];m=list(m0);q=list(q0)
 for phase in range(3):
  pairs=[tanh_pair(x,v,b)for x,v in zip(m,q,strict=True)]
  m=[(1-d)*m[i]+d*g*sum((j[i][k]*pairs[k][0]for k in range(n)),0)for i in range(n)]
  q=[(1-d)*q[i]+d*g*sum((j[i][k]*pairs[k][1]for k in range(n)),0)for i in range(n)]
  observe('native_even_odd_phase',dict(phase=phase,even=serial(m),odd=serial(q)))
 return [-m[i]-m0[i]for i in range(n)]+[-q[i]+q0[i]for i in range(n)]+[(sum((x*x for x in q0),0)-1)*rational(Fraction(1,2))]


def serial(x):
 if isinstance(x,MixedJet):return dict(f=ball_json(x.f),g=vector_json(x.g),d=ball_json(x.d),m=vector_json(x.m))
 return [serial(v)for v in x]


def finite(A):return all(v.is_finite()for row in A.tolist()for v in row)


def augment(family,z,b,n,observe=lambda *_:None):
 """H(U,g,b)=0 and leak_b=0, all26 columns and external-b derivative.

U=(m,leak,q) has25 components, g is the actual gain. H_U inverse
uncertainty is retained in U_b and its implicit derivatives.
"""
 N=2*n+2;K=N-1;require(1<=n<=12 and len(z)==N and all(v.is_finite()for v in z+[b]),'full finite amplitude-fold variables')
 values=z+[b];first=[MixedJet.variable(v,N+1,i)for i,v in enumerate(values)];h=family(first[:-1],first[-1]);require(len(h)==K and all(len(v.g)==N+1 for v in h),'all actual paired equations/columns')
 A=arb_mat([[v.g[k]for k in range(K)]for v in h]);hb=arb_mat([[v.g[N]]for v in h]);require(finite(A)and finite(hb),'finite paired Jacobian and amplitude derivative')
 # A is nonsingular throughout the supplied box, not merely at its midpoint.
 C=midpoint_matrix(midpoint_matrix(A).inv());defect=eye(K)-C*A;bound=norm_inf_upper(defect);require(bound.is_finite()and bound<1,'strict full paired inverse bound')
 inverse=(C*A).inv()*C;require(finite(inverse),'validated finite paired inverse');v=-inverse*hb
 observe('paired_implicit_solve',dict(matrix=matrix_json(A),amplitude_derivative=matrix_json(hb),preconditioner=matrix_json(C),defect=matrix_json(defect),bound=ball_json(bound),inverse=matrix_json(inverse),solution=matrix_json(v)))
 direction=[v[i,0]for i in range(K)]+[arb(0),arb(1)]
 second=[MixedJet.variable(x,N+1,i,direction[i])for i,x in enumerate(values)];mixed=family(second[:-1],second[-1]);require(len(mixed)==K and all(len(x.m)==N+1 for x in mixed),'all mixed implicit derivative columns')
 dh=arb_mat([[x.m[k]for k in range(N+1)]for x in mixed]);dv=-inverse*dh;require(finite(dv),'finite full fold gradient')
 residual=arb_mat([[x.f]for x in h]+[[v[n,0]]]);jac=arb_mat([[x.g[k]for k in range(N)]for x in h]+[[dv[n,k]for k in range(N)]]);parameter=arb_mat([[x.g[N]]for x in h]+[[dv[n,N]]])
 require(finite(residual)and finite(jac)and finite(parameter),'finite full augmented fold')
 observe('amplitude_fold_return',dict(input=vector_json(z),squared_amplitude=ball_json(b),value=matrix_json(residual),jacobian=matrix_json(jac),parameter_derivative=matrix_json(parameter),paired_tangent=matrix_json(v)))
 return residual,jac,parameter,v


def augmented(j,z,b,observe=lambda *_:None):
 n=len(j);require(z[-1]>0 and 0<z[n]<=1 and all(v.is_exact()and v.is_finite()for row in j for v in row),'fixed exact map and physical gain/leak')
 return augment(lambda x,a:paired_family(j,x,a,observe),z,b,n,observe)
