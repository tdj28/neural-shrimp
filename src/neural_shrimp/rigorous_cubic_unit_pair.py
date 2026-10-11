"""Cubic native unit-pair predictors with rigorous quartic residual remainder.

Source preparation only. Ten coefficients in R[u,e]/(u^5,e^2) retain F and DFq
through order4. The full state/mode/rate/angle path has no derivative tensor.
"""
from fractions import Fraction
from math import factorial
from flint import arb,arb_mat
from .rigorous_eigenpair_locus import require,finite
from .rigorous_orbits import exact,vector_json,matrix_json,ball_json


def _tanh_polynomials():
    rows=[[0,1]]
    for _ in range(5):
        old=rows[-1];derivative=[(i+1)*old[i+1]for i in range(len(old)-1)];out=[0]*(len(derivative)+2)
        for i,v in enumerate(derivative):out[i]+=v;out[i+2]-=v
        rows.append(out)
    return tuple(tuple(v)for v in rows)
_TANH=_tanh_polynomials()


class QuarticPathJet:
    __slots__=('a','b')
    def __init__(self,a,b=None):
        require(len(a)==5 and (b is None or len(b)==5),'ten path coefficients')
        self.a=tuple(arb(v)for v in a);self.b=tuple(arb(v)for v in (b if b is not None else [0]*5))
    @classmethod
    def constant(cls,v):return cls([v,0,0,0,0])
    def __add__(self,v):
        if not isinstance(v,QuarticPathJet):v=QuarticPathJet.constant(v)
        return QuarticPathJet([x+y for x,y in zip(self.a,v.a,strict=True)],[x+y for x,y in zip(self.b,v.b,strict=True)])
    __radd__=__add__
    def __neg__(self):return self*-1
    def __sub__(self,v):return self+-v
    def __rsub__(self,v):return -self+v
    def __mul__(self,v):
        if not isinstance(v,QuarticPathJet):return QuarticPathJet([x*v for x in self.a],[x*v for x in self.b])
        return QuarticPathJet([sum((self.a[i]*v.a[k-i]for i in range(k+1)),arb(0))for k in range(5)],[sum((self.a[i]*v.b[k-i]+self.b[i]*v.a[k-i]for i in range(k+1)),arb(0))for k in range(5)])
    __rmul__=__mul__
    def __truediv__(self,v):
        require(not isinstance(v,QuarticPathJet),'scalar division only')
        return QuarticPathJet([x/v for x in self.a],[x/v for x in self.b])
    def tanh(self):
        # The nilpotent ideal (u,e) has sixth power zero. Derivative polynomials
        # obey P_(k+1)(t)=(1-t²)P'_k(t), so orders0..5 suffice exactly in this ring.
        t=self.a[0].tanh();nil=QuarticPathJet([0,*self.a[1:]],self.b);power=QuarticPathJet.constant(1);result=QuarticPathJet.constant(0)
        for order,coeff in enumerate(_TANH):
            derivative=arb(0)
            for value in reversed(coeff):derivative=derivative*t+value
            result=result+power*derivative/factorial(order)
            if order<5:power=power*nil
        return result
    def serialized(self):return [vector_json(self.a),vector_json(self.b)]


def cubic_path(center,slope,curvature,cubic,d):
    """cubic is the coefficient of d³, not the third derivative."""
    return QuarticPathJet([center+slope*d+curvature*d*d/2+cubic*d*d*d,slope+curvature*d+3*cubic*d*d,curvature/2+3*cubic*d,cubic,0])


def entire_quartic(t):
    require(isinstance(t,QuarticPathJet)and all(v.is_zero()for v in t.b),'angle has no mode direction')
    require(t.a[0].is_finite()and abs(t.a[0]).upper()<=exact('1/4'),'whole entire angle domain')
    T=abs(t.a[0]).upper();nil=QuarticPathJet([0,*t.a[1:]]);powers=[QuarticPathJet.constant(1)]
    for _ in range(4):powers.append(powers[-1]*nil)
    def derivative(offset,j):
        result=arb(0)
        for k in range(20,j-1,-1):result=result*t.a[0]+exact(str(Fraction((-1)**k*factorial(k),factorial(k-j)*factorial(2*k+offset))))
        # For |t|<=1/4, j<=4 and k>=21 the successive absolute derivative
        # series terms have ratio <= ((k+1)/(k+1-j))*1/4/((2k+1)(2k+2)) <1/2.
        tail=2*exact(str(Fraction(factorial(21),factorial(21-j)*factorial(42+offset))))*T**(21-j)
        return result+arb(0,tail)
    def compose(offset):return sum((powers[j]*derivative(offset,j)/factorial(j)for j in range(5)),QuarticPathJet.constant(0))
    c,e=compose(0),compose(1);return c,e,t*e


def cubic_residual_coefficients(model,center,slope,curvature,cubic,d,period,normal,origin,parameter,*,tick=lambda:None,observe=lambda k,v:None):
    n=model.n;m=3*n;size=m*period+3
    require(type(period)is int and 1<=period<=18 and 2<=n<=13,'native bounded phase dimensions')
    require(all(len(v)==size for v in [center,slope,curvature,cubic,normal,origin]),'full cubic native dimensions')
    require(all(v.is_finite()and v.is_exact()for row in [center,slope,curvature,cubic,normal,origin]for v in row)and d.is_finite()and parameter.is_exact()and parameter.is_finite(),'finite exact predictor/gauge')
    z=[cubic_path(center[i],slope[i],curvature[i],cubic[i],d)for i in range(size)]
    require(z[-3].a[0]>0 and z[-2].a[0]>0,'positive whole native rates');c,e,dd=entire_quartic(z[-1]);rows=[]
    for phase in range(period):
        off=phase*m;after=((phase+1)%period)*m;both=[]
        for name,group in [('q',n),('r',2*n)]:
            state=[QuarticPathJet(z[off+i].a,z[off+group+i].a)for i in range(n)]
            observe('cubic_path_input',{'phase':phase,'direction':name,'d':ball_json(d),'inputs':[v.serialized()for v in state+z[-3:-1]]});tick();ret=model.cusp_step(state,*z[-3:-1])
            require(len(ret)==n and all(isinstance(v,QuarticPathJet)and all(a.is_finite()for a in v.a+v.b)for v in ret),'full quartic native result')
            observe('cubic_path_return',{'phase':phase,'direction':name,'returns':[v.serialized()for v in ret]});both.append(ret)
        qret,rret=both
        rows.extend([[qret[i].a[k]-z[after+i].a[k]for k in range(5)]for i in range(n)])
        for group,ret in [(n,qret),(2*n,rret)]:
            for i in range(n):
                close=z[after+group+i]
                if phase==period-1:close=c*z[n+i]-dd*z[2*n+i]if group==n else e*z[n+i]+c*z[2*n+i]
                rows.append([ret[i].b[k]-close.a[k]for k in range(5)])
    norm=(sum((v*v for v in z[n:2*n]),QuarticPathJet.constant(0))-1)/2
    orth=sum((z[n+i]*z[2*n+i]for i in range(n)),QuarticPathJet.constant(0))
    gauge=sum((normal[i]*(z[i]-origin[i])for i in range(size)),QuarticPathJet.constant(0))-QuarticPathJet([parameter+d,1,0,0,0])
    rows.extend([list(norm.a),list(orth.a),list(gauge.a)]);result=arb_mat(rows);require(finite(result),'full finite quartic residual')
    observe('cubic_path_assembly_return',{'d':ball_json(d),'coefficients':matrix_json(result)});return result
