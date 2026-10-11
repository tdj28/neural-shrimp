"""Scalar-objective second-order AD with Arb matrix derivative storage.

The expression tree is differentiated generically. No producer gradient,
Hessian or native map is called; bias-free and biased layouts are supported.
"""
from fractions import Fraction
from flint import arb,arb_mat

def rational(v):
    q=Fraction(v);return arb(q.numerator)/q.denominator

class Jet:
    def __init__(self,value,n,index=None):
        self.v=value if isinstance(value,arb)else arb(value)
        self.g=arb_mat(n,1);self.h=arb_mat(n,n)
        if index is not None:self.g[index,0]=1
    @classmethod
    def parts(cls,v,g,h):
        a=object.__new__(cls);a.v,a.g,a.h=v,g,h;return a
    def __add__(self,b):
        if not isinstance(b,Jet):return self.parts(self.v+b,self.g,self.h)
        if self.g.nrows()!=b.g.nrows():raise ValueError('jet dimension')
        return self.parts(self.v+b.v,self.g+b.g,self.h+b.h)
    __radd__=__add__
    def __mul__(self,b):
        if not isinstance(b,Jet):return self.parts(self.v*b,self.g*b,self.h*b)
        if self.g.nrows()!=b.g.nrows():raise ValueError('jet dimension')
        return self.parts(self.v*b.v,self.g*b.v+b.g*self.v,self.h*b.v+b.h*self.v+self.g*b.g.transpose()+b.g*self.g.transpose())
    __rmul__=__mul__
    def tanh(self):
        v=self.v.tanh();one=1-v*v;two=-2*v*one
        return self.parts(v,self.g*one,self.h*one+(self.g*self.g.transpose())*two)

def objective_ad(inp,theta):
    s=inp['network'];d,w,b=s['d'],s['width'],s['bias'];n=len(theta)
    if inp.get('schema')!='neural-shrimp-c1-input-v1'or s['activation']!='tanh'or type(b)is not bool or type(d)is not int or type(w)is not int or not(1<=d<=8 and 1<=w<=8):raise ValueError('bounded tanh network')
    hidden=d*w+(w if b else 0)
    if n!=hidden+w+int(b)or not 1<=n<=72 or not all(v.is_finite()for v in theta):raise ValueError('full finite parameter vector')
    if not inp['x']or len(inp['x'])!=len(inp['y'])or any(len(x)!=d for x in inp['x']):raise ValueError('complete exact dataset')
    ins,outs,scale,ridge=[rational(s[k])for k in ['in_scale','out_scale','loss_scale','ridge']]
    rh,ro=rational(inp['lr_hidden']),rational(inp['lr_output'])
    if not all(v>0 for v in [ins,outs,scale,rh,ro])or not ridge>=0:raise ValueError('positive map scales')
    variables=[Jet(x,n,i)for i,x in enumerate(theta)];loss=Jet(0,n)
    for x,y in zip(inp['x'],inp['y'],strict=True):
        pred=variables[-1]if b else Jet(0,n)
        for unit in range(w):
            activation=variables[d*w+unit]if b else Jet(0,n)
            for k in range(d):activation=activation+variables[k*w+unit]*(ins*rational(x[k]))
            pred=pred+variables[hidden+unit]*activation.tanh()*outs
        residual=pred+(-rational(y));loss=loss+(residual*residual)*(scale/len(inp['x']))
    for variable in variables:loss=loss+(variable*variable)*(ridge/2)
    rates=[rh]*hidden+[ro]*(n-hidden)
    return [theta[i]-rates[i]*loss.g[i,0]for i in range(n)],arb_mat([[int(i==j)-rates[i]*loss.h[i,j]for j in range(n)]for i in range(n)])
