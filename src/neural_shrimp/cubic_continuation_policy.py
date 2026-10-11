"""Finite proposal-only width/gauge policy; actual full proof always decides."""
from fractions import Fraction as F

def next_step(exponent,strong,*,root_passed,joined,other_gates,contraction):
    if type(exponent)is not int or not -9<=exponent<=-3 or type(strong)is not int or not 0<=strong<=1:raise ValueError('bounded width and strong-pass counter')
    if not other_gates or(root_passed and not joined):return dict(action='stop',halfwidth=exponent,strong=0,rotate=False)
    if not root_passed:
        return dict(action='halve'if exponent>-9 else'stop',halfwidth=max(-9,exponent-1),strong=0,rotate=False)
    q=F(contraction)
    if not 0<=q<1:raise ValueError('passed actual contraction')
    rotate=q>=F(1,4);count=strong+1 if q<F(1,5)else 0
    width=exponent
    if count==2:width=min(-3,exponent+1);count=0
    return dict(action='advance',halfwidth=width,strong=count,rotate=rotate)
