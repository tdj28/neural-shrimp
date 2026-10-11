"""Finite adaptive width policy; proposals never advance an unproved root."""
from fractions import Fraction as F

def require(v,s):
 if not v:raise ValueError(s)

def next_state(width,strong,*,uniform,primitive,joined,contraction,minimum=-8,maximum=-1):
 require(type(width)is int and minimum<=width<=maximum and type(strong)is int and 0<=strong<=1,'fixed bounded width state')
 if not primitive:return dict(action='stop',reason='primitive_gate_failed',width=width,strong=0)
 if uniform and not joined:return dict(action='stop',reason='successful_root_join_failed',width=width,strong=0)
 if not joined:return dict(action='stop',reason='association_failed',width=width,strong=0)
 if not uniform:
  return dict(action='retry',reason='completed_root_bound_failed',width=width-1,strong=0)if width>minimum else dict(action='stop',reason='minimum_width_bound_failed',width=width,strong=0)
 good=contraction<F(1,5)
 if good and strong:return dict(action='advance',reason='two_strong_passes',width=min(maximum,width+1),strong=0)
 return dict(action='advance',reason='qualified_root_and_join',width=width,strong=int(good))

def next_parameter(known,width):
 # Travel toward decreasing physical hidden rate under the exact fixed gauge.
 require(type(width)is int and -8<=width<=-1,'bounded dyadic width')
 right=F(known);half=F(2)**width;return str(right-half),str(right-2*half),str(right)
