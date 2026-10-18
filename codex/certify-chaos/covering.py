"""Finite Arb face obligations for a one-unstable-coordinate covering.
Input transverse_bound must be uniform on the entire source h-set, not sampled.
This checker alone neither constructs that enclosure nor proves a Qwen graph.
"""
from flint import arb, fmpq

def q(n,d=1):return arb(fmpq(n,d))
def pack(x):
 if not x.is_finite():raise ValueError('finite enclosure required')
 return {'mid':list(map(int,x.mid().man_exp())),'rad':list(map(int,x.rad().man_exp()))}

def check(left_face,right_face,target_low,target_high,transverse_bound,target_radius):
 for x in [left_face,right_face,target_low,target_high,transverse_bound,target_radius]:
  if not x.is_finite():raise ValueError('invalid enclosure')
 if not target_low<target_high or not target_radius>0 or not transverse_bound>=0:raise ValueError('invalid target or bound')
 forward=[target_low-left_face,right_face-target_high];reverse=[target_low-right_face,left_face-target_high]
 orientation=1 if all(x>0 for x in forward) else (-1 if all(x>0 for x in reverse) else 0)
 margins=forward if orientation==1 else reverse if orientation==-1 else forward
 stable=target_radius-transverse_bound
 return {'covered':bool(orientation and stable>0),'orientation':orientation,'exit_margins':[pack(x) for x in margins],'containment_margin':pack(stable),'stable_pass':bool(stable>0),'exit_pass':bool(orientation)}

def rational_householder(direction):
 """For exact rational nonzero v: Q=I-2vv^T/(v^Tv), Q^TQ=I exactly.
 The returned direction is Qe1. A supplied binary64 approximate unit u is
 not assumed exactly unit; using v=e1-u changes u unless its norm is exact1.
 """
 from fractions import Fraction
 v=[Fraction(x) for x in direction];s=sum(x*x for x in v)
 if s==0:raise ValueError('nonzero Householder vector required')
 u=[Fraction(int(i==0))-2*x*v[0]/s for i,x in enumerate(v)]
 assert sum(x*x for x in u)==1
 # Rank-one coefficient in Q^2-I vanishes as an exact rational.
 assert -4/s+4*s/(s*s)==0
 return u,s
