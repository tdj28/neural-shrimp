"""Unrelated signed fixture and durable native acquisition adapter."""
from fractions import Fraction as F
from flint import arb
from neural_shrimp.rigorous_orbits import exact,vector_json,ball_json
from neural_shrimp.rigorous_folds import MixedJet
from neural_shrimp.rigorous_eigenpair_path import PathJet

class SyntheticFlip:
 def __init__(self,n=13):self.n=n
 def cusp_step(self,z,h,o):
  x=z[0]+sum((z[i]*arb((1,-i))for i in range(1,self.n)),0)
  fx=-exact('1/2')-exact('3/2')*x+exact('1/2')*x*x+exact('1/2')*x*x*x+(h-1)+(o-1)*x
  rest=[v/2 if isinstance(v,PathJet)else v*exact('1/2')for v in z[1:]]
  return [fx-sum((rest[i-1]*arb((1,-i))for i in range(1,self.n)),0)]+rest

def synthetic_fixture(n=13,period=18):
 initial=[]
 for k in range(period):initial += [str((-1)**k)]+['0']*(n-1)+[str((-1)**(k//2))]+['0']*(n-1)
 initial+=['1','1'];size=len(initial)
 return initial,dict(period=period,sigma=-1,normal=['0']*(size-2)+['1','0'],origin=initial,parameter='0',halfwidth_exponent=-12,radius_exponent=-16,precision=256,newton_steps=2,anchor=dict(box=[exact(v)for v in initial],period=period,sigma=-1,dimension=n))

def encode(v):
 if isinstance(v,MixedJet):return dict(kind='mixed',f=ball_json(v.f),g=vector_json(v.g),d=ball_json(v.d),m=vector_json(v.m))
 if isinstance(v,PathJet):return dict(kind='path',a=vector_json(v.a),b=vector_json(v.b))
 raise TypeError('only complete native mixed or quadratic path jets')

class RetainedMap:
 def __init__(self,model,observe):self.model=model;self.n=model.n;self.observe=observe;self.calls=0
 def cusp_step(self,z,h,o):
  index=self.calls;self.calls+=1;self.observe('native_input',dict(index=index,inputs=[encode(v)for v in list(z)+[h,o]]))
  result=self.model.cusp_step(z,h,o);self.observe('native_return',dict(index=index,returns=[encode(v)for v in result]));return result
