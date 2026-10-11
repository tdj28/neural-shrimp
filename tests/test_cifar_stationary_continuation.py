import numpy as np
import pytest
from neural_shrimp import cifar_stationary_continuation as C

def test_inexact_point_seven_five_direction_uses_actual_merit(monkeypatch):
 # Deliberately inexact direction has eta .75 and decreases a true quadratic.
 def fake(A,b,**kw):
  p=b*.25;A@p;kw['callback'](p);return p,1
 monkeypatch.setattr(C,'minres',fake)
 origin=np.array([10.]);initial=np.array([9.99]);grad=lambda x:x-9.9;hv=lambda x,v:v
 x,g,out=C.stationary_from_seed(initial,grad(initial),origin,grad,hv)
 assert len(out['steps'])==4 and all(s['forcing']==pytest.approx(.75)for s in out['steps'])
 assert all(s['accepted_index']==0 for s in out['steps'])and abs(g[0])<abs(grad(initial)[0])
 assert not out['passed']

def test_global_origin_is_not_reset(monkeypatch):
 def fake(A,b,**kw):
  p=b.copy();A@p;kw['callback'](p);return p,1
 monkeypatch.setattr(C,'minres',fake)
 origin=np.array([10.]);initial=np.array([10.49]);grad=lambda x:x-11.;hv=lambda x,v:v
 x,g,out=C.stationary_from_seed(initial,grad(initial),origin,grad,hv)
 assert out['relative_origin_distance']<=.05 and x[0]<=10.5
 assert out['steps'][0]['accepted_index']>=6
 assert not out['steps'][0]['trials'][0]['within_trust']
 with pytest.raises(ValueError,match='global trust'):C.stationary_from_seed(np.array([10.51]),np.array([1.]),origin,grad,hv)

def test_forcing_over_point_nine_stops_without_trials(monkeypatch):
 def fake(A,b,**kw):
  p=b*.01;A@p;kw['callback'](p);return p,1
 monkeypatch.setattr(C,'minres',fake);calls=[]
 x,g,out=C.stationary_from_seed(np.array([1.]),np.array([1.]),np.array([1.]),lambda x:calls.append(x),lambda x,v:v)
 assert out['status']=='true_forcing_failed'and not calls and not out['steps'][0]['trials']
