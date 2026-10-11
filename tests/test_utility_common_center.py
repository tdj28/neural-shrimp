import numpy as np
import pytest
from neural_shrimp.utility_common_center import reduce_contrast,evaluate
from neural_shrimp.gdmaps import TwoLayerNet

def test_moving_center_quadratic_is_exact_and_old_gradient_is_not():
 h=np.diag(np.arange(1.,14.));t=np.linspace(.1,.3,13);g=h@t;events=[]
 r=evaluate(t,g,np.array([.03,.01]),lambda z:h@z,lambda z:float(z@h@z/2),lambda k,v:events.append((k,v)))
 assert abs(r['centered_residual'])<1e-15
 assert abs(r['original_state_first_order_prediction']-r['loss_difference_plus_minus'])>1e-7
 assert [v['kind']for k,v in events if k=='operation_return']==['gradient_center','loss_plus','loss_minus']

def test_cubic_odd_remainder_coefficient():
 z=.7;v=.04;f=lambda x:x**3+2*x*x;g=lambda x:3*x*x+4*x
 r=reduce_contrast(f(z+v),f(z-v),np.array([g(z)]),np.array([v]),np.array([0.]))
 assert r['centered_residual']==pytest.approx(2*v**3,abs=1e-15)

def test_same_neural_loss_and_center_gradient_with_bias_ridge():
 model=TwoLayerNet(2,3,ridge=.001);x=np.linspace(-.8,.7,12).reshape(6,2);y=np.sin(x[:,0]);t=np.linspace(-.2,.5,13);g=model.gradient(t[None],x,y)[0]
 for i in range(13):
  e=np.zeros(13);e[i]=1e-6
  fd=(model.loss((t+e)[None],x,y)-model.loss((t-e)[None],x,y))[0]/2e-6
  assert g[i]==pytest.approx(fd,abs=1e-10)
 r=evaluate(t,g,np.array([.3,.1]),lambda z:model.gradient(z[None],x,y)[0],lambda z:model.loss(z[None],x,y)[0]);assert np.isfinite(r['centered_residual'])

def test_nonfinite_return_retained_before_failure():
 events=[]
 with pytest.raises(ValueError,match='finite returned gradient'):evaluate(np.arange(13.),np.arange(1.,14.),np.array([.1,.01]),lambda z:np.full(13,np.nan),lambda z:0.,lambda k,v:events.append(k))
 assert events[-1]=='operation_return'

def test_deadline_after_gradient_return_is_durable():
 events=[]
 def check():
  if events and events[-1]=='operation_return':raise TimeoutError('synthetic')
 with pytest.raises(TimeoutError):evaluate(np.arange(13.),np.arange(1.,14.),np.array([.1,.01]),lambda z:z,lambda z:0.,lambda k,v:events.append(k),check)
 assert events[-1]=='operation_return'


def test_launcher_byte_allowance_fails_closed(tmp_path):
 import importlib.util
 from pathlib import Path
 s=importlib.util.spec_from_file_location('center_launcher',Path(__file__).parents[1]/'tools/launch_utility_common_center.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 (tmp_path/'run').mkdir();(tmp_path/'run'/'data').write_bytes(b'x'*1000)
 assert m.projected_final_bytes(tmp_path,{})['within_bytes']
 (tmp_path/'process.log').write_bytes(b'x'*(1024**2))
 assert not m.projected_final_bytes(tmp_path,{})['within_bytes']


def test_result_supports_console_json_without_numpy_scalars():
 import json
 r=reduce_contrast(.4,.3,np.array([.1]),np.array([.2]),np.array([.3]))
 assert json.loads(json.dumps(r))==r
