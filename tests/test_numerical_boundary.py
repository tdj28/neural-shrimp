import numpy as np
import pytest
from neural_shrimp.numerical_boundary import TwoRateBoundary,null_tangent,correct_section
from neural_shrimp.gdmaps import TwoLayerNet


def model():
 return dict(schema='neural-shrimp-c1-input-v1',network=dict(d=1,width=1,activation='tanh',bias=True,in_scale='3/2',out_scale='2/3',loss_scale='1/2',ridge='1/100'),x=[['-1/2'],['1/3']],y=['1/5','-1/7'],lr_hidden='1/10',lr_output='1/20')

@pytest.mark.parametrize('sigma',[-1,1])
@pytest.mark.parametrize('period',[1,3])
def test_two_rate_augmented_jacobian(sigma,period):
 f=TwoRateBoundary(model(),period,sigma);z=np.r_[np.arange(4)/10,.1,.05,np.array([1,2,-1,3])/np.sqrt(15)]
 a=f(z);step=2.**-17
 fd=np.column_stack([(f(z+np.eye(len(z))[i]*step)['value']-f(z-np.eye(len(z))[i]*step)['value'])/(2*step) for i in range(len(z))])
 np.testing.assert_allclose(a['jacobian'],fd,rtol=2e-7,atol=3e-9)


def test_native_map_and_chain_rule():
 f=TwoRateBoundary(model(),3,1);z=np.r_[np.arange(4)/10,.1,.05,np.array([1,2,-1,3])/np.sqrt(15)];d=f(z)
 net=TwoLayerNet(1,1,in_scale=1.5,out_scale=2/3,ridge=.01);x=np.array([[-.5],[1/3]]);y=np.array([.2,-1/7]);state=z[:4].copy();m=np.eye(4);rates=net.rate_vector([.1],[.05])[0]
 for _ in range(3):
  m=(np.eye(4)-rates[:,None]*net.hessian(state,x,y))@m;state-=rates*net.gradient(state[None,:],x,y)[0]
 np.testing.assert_allclose(d['orbit'][-1],state,atol=1e-15);np.testing.assert_allclose(d['monodromy'],m,atol=1e-15)
 assert all(not hasattr(v,'g') for v in f.model.rates)


def cusp(z):
 x,a,b=z
 return {'value':np.array([x**3+a*x+b,3*x*x+a]),'jacobian':np.array([[3*x*x+a,x,1],[6*x,1,0]])}


def test_cusp_continuation_through_turning_point():
 z=np.array([-.2,-.12,-.016]);t,_=null_tangent(cusp(z)['jacobian'],np.array([1.,0,0]));scale=np.ones(3)
 for i in range(20):
  out=correct_section(cusp,z+.025*t,t,scale);assert out['qualified'];z=out['state'];t,_=null_tangent(cusp(z)['jacobian'],t)
  np.testing.assert_allclose(z[1:],[-3*z[0]**2,2*z[0]**3],atol=1e-9)
 assert z[0]>.1


def test_rank_failure():
 with pytest.raises(ValueError):null_tangent(np.zeros((2,3)))


def test_stop_retained_by_caller():
 seen=[]
 def fail(z):seen.append(z.copy());raise TimeoutError('unit budget')
 with pytest.raises(TimeoutError):correct_section(fail,np.zeros(3),np.array([1.,0,0]),np.ones(3))
 assert len(seen)==1

@pytest.mark.parametrize('a',[np.empty((0,1)),np.ones(2),np.full((2,3),np.nan)])
def test_invalid_tangent_matrix(a):
 with pytest.raises(ValueError):null_tangent(a)

def test_driver_journal_text_wrapper(tmp_path):
 import gzip,os
 with gzip.open(tmp_path/'journal.gz','xt') as log:
  log.write('retained\n');log.flush();os.fsync(log.buffer.fileobj.fileno())
 with gzip.open(tmp_path/'journal.gz','rt') as log:assert log.read()=='retained\n'

def test_extended_coordinate_exactness():
 from neural_shrimp.numerical_boundary import coordinate_ball
 from flint import ctx
 with ctx.workprec(128):
  x=np.longdouble(1)+np.ldexp(np.longdouble(1),-60)
  assert coordinate_ball(x)-1>0
  assert coordinate_ball(float(x))==1

def test_extended_corrector_preserves_small_update():
 root=np.longdouble(1)+np.ldexp(np.longdouble(1),-60)
 def f(z):return {'value':np.array([float((z[0]-root)*2.**50)]),'jacobian':np.array([[2.**50,0.]])}
 p=np.array([1,0],dtype=np.longdouble);out=correct_section(f,p,np.array([0.,1.]),np.ones(2))
 assert out['qualified'] and out['state'][0]==root

def test_constructor_uses_requested_precision_before_parsing_constants():
 from flint import ctx,arb
 with ctx.workprec(53):
  f=TwoRateBoundary(model(),3,1,precision=128)
  assert f.model.x[1][0].rad()<arb(2)**-120
  assert f.model.ridge.rad()<arb(2)**-120
 with ctx.workprec(53):
  high=TwoRateBoundary(model(),3,1,precision=256)
  assert high.model.x[1][0].rad()<arb(2)**-248
 z=np.r_[np.arange(4)/10,.1,.05,np.array([1,2,-1,3])/np.sqrt(15)]
 np.testing.assert_array_equal(f(z)['value'],high(z)['value'])


def test_exact_saved_coordinate_resume(tmp_path):
 import importlib.util
 from pathlib import Path
 p=Path(__file__).parents[1]/'tools/explore_s2_boundary_curves.py'
 spec=importlib.util.spec_from_file_location('boundary_runner',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 x=np.longdouble(1)+np.ldexp(np.longdouble(1),-60)
 assert m.exact_coordinate(m.coordinates([x])[0])==x
 with pytest.raises(ValueError):m.exact_coordinate('1/3')
 with pytest.raises(ValueError):m.exact_coordinate(str((2**80+1))+'/'+str(2**80))
