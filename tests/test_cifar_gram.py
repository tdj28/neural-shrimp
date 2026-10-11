import numpy as np
import pytest
from neural_shrimp.cifar_gram import input_basis,RightScaling
from neural_shrimp.cifar_linear_diagnostic import solve_direction

def fixture(rank_deficient=False):
 x=np.array([[1.,2.,-1.,.5],[0.,1.,2.,-.5],[2.,-.3,1.,1.2]])
 if rank_deficient:x[1]=x[0]
 records={};calls=[]
 def call(k,f,*a,**kw):calls.append(k);return f(*a,**kw)
 u,l=input_basis(x,lambda k,v:records.update({k:v}),call);return x,u,l,records,calls

@pytest.mark.parametrize('rank_deficient',[False,True])
def test_augmented_gram_basis_complete_even_when_rank_deficient(rank_deficient):
 x,u,l,records,calls=fixture(rank_deficient);z=np.column_stack((x,np.ones(len(x))));assert calls==['qr','eigh']and u.shape==(5,3)
 np.testing.assert_allclose((u*l)@u.T,z.T@z/len(x),atol=2e-14);np.testing.assert_allclose(u.T@u,np.eye(3),atol=2e-14)

def test_mode_surrogate_matches_dense_commuting_products_and_packing():
 x,u,l,_,_=fixture();d,w,k=4,2,2;n=(d+1)*w+w*k+k;c0=np.array([.4,-.1]);c1=np.array([-.2,.7]);di0=np.linspace(.1,1,n);di1=np.linspace(.3,.7,n);eta=.03;lam=.04;scale=RightScaling(u,l,c0,c1,di0,di1,eta,lam,d,w,k);z=np.column_stack((x,np.ones(len(x))));gram=z.T@z/len(x);v=np.cos(np.arange(n)*.7);actual=scale.apply(v);expected=np.zeros(n)
 for j in range(w):
  h0=c0[j]*gram+lam*np.eye(d+1);h1=c1[j]*gram+lam*np.eye(d+1);a=(np.eye(d+1)-eta*h1)@(np.eye(d+1)-eta*h0)-np.eye(d+1);ev,evec=np.linalg.eigh(a);s=(evec*(1/np.maximum(np.abs(ev),scale.floor)))@evec.T;expected[j:(d+1)*w:w]=s@v[j:(d+1)*w:w]
 expected[(d+1)*w:]=scale.output*v[(d+1)*w:];np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-10)

def test_complement_exact_ridge_action_and_positive_self_adjoint():
 _,u,l,_,_=fixture();d,w,k=4,2,1;n=(d+1)*w+w*k+k;s=RightScaling(u,l,[.4,-.1],[.2,.3],np.ones(n),np.ones(n),.03,.04,d,w,k);v=np.sin(np.arange(n));z=np.cos(np.arange(n));hv=v[:10].reshape(5,2);hv=hv-u@(u.T@hv);complement=np.r_[hv.ravel(),np.zeros(3)];np.testing.assert_allclose(s.apply(complement),s.complement*complement,rtol=2e-12,atol=1e-10);assert v@s.apply(v)>0;np.testing.assert_allclose(v@s.apply(z),z@s.apply(v),rtol=1e-12,atol=1e-11)

def test_right_residual_stays_physical():
 a=np.array([[2.,.7],[.3,1.]]);s=np.array([[2.,.5],[.5,1.]]);r=np.array([1.,-2.]);y,ap,error,m=solve_direction(lambda y:a@(s@y),r,restart=2,cycles=1,rtol=1e-10,forcing_limit=.5,max_matvec=4,explicit_zero_start=True);np.testing.assert_allclose(error,a@(s@y)+r,atol=1e-14);assert m['forcing_test_pass']

def test_invalid_basis_fails_after_durable_return():
 x,_,_,_,_=fixture();saved=[]
 def call(k,f,*a,**kw):
  out=f(*a,**kw)
  if k=='eigh':out=(out[0]*np.nan,out[1])
  return out
 with pytest.raises(ValueError):input_basis(x,lambda k,v:saved.append(k),call)
 assert saved[-1]=='eigh'

def test_full_driver_and_launcher_fixed_caps():
 import importlib.util
 from pathlib import Path
 root=Path(__file__).resolve().parents[1]
 def module(name,path):
  s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 driver=module('gram_driver',root/'tools/explore_cifar_gram.py');launcher=module('gram_launcher',root/'tools/launch_cifar_gram.py')
 assert driver.CAPS==dict(action=402,j2vp=402,hvp=804,direct_gradient=28,analytic_diagonal=4,qr=2,eigh=2,right_application=407)
 assert launcher.STORE_BYTES==255*1024**2 and launcher.SUPERVISOR_SECONDS==140.
