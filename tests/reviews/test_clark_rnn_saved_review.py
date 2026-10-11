import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/reviews'))
import review_clark_rnn_math as R
from neural_shrimp.clark_rnn import iterate,period_proposals
from neural_shrimp.clark_rnn_boundaries import jets,augmented,qualify_boundary

@pytest.mark.parametrize('n,p',[(2,1),(3,5),(4,3)])
def test_independent_lifted_jets(n,p):
 j=np.arange(n*n).reshape(n,n)/17-.3;x=np.linspace(-.2,.4,n);g=1.3;d=.7
 end,u,h,c,js=jets(x,j,g,d,p);a,b=R.lifted_jets(c,j,g,d)
 np.testing.assert_allclose(a,u,rtol=2e-13,atol=2e-13);np.testing.assert_allclose(b,h,rtol=2e-13,atol=2e-13)
 R.native(c,j,g,d,js,u[:,:n],end)

@pytest.mark.parametrize('mutation',['native_order','hessian','mixed','normalization','alpha'])
def test_flip_saved_tamper(mutation):
 j=np.diag([-1.,.1]);a,m=qualify_boundary(np.array([0.,0.,2/3,1.,0.]),j,2.,1,-1);m.update(gain=2.)
 assert R.boundary(a,m,j)['qualified']
 if mutation=='native_order':a['phase_jacobians'][0,0,0]+=.01
 if mutation=='hessian':a['second'][0,0,0]+=.01
 if mutation=='mixed':a['second'][0,0,3]+=.01
 if mutation=='normalization':a['left_vector']*=2
 if mutation=='alpha':m['total_flip_alpha_delta']+=.1
 with pytest.raises(ValueError):R.boundary(a,m,j)


def test_periods_saved_tail_independent():
 t=np.arange(513);tail=np.stack([t%3,t%5],axis=1)[:,:,None].astype(float)
 a,b=R.periods(tail);c,d=period_proposals(tail)
 np.testing.assert_array_equal(a,[3,5]);np.testing.assert_array_equal(a,c);np.testing.assert_equal(b,d)


def test_zero_eigenvector_rejected():
 with pytest.raises(ValueError):R.spectrum(np.eye(2),np.ones(2),np.zeros((2,2)))
