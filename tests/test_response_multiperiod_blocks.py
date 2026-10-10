import numpy as np
import pytest
from neural_shrimp.response_multiperiod_blocks import block_jacobi,probe
from neural_shrimp.response_spectrum import difference_matrix,response

@pytest.mark.parametrize('periods',[2,4])
def test_fixed_multiperiod_blocks_match_whole_dense_normal(periods):
 a=np.array([[[.7,.3],[-.1,1.1]],[[.9,-.2],[.4,.6]]]);p,n,_=a.shape;steps=16*p;eta=.01;events=[];m,stats,blocks=block_jacobi(a,steps,-.05,eta,periods,lambda k,v:events.append((k,v)));k=difference_matrix(a,steps,-.05,'free').toarray();h=k.T@k+eta**2*np.eye((steps+1)*n);v=np.sin(np.arange(len(h)));expected=np.zeros_like(v)
 for name,e in events:
  if name=='block_input':
   s,t=e['start']*n,e['end']*n;np.testing.assert_allclose(e['matrix'],h[s:t,s:t],atol=2e-16);expected[s:t]=np.linalg.solve(h[s:t,s:t],v[s:t])
 np.testing.assert_allclose(m@v,expected,rtol=2e-12,atol=2e-12);assert stats['cholesky']==3;assert len(blocks)==16//periods+1;assert blocks[-1][1]-blocks[-1][0]==n
 x=np.cos(np.arange(len(h)));assert x@(m@x)>0;np.testing.assert_allclose(x@(m@v),v@(m@x),rtol=2e-12)

@pytest.mark.parametrize('periods',[2,4])
def test_fixed_multiperiod_probe_direct_reference_counts_and_durability(periods):
 a=np.array([[[.7,.1],[0.,.6]],[[.8,-.1],[.05,.65]]]);events=[];r=probe(a,periods,lambda k,v:events.append((k,v)));direct=response(a,32,-.05,.01,'free');assert r['solve_qualified'];np.testing.assert_allclose(r['readout_column'],direct['response_block'][:,0],atol=1e-8);assert sum(k=='iterate'for k,v in events)==r['iterations'];assert any(k=='solution'for k,v in events);assert r['counts']['JVP']==32*r['counts']['K']and r['counts']['VJP']==32*r['counts']['KT'];assert r['preconditioner']['triangular_solves']==2*(16//periods+1)*r['preconditioner']['applications']

@pytest.mark.parametrize('periods',[1,3,8,True])
def test_unplanned_sizes_rejected(periods):
 with pytest.raises(ValueError):block_jacobi(np.array([[[.8]]]),16,-.05,.01,periods)

def test_durable_factor_return_precedes_failed_residual(monkeypatch):
 import neural_shrimp.response_multiperiod_blocks as m
 events=[];monkeypatch.setattr(m,'cholesky',lambda h,lower:np.eye(len(h)))
 with pytest.raises(ValueError,match='factor residual'):m.block_jacobi(np.array([[[.8]]]),16,-.05,.01,2,lambda k,v:events.append(k))
 assert events[-2:]==['factor','factor_check']
