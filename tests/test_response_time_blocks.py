import numpy as np
import pytest
from neural_shrimp.response_time_blocks import block_jacobi,probe
from neural_shrimp.response_spectrum import difference_matrix,response

def test_blocks_match_dense_normal_including_both_free_endpoints():
 a=np.array([[[.7,.3],[-.1,1.1]],[[.9,-.2],[.4,.6]]]);p,n,_=a.shape;steps=6;eta=.01;events=[];m,stats,blocks=block_jacobi(a,steps,-.05,eta,lambda k,v:events.append((k,v)));k=difference_matrix(a,steps,-.05,'free').toarray();h=k.T@k+eta**2*np.eye((steps+1)*n);v=np.sin(np.arange(len(h)));expected=np.zeros_like(v)
 for name,e in events:
  if name=='block_input':
   s,t=e['start']*n,e['end']*n;np.testing.assert_allclose(e['matrix'],h[s:t,s:t],atol=2e-16);expected[s:t]=np.linalg.solve(h[s:t,s:t],v[s:t])
 np.testing.assert_allclose(m@v,expected,rtol=2e-12,atol=2e-12);assert stats['cholesky']==3

def test_transpose_nonnormal_and_symmetric_positive_preconditioner():
 a=np.array([[[.7,1.2],[.1,.9]]]);m,_,_=block_jacobi(a,4,-.05,.01);x=np.sin(np.arange(10));y=np.cos(np.arange(10));assert x@(m@x)>0;np.testing.assert_allclose(x@(m@y),y@(m@x),rtol=1e-12)

@pytest.mark.parametrize('pre',[False,True])
def test_probe_matches_direct_tiny_and_retains_all_returned_states(pre):
 a=np.array([[[.7,.1],[0.,.6]],[[.8,-.1],[.05,.65]]]);events=[];r=probe(a,pre,lambda k,v:events.append((k,v)));direct=response(a,32,-.05,.01,'free');assert r['solve_qualified'];np.testing.assert_allclose(r['readout_column'],direct['response_block'][:,0],atol=1e-8);assert sum(k=='iterate'for k,v in events)==r['iterations'];assert any(k=='solution'for k,v in events);assert r['counts']['JVP']==32*r['counts']['K']and r['counts']['VJP']==32*r['counts']['KT']

def test_interrupt_after_iterate_preserves_return():
 a=np.array([[[.8]]]);events=[]
 def emit(k,v):
  events.append(k)
  if k=='iterate':raise RuntimeError('synthetic interruption')
 with pytest.raises(RuntimeError,match='synthetic'):probe(a,True,emit)
 assert events[-1]=='iterate'
