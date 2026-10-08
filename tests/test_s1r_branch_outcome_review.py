import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('audit_s1r',Path(__file__).parents[1]/'tools/audit_s1r_branch_outcome.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_independent_gradient_and_hessian():
    model=dict(d=2,width=2,activation='tanh',bias=False,in_scale=.5,out_scale=.125,loss_scale=1.,ridge=.001)
    t=np.array([.2,-.1,.4,-.3,.5,.6]);x=np.array([[.2,.1],[-.5,.7],[1.,-.1]]);y=np.array([.2,-.3,.1]);h=2e-6
    _,g,H=audit.loss_gradient_hessian(t,x,y,model);basis=np.eye(6)
    fdg=np.array([(audit.loss_gradient_hessian(t+h*v,x,y,model)[0]-audit.loss_gradient_hessian(t-h*v,x,y,model)[0])/(2*h) for v in basis])
    fdH=np.column_stack([(audit.loss_gradient_hessian(t+h*v,x,y,model)[1]-audit.loss_gradient_hessian(t-h*v,x,y,model)[1])/(2*h) for v in basis])
    np.testing.assert_allclose(g,fdg,atol=2e-10,rtol=2e-8);np.testing.assert_allclose(H,fdH,atol=2e-10,rtol=2e-8)
    np.testing.assert_allclose(H,H.T,atol=1e-15)


def test_phase_swap_ambiguity_and_remote_reject():
    p=np.array([[0.,0.],[1.,-1.]])
    assert audit.pair_match(p,p)['phase']==0
    assert audit.pair_match(p,p[::-1])['phase']==1
    assert audit.pair_match(p,p[::-1])['passed']
    assert not audit.pair_match(p,np.zeros((2,2)))['passed']
    assert not audit.pair_match(p,p+.1)['passed']


def test_manifest_corruption_rejected(tmp_path):
    (tmp_path/'data').write_bytes(b'original');m=dict(files={'data':audit.entry(tmp_path/'data')})
    (tmp_path/'manifest.json').write_text(json.dumps(m));sha=audit.digest((tmp_path/'manifest.json').read_bytes())
    audit.inventory(tmp_path,sha);(tmp_path/'data').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact hash'):audit.inventory(tmp_path,sha)


def test_nonfinite_state_rejected():
    m=dict(d=1,width=1,activation='tanh',bias=False)
    with pytest.raises(ValueError,match='independent state'):audit.loss_gradient_hessian(np.array([float('nan'),0]),np.ones((1,1)),np.ones(1),m)
