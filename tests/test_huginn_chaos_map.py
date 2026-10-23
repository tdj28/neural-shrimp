import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
import huginn_chaos_map as m


def test_axes_include_native():
    p=json.loads((Path(__file__).parents[1]/'docs/huginn-chaos-map-plan.json').read_text())
    l,a=m.axes(p)
    assert len(l)==len(a)==32 and 1 in l and 0 in a
    assert np.all(np.diff(l)>0) and np.all(np.diff(a)>0)


def test_period_requires_multiple_full_state_closures():
    t=np.array([[(k%3,2*(k%3))] for k in range(80)],dtype=float)
    assert m.recurrence_candidates(t,1e-5)[0]==3
    t[-2,0,1]+=0.1
    assert m.recurrence_candidates(t,1e-5)[0]==0


def test_cpu_end_to_end_contracting_map(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    import explore_latent_loop as ell
    from types import SimpleNamespace
    class Fake:
        dtype=torch.float32
        token_ids=[1,2]
        def __init__(self,*a,**kw):self.model=SimpleNamespace(core_block_forward=lambda h,*a:(self.F(h),None));self.e=None;self.freqs=None;self.s=None
        def setup(self,p):return torch.ones(1,2,3)
        def F(self,h):return .5*h+1+(0 if self.s is None else self.s)
        def set_steering(self,layer,rows):self.s=torch.as_tensor(rows,dtype=self.dtype)[:,None,:]
    monkeypatch.setattr(ell,'HuginnLoop',Fake);monkeypatch.setattr(ell,'DATA',tmp_path)
    for name in ('set_per_process_memory_fraction','synchronize'):monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'CPU analytic fixture')
    tensor=torch.tensor
    monkeypatch.setattr(torch,'tensor',lambda *a,**kw:tensor(*a,**{k:v for k,v in kw.items() if k!='device'}))
    p=json.loads((Path(__file__).parents[1]/'docs/huginn-chaos-map-plan.json').read_text());p.update(resolution=2,batch=2,transient=8,measure=80)
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(p))
    monkeypatch.setattr(sys,'argv',['worker','--plan',str(plan),'--source-commit','a'*40,'--mode','scan','--out','test'])
    m.main();root=tmp_path/'explore/test';a=np.load(root/'map.npz')
    assert a['lyapunov'].shape==(2,4)
    expected=np.log(abs(1-.5*a['cells'][:,0]))
    assert np.max(abs(a['lyapunov']-expected))<.002
    assert np.all(a['recurrence_candidate']==1)
    for file,digest in json.loads((root/'manifest.json').read_text()).items():assert hashlib.sha256((root/file).read_bytes()).hexdigest()==digest
