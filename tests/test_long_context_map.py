import sys,json,hashlib
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from long_context_map import grid,wait_for_ack
from long_context_collect import collect_checkpoint,sha

def test_grid():
 g=grid();assert len(set(g))==256
 assert g[0]==(-.3,0.) and g[-1]==(.3,.1)

def test_barrier_fails_closed(tmp_path):
 with pytest.raises(TimeoutError):wait_for_ack(tmp_path/'ack','hash','source',lambda:None,timeout=0)
 (tmp_path/'ack').write_text(json.dumps(dict(source='wrong',manifest_sha256='hash')))
 with pytest.raises(TimeoutError):wait_for_ack(tmp_path/'ack','hash','source',lambda:None,timeout=0)
 with pytest.raises(RuntimeError):wait_for_ack(tmp_path/'ack','hash','source',lambda:(_ for _ in ()).throw(RuntimeError('deadline')))

def files():
 m=json.dumps({'cell.npz':sha(b'evidence')}).encode()
 return {'cell.npz':b'evidence','manifest.json':m,'checkpoint.json':json.dumps(dict(source='source',manifest_sha256=sha(m),final=True)).encode()}

def test_retrieval_corrupt_then_resume(tmp_path):
 data=files();out=tmp_path/'data';out.mkdir();acks=[]
 def write(n,b):acks.append((n,b))
 data['cell.npz']=b'bad'
 with pytest.raises(ValueError):collect_checkpoint(data.__getitem__,write,out,'source')
 assert not acks and (out/'cell.npz.partial').exists()
 data['cell.npz']=b'evidence'
 assert collect_checkpoint(data.__getitem__,write,out,'source')['complete']
 assert len(acks)==1
 (out/'cell.npz').write_bytes(b'bad')
 with pytest.raises(ValueError):collect_checkpoint(data.__getitem__,write,out,'source')
 assert len(acks)==1

def test_interrupted_and_missing(tmp_path):
 data=files();out=tmp_path/'data';out.mkdir();acks=[]
 del data['cell.npz']
 with pytest.raises(KeyError):collect_checkpoint(data.__getitem__,lambda *a:acks.append(a),out,'source')
 assert not acks
 with pytest.raises(ValueError):collect_checkpoint(data.__getitem__,lambda *a:acks.append(a),out,'wrong')

def test_scope():
 from runpod_scan_group import validate_orbit_job
 source='a'*40
 validate_orbit_job(f'lcm1 700 long_context_map.py --source-commit {source} --out llm-soft-loop/pod64/lcm1/map-01',source)
 with pytest.raises(ValueError):validate_orbit_job(f'lcm1 701 long_context_map.py --source-commit {source} --out llm-soft-loop/pod64/lcm1/map-01',source)

@pytest.mark.parametrize('fault,expected', [('collapse',1),('nan',2),('inf',2)])
def test_twin_failure_preserves_base_and_masks_growth(fault,expected):
 import torch
 import numpy as np
 from long_context_map import measure_cell
 class Loop:
  def step(self,x,temps):
   y=x.clone()
   y[0]+=1
   y[1]=y[0] if fault=='collapse' else float(fault)
   y[2]=y[0]+.25
   return y,None
 loop=Loop();loop.torch=torch
 out=measure_cell(loop,torch.ones(3,1,2),[1]*3,[.1,.1],lambda:None,steps=4)
 assert out['states'].shape==(4,1,2)
 assert np.isfinite(out['states']).all()
 assert out['growth_status'][:,0].tolist()==[expected,3,3,3]
 assert np.isnan(out['growth'][:,0]).all()
 assert np.isfinite(out['growth'][:,1]).all()
 assert out['cell_status']=='complete_base_censored_growth'


def test_base_failure_is_explicit_partial():
 import torch
 from long_context_map import measure_cell
 class Loop:
  def step(self,x,temps):return x*float('nan'),None
 loop=Loop();loop.torch=torch
 out=measure_cell(loop,torch.ones(3,1,2),[1]*3,[.1,.1],lambda:None)
 assert out['cell_status']=='nonfinite_base' and out['failed_step']==0
 assert len(out['states'])==0


def test_valid_growth_matches_original_algorithm():
 import torch
 import numpy as np
 from long_context_map import measure_cell
 class Loop:
  def step(self,x,temps):return x*.5,None
 loop=Loop();loop.torch=torch
 x=torch.tensor([[[1.,1.]],[[1.125,1.]],[[1.,1.25]]])
 out=measure_cell(loop,x,[1]*3,[.125,.25],lambda:None,steps=8)
 np.testing.assert_allclose(out['growth'],np.log(.5),rtol=1e-6)
 assert (out['growth_status']==0).all()


def test_recovery_scope_is_bounded():
 from runpod_scan_group import validate_orbit_job
 source='a'*40
 cmd=f'lcm1r 90 long_context_map.py --source-commit {source} --out llm-soft-loop/pod64/lcm1/recovery-01 --short-recovery'
 validate_orbit_job(cmd,source)
 with pytest.raises(ValueError):validate_orbit_job(cmd.replace('lcm1r 90','lcm1r 91'),source)
 with pytest.raises(ValueError):validate_orbit_job(cmd.replace('--short-recovery','--local-no-ack'),source)
