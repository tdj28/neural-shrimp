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
