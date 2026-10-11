import os,sys,json,subprocess
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import review_tiny_lm_parity_order as W

def apply(ms,order,u):
 for i in order:u=ms[i]@u
 return u

def test_commuting_time_variation_can_fail_mean_but_not_parity():
 u=np.array([1.,.5]);ms=[np.diag([1+i/50, .5+i/100])for i in range(16)];v={k:apply(ms,o,u)for k,o in W.ORDERS.items()};r,_=W.reduce(u,v['P'],v['R'],v['Q'],v['T'],np.linalg.matrix_power(sum(ms)/16,16)@u,u)
 assert r['qualified']and r['even_relative_ordered']<1e-12 and r['products']['P']['gain']>0
 assert np.linalg.norm(v['P']-np.linalg.matrix_power(sum(ms)/16,16)@u)>.01

def test_noncommuting_dense_products_and_saved_vector_corruption():
 a=np.diag([1.3,.4]);q=np.array([[1.,-1.],[1.,1.]])/np.sqrt(2);b=q@a@q.T;ms=[a,b]*8;u=np.array([1.,0.]);v={k:apply(ms,o,u)for k,o in W.ORDERS.items()};r,d=W.reduce(u,v['P'],v['R'],v['Q'],v['T'],u,u)
 assert r['qualified']and r['even_relative_ordered']>.2
 assert np.allclose(v['Q']-v['P'],d['parity_even']+d['grouped_odd']-d['chronological_odd'])
 assert W.derived(d['parity_even'],d['parity_even'],list(v.values()))==0
 with pytest.raises(ValueError):W.derived(d['parity_even']+1e-3,d['parity_even'],list(v.values()))

def test_exact_native_event_direction_mutation_rejected(tmp_path):
 from hashlib import sha256
 (tmp_path/'objects').mkdir();(tmp_path/'events').mkdir()
 def obj(v):
  import io
  b=io.BytesIO();np.save(b,v,allow_pickle=False);raw=b.getvalue();h=sha256(raw).hexdigest();n='objects/'+h+'.npy';(tmp_path/n).write_bytes(raw);return dict(path=n,bytes=len(raw),sha256=h,dtype='float64',shape=list(v.shape))
 u=np.ones(15776);a=dict(case='x',operation='hvp',sequence=1,index=1,state=obj(u),direction=obj(2*u));b=dict(a,returned=obj(3*u),seconds=.1)
 for name,data in [('00001-input',a),('00001-return',b)]:(tmp_path/'events'/f'{name}.json').write_text(json.dumps(data))
 ev=W.M.Events(tmp_path,lambda:None)
 with pytest.raises(ValueError):ev.call('x','hvp',u,u)
 ev=W.M.Events(tmp_path,lambda:None);assert np.array_equal(ev.call('x','hvp',u,2*u),3*u)

def test_actual_metadata_forbids_scientific_arrays(monkeypatch):
 monkeypatch.setattr(np,'load',lambda *_a,**_k:(_ for _ in()).throw(AssertionError('target arrays opened')))
 inv,out,mean=W.metadata(lambda:None);assert len(inv)>10 and out['event_pairs']==224 and mean['counts']['hvp']==1088

def test_complete_archive_import(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path.insert(0,'tools');import review_tiny_lm_parity_order as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 out=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert out.returncode==0,out.stderr
