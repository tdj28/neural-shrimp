import os,sys,json,subprocess
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import review_tiny_lm_pair_blocks as W

def apply(ms,order,u):
 for i in order:u=ms[i]@u
 return u

def test_pairs_preserve_identical_two_step_dynamics():
 a=np.diag([1.3,.4]);q=np.array([[1.,-1.],[1.,1.]])/np.sqrt(2);b=q@a@q.T;ms=[a,b]*8;u=np.array([1.,0.])
 p=apply(ms,range(16),u);r=apply(ms,range(15,-1,-1),u);x=apply(ms,W.ORDERS['U'],u);y=apply(ms,W.ORDERS['V'],u)
 row,vec=W.reduce(u,p,r,x,y,u,u,10*u)
 assert row['qualified']and row['even_norm']<1e-12 and np.array_equal(x,p)and np.array_equal(y,r)
 assert W.P.derived(vec['pair_block_even'],vec['pair_block_even'],[p,r,x,y])==0
 with pytest.raises(ValueError):W.P.derived(vec['pair_block_even']+1e-3,vec['pair_block_even'],[p,r,x,y])

def test_distinct_gates_require_every_original_and_every_qualification():
 rows=[dict(seed=s,rate_role=r,qualified=True,comparison=dict(even_relative_ordered=1.,even_relative_parity_even=.01))for s in(28161,28162)for r in('original','quarter_step')]
 assert all(W.gates(rows).values())
 rows[2]['comparison']['even_relative_parity_even']=.2
 assert W.gates(rows)['material_in_both_originals']and not W.gates(rows)['tenfold_attenuation_in_both_originals']
 rows[3]['qualified']=False
 assert not any(W.gates(rows).values())
 with pytest.raises(ValueError):W.gates(rows[:-1])

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
 inv,out,auth,mean=W.metadata(lambda:None);assert len(inv)>10 and out['event_pairs']==128 and auth['retained_hvp_checked']==224 and mean['counts']['hvp']==1088

def test_complete_archive_import(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path.insert(0,'tools');import review_tiny_lm_pair_blocks as w;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 out=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert out.returncode==0,out.stderr
