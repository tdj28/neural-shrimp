import importlib.util,json,pathlib,sys,os,subprocess,resource
import numpy as np
import pytest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews')]
import audit_cifar_curved_saved as W
from review_cifar_curved_math import arithmetic,path

def fixture(n=4):
 x=np.ones(n);g=np.full(n,.1);p=-g;a=p/10;states=[path(x,p,a,t)for t in(1/32,1/64)];grads=[g+y-x for y in states];return x,g,p,a,states,grads

def test_independent_scalar_path_and_first_selection():
 x,g,p,a,ys,gs=fixture();r=arithmetic(x,g,x,p,p,p,a,a,a,ys,gs)
 assert r['accepted_index']==0 and all(z['accepted']for z in r['trials'])and r['tangent']['true_forcing']==0
 for t,y in zip([1/32,1/64],ys):assert np.array_equal(y,x+t*p+t*t*a)

def test_reject_changed_path_and_actual_action():
 x,g,p,a,ys,gs=fixture();ys[1]=ys[1]+1e-8
 with pytest.raises(ValueError,match='coordinates'):arithmetic(x,g,x,p,p,p,a,a,a,ys,gs)
 x,g,p,a,ys,gs=fixture()
 with pytest.raises(ValueError,match='acceleration'):arithmetic(x,g,x,p,p,p,a,2*a,a,ys,gs)

def test_failed_first_is_retained_not_best_selected():
 x,g,p,a,ys,gs=fixture();gs[0]=g*2
 r=arithmetic(x,g,x,p,p,p,a,a,a,ys,gs);assert r['accepted_index']==1 and not r['trials'][0]['accepted']
 gs=[g*.98,g*.9];r=arithmetic(x,g,x,p,p,p,a,a,a,ys,gs);assert r['accepted_index']==0 and r['trials'][1]['merit']<r['trials'][0]['merit']

def test_actual_metadata_has_no_array_reads(monkeypatch):
 class D:
  def tick(self):pass
  def save(self,*a,**kw):pass
 def bad(*a,**kw):raise AssertionError('scientific arrays forbidden')
 monkeypatch.setattr(np,'load',bad);b,o,p,r=W.metadata(D());assert len(b)==5 and len(o)==3 and len(p)==523

def test_wrong_native_identity_or_event_order_reject(tmp_path):
 r=[dict(kind='native_input',operation='hvp',index=1,state=dict(file='s.npy',identity={'sha256':'a','bytes':1}),direction=dict(file='v.npy',identity={'sha256':'b','bytes':1})),dict(kind='operation_started',operation='hvp',index=1),dict(kind='operation_returned',operation='hvp',index=1,seconds=.1),dict(kind='native_output',operation='hvp',index=1,result=dict(file='r.npy',identity={'sha256':'c','bytes':1}))]
 inv={e['file']:e['identity']for e in[r[0]['state'],r[0]['direction'],r[-1]['result']]};p=tmp_path/'events.jsonl';p.write_text('\n'.join(json.dumps(v)for v in r));assert len(W.events(tmp_path,inv)[0])==1
 inv['r.npy']=dict(sha256='x',bytes=1)
 with pytest.raises(ValueError,match='pointer'):W.events(tmp_path,inv)
 inv['r.npy']=r[-1]['result']['identity'];r[2]['index']=2;p.write_text('\n'.join(json.dumps(v)for v in r))
 with pytest.raises(ValueError,match='matches'):W.events(tmp_path,inv)

def test_archive_import_and_failure_closing_reserve(tmp_path,monkeypatch):
 for n in W.FILES:
  q=tmp_path/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','tools/reviews'];import audit_cifar_curved_saved as w;r=Path.cwd();a={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert a<=set(w.FILES),a-set(w.FILES)"
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');z=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=env);assert z.returncode==0,z.stderr
 monkeypatch.setattr(W,'OUT',tmp_path/'output');s=W.Work();s.used=W.CAPS['raw_bytes']-W.CAPS['reserve_bytes']
 with pytest.raises(ValueError,match='reserved'):s.save('regular.json',{})
 s.save('failed.json',{'passed':False},True)


def test_real_preimport_address_space(tmp_path):
 import ast
 tree=ast.parse((ROOT/'tools/audit_cifar_curved_saved.py').read_text());tree.body=tree.body[:-1]
 # Preserve own helper discovery while executing the actual startup body as main.
 code=ast.unparse(tree).replace("ROOT=Path(__file__).resolve().parents[1]", "ROOT=Path("+repr(str(ROOT))+")")
 code=code.replace("ROOT = Path(__file__).resolve().parents[1]", "ROOT = Path("+repr(str(ROOT))+")")
 p=tmp_path/'import.py';p.write_text(code)
 def limit():resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,3*1024**3))
 z=subprocess.run([sys.executable,'-B',str(p)],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),preexec_fn=limit,capture_output=True,text=True);assert z.returncode==0,z.stderr
