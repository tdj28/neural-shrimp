import sys,copy,subprocess,os
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_cifar_nested512 as R

def fixture():
 rng=np.random.default_rng(51);g=rng.normal(size=8);h=rng.normal(size=(8,8));h=(h+h.T)/2;V=[-g/np.linalg.norm(g)];W=[];rows=[]
 for j in range(5):
  W.append(h@V[-1])
  if j==4:break
  q,c=R.pair_project(np.column_stack(V),W[-1]);rows.append(dict(coefficients=[a.tolist()for a in c],residual_norm=np.linalg.norm(q),raw_action_norm=np.linalg.norm(W[-1])));V.append(q/np.linalg.norm(q))
 return np.column_stack(V),np.column_stack(W),g,rows

def test_all_recurrence_and_corruptions():
 V,W,g,rows=fixture();assert len(R.recurrence(V,W,g,rows))==4
 for kind in ('vector','coefficient','residual','gradient'):
  v,w,gg,rr=V.copy(),W.copy(),g.copy(),copy.deepcopy(rows)
  if kind=='vector':v[0,2]+=.01
  elif kind=='coefficient':rr[1]['coefficients'][1][0]+=.01
  elif kind=='residual':rr[1]['residual_norm']*=2
  else:gg=-gg
  with pytest.raises(ValueError):R.recurrence(v,w,gg,rr)

def test_fresh_action_and_improvement_gates():
 g=np.ones(5);r=R.comparison(g,-.35*g,-.58*g,-.58*g,0,True);assert r['strict_forcing_pass']and r['material_improvement']
 assert not R.comparison(g,-.35*g,-.58*g,-.5*g,0,True)['qualified']
 assert not R.comparison(g,-.35*g,-.58*g,-.58*g,0,False)['qualified']

def test_complete_event_sequence_rejects_missing_or_duplicate():
 events=[];manifest={'x':{},'v':{},'w':{}}
 def call(i):
  q=dict(operation='hvp',index=i);return [dict(kind='native_input',state=dict(file='x',identity={}),direction=dict(file='v',identity={}),**q),dict(kind='operation_started',**q),dict(kind='operation_returned',seconds=.01,**q),dict(kind='native_output',result=dict(file='w',identity={}),**q)]
 events+=call(1)
 for i in range(256):events.append(dict(kind='nested_reorthogonalization',from_column=255+i,status='next_vector_qualified'));events+=call(i+2)
 events+=call(258);assert len(R.events_bound(events,manifest)[0])==258
 for changed in (events[:-1],events[:4]+events[5:],events+[events[-1]]):
  with pytest.raises((ValueError,KeyError)):R.events_bound(changed,manifest)

def test_archive_import_closure(tmp_path):
 for n in R.FILES:
  q=tmp_path/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=[str(Path.cwd()/'tools/reviews')];import review_cifar_nested512 as r;loaded={str(Path(m.__file__).resolve().relative_to(Path.cwd())) for m in sys.modules.values() if getattr(m,'__file__',None) and Path.cwd() in Path(m.__file__).resolve().parents};assert loaded<=set(r.FILES),loaded-set(r.FILES)"
 subprocess.run([sys.executable,'-I','-c',code],cwd=tmp_path,check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))

def test_caps_and_source_scope():
 assert len(R.EXEC)==40 and 'cifar_nested_krylov'not in Path(R.__file__).read_text()
 s=(ROOT/'tools/reviews/launch_cifar_nested512_review.py').read_text();assert "'internal_seconds':300"in s and "'external_seconds':315"in s and '4*1024**2'in s and '10*1024**2'in s
