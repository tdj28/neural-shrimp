import copy,sys,os,subprocess
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'tools/reviews')]
import review_cifar_current512 as R


def test_actual_trust_skips_then_first_acceptance():
 x=np.array([1.,0.]);g=np.array([.01,0.]);p=np.array([.16,0.]);hp=-.8*g
 # alpha1,.5 exceed5%; alpha.25 is strictly interior but fails merit.
 trials=[None,None,(x+.25*p,2*g),(x+.125*p,.9*g)]
 a,y,gy=R.acceptance(x,x,g,p,hp,hp,trials,True)
 assert a['accepted_index']==3 and a['direction_gate']['passed']and not a['stationarity_qualified']and np.array_equal(y,x+.125*p)
 bad=copy.deepcopy(trials);bad[0]=(x+p,g)
 with pytest.raises(ValueError):R.acceptance(x,x,g,p,hp,hp,bad,True)
 bad=copy.deepcopy(trials);bad[-1]=(x+.1*p,.9*g)
 with pytest.raises(ValueError):R.acceptance(x,x,g,p,hp,hp,bad,True)
 with pytest.raises(ValueError):R.acceptance(x,x,g,p,hp,-.3*g,trials,True)
 with pytest.raises(ValueError):R.acceptance(x,x,g,p,hp,hp,trials,False)


def synthetic_events():
 out=[];mf={'x':{},'v':{},'w':{}};counts={'hvp':0,'direct_gradient':0}
 def call(op):
  counts[op]+=1;q=dict(operation=op,index=counts[op]);v=dict(kind='native_input',state=dict(file='x',identity={}),**q)
  if op=='hvp':v['direction']=dict(file='v',identity={})
  out.extend([v,dict(kind='operation_started',**q),dict(kind='operation_returned',seconds=.1,**q),dict(kind='native_output',result=dict(file='w',identity={}),**q)])
 call('direct_gradient');call('hvp');call('hvp')
 for _ in range(4):call('direct_gradient')
 for i in range(511):out.append(dict(kind='nested_reorthogonalization',from_column=i,status='next_vector_qualified'));call('hvp')
 call('hvp');out.append(dict(kind='current_direction_gate',passed=True))
 for i in range(7):
  if i>=3:call('direct_gradient')
  out.append(dict(kind='current_merit_trial',index=i,within_trust=i>=3,accepted=i==6))
 return out,mf


def test_complete_native_order_and_hidden_trial_corruptions():
 events,manifest=synthetic_events();counts,calls,rows,gate,trials=R.events_bound(events,manifest);assert counts==dict(hvp=514,direct_gradient=9)and len(rows)==511 and len(trials)==7
 for es in [events[:-1],events+[events[-1]],events[:28]+events[29:]]:
  with pytest.raises((ValueError,KeyError)):R.events_bound(es,manifest)
 bad=copy.deepcopy(events);next(e for e in bad if e['kind']=='nested_reorthogonalization')['from_column']=2
 with pytest.raises(ValueError):R.events_bound(bad,manifest)


def test_profile_bounds_and_no_native_or_producer_reducer_import():
 s=Path(R.__file__).read_text();assert 'import explore_cifar'not in s and 'cifar_current_correction import'not in s and 'native_calls=0'in s
 assert len(R.EXEC)==40 and all(len(x)==64 for x in R.PINS)


def test_archived_import_source_closure(tmp_path):
 for n in R.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=[str(Path.cwd()/'tools/reviews')];import review_cifar_current512 as r;loaded={str(Path(m.__file__).resolve().relative_to(Path.cwd())) for m in sys.modules.values() if getattr(m,'__file__',None) and Path.cwd() in Path(m.__file__).resolve().parents};assert loaded<=set(r.FILES),loaded-set(r.FILES)"
 subprocess.run([sys.executable,'-I','-c',code],cwd=tmp_path,check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))


def test_native_metadata_actual_stream_without_vectors():
 import json
 events=[json.loads(v)for v in(R.PARENT/'run/events.jsonl').read_text().splitlines()];manifest=R.M.read(R.PARENT/'run/manifest.json')['files']
 counts,calls,rows,gate,trials=R.events_bound(events,manifest)
 assert counts==dict(hvp=514,direct_gradient=9)and len(rows)==511 and len(trials)==7 and all(v['state']=='state-000.npy'for (op,_),v in calls.items()if op=='hvp')
