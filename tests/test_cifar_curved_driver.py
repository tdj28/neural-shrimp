import copy,json,os,resource,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import explore_cifar_curved_path as W
import launch_cifar_curved_path as L

def test_card_rejects_budget_and_missing_authority():
 c=W.read(ROOT/W.CARD);c['status']='reviewed';W.check_card(c)
 for key,value in [('status','source_prepared'),('settings',dict(c['settings'],alphas=[1/16,1/32])),('operation_caps',dict(c['operation_caps'],hvp=2)),('remainder',{})]:
  d=copy.deepcopy(c);d[key]=value
  with pytest.raises(ValueError):W.check_card(d)
 assert W.OUT==L.OUT

def test_forecast_uses_complete_fixed_native_and_vector_retention():
 z=W.forecast(.001,dict(hvp=1.2,direct_gradient=1.),5.,1024**2,3*1024**2)
 assert z['passed']and z['retained_vector_ceiling']==32 and z['forecast_bytes']>40*1024**2
 assert not W.forecast(.001,dict(hvp=100.,direct_gradient=1.),5.,1,1)['passed']
 with pytest.raises(ValueError):W.forecast(float('nan'),dict(hvp=1.,direct_gradient=1.),1.,1,1)

def test_supervisor_requires_both_fixed_points_and_exact_calls():
 row=dict(status='curved_path_comparison_complete',accepted_orbit=False,rigorous_certificate=False,profile=dict(passed=True),operation_caps=W.CAPS,operation_attempted=dict(action=0,j2vp=0,hvp=1,direct_gradient=2),operation_returned=dict(action=0,j2vp=0,hvp=1,direct_gradient=2),direction_applied=True,correction=dict(accepted_index=0,trials=[dict(admitted=True),dict(admitted=True)]))
 assert L.scientific_complete(row)
 row['operation_returned']['direct_gradient']=1;assert not L.scientific_complete(row)
 row['operation_attempted']['direct_gradient']=1;assert not L.scientific_complete(row)
 row['correction']['trials'][1]['admitted']=False;assert L.scientific_complete(row)
 row['correction']['trials'].pop();assert not L.scientific_complete(row)

def test_native_return_retained_before_post_call_budget_stop(tmp_path):
 store=W.Store(tmp_path/'run',limit=8*1024**2);b=W.Budget(store.event)
 class Problem:
  def hvp(self,x,v):
   def work():b.stopped=True;return W.B.torch.ones(98666,dtype=W.B.torch.float64)
   return b.call('hvp',work)
 with pytest.raises(ValueError):W.E.E.Native(Problem(),store,b).hvp(np.zeros(98666),np.ones(98666))
 events=[json.loads(s)for s in(tmp_path/'run/events.jsonl').read_text().splitlines()]
 assert [e['kind']for e in events]==['native_input','operation_started','operation_returned','native_output']
 assert b.attempted==b.returned and b.returned['hvp']==1
 assert (tmp_path/'run'/events[-1]['result']['file']).is_file()

def test_exact_actual_binding_before_opening_target_arrays(monkeypatch,tmp_path):
 W.install();c=W.read(ROOT/W.CARD);c['status']='reviewed'
 inv={n:W.entry(ROOT/n)['sha256']for n in W.B.SOURCES};old=W.read(W.A.PARENT/'run/result.json');store=W.Store(tmp_path/'run',limit=16*1024**2);b=W.Budget(store.event)
 def forbidden(*a,**kw):raise AssertionError('scientific arrays opened during preflight')
 monkeypatch.setattr(np,'load',forbidden)
 r=W.bind(c,b,store,inv,old['runtime']);assert r[-1]==dict(hvp=514,direct_gradient=9)and not any(b.attempted.values())

def test_complete_archive_import_and_inherited_as(tmp_path):
 W.install()
 for n in W.B.SOURCES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import explore_cifar_curved_path as w;w.install();r=Path.cwd();a={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert a<=set(w.B.SOURCES),a-set(w.B.SOURCES)"
 z=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True);assert z.returncode==0,z.stderr
 # Execute module as __main__ but stop before scientific clean-source guard.
 import ast
 tree=ast.parse((tmp_path/'tools/explore_cifar_curved_path.py').read_text());tree.body=tree.body[:-1]
 p=tmp_path/'tools/import_preflight.py';p.write_text(ast.unparse(tree))
 def limits():resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,12*1024**3))
 z=subprocess.run([sys.executable,'-B',str(p)],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),preexec_fn=limits,capture_output=True,text=True);assert z.returncode==0,z.stderr
