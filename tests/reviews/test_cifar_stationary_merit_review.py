import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('review',ROOT/'tools/reviews/review_cifar_stationary_merit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def fixture():
 x=np.ones(7);g=x*.01;p=-x*.01;hp=-g*.25
 return x,g,p,hp

def test_relaxed_descent_and_true_merit():
 x,g,p,hp=fixture();r=m.arithmetic(x,g,p,hp,[(x+p,g*.8)])
 assert r['direction_gate']['forcing']==pytest.approx(.75)
 assert r['direction_gate']['passed']and not r['direction_gate']['old_forcing_pass']
 assert r['merit_ratio']==pytest.approx(.64)and r['accepted_index']==0

def test_exact_candidate_and_stop():
 x,g,p,hp=fixture()
 with pytest.raises(ValueError,match='candidate'):m.arithmetic(x,g,p,hp,[(x+p+.0001,g*.8)])
 with pytest.raises(ValueError,match='after acceptance'):m.arithmetic(x,g,p,hp,[(x+p,g*.8),(x+.5*p,g*.8)])

def test_false_report_and_small_scalar_mismatch():
 with pytest.raises(ValueError):m.compare({'accepted':True},{'accepted':False})
 with pytest.raises(ValueError):m.close(1e-7,1e-7+1e-16,'small merit')

def test_native_event_hash_and_state(tmp_path):
 # Synthetic event shape matches the native producer; no model is imported.
 n=98666;x=np.ones(n);y=x*.1
 for name,a in [('state.npy',x),('return.npy',y)]:np.save(tmp_path/name,a)
 item=lambda name:dict(file=name,identity=m.entry(tmp_path/name))
 events=[dict(kind='native_input',operation='direct_gradient',index=1,state=item('state.npy')),dict(kind='operation_started',operation='direct_gradient',index=1),dict(kind='operation_returned',operation='direct_gradient',index=1,seconds=.1),dict(kind='native_output',operation='direct_gradient',result=item('return.npy'))]
 assert m.event_block(events,0,'direct_gradient',1,tmp_path,x,y)==4
 events[0]['state']['identity']['sha256']='0'*64
 with pytest.raises(ValueError,match='binding'):m.event_block(events,0,'direct_gradient',1,tmp_path,x,y)

def test_archive_import_closure(tmp_path):
 for name in m.FILES:
  p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
 code="import importlib.util,sys;from pathlib import Path;p=Path(sys.argv[1]);s=importlib.util.spec_from_file_location('audit',p/'tools/reviews/review_cifar_stationary_merit.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert not any(k.startswith('neural_shrimp') or k.startswith('torch') for k in sys.modules)"
 subprocess.run([sys.executable,'-I','-c',code,str(tmp_path)],check=True,cwd=tmp_path)


def test_launcher_hard_as_allows_reviewed_child_limit():
 text=(ROOT/'tools/reviews/launch_cifar_stationary_merit_review.py').read_text()
 assert 'resource.RLIMIT_AS,(512*1024**2,2*1024**3)'in text
 assert 'resource.RLIMIT_AS,(2*1024**3,)*2)'in (ROOT/'tools/reviews/review_cifar_stationary_merit.py').read_text()
