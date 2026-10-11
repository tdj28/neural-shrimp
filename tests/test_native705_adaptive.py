import copy,sys,subprocess,json
from pathlib import Path
import pytest
from flint import arb,ctx
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
import probe_native705_adaptive as W
import launch_native705_adaptive as L
from neural_shrimp.rigorous_orbits import vector_json


def test_fixed_policy_caps_and_halfwidth_geometry():
 assert L.LIMITS['attempts']==32 and L.LIMITS['ticks']==10592 and L.LIMITS['native_calls']==10368
 for e in range(-9,-4):
  v=W.settings(e);assert v['radius_exponent']==-10 and v['newton_steps']==2 and v['precision']==256
  assert W.exact(v['parameter'])-arb((1,e))==-arb(1)/2048
 for e in [-10,-4,True]:
  with pytest.raises(ValueError):W.settings(e)


def fixture():
 zero=vector_json([arb(0)]*705);one=zero.copy();one[0]=vector_json([arb(1)])[0]
 c=dict(shooting_dimension=705,dimension=13,period=18,normal=['1']+['0']*704,origin=['0']*705,center=zero,slope=one,curvature=zero,cubic=zero,weights=['1']*705,settings={'halfwidth_exponent':-5,'radius_exponent':-10},parameter_center='0')
 return c


def test_all705_projection_in_new_gauge_never_inherits_old_scalar():
 with ctx.workprec(256):
  c=fixture();box=vector_json([arb(1)/64]+[arb(0)]*704);a={'box':box};m=W.projected_join(a,c)
  assert m['passed']and len(m['coordinate_inclusions'])==705 and all(m['coordinate_inclusions'])
  bad=copy.deepcopy(a);bad['box'][704]=vector_json([arb(1)/512])[0];assert not W.projected_join(bad,c)['passed']
  bad=copy.deepcopy(a);bad['exact_parameter']='0'
  with pytest.raises(ValueError):W.projected_join(bad,c)


def test_completed_failure_does_not_hide_missing_native_or_changed_dimension():
 c=fixture();c['gates']={'uniform_full_native_unit_pair_locus':False,'primitive_period':True,'realified_frame_independent':True,'whole_positive_angle':True,'anchor_identified':False};c['bound']={'passed':False,'coordinate_inclusions':[False]*705}
 d=dict(attempt=1,ticks=331,events=W.EXPECTED,connected=False,status='completed_bounds_failed',passed=False,gates=c['gates'],strict_inclusions=0)
 assert L.complete_result(d,c,1,{'halfwidth_exponent':-5,'parameter':'0'})
 for key,value in [('ticks',330),('status','failed_or_interrupted')]:
  bad=copy.deepcopy(d);bad[key]=value
  with pytest.raises(ValueError):L.complete_result(bad,c,1,{'halfwidth_exponent':-5,'parameter':'0'})
 bad=copy.deepcopy(c);bad['shooting_dimension']=704
 with pytest.raises(ValueError):L.complete_result(d,bad,1,{'halfwidth_exponent':-5,'parameter':'0'})


def test_worker_and_launcher_actual_entrypoints_and_source_closure():
 root=Path(__file__).parents[1]
 for name in ['probe_native705_adaptive.py','launch_native705_adaptive.py']:
  run=subprocess.run([sys.executable,str(root/'tools'/name),'--help'],capture_output=True,text=True);assert run.returncode==0,run.stderr
 code="import sys,json;sys.path.insert(0,'tools');import launch_native705_adaptive as L;from pathlib import Path;print(json.dumps(sorted(str(Path(m.__file__).resolve().relative_to(L.ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(L.ROOT)and str(Path(m.__file__).resolve().relative_to(L.ROOT))not in L.W.SOURCES)))"
 out=subprocess.check_output([sys.executable,'-c',code],cwd=root,text=True);assert json.loads(out)==[]
