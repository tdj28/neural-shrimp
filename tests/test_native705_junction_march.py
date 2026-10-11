import copy,sys,subprocess,time
from pathlib import Path
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import ball_json
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_junction_march as W
import bind_native705_junction_march as B
import launch_native705_junction_march as L
import launch_native705_connected as OLD
import native705_completion_event as E
from test_native705_maintained import fixture

def test_actual_parent_and_endpoint_and_lift_authority():
 r=W.read(ROOT/'docs/reviews/native705-maintained003.json');last=B.contract(r['outcome'],r)
 assert last['attempt']==24 and B.STATE==(-3,0,'same_gauge')and W.entry(B.PREVIOUS/'connected-endpoint.json')['sha256']==B.ENDPOINT
 for k,v in [('connected_count',23),('attempted',23),('fallback_count',1)]:
  bad=copy.deepcopy(r);bad['outcome'][k]=v
  with pytest.raises(ValueError):B.contract(bad['outcome'],bad)
 p=W.read(L.POLICY);lift=W.read(ROOT/'docs/reviews/native705-junction-lift003.json')
 assert p['initial_parent']==B.PARENT and p['initial_state']==list(B.STATE)
 assert p['completion_target']['outer']==lift['outer']['sha256']and p['completion_target']['lift']==lift['lift']
 assert lift['outcome']['native_calls']==108 and lift['outcome']['events']==228 and lift['outcome']['passed']

def synthetic_chart():
 p=fixture();p.update(dimension=13,period=18,shooting_dimension=705,parameter_center='0',parameter_endpoints=['-1/8','1/8'],settings=dict(halfwidth_exponent=-3,radius_exponent=-10),bound=dict(passed=True,coordinate_inclusions=[True]*705),gates=dict(uniform_full_native_unit_pair_locus=True,primitive_period=True,realified_frame_independent=True,whole_positive_angle=True,anchor_identified=False))
 return p

def test_full705_membership_and_critical_corruptions():
 with ctx.workprec(480):
  p=synthetic_chart();z=[ball_json(arb(0))for _ in range(705)];z[-1]=ball_json(arb(1)/32)
  t=dict(box=z,authority={'synthetic':True});begin=time.monotonic();a=E.membership(t,p)
  assert a['passed']and a['strict_inclusions']==705 and time.monotonic()-begin<2
  for index in [0,13,26,700,702,703]:
   bad=copy.deepcopy(t);bad['box'][index]=ball_json(arb(1)/512)
   q=E.membership(bad,p);assert not q['passed']and not q['known_root_join']['coordinate_inclusions'][index]
  bad=copy.deepcopy(t);bad['box'][-1]=ball_json(arb(1)/4)
  assert not E.membership(bad,p)['known_root_join']['parameter_inside']
  bad=copy.deepcopy(p);bad['bound']['coordinate_inclusions'][700]=False
  with pytest.raises(ValueError):E.membership(t,bad)

def test_event_cannot_use_an_invented_exact_scalar():
 with ctx.workprec(480):
  p=synthetic_chart();z=[ball_json(arb(0))for _ in range(705)];z[-1]=ball_json(arb(1)/4)
  t=dict(box=z,authority={},exact_parameter='0');assert not E.membership(t,p)['passed']

def test_old_defaults_and_fixed_costs_and_entrypoints():
 assert OLD.COMPLETION_BINDER is OLD.COMPLETION_TEST is None
 assert OLD.INITIAL_STATE==(-6,0,'same_gauge')and OLD.WORKER_PATH.name=='probe_native705_connected.py'
 assert L.LIMITS==dict(OLD.LIMITS,initial_halfwidth=-3)
 p=W.read(L.POLICY);assert24=24*p['forecast']['one_fresh_chart_seconds']+8*p['fallback_forecast']['forecast_seconds']+300;assert assert24<8400
 with pytest.raises(ValueError):L.policy(ROOT/'docs/plans/native705-resumed-policy.json')
 for n in ['probe_native705_junction_march.py','launch_native705_junction_march.py','probe_native705_junction_fallback.py']:
  result=subprocess.run([sys.executable,str(ROOT/'tools'/n),'--help'],capture_output=True,text=True);assert result.returncode==0 and 'usage:'in result.stdout
 loaded={str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)}
 assert{n for n in loaded if not n.startswith('tests/')}<=set(W.SOURCES)
 for n in W.SOURCES:assert(ROOT/n).is_file()
