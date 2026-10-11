import json,sys,copy,subprocess
from pathlib import Path
from fractions import Fraction as Q
import pytest
from flint import ctx
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_connected as W
import launch_native705_connected as L
import probe_native705_endpoint_fallback as E
from test_native705_maintained import fixture

def test_unmodified_kernel_and_same_or_changed_gauge_controls(monkeypatch):
 with ctx.workprec(256):
  p=fixture();a=L.anchor_from(p);pin={'bytes':1,'sha256':'p'};monkeypatch.setattr(W,'entry',lambda _:pin)
  for mode in ['same_gauge','rotated_gauge']:
   card=L.make_card('s',24,-6,mode,{},p,a,{}, {},pin);assert W.validate_card(card,'s',24)
   assert ('exact_parameter'in card['anchor'])==(mode=='same_gauge')
  with pytest.raises(ValueError):W.validate_card(card,'s',25)
  card['anchor']['exact_parameter']='0'
  with pytest.raises(ValueError):W.validate_card(card,'s',24)

def test_all_profiles_and_literal_caps_and_source_closure():
 policy=W.read(ROOT/'docs/plans/native705-connected-policy.json');assert policy['limits']==L.LIMITS and policy['initial_parent']==L.PARENT
 f=policy['forecast'];g=policy['fallback_forecast']
 assert 24*f['one_fresh_chart_seconds']+8*g['forecast_seconds']+300<8400
 assert 24*f['compressed_storage_bytes']+8*g['forecast_bytes']+16*1024**2<8*1024**3
 assert L.LIMITS['native_calls']==8064 and L.LIMITS['fallback_replays']==576
 loaded={str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)}
 assert{n for n in loaded if not n.startswith('tests/')}<=set(W.SOURCES)
 for n in W.SOURCES:assert(ROOT/n).is_file()

def test_actual_retained_parent_geometry_without_new_calls():
 r=W.read(ROOT/'docs/reviews/native705-rechart-refinement002.json');assert L.PARENT['source']==r['source']and L.PARENT['outer']==r['outer']['sha256']and L.PARENT['endpoint']==r['connected_endpoint']['sha256']
 ep=W.read(Path(r['root'])/'run/connected-endpoint.json');assert len(ep['endpoint'])==705 and ep['chart']['parameter_endpoints'][1]=='63/2048'
 old=W.read(W.REF.L.BASE/'outcome.json');assert old['attempts'][4]['root_passed']and not old['attempts'][4]['joined']
 assert r['outcome']['strict_inclusions']==705 and r['outcome']['native_calls']==72

def test_entrypoints_help_and_unlaunchable_prepared_policy():
 for name in ['probe_native705_connected.py','launch_native705_connected.py','probe_native705_endpoint_fallback.py']:
  out=subprocess.run([sys.executable,str(ROOT/'tools'/name),'--help'],capture_output=True,text=True);assert out.returncode==0 and'usage:'in out.stdout
 with pytest.raises(ValueError):L.policy(ROOT/'docs/plans/native705-maintained-policy.json')

def test_child_rejects_wrong_card_source_before_acquisition():
 with pytest.raises(ValueError):E.binding({'kind':'wrong','source':'wrong'},{},lambda:None)
