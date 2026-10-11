import copy,json,subprocess,sys
from pathlib import Path
import pytest
from flint import ctx
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_resumed as W
import bind_native705_resumed as B
import launch_native705_resumed as L
import launch_native705_connected as OLD
import probe_native705_endpoint_fallback as F

def test_actual_complete_parent_and_inherited_policy_state():
 r=W.read(ROOT/'docs/reviews/native705-maintained002.json');last=B.contract(r['outcome'],r)
 assert last['attempt']==24 and B.STATE==(-3,1,'same_gauge')
 assert r['outer']['sha256']==B.OUTER and r['connected_endpoint']['sha256']==B.ENDPOINT
 assert W.entry(B.PREVIOUS/'connected-endpoint.json')['sha256']==B.ENDPOINT
 for k,v in [('connected_count',19),('attempted',23),('fallback_count',1)]:
  bad=copy.deepcopy(r);bad['outcome'][k]=v
  with pytest.raises(ValueError):B.contract(bad['outcome'],bad)
 bad=copy.deepcopy(r);bad['outcome']['attempts'][23]['joined']=False
 with pytest.raises(ValueError):B.contract(bad['outcome'],bad)

def test_actual_next_finite_card_and_entire_known_scalar(monkeypatch):
 with ctx.workprec(256):
  # Actual retained finite proposal/endpoint; no equations or derivative calls.
  r=W.read(B.PREVIOUS/'attempt024/run/certificate.json.gz');a=OLD.anchor_from(r);pin={'bytes':1,'sha256':'p'}
  monkeypatch.setattr(OLD,'W',W);monkeypatch.setattr(W.C,'BASE',W.BASE);monkeypatch.setattr(W.C,'entry',lambda _:pin)
  c=OLD.make_card('s',1,-3,'same_gauge',{},r,a,{}, {},pin)
  assert W.validate_card(c,'s',1)and c['anchor']['exact_parameter']==r['parameter_endpoints'][1]
  assert c['settings']['halfwidth_exponent']==-3 and len(c['initial'])==705
  for v in c['initial']:assert W.W.exact_proposal(v)

def test_old_defaults_unchanged_and_unlaunchable_prepared_resume():
 assert OLD.POLICY_PATH==ROOT/'docs/plans/native705-connected-policy.json'
 assert OLD.WORKER_PATH.name=='probe_native705_connected.py'and OLD.INITIAL_STATE==(-6,0,'same_gauge')
 assert F.POLICY_PATH==OLD.POLICY_PATH
 assert L.LIMITS==dict(OLD.LIMITS,initial_halfwidth=-3)
 p=W.read(L.POLICY);assert p['initial_parent']==B.PARENT and p['initial_state']==list(B.STATE)
 with pytest.raises(ValueError):L.policy(ROOT/'docs/plans/native705-connected-policy.json')

def test_entrypoints_and_archive_closure():
 for name in ['probe_native705_resumed.py','launch_native705_resumed.py','probe_native705_resumed_fallback.py']:
  r=subprocess.run([sys.executable,str(ROOT/'tools'/name),'--help'],capture_output=True,text=True);assert r.returncode==0 and'usage:'in r.stdout
 loaded={str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)}
 assert{n for n in loaded if not n.startswith('tests/')}<=set(W.SOURCES)
 for n in W.SOURCES:assert(ROOT/n).is_file()
