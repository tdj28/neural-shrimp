"""Retain the entire unlaunched panel; saved selection metadata only."""
import hashlib,json,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
SEL=Path('/data2/neural-shrimp-data/audits/EXP-022/selection-001/selection.json')
SHA='b8bcf7f11fc82623993e07af54cc3d6a42a338d18561cc148f4316329952416b'
def bind(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(name,obj):
 with (OUT/name).open('x') as f:json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
start=time.monotonic();assert bind(SEL)['sha256']==SHA
r=json.loads(SEL.read_text());slots=r['slots'];missing=[s['id'] for s in slots if not s['available']]
assert missing==['core-rho025','core-rho050']
workers=[];nested=[]
def row(ident,kind,slot,**more):
 return dict(id=ident,kind=kind,cycle_id=slot['id'],historical_slot_available=slot['available'],
  status='not_launched',reason='full_panel_missing_historical_support',ivp_started=False,ivp_completed=False,**more)
for s in slots:
 c=s['id']
 for m in ('tight-dop853','tight-rk45'):
  ident=c+'--'+m;workers.append(row(ident,'shoot',s))
  for i in range(25):nested.append(row(ident+f'--trial-{i:02d}','shoot_trial',s,worker_id=ident))
 ref=c+'--reference-pair';workers.append(row(ref,'reference',s))
 for candidate in ('tight-dop853','tight-rk45'):
  for m in ('tight-dop853','tight-rk45'):
   nested.append(row(c+'--verify--'+candidate+'--'+m,'cross_forward',s,worker_id=ref))
 for m in ('tight-dop853','tight-rk45'):nested.append(row(c+'--adjoint--'+m,'adjoint',s,worker_id=ref))
 for i in range(8):
  for m in ('dop853','rk45'):
   ident=c+f'--p{i:02d}--reference--'+m
   workers.append(row(ident,'tangent',s,phase_index=i))
   for seg in ('pre','post'):nested.append(row(ident+'--'+seg,'tangent_segment',s,worker_id=ident,segment=seg))
assert len(workers)==len({x['id'] for x in workers})==114
assert len(nested)==len({x['id'] for x in nested})==528
branches=[]
for b in r['branch_audits']:
 nodes=b['nodes'];p=[n for n in nodes if n['in_initial_attracting_prefix']];eligible=[n for n in nodes if n['eligible']]
 branches.append(dict(branch_id=b['branch_id'],nodes=len(nodes),prefix_nodes=len(p),prefix_stop=b['prefix_stop'],eligible_nodes=len(eligible),
  prefix_positive_real_dominant_min=min(n['dominant_rho'] for n in eligible),prefix_positive_real_dominant_max=max(n['dominant_rho'] for n in eligible)))
core=r['branch_audits'][0]['nodes'];brackets=[]
for target,i,j in ((.25,14,15),(.5,15,16)):
 a,b=core[i],core[j]
 assert a['eligible'] and b['eligible'] and a['dominant_rho']<target<b['dominant_rho']
 fraction=(target-a['dominant_rho'])/(b['dominant_rho']-a['dominant_rho'])
 near=a if fraction<=.5 else b
 brackets.append(dict(target=target,left={k:a[k] for k in ('node_id','index','alpha','gamma','period','dominant_rho','first_maximum_state','node_metadata_sha256')},
  right={k:b[k] for k in ('node_id','index','alpha','gamma','period','dominant_rho','first_maximum_state','node_metadata_sha256')},
  fraction=fraction,proposed_interpolated_gamma=a['gamma']+fraction*(b['gamma']-a['gamma']),proposed_nearest_endpoint_index=near['index'],
  endpoint_scaled_state_separation=max(abs(x-y)/max(1.,abs(z)) for x,y,z in zip(a['first_maximum_state'],b['first_maximum_state'],near['first_maximum_state'])),
  endpoint_relative_period_separation=abs(a['period']-b['period'])/near['period'],
  new_parameter_acquisition_performed=False))
save('not-launched-ledger.json',dict(workers=workers,nested_ivp_slots=nested))
report=dict(experiment='EXP-022',status='historical_support_incomplete_no_neural_launch',
 selection=bind(SEL),planned_slots=6,available_slots=4,missing_slots=missing,branches=branches,
 slots=[{k:s[k] for k in ('id','available','status','node_id','dominant_rho','alpha','gamma','period')} for s in slots],
 intended_workers=114,intended_nested_ivp_slots=528,workers_launched=0,ivps_launched=0,
 matching_pairs_planned=2,matching_pairs_with_both_historical_candidates=0,
 prospective_gap_brackets=brackets,ledger=bind(OUT/'not-launched-ledger.json'),
 interpretation='The discrete historical core path skipped both fixed bins within its attracting prefix. This proves insufficient sampled metadata support, not absence of comparable attracting cycles. No new memory performance, gain or cycle qualification was measured.',
 original_studies_requalified=False,selector_rules_changed_after_availability=False,
 author_scope='Selector author closure and arithmetic report; root performs separate independent selection audit.',
 wall_seconds=time.monotonic()-start,source=bind(Path(__file__)))
save('report.json',report)
print(json.dumps(dict(report=bind(OUT/'report.json'),ledger=bind(OUT/'not-launched-ledger.json'),source=bind(Path(__file__))),indent=2))
