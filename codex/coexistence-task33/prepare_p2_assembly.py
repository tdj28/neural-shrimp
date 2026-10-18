import sys,json,hashlib
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');D=R/'codex/coexistence-task33';RAW=Path('/data2/neural-shrimp-data/codex/coexistence');sys.path.insert(0,str(D))
import assembly_core as a
sha=a.points.sha
def pin(p):p=Path(p);return dict(path=str(p),sha256=sha(p))
def put(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
pp=json.loads((D/'p2-point-bridge-plan.json').read_text());packet=json.loads((D/'p2-dispatch-001.json').read_text());dp=json.loads((R/packet['plans'][0]['plan']).read_text());definition=json.loads(Path(dp['definition_path']).read_text())
review=D/'assembly-source-review.md'
review.write_text('''# Task33 assembly source and hypothesis review

Root source review, 18 October 2026. This isolated caller uses the previously reviewed scalar/Jet/streamed exact-real evaluator and generic shooting gates unchanged. The historical evaluator review states the P8 cell; its analytic argument extends to this explicitly different fixed binary64 alpha .06468 and log10T -.148: steering remains affine and temperature exp(log(10)*theta) remains positive. Stored assets/prefix/layer are independently bound by the Task33 point and derivative adapters. Period2 changes only the number of shooting phases, not the analytic graph or the generic gate proof. This is an explicit scoped extension, not a relabelled P8 certificate.

Root read the assembly caller, point bridge, derivative adapter, original closure schema and generic gate wrapper. Point images retain original2048-bit radii under outward1024-bit decoding; every native centre is exact at1024. Signed original whole-box derivatives cover every coordinate once per phase. Full point/derivative map, frozen domain, actual native matrices and exact dyadic A/B proposals are bound before conditional gates. Original primal intersection agreement is a consistency check, not an enclosure proof. Exact source-set checks and review binding prohibit a missing dependency or different map. Original outer commands and producer termination receipts are crossbound; no remote process query. Assembly uses original facts, no model reevaluation.

The mathematical obligations remain the accepted C1 map/whole-box enclosure premises plus strict shooting inclusion/contraction, all proper-divisor separations and signed monodromy-power norms. Failed gates are inconclusive sufficient-condition failures. The generic wrapper never releases a certificate; every gate and witness is retained for independent saved-evidence review. Token labels and fixed-start basin reachability are excluded. Arithmetic relies on the pinned trusted Arb implementation. No claim is made before actual gates and final review.

Budget: accepted full-size eight-phase synthetic gate workload took988.804s/6.731GiB RSS at1024bits, while this P2 workload has two phases; source/28original-return authentication and native conversion add cost. Admit oneCPU,18GiBAS,3500CPU/3600wall,2GiB output,32GiBstartup/16GiBruntime floor. This is a bounded forecast, not a guaranteed completion time. No retry or extension. Source delta controls check required pins/review scopes/map binding and original closure refusals; unchanged gate mathematics controls are reused.
''')
reviews={x:pin(review)for x in ['point_evaluator','derivative_evaluator','map_regularity']};reviews['gate_mathematics']=pin(R/'codex/certify-llm/shooting-gates-review.md')
# Preserve historical review authority explicitly in the new binding.
historical=pin(R/'codex/certify-llm/final-evaluator-obligations-review.md')
rows=[]
for file in ['return-p2a-operational.json','return-p3bd-p2f-operational.json','return-1626-operational.json']:
 for row in json.loads((D/file).read_text())['attempts']:
  name=row['attempt']
  if not name.startswith('p2-'):continue
  if 'outer_closure_path' in row:outer=Path(row['outer_closure_path']);term=Path(row['termination_record_path'])
  else:
   outer=RAW/'return-t33-p2-a-001/explore/llm-soft-loop/pod64/certpack/t33-p2-a'/name/'run.json'
   term=Path('/data2/neural-shrimp-data/codex/remote/t33-p2-a/returned/20261018T150059Z/return-summary.json')
  assignment=next(x for x in packet['plans']if x['attempt']==name)
  rows.append(dict(attempt=str(RAW/name),receipt_sha256=row['original_archive_receipt_sha256'],source_commit='d627b74affa176037f553332d6fb5b26d1dd0b37',phase_index=assignment['phase'],start_column=assignment['start_column'],outer_closure=pin(outer),termination=pin(term)))
rows.sort(key=lambda x:(x['phase_index'],x['start_column']));assert len(rows)==28
prefix=dict(attempt=dp['prefix_attempt'],receipt_sha256=dp['prefix_receipt_sha256'],map_source_sha256=dp['prefix_map_source_sha256'],cache_sha256=dp['prefix_cache_sha256'],source_commit=dp['prefix_source_commit'],receipt_path=dp['prefix_receipt'],admission_path=dp['prefix_admission_path'],admission_sha256=dp['prefix_admission_sha256'])
sources={x:sha(R/x) for x in sorted(a.REQUIRED_SOURCES)}
binding=dict(status='source_reviewed_conditional_enclosures',map=definition['map'],derivative_source_sha256=dp['source_sha256'],reviews=reviews,assembly_source_sha256=sources,point_plan=pin(D/'p2-point-bridge-plan.json'),historical_evaluator_review=historical)
put(D/'p2-assembly-001-review-binding.json',binding)
card=dict(period=2,assembly_source_sha256=sources,reviews=reviews,review_binding=pin(D/'p2-assembly-001-review-binding.json'),definition=pp['definition'],point_plan=pin(D/'p2-point-bridge-plan.json'),map=definition['map'],shards=rows,campaign_source='d627b74affa176037f553332d6fb5b26d1dd0b37',derivative_backend=packet['expected_backend'],derivative_source_sha256=dp['source_sha256'],derivative_prefix=prefix,preconditioner=pin(RAW/'phase-proposal-001/p2/preconditioner.npz'))
put(D/'p2-assembly-001-card.json',card)
a.authenticate_sources(card,R)
for row in rows:a.authenticate_closure(row)
print('Exact source/review authority and28 original closure assignments passed; no matrix/model/gate run.')
