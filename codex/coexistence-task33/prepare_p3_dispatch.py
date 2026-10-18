import sys,json,hashlib
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path[:0]=[str(R/'codex/coexistence-task33'),str(R/'codex/certify-llm')]
import derivative_shard as m
import torch
from flint import ctx
ctx.prec=1024;ctx.threads=1
raw=m.OWNED; folder=R/'codex/coexistence-task33';pre=raw/'prefix-001';sup=raw/'prefix-001-supervision';df=raw/'p3-final-domain-001.json'
sha=m.sha
ent=lambda p:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,v):assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n')
meta=json.loads((pre/'inputs.json').read_text());rec=json.loads((sup/'receipt.json').read_text());adm=json.loads((sup/'admission.json').read_text())
backend=m.backend_definition(torch.__version__);bundle=m.source_bundle();mp=m.map_definition(meta['pinned_assets'],meta['prefix_ids'])
plans=[]
for phase in range(3):
 for col in range(0,896,64):
  name=f'p3-derivative-p{phase}-c{col:03d}-001'
  plan=dict(schema=m.SCHEMA,definition_path=str(df),definition_sha256=sha(df),map_sha256=m.domain.canonical_hash(mp),backend_sha256=m.domain.canonical_hash(backend),source_sha256=bundle['source_sha256'],evaluator_sha256=m.domain.canonical_hash(bundle),prefix_attempt=str(pre),prefix_receipt_sha256=sha(sup/'receipt.json'),prefix_map_source_sha256={x['relative_path']:x['sha256'] for x in adm['source_blobs']}['codex/certify-llm/ball_qwen.py'],prefix_cache_sha256=sha(pre/'prefix-cache.json.gz'),phase_index=phase,start_column=col,precision=1024,token_chunk=1024,envelope=m.ENVELOPE,period=3,prefix_source_commit=rec['source_commit'],prefix_output=str(pre),prefix_receipt=str(sup/'receipt.json'),prefix_admission_path=str(sup/'admission.json'),prefix_admission_sha256=sha(sup/'admission.json'))
  definition,authority,_=m.admit_definition(plan,meta['pinned_assets'],meta['prefix_ids'],torch.__version__)
  path=folder/(name+'-plan.json');save(path,plan)
  plans.append(dict(attempt=name,plan=str(path.relative_to(R)),plan_sha256=sha(path),phase=phase,start_column=col,command=f'python -I -B -u codex/coexistence-task33/launch.py --attempt {name} --driver derivative_shard --plan {name}-plan.json --threads 1 --seconds 3600'))
up=[df,sup/'receipt.json',sup/'admission.json']+[Path(v['path']) for v in rec['output_files']]
packet=dict(status='reviewed_dispatch_requires_Claude_owned_capacity_and_existing_ceiling',period=3,source_commit='Exact privately pushed commit specified in standalone comms; checkout that commit for every attempt.',expected_backend=backend,expected_assets=meta['pinned_assets'],source_sha256=bundle['source_sha256'],launcher_sha256=sha(folder/'launch.py'),collector_sha256=sha(folder/'returned_shard.py'),definition=ent(df),upstream_files=[ent(p) for p in up],plans=plans,measured_cost=dict(basis='Closed calibration002: same backend,1024bits,64columns/fullvocabulary, earlier P8 domain. New-domain cost is forecast only.',per_shard_worker_seconds=1866.785209,total_serial_forecast_seconds=42*1866.785209,per_shard_hard_wall_seconds=3600,per_shard_memory_gib=18,per_shard_cpu_threads=1,retained_storage_reservation_gib=12),preflight=['Exact scientific source,all42plan hashes,all upstream files and model assets; real absolute paths, no symlink relocation.','All runtime build/module versions as calibration002; no spoofing.','Claude exclusively records approved ceiling, exact pod ownership, effective cgroup CPU/RAM admission, bounded setup/retrieval/cleanup and verified termination. Each distinct worker reserves18GiB plus shared4GiB per pod, CPU1. Packing count limited by actual effective capacity and approved ceiling.','Each attempt fresh and run at most once; no automatic retries, extensions or precision change. Retain all failures.','Capture complete original source/manifest/plan/native outputs/events/stdout/stderr/receipt and exact writer termination evidence; retrieve and hash before terminating dedicated resources.','Return each completed attempt promptly, do not wait for all42. Use isolated returned_shard.py for operational admission; original signed endpoint audit and strict mathematical gates follow.'],claim='Derivative enclosures only. No certificate, token label or basin reachability claim from completion.',required_import_versions=json.loads((R/'codex/certificate-ladder-task34/calibration-002.json').read_text())['required_import_versions'])
save(folder/'p3-dispatch-001.json',packet)
print('42 concrete plans admitted without model calls; packet',sha(folder/'p3-dispatch-001.json'))
