"""One finite Task34 signed-return admission; no model calls or proof claim."""
import argparse, hashlib, json, os, resource, sys, time, traceback
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
HERE=R/'codex/certificate-ladder-task34'
RAW=Path('/data2/neural-shrimp-data/codex/certificate-ladder')
SOURCE='42508b68a733acc4216b40d7b60215de36c5d543'
PACKET_SHA='3167eeeec4329eeb1ed8f01ab9738a8173decaabaa021831ab1153a6dcd3c1bc'

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024**2),b''):h.update(x)
 return h.hexdigest()

def assignments(rows,plans):
 expected={p['attempt']:(p['phase'],p['start_column'])for p in plans}
 if len(expected)!=42 or len(plans)!=42:raise ValueError('exact 42 planned attempts required')
 actual=[r['attempt']for r in rows]
 if len(actual)!=42 or len(set(actual))!=42 or set(actual)!=set(expected):raise ValueError('complete unique attempt coverage required')
 coverage=set()
 for name in actual:
  phase,start=expected[name]
  if type(phase)is not int or phase not in range(3) or type(start)is not int or start not in range(0,896,64):raise ValueError('exact phase/block allocation required')
  cells={(phase,j)for j in range(start,start+64)}
  if cells&coverage:raise ValueError('duplicate phase-column allocation')
  coverage|=cells
 if coverage!={(i,j)for i in range(3)for j in range(896)}:raise ValueError('complete 2688 phase-column coverage required')
 return expected

def run(op_path,op_sha):
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
 resource.setrlimit(resource.RLIMIT_CPU,(240,)*2)
 start=time.monotonic()
 if op_path!=RAW/'return-cascade-p3-002/operational.json' or op_path.is_symlink() or digest(op_path)!=op_sha:raise ValueError('exact operational admission required')
 op=json.loads(op_path.read_text()); rows=op['attempts']
 packet_path=HERE/'cascade-p3-dispatch-001.json'
 if digest(packet_path)!=PACKET_SHA:raise ValueError('independent dispatch packet differs')
 packet=json.loads(packet_path.read_text());expected=assignments(rows,packet['plans'])
 if any(r['source_commit']!=SOURCE for r in rows):raise ValueError('wrong original scientific source')
 destination=RAW/'return-cascade-p3-endpoints-001';destination.mkdir()
 out=dict(status='running',certificate_passed=False,scientific_interpretation=False,model_calls=0,source_commit=SOURCE,packet_sha256=PACKET_SHA,operational_report_path=str(op_path),operational_report_sha256=op_sha,receiver_source_sha256=digest(__file__),audits=[])
 def save(name):
  (destination/name).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 try:
  sys.path[:0]=[str(HERE),str(R/'codex/certify-llm')]
  import derivative_bridge as b
  b.ctx.prec=1024;b.ctx.threads=1
  plan=json.loads((R/packet['plans'][0]['plan']).read_text())
  definition=json.loads(Path(plan['definition_path']).read_text())
  prefix=dict(attempt=plan['prefix_attempt'],receipt_sha256=plan['prefix_receipt_sha256'],map_source_sha256=plan['prefix_map_source_sha256'],cache_sha256=plan['prefix_cache_sha256'],source_commit=plan['prefix_source_commit'],receipt_path=plan['prefix_receipt'],admission_path=plan['prefix_admission_path'],admission_sha256=plan['prefix_admission_sha256'])
  anchor=b.trust_anchors(plan['definition_path'],plan['definition_sha256'],definition['map'],packet['expected_backend'],plan['source_sha256'],prefix)
  pre=b.authenticate_prefix(anchor)
  out.update(adapter_sha256=b.digest(b.__file__),prefix_authentication=pre)
  save('progress.json')
  coverage=set()
  for row in rows:
   ref=dict(attempt=str(b.OWNED/row['attempt']),receipt_sha256=row['original_archive_receipt_sha256'],source_commit=SOURCE)
   shard,audit=b.authenticate_shard(ref,anchor,pre)
   phase,col=expected[row['attempt']];columns=shard['direction_indices']
   if shard['phase_index']!=phase or columns!=list(range(col,col+64)):raise ValueError('actual shard allocation differs')
   if len(shard['derivatives'])!=896 or len(shard['directions'])!=896:raise ValueError('complete original rows required')
   for i,(directions,values)in enumerate(zip(shard['directions'],shard['derivatives'])):
    if len(values)!=64 or len(directions)!=64:raise ValueError('complete block width required')
    if any(pair!=(int(i==j),0)for pair,j in zip(b.coverage._records(directions,64,'original basis',exact=True),columns)):raise ValueError('positive standard basis differs')
    b.coverage._records(values,64,'original derivative')
    for v in values:b.coverage._decode(v)
   b.coverage._records(shard['primal'],896,'original primal')
   for v in shard['primal']:b.coverage._decode(v)
   cells={(phase,j)for j in columns}
   if cells&coverage:raise ValueError('duplicate actual phase-column coverage')
   coverage|=cells
   audit.update(attempt=row['attempt'],original_derivative_components_checked=896*64,original_positive_basis_components_checked=896*64,original_primal_components_checked=896)
   out['audits'].append(audit);save('progress.json')
   print(row['attempt'],'signed endpoints/basis admitted',flush=True)
  if coverage!={(i,j)for i in range(3)for j in range(896)}:raise ValueError('incomplete actual campaign coverage')
  out.update(status='all_original_signed_endpoints_admitted',complete_campaign_coverage=True,unique_phase_columns=len(coverage),attempt_count=len(rows),seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  save('endpoints.json')
 except BaseException as e:
  out.update(status='failed',failure=type(e).__name__+': '+str(e),seconds=time.monotonic()-start)
  save('failure.json');raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--operational',type=Path,required=True);ap.add_argument('--operational-sha256',required=True);args=ap.parse_args()
 run(args.operational,args.operational_sha256)
