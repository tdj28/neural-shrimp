import sys,json,time,resource
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path[:0]=[str(R/'codex/coexistence-task33'),str(R/'codex/certify-llm')]
import derivative_bridge as b
from flint import ctx
ctx.prec=1024;ctx.threads=1
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);resource.setrlimit(resource.RLIMIT_CPU,(120,)*2)
start=time.monotonic();folder=R/'codex/coexistence-task33';audits=[]
rows=json.loads(Path('/tmp/returns-1626-operational.json').read_text())['attempts']
for row in rows:
 packet=json.loads((folder/f"p{row['period']}-dispatch-001.json").read_text());plan=json.loads((R/packet['plans'][0]['plan']).read_text());definition=json.loads(Path(plan['definition_path']).read_text())
 prefix=dict(attempt=plan['prefix_attempt'],receipt_sha256=plan['prefix_receipt_sha256'],map_source_sha256=plan['prefix_map_source_sha256'],cache_sha256=plan['prefix_cache_sha256'],source_commit=plan['prefix_source_commit'],receipt_path=plan['prefix_receipt'],admission_path=plan['prefix_admission_path'],admission_sha256=plan['prefix_admission_sha256'])
 a=b.trust_anchors(plan['definition_path'],plan['definition_sha256'],definition['map'],packet['expected_backend'],plan['source_sha256'],prefix);pre=b.authenticate_prefix(a)
 ref=dict(attempt=str(b.OWNED/row['attempt']),receipt_sha256=row['original_archive_receipt_sha256'],source_commit=row['source_commit'])
 shard,audit=b.authenticate_shard(ref,a,pre)
 columns=shard['direction_indices'];assert len(shard['derivatives'])==896 and len(shard['directions'])==896
 for i,(directions,values)in enumerate(zip(shard['directions'],shard['derivatives'])):
  assert len(values)==len(directions)==64
  assert all(pair==(int(i==j),0)for pair,j in zip(b.coverage._records(directions,64,'original basis',exact=True),columns))
  b.coverage._records(values,64,'original derivative')
  for v in values:b.coverage._decode(v)
 b.coverage._records(shard['primal'],896,'original primal')
 for v in shard['primal']:b.coverage._decode(v)
 audit.update(original_derivative_components_checked=896*64,original_positive_basis_components_checked=896*64,complete_campaign_coverage=False)
 audits.append(audit);print(row['attempt'],'signed original endpoint/basis/output admission passed',flush=True)
out=dict(certificate_passed=False,partial_campaign=True,scientific_interpretation=False,adapter_sha256=b.digest(b.__file__),audits=audits,seconds=time.monotonic()-start)
Path('/tmp/returns-1626-endpoints.json').write_text(json.dumps(out,indent=2)+'\n')
