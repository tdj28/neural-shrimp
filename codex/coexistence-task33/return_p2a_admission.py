import json,sys,tarfile,hashlib
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path.insert(0,str(R/'codex/coexistence-task33'))
import returned_shard as rs
raw=Path('/data2/neural-shrimp-data');base=raw/'codex/remote/t33-p2-a/returned/20261018T150059Z';folder=raw/'codex/coexistence/return-t33-p2-a-001/explore/llm-soft-loop/pod64/certpack/t33-p2-a'
summary=json.loads((base/'return-summary.json').read_text());assert summary['terminated']['terminated'] and not summary['terminated']['in_inventory'];packet=json.loads((R/'codex/coexistence-task33/p2-dispatch-001.json').read_text());assigned={x['attempt']:x for x in packet['plans']};results=[]
for row in summary['attempts']:
 name=row['attempt'];tar=folder/(name+'.tar');assert rs.sha(tar)==row['tar_sha256'];attempt=raw/'codex/coexistence'/name
 with tarfile.open(tar)as t:
  seen=set()
  for m in t.getmembers():
   p=Path(m.name)
   assert not p.is_absolute() and '..'not in p.parts and p.is_relative_to(Path('codex/coexistence')/name) and m.name not in seen and (m.isfile()or m.isdir());seen.add(m.name)
  if attempt.exists():
   for m in t.getmembers():
    if m.isfile():assert (raw/m.name).read_bytes()==t.extractfile(m).read()
  else:t.extractall(raw,filter='data')
  member=t.getmember('codex/coexistence/'+name+'/receipt.json');original_receipt_hash=hashlib.sha256(t.extractfile(member).read()).hexdigest()
 run=rs.read(folder/name/'run.json');assert run['success'] and run['closure_verified'] and run['exit_code']==0 and run['watchdog_stop']is None and run['alive_after_exit']==[] and run['alive_after_cleanup']==[]
 a=assigned[name];plan=rs.read(R/a['plan']);r=dict(phase_index=a['phase'],start_column=a['start_column'],plan=dict(name=Path(a['plan']).name,sha256=a['plan_sha256']))
 op=rs.completed(attempt,r,rs.common(plan),source='d627b74affa176037f553332d6fb5b26d1dd0b37',expected_receipt=original_receipt_hash);results.append(dict(attempt=name,operational=op,outer_closure_sha256=rs.sha(folder/name/'run.json'),archive_sha256=row['tar_sha256'],producer_summary_receipt_sha256=row['receipt_sha256'],original_archive_receipt_sha256=original_receipt_hash,summary_receipt_matches=row['receipt_sha256']==original_receipt_hash))
out=dict(status='five_original_returns_operationally_admitted',whole_archive_sha256=summary['results_sha256'],termination_record_sha256=rs.sha(base/'return-summary.json'),attempts=results,scientific_endpoint_audit_pending=True,certificate_passed=False)
Path('/tmp/t33-p2-a-operational.json').write_text(json.dumps(out,indent=2)+'\n');print('Five complete original inventories/source/plan/closure operationally admitted; endpoint audit pending')
