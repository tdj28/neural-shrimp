import json,sys,tarfile,hashlib,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
resource.setrlimit(resource.RLIMIT_CPU,(150,)*2)
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path.insert(0,str(R/'codex/coexistence-task33'))
import returned_shard as rs
raw=Path('/data2/neural-shrimp-data');allresults=[]
configs=[('t33-p3-a', '20261018T154914Z', 'b9302fa074430a12769c2c15d94e7ab5458bc0d946c5da4795606093e6474ea8', 3), ('t33-p3-i', '20261018T154809Z', '6f2a60537d43d14322f311dbed62ee1de9b78aa98d24893018499888949c3f41', 3)]
for pod,stamp,whole,period in configs:
 base=raw/'codex/remote'/pod/'returned'/stamp;archive=base/'results.tgz';assert rs.sha(archive)==whole
 dest=raw/'codex/coexistence'/('return-'+pod+'-001');assert not dest.exists();dest.mkdir()
 with tarfile.open(archive)as t:
  seen=set()
  for m in t.getmembers():
   p=Path(m.name);assert not p.is_absolute() and '..'not in p.parts and m.name not in seen and (m.isfile()or m.isdir());seen.add(m.name)
  t.extractall(dest,filter='data')
 folder=dest/'explore/llm-soft-loop/pod64/certpack'/pod
 summary=rs.read(base/'return-summary.json');assert summary['results_sha256']==whole and summary['terminated']['terminated'] and not summary['terminated']['in_inventory'] and summary['not_started']==[]
 packet=rs.read(R/f'codex/coexistence-task33/p{period}-dispatch-001.json');assigned={x['attempt']:x for x in packet['plans']}
 source={2:'d627b74affa176037f553332d6fb5b26d1dd0b37',3:'13e5a20305b9af45a542188c40d7f276aa60f8b0'}[period]
 for row in summary['attempts']:
  name=row['attempt'];tar=folder/(name+'.tar');assert rs.sha(tar)==row['tar_sha256'];attempt=raw/'codex/coexistence'/name;assert not attempt.exists()
  with tarfile.open(tar)as t:
   seen=set()
   for m in t.getmembers():
    p=Path(m.name);assert not p.is_absolute() and '..'not in p.parts and p.is_relative_to(Path('codex/coexistence')/name) and m.name not in seen and (m.isfile()or m.isdir());seen.add(m.name)
   original=hashlib.sha256(t.extractfile('codex/coexistence/'+name+'/receipt.json').read()).hexdigest();assert original==row['receipt_sha256'];t.extractall(raw,filter='data')
  run=rs.read(folder/name/'run.json');assert run['success'] and run['closure_verified'] and run['exit_code']==0 and run['watchdog_stop']is None and run['alive_after_exit']==[] and run['alive_after_cleanup']==[]
  a=assigned[name];plan=rs.read(R/a['plan']);ref=dict(phase_index=a['phase'],start_column=a['start_column'],plan=dict(name=Path(a['plan']).name,sha256=a['plan_sha256']))
  op=rs.completed(attempt,ref,rs.common(plan),source=source,expected_receipt=original)
  allresults.append(dict(attempt=name,period=period,source_commit=source,operational=op,outer_closure_path=str(folder/name/'run.json'),outer_closure_sha256=rs.sha(folder/name/'run.json'),whole_archive_sha256=whole,archive_sha256=row['tar_sha256'],original_archive_receipt_sha256=original,summary_receipt_matches=True,termination_record_path=str(base/'return-summary.json'),termination_record_sha256=rs.sha(base/'return-summary.json')))
  print(name,'operational admission passed',flush=True)
Path('/tmp/returns-p3ai-operational.json').write_text(json.dumps(dict(attempts=allresults,certificate_passed=False,scientific_endpoint_audit_pending=True),indent=2)+'\n')
