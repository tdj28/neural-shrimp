import json,hashlib,subprocess,gzip,sys
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');raw=Path('/data2/neural-shrimp-data/codex/certificate-ladder');sup=raw/'prefix-cascade-p3-001-supervision';base=raw/'prefix-cascade-p3-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((sup/'receipt.json').read_text());a=json.loads((sup/'admission.json').read_text())
assert r['returncode']==0 and r['failure'] is None and r['worker_reaped'] and r['worker_group_session_absent']
for e in r['output_files']:assert Path(e['path']).stat().st_size==e['bytes'] and sha(e['path'])==e['sha256']
for e in a['source_blobs']:
 b=subprocess.check_output(['git','show',r['source_commit']+':'+e['relative_path']],cwd=R);assert hashlib.sha256(b).hexdigest()==e['sha256']
ids={r['worker_pid'],r['supervisor']['pid']}
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  s=(p/'stat').read_text().rsplit(')',1)[1].split();assert not ({int(p.name),int(s[2]),int(s[3])}&ids)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
sys.path.insert(0,str(R/'codex/certify-llm'))
from cache_io import unpack
from flint import ctx
ctx.prec=2048
with gzip.open(base/'prefix-cache.json.gz','rt') as f:data=json.load(f)
assert len(data)==24;count=0
for layer in data:
 assert len(layer)==2
 for field in layer:
  assert len(field)==37
  for token in field:
   assert len(token)==2
   for head in token:
    assert len(head)==64
    for v in head:assert unpack(v).is_finite();count+=1
meta=json.loads((base/'inputs.json').read_text());assert meta['alpha_ratio']==list((.0650).as_integer_ratio()) and meta['log10_temperature_ratio']==list((-.14798).as_integer_ratio()) and meta['source_commit']==r['source_commit']
assert meta['cell']=='cascade-p3'
report=json.loads((base/'report.json').read_text());assert report['completed'] and report['layers']==list(range(24)) and report['precision']==2048
out=dict(source=r['source_commit'],receipt_sha256=sha(sup/'receipt.json'),report_sha256=sha(base/'report.json'),outputs_checked=len(r['output_files']),source_blobs_checked=len(a['source_blobs']),prefix_balls_checked=count,original_groups_absent=True,session_reaped_exit=0,external_seconds=r['external_elapsed_seconds'],report=report)
(R/'codex/certificate-ladder-task34/prefix-cascade-p3-001-completion.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
