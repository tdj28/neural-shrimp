from pathlib import Path
import subprocess,json,hashlib,re,time,tarfile,datetime
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');B=Path('/data2/neural-shrimp-data');O=Path('/tmp/task26a-source-metadata-001');O.mkdir(exist_ok=False);current='ea4f5c88255fd755cedc39f891f36530aca4a93f';original='a9c411ff1bf992a20f595039000c55be15c6d12c';files={};io=[]
def sh(b):return hashlib.sha256(b).hexdigest()
def pin(p):
 p=Path(p);a=p.stat();h=hashlib.sha256();t=time.monotonic()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 z=p.stat();assert(a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns);io.append(dict(bytes=a.st_size,seconds=time.monotonic()-t));return dict(path=str(p),sha256=h.hexdigest(),bytes=a.st_size)
def show(c,p):return subprocess.check_output(['git','show',c+':'+p],cwd=R)
def put(p,b):
 q=O/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);files[p]=pin(q);return files[p]
def js(p,v):return put(p,(json.dumps(v,indent=2)+'\n').encode())
for p in ['docs/repro-task-test.md','docs/repro-task-problems.json','tools/repro_task.py','tools/explore_llm_attractors.py']:
 put('original/'+p,show(original,p))
put('current/tools/repro_task.py',show(current,'tools/repro_task.py'))
put('repro-task-source-delta.diff',subprocess.check_output(['git','diff',original,current,'--','tools/repro_task.py','docs/repro-task-test.md','docs/repro-task-problems.json'],cwd=R))
runner=show(original,'tools/runpod_soft_runner.sh').decode().splitlines();i=next(i for i,s in enumerate(runner)if s.startswith('  reprotask)'));j=next(j for j in range(i+1,len(runner))if runner[j].strip()==';;');put('original/reprotask-runner-command.sh',('\n'.join(runner[i:j+1])+'\n').encode())
log=show(current,'docs/research-log.md').decode().splitlines();blocks=[]
for i,line in enumerate(log):
 if line.startswith('## 2026-10-15 ')and ('REPRO-T' in line):
  j=next((j for j in range(i+1,len(log))if log[j].startswith('## ')),len(log));block=log[i:j];blocks.extend(s for s in block if not re.search(r'(rate cap|\$\d|cost ceiling)',s,re.I));blocks.append('')
put('repro-task-log-card-and-results.md',('\n'.join(blocks)+'\n').encode())
comms=show(current,'comms/claude-to-codex.md').decode();a=comms.index('**Task 26**');z=comms.index('**Advance notice',a);put('assignment-task26.md',comms[a:z].encode());a=comms.index('## 2026-10-15T07:57Z');put('queue-reprioritization.md',comms[a:].encode())
paper=show(current,'paper/llm/main.tex').decode();a=paper.index('\\label{sec:consequences}');a=paper.rfind('\\section',0,a);z=paper.find('\\section',a+8);put('paper-consequences-section.tex',paper[a:z if z>=0 else len(paper)].encode())
root=B/'runpod/soft64/20261015T070508Z';v=json.loads((root/'result.json').read_text());q={k:v.get(k)for k in ['pod_id','gpu_type','commit','jobset','started','runner_returncode','hash_verified','local_sha256','remote_sha256','wall_seconds']};assert v['commit']==original;q['upload']={k:v.get('upload',{}).get(k)for k in ['code_bytes','data_bytes','code_sha256','data_sha256','extra_data','jobset_args']};q['receipt']=pin(root/'result.json');q['archive']=pin(root/'results.tgz');assert q['archive']['sha256']==v['local_sha256']==v['remote_sha256'] and v['hash_verified']
q['stdout']=pin(root/'runner-stdout.txt');q['status_lines']=[s for s in (root/'runner-stdout.txt').read_text().splitlines()if re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ (?:start [A-Za-z0-9_.-]+|end [A-Za-z0-9_.-]+ status -?\d+)',s)]
allow=['event','time','pod_id','commit','gpu_type','archive_bytes','data_bytes','hash_verified','sha256','seconds_after_create'];lr=[]
for i,b in enumerate((B/'runpod/pods.jsonl').read_bytes().splitlines()):
 d=json.loads(b)
 if d.get('pod_id')==v['pod_id']:lr.append(dict(line_number=i+1,original_line_sha256=sh(b),fields={k:d[k]for k in allow if k in d}))
q['ledger_records']=lr;q['recorded_termination_verified']=any(r['fields'].get('event')=='terminated_verified'for r in lr);assert q['recorded_termination_verified'];raw=B/'explore/llm-soft-loop/pod64/repro/t05';inputs=[]
with tarfile.open(root/'results.tgz','r:gz')as tf:
 for p in sorted(raw.iterdir()):
  if not p.is_file():continue
  pp=pin(p);rel=str(p.relative_to(B));matches=[m for m in tf.getmembers()if m.isfile()and m.name.lstrip('./')==rel];assert len(matches)==1
  with tf.extractfile(matches[0])as f:b=f.read()
  assert len(b)==pp['bytes']and sh(b)==pp['sha256'];inputs.append(dict(file=pp,archive_member=matches[0].name,archive_member_match=True,content_decoded=False))
q['selected_files']=inputs;q['source_contract']=files['original/docs/repro-task-test.md'];q['historical_limits']=['Original recorded source/upload/archive/termination finite authority; no historical loaded model-byte reconstruction or new remote query.','No target runs/progress payload decoded in this packet. Producer summary/paper claims exposed from source log; independent statistics remain pending.','Frozen source/card is the exact original uploaded commit; no separate historical remote push receipt is inferred.']
js('original-run-authority.json',q)
manifest=dict(source_snapshot=current,original_source_commit=original,frozen_design_addition_commit=original,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,hash_io=dict(bytes=sum(x['bytes']for x in io),seconds=sum(x['seconds']for x in io)),scope='Source/design/paper/log and original local receipt/archive-member/termination metadata only; target runs/progress JSONL were hashed as opaque bytes, never parsed. No model, producer analysis or statistical execution.');(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(path=str(O),manifest_sha256=sh((O/'manifest.json').read_bytes()),original_authority_sha256=files['original-run-authority.json']['sha256'],target_bytes=sum(x['file']['bytes']for x in inputs),hash_io=manifest['hash_io'])))
