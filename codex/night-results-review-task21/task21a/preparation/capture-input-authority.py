from pathlib import Path
import os,resource,json,hashlib,re,tarfile,time,subprocess,datetime
os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)));resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');BASE=Path('/data2/neural-shrimp-data');RAW=BASE/'explore/llm-soft-loop/pod64';OUT=Path('/tmp/task21a-input-authority-001');OUT.mkdir(exist_ok=False)
tic=time.monotonic();io=[];files={}
def digest(b):return hashlib.sha256(b).hexdigest()
def pin(p):
 p=Path(p);before=p.stat();h=hashlib.sha256();t=time.monotonic()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 after=p.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino);io.append(dict(path=str(p),bytes=before.st_size,seconds=time.monotonic()-t))
 return dict(path=str(p),sha256=h.hexdigest(),bytes=before.st_size)
def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n');files[name]=pin(p);return files[name]
def export(name,b):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);files[name]=pin(p);return files[name]
records=json.loads(Path('/tmp/task21a-source-supplement-001/original-run-metadata.json').read_text())['runs'];ledger=BASE/'runpod/pods.jsonl';lrows=[(i+1,line,json.loads(line))for i,line in enumerate(ledger.read_bytes().splitlines())];allowed=['event','time','pod_id','commit','gpu_type','archive_bytes','data_bytes','hash_verified','sha256','seconds_after_create']
original=[];inputs=[];source={}
for old in records:
 root=Path(old['receipt']['out']);receipt=pin(root/'result.json');v=json.loads((root/'result.json').read_text());q={k:v.get(k)for k in ['pod_id','gpu_type','commit','jobset','started','runner_returncode','hash_verified','local_sha256','remote_sha256','wall_seconds']};q['upload']={k:v.get('upload',{}).get(k)for k in ['code_bytes','data_bytes','code_sha256','data_sha256','extra_data','jobset_args']}
 archive=pin(root/'results.tgz');assert v['hash_verified']is True and archive['sha256']==v['local_sha256']==v['remote_sha256']
 stdout=root/'runner-stdout.txt';q['status_lines']=[s for s in stdout.read_text().splitlines()if re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ (?:start [A-Za-z0-9_.-]+|end [A-Za-z0-9_.-]+ status -?\d+)',s)]
 q['ledger_records']=[dict(line_number=i,original_line_sha256=digest(b),fields={k:d[k]for k in allowed if k in d})for i,b,d in lrows if d.get('pod_id')==v['pod_id']]
 q['recorded_termination_verified']=any(r['fields'].get('event')=='terminated_verified'for r in q['ledger_records']);assert q['recorded_termination_verified']
 name=q['upload']['jobset_args'].split()[0];folder=q['jobset'];folder='cascade'if folder=='cascade'else'superstable';current=RAW/folder/name;selected=[p for p in sorted(current.iterdir())if p.is_file()and p.suffix in ['.json','.jsonl']];assert selected
 matched=[]
 with tarfile.open(root/'results.tgz','r:gz')as tf:
  members=tf.getmembers()
  for p in selected:
   pp=pin(p);rel=str(p.relative_to(BASE));matches=[m for m in members if m.isfile()and m.name.lstrip('./')==rel];assert len(matches)==1,(rel,len(matches));m=matches[0];h=hashlib.sha256();n=0
   with tf.extractfile(m)as f:
    for b in iter(lambda:f.read(1024**2),b''):h.update(b);n+=len(b)
   assert n==pp['bytes']and h.hexdigest()==pp['sha256'];row=dict(id=old['name']+'/'+p.name,run=old['name'],kind=folder,case=name,file=pp,archive_member=m.name,archive_sha256=archive['sha256'],original_source_commit=v['commit'],writer_disposition='original retrieved job output; scientific failure or partial-result disposition retained in status_lines',selected_for_saved_review=True)
   matched.append(row);inputs.append(row)
 q.update(name=old['name'],receipt=receipt,archive=archive,stdout=pin(stdout),members=matched);original.append(q)
 if v['commit']not in source:
  b=subprocess.check_output(['git','-C',str(R),'show',v['commit']+':tools/llm_feigenbaum.py']);source[v['commit']]=export('source/'+v['commit']+'/llm_feigenbaum.py',b)
# Completed local c: terminal done marker is a source-defined completion record, not invented process proof.
localdir=RAW/'cascade/crescent-t-0.156-c';progress=localdir/'progress.jsonl';log=BASE/'scratch/fb156c.log';ptext=progress.read_text();ltext=log.read_text();assert re.search(r'"done"\s*:\s*true',ptext.splitlines()[-1])and re.search(r'"done"\s*:\s*true',ltext.splitlines()[-1])
local=[]
for p in [localdir/'cascade.json',progress]:
 row=dict(id='local-FB156c/'+p.name,run='local-FB156c',kind='cascade',case='crescent-t-0.156-c',file=pin(p),selected_for_saved_review=True,writer_disposition='terminal done:true in both progress.jsonl and producer stdout; declared complete in pinned research log',source_commit=None,source_limit='Historical loaded-source commit and process births/exit receipt were not independently captured; original run card/current source and completed producer statement provide finite authority only.')
 inputs.append(row);local.append(row)
local_authority=dict(progress=pin(progress),stdout=pin(log),terminal_done_true=True,pinned_card_commit='40a2b11769696d94c1a6b4732c1e7fe9d57abac0',complete_result_statement_commit='b5af43a7b3259b00e2e7cd573c097f999f6587f9',observed_original_births=None,original_process_closure_independently_observed=False,limitations=['No query of Claude GPU/process or pods was performed.','Local completed output has no independent historical source capsule/exit receipt; no source identity or termination is manufactured.'])
pending=[]
for case,logname in [('crescent-t-0.156-d','fb156d.log'),('crescent-t-0.152-c','fb152c.log')]:
 p=RAW/'cascade'/case;pending.append(dict(case=case,selected_for_saved_review=False,disposition='No terminal completion authority captured; exclude numerical prefix. Remains pending/cannot-check as a completed run.',files=[dict(path=str(x),bytes=x.stat().st_size,mtime_ns=x.stat().st_mtime_ns)for x in sorted(p.iterdir())if x.is_file()],producer_stdout_path=str(BASE/'scratch'/logname)))
a=save('original-authority.json',dict(schema='task21a-original-input-authority-v1',original_runs=original,local_completed_c=local_authority,pending_local=pending,read_scope='Source, command/receipt/status/termination metadata, streamed byte hashes and unique archive member equality. Local c terminal done booleans only; no scientific numeric fields decoded. Pending d/c numerical prefixes excluded.',limits=['Original remote source/upload/archive/termination records are finite producer authority; no model/asset reconstruction or new pod query.','FBD e locator refusal remains an original failure distinct from later inferred root.','No arbitrary historical process identities invented.']))
m=save('inputs.json',dict(schema='task21a-saved-json-inputs-v1',status='prepared_unlaunchable',source_snapshot='9db1cb415a21a92f8e2576996aa42277a5af7925',frozen_pf2_commit='bb5bd4181d1e1debf7e1bef5cd04061f29f05c5d',authority=a,files=inputs,excluded=pending,source_files=source,scientific_values_decoded=False,selected_file_bytes=sum(x['file']['bytes']for x in inputs)))
manifest=dict(files=files,hash_io=dict(bytes=sum(x['bytes']for x in io),seconds=sum(x['seconds']for x in io)),elapsed_seconds=time.monotonic()-tic,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(inputs=m,authority=a,manifest_sha256=digest((OUT/'manifest.json').read_bytes()),selected_files=len(inputs),selected_bytes=sum(x['file']['bytes']for x in inputs),hash_io=manifest['hash_io'],elapsed_seconds=manifest['elapsed_seconds'])))
