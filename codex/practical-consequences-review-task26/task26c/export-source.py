from pathlib import Path
import subprocess,json,hashlib,re,time,tarfile,datetime,zipfile,ast,struct,resource
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(55,55))
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');B=Path('/data2/neural-shrimp-data');O=Path('/tmp/task26cd-source-metadata-001');O.mkdir();current='666e9b72e9b384755bf52469dbf1ed6c1d4f89f3';files={};io=[]
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
source_sets=[('mech','b203fb56d0daa41441a926097baa7c6743d42b8a',['tools/steer_gain.py','tools/gain_split.py']),('xo','92f1a20eea7794ca6348cf773e3757026568775e',['tools/crossover.py','docs/xo-cells.json'])]
for name,ref,paths in source_sets:
 for p in paths:put('original/'+name+'/'+p,show(ref,p))
 runner=show(ref,'tools/runpod_soft_runner.sh').decode().splitlines();i=next(i for i,s in enumerate(runner)if s.startswith('  '+name+')'));j=next(j for j in range(i+1,len(runner))if runner[j].strip()==';;');put('original/'+name+'/runner-command.sh',('\n'.join(runner[i:j+1])+'\n').encode())
for p in ['tools/crossover.py','paper/llm/figures/fig-mechanism.json']:put('current/'+p,show(current,p))
for name,ref,paths in source_sets:put(name+'-source-delta.diff',subprocess.check_output(['git','diff',ref,current,'--',*paths],cwd=R))
logb=show(current,'docs/research-log.md');log=logb.decode().splitlines(keepends=True);scopes=[];out=[]
for i,line in enumerate(log):
 if line.startswith('## 2026-10-15 ')and ('07:14 UTC: run (exploratory): XO,'in line or 'XO result:'in line):
  j=next((j for j in range(i+1,len(log))if log[j].startswith('## ')),len(log));block=log[i:j];omit=[]
  for k,s in enumerate(block):
   if re.search(r'\$\d|\bcap \d+(?:\.\d+)?/h|cost ceiling|rate cap',s,re.I):omit.append({'line':i+k+1,'sha256':sh(s.encode())})
   else:out.append(s)
  scopes.append({'first_line':i+1,'last_line':j,'original_block_sha256':sh(''.join(block).encode()),'omitted_private_cost_lines':omit})
put('xo-log-card-and-results-scoped.md',''.join(out).encode());js('log-source-map.json',{'source_snapshot':current,'original_log_sha256':sh(logb),'scopes':scopes})
allow=['event','time','pod_id','commit','gpu_type','archive_bytes','data_bytes','hash_verified','sha256','seconds_after_create'];ledger=[]
for i,b in enumerate((B/'runpod/pods.jsonl').read_bytes().splitlines()):
 d=json.loads(b);ledger.append((i+1,sh(b),{k:d[k]for k in allow if k in d}))
configs=[('amap-a','20261015T064306Z','mech'),('amap-b','20261015T064309Z','mech'),('bmap-a','20261015T064312Z','mech'),('bmap-b','20261015T064315Z','mech'),('cells','20261015T071443Z','xo')];runs=[]
for name,stamp,kind in configs:
 root=B/'runpod/soft64'/stamp;v=json.loads((root/'result.json').read_text());expected=source_sets[0 if kind=='mech'else 1][1];assert v['commit']==expected and v['jobset']==kind
 q={k:v.get(k)for k in ['pod_id','gpu_type','commit','jobset','started','runner_returncode','hash_verified','local_sha256','remote_sha256','wall_seconds']};q.update(name=name,receipt=pin(root/'result.json'),archive=pin(root/'results.tgz'),stdout=pin(root/'runner-stdout.txt'));assert q['archive']['sha256']==v['local_sha256']==v['remote_sha256']and v['hash_verified'];q['upload']={k:v.get('upload',{}).get(k)for k in ['code_bytes','data_bytes','code_sha256','data_sha256','extra_data','jobset_args']};q['status_lines']=[s for s in(root/'runner-stdout.txt').read_text().splitlines()if re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ (?:start [A-Za-z0-9_.-]+|end [A-Za-z0-9_.-]+ status -?\d+)',s)];q['ledger_records']=[dict(line_number=i,original_line_sha256=h,fields=d)for i,h,d in ledger if d.get('pod_id')==v['pod_id']];q['recorded_termination_verified']=any(x['fields'].get('event')=='terminated_verified'for x in q['ledger_records']);assert q['recorded_termination_verified']
 raw=B/'explore/llm-soft-loop/pod64'/kind/name;q['selected_files']=[]
 with tarfile.open(root/'results.tgz','r:gz')as tf:
  members=tf.getmembers()
  for p in sorted(raw.iterdir()):
   if not p.is_file():continue
   row=pin(p);rel=str(p.relative_to(B));matches=[m for m in members if m.isfile()and m.name.lstrip('./')==rel];assert len(matches)==1
   h=hashlib.sha256();n=0
   with tf.extractfile(matches[0])as f:
    for bb in iter(lambda:f.read(1024**2),b''):h.update(bb);n+=len(bb)
   assert n==row['bytes']and h.hexdigest()==row['sha256'];item=dict(file=row,archive_member=matches[0].name,archive_member_match=True,scientific_payload_decoded=False)
   if p.suffix=='.npz':
    headers=[]
    with zipfile.ZipFile(p)as zp:
     for member in zp.namelist():
      with zp.open(member)as f:
       assert f.read(6)==b'\x93NUMPY';major,minor=f.read(2);size=struct.unpack('<H'if major==1 else'<I',f.read(2 if major==1 else 4))[0];headers.append({'member':member,'header':ast.literal_eval(f.read(size).decode('latin1').strip())})
    item['npy_headers']=headers
   q['selected_files'].append(item)
 runs.append(q)
js('original-run-authority.json',{'schema':'task26cd-original-metadata-v1','runs':runs,'scope':'Exactly original four MECH3/4 partitions and original 24-cell XO. Later XO7/gratitude/random1 excluded. Selected scientific JSON/JSONL are opaque bytes; NPZ headers only. No model/producer/statistical execution.','historical_limits':['Recorded source/upload/retrieval/archive-member/termination basis; no original loaded model-byte or process-birth reconstruction.','Source-log/paper summaries were exposed; no independent numerical acceptance inferred.']})
manifest={'source_snapshot':current,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'hash_io':{'bytes':sum(x['bytes']for x in io),'seconds':sum(x['seconds']for x in io)},'target_bytes':sum(x['file']['bytes']for q in runs for x in q['selected_files']),'resource_scope':'Metadata capture CPU1/nice10/4GiBAS/55CPU/60wall; no numerical profile.'};p=O/'manifest.json';p.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'path':str(O),'manifest_sha256':sh(p.read_bytes()),'authority_sha256':files['original-run-authority.json']['sha256'],'target_bytes':manifest['target_bytes'],'hash_io':manifest['hash_io']}))
