from pathlib import Path
import subprocess,json,hashlib,re,time,tarfile,datetime,zipfile,ast,struct
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');B=Path('/data2/neural-shrimp-data');O=Path('/tmp/task26be-source-metadata-001');O.mkdir();current='666e9b72e9b384755bf52469dbf1ed6c1d4f89f3';original='dbd125a70dd1b1a1f54a12c1ff2fde275b4c7a49';files={};io=[]
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
for p in ['tools/repro_noise.py','docs/h2-horizon-cells.json']:
 put('original/'+p,show(original,p))
for p in ['tools/plot_repro.py','tools/repro_noise.py','tools/steer_gain.py','tools/gain_split.py','tools/plot_mechanism.py']:
 put('current/'+p,show(current,p))
put('repro-noise-source-delta.diff',subprocess.check_output(['git','diff',original,current,'--','tools/repro_noise.py'],cwd=R))
runner=show(original,'tools/runpod_soft_runner.sh').decode().splitlines();i=next(i for i,s in enumerate(runner)if s.startswith('  repro)'));j=next(j for j in range(i+1,len(runner))if runner[j].strip()==';;');put('original/repro-runner-command.sh',('\n'.join(runner[i:j+1])+'\n').encode())
logb=show(current,'docs/research-log.md');log=logb.decode().splitlines(keepends=True);excerpts=[]
for label,condition in [('repro1',lambda s:'06:51 UTC: run (exploratory): REPRO,' in s or '07:09 UTC: REPRO1 result:' in s),('mechanism',lambda s:'06:42 UTC: run (exploratory): MECH,' in s or '07:14 UTC: MECH3 result:' in s or '07:19 UTC: MECH4 result:' in s)]:
 out=[];scopes=[]
 for i,line in enumerate(log):
  if line.startswith('## ') and condition(line):
   j=next((j for j in range(i+1,len(log))if log[j].startswith('## ')),len(log));block=log[i:j];kept=[];omitted=[]
   for k,s in enumerate(block):
    if re.search(r'\$\d|\bcap \d+(?:\.\d+)?/h|cost ceiling|rate cap',s,re.I):omitted.append({'line':i+k+1,'sha256':sh(s.encode()),'reason':'private rate/cost text omitted'})
    else:kept.append(s)
   out.extend(kept);scopes.append({'first_line':i+1,'last_line':j,'original_block_sha256':sh(''.join(block).encode()),'omitted_lines':omitted})
 put(label+'-log-card-and-results-scoped.md',''.join(out).encode());excerpts.append({'name':label,'original_path':'docs/research-log.md','original_file_sha256':sh(logb),'scopes':scopes})
pb=show(current,'paper/llm/main.tex');p=pb.decode();a=p.index('\\paragraph{Steering makes tokens compete.}');z=p.find('\\paragraph',a+12)
if z<0:z=p.find('\\section',a+12)
put('paper-steering-makes-tokens-compete.tex',p[a:z].encode())
a=p.index('\\label{sec:consequences}');a=p.rfind('\\section',0,a);z=p.find('\\section',a+8);put('paper-consequences-section.tex',p[a:z if z>=0 else len(p)].encode())
js('excerpt-source-map.json',{'snapshot':current,'log_excerpts':excerpts,'paper_original_sha256':sh(pb),'no_source_changes':True})
root=B/'runpod/soft64/20261015T065110Z';v=json.loads((root/'result.json').read_text());assert v['commit']==original
q={k:v.get(k)for k in ['pod_id','gpu_type','commit','jobset','started','runner_returncode','hash_verified','local_sha256','remote_sha256','wall_seconds']};q['upload']={k:v.get('upload',{}).get(k)for k in ['code_bytes','data_bytes','code_sha256','data_sha256','extra_data','jobset_args']};q['receipt']=pin(root/'result.json');q['archive']=pin(root/'results.tgz');assert q['archive']['sha256']==v['local_sha256']==v['remote_sha256'] and v['hash_verified']
q['stdout']=pin(root/'runner-stdout.txt');q['status_lines']=[s for s in(root/'runner-stdout.txt').read_text().splitlines()if re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ (?:start [A-Za-z0-9_.-]+|end [A-Za-z0-9_.-]+ status -?\d+)',s)]
allow=['event','time','pod_id','commit','gpu_type','archive_bytes','data_bytes','hash_verified','sha256','seconds_after_create'];lr=[]
for i,b in enumerate((B/'runpod/pods.jsonl').read_bytes().splitlines()):
 d=json.loads(b)
 if d.get('pod_id')==v['pod_id']:lr.append(dict(line_number=i+1,original_line_sha256=sh(b),fields={k:d[k]for k in allow if k in d}))
q['ledger_records']=lr;q['recorded_termination_verified']=any(r['fields'].get('event')=='terminated_verified'for r in lr);assert q['recorded_termination_verified'];raw=B/'explore/llm-soft-loop/pod64/repro/h2cells';inputs=[]
with tarfile.open(root/'results.tgz','r:gz')as tf:
 members=tf.getmembers()
 for pp in sorted(raw.iterdir()):
  if not pp.is_file():continue
  row=pin(pp);rel=str(pp.relative_to(B));matches=[m for m in members if m.isfile()and m.name.lstrip('./')==rel];assert len(matches)==1
  with tf.extractfile(matches[0])as f:blob=f.read()
  assert len(blob)==row['bytes']and sh(blob)==row['sha256'];item=dict(file=row,archive_member=matches[0].name,archive_member_match=True,scientific_payload_decoded=False)
  if pp.suffix=='.npz':
   hdr=[]
   with zipfile.ZipFile(pp)as zp:
    for name in zp.namelist():
     with zp.open(name)as f:
      assert f.read(6)==b'\x93NUMPY';major,minor=f.read(2);size=struct.unpack('<H'if major==1 else'<I',f.read(2 if major==1 else 4))[0];header=ast.literal_eval(f.read(size).decode('latin1').strip());hdr.append(dict(member=name,header=header))
   item['npy_headers']=hdr
  inputs.append(item)
q['selected_files']=inputs;q['historical_limits']=['Original recorded source/upload/retrieval/unique archive-member and termination basis; no new remote or model-asset-byte reconstruction.','Target scientific JSON is opaque hashed bytes; NPZ header shapes/dtypes only. No scientific values decoded or producer code executed.','Exposed source log/paper summaries are not independent numerical acceptance.']
js('repro1-original-authority.json',q)
manifest={'source_snapshot':current,'original_repro1_source_commit':original,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'hash_io':{'bytes':sum(x['bytes']for x in io),'seconds':sum(x['seconds']for x in io)},'scope':'Task26(b) source/card/local original retrieval metadata and Task26(e) mechanism source/log/paper only. No target scientific outcome decoding or execution.'};(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'path':str(O),'manifest_sha256':sh((O/'manifest.json').read_bytes()),'authority_sha256':files['repro1-original-authority.json']['sha256'],'target_bytes':sum(x['file']['bytes']for x in inputs),'hash_io':manifest['hash_io']}))
