from pathlib import Path
import os,resource,json,hashlib,re,tarfile,time,subprocess,datetime
os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)));resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2)
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');BASE=Path('/data2/neural-shrimp-data');OUT=Path('/tmp/task21b-input-authority-002');OUT.mkdir(exist_ok=False)
tic=time.monotonic();io=[];files={};rows=[];runs=[]
def digest(b):return hashlib.sha256(b).hexdigest()
def pin(p):
 p=Path(p);a=p.stat();h=hashlib.sha256();t=time.monotonic()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 z=p.stat();assert(a.st_size,a.st_mtime_ns,a.st_ino)==(z.st_size,z.st_mtime_ns,z.st_ino);io.append(dict(path=str(p),bytes=a.st_size,seconds=time.monotonic()-t));return dict(path=str(p),sha256=h.hexdigest(),bytes=a.st_size)
def export(name,b):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);files[name]=pin(p);return files[name]
def save(name,v):return export(name,(json.dumps(v,indent=2)+'\n').encode())
def gitshow(commit,path):return subprocess.check_output(['git','-c','core.preloadIndex=false','-c','core.packedGitWindowSize=32m','-c','core.packedGitLimit=64m','show',commit+':'+path],cwd=R)
ledger=BASE/'runpod/pods.jsonl';lrows=[(i+1,b,json.loads(b))for i,b in enumerate(ledger.read_bytes().splitlines())];allow=['event','time','pod_id','commit','gpu_type','archive_bytes','data_bytes','hash_verified','sha256','seconds_after_create']
specs=[('KY1','20261015T042138Z','spectrum-ky-philosophy','docs/ky-cells-philosophy.json',list(range(20)),[1000,2000],{'high':list(range(10)),'head':list(range(10,20))}),('KY2','20261015T042142Z','spectrum-ky-hostility','docs/ky-cells-hostility.json',list(range(10)),[1000,2000],{'host':list(range(10))}),('TC','20261015T004735Z','tc-spectrum','docs/h2-horizon-cells.json',[10,47,56,143,240,293,242,406],[250,500,1000,2000],{'prior_headline':[10,47,56,240,293,242],'periodic_control':[143,406]})]
for rid,stamp,folder,cp,indices,checkpoints,groups in specs:
 root=BASE/'runpod/soft64'/stamp;v=json.loads((root/'result.json').read_text());commit=v['commit'];rp=pin(root/'result.json');ap=pin(root/'results.tgz');sp=pin(root/'runner-stdout.txt')
 assert v['hash_verified'] is True and v['local_sha256']==v['remote_sha256']==ap['sha256']
 statuses=[s for s in (root/'runner-stdout.txt').read_text().splitlines() if re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ (?:start [A-Za-z0-9_.-]+|end [A-Za-z0-9_.-]+ status -?\d+)',s)]
 lr=[dict(line_number=i,original_line_sha256=digest(b),fields={k:d[k] for k in allow if k in d})for i,b,d in lrows if d.get('pod_id')==v['pod_id']];terminated=any(x['fields'].get('event')=='terminated_verified'for x in lr);assert terminated
 p=BASE/'explore/llm-soft-loop/pod64/hidden'/folder/'lyapunov-spectrum.json';pp=pin(p);rel=str(p.relative_to(BASE))
 with tarfile.open(root/'results.tgz','r:gz')as tf:
  ms=[m for m in tf.getmembers()if m.isfile()and m.name.lstrip('./')==rel];assert len(ms)==1;(m,)=ms
  with tf.extractfile(m)as f:b=f.read()
  assert len(b)==pp['bytes']and digest(b)==pp['sha256']
 # Read only source-defined metadata fields, never parse exponent/entropy/checkpoint vectors.
 text=p.read_text();mark=text.index('\n "cells":');args=json.loads(text[:mark].rstrip().rstrip(',')+'\n}')['args']
 got=[int(x)for x in re.findall(r'^\s+"index": (\d+),?$',text,re.M)]
 cm=re.search(r'\n "complete": (true|false)\s*\n}\s*$',text);assert cm;complete=cm.group(1)=='true'
 assert got==indices[:len(got)] and args['cells_index']==indices and args['k']==4 and args['transient']==1000 and args['record']==2000 and args['checkpoints']==checkpoints and args['cells_file']==cp
 assert complete==(len(got)==len(indices));assert len(got)=={'KY1':18,'KY2':10,'TC':7}[rid]
 source=export('source/'+rid+'/lyapunov_spectrum.py',gitshow(commit,'tools/lyapunov_spectrum.py'));cells=export('source/'+rid+'/'+Path(cp).name,gitshow(commit,cp));cv=json.loads(Path(cells['path']).read_text());assert max(indices)<len(cv['cells'])
 runner=gitshow(commit,'tools/runpod_soft_runner.sh');ls=runner.decode().splitlines();tag='  tangentcheck)'if rid=='TC'else'  spectrum)';i=next(i for i,s in enumerate(ls)if s.startswith(tag));j=next(j for j in range(i+1,len(ls))if ls[j].strip()==';;');runner_excerpt=export('source/'+rid+'/runner-command.sh',('\n'.join(ls[i:j+1])+'\n').encode())
 if rid=='TC':
  selection=export('source/TC/tc-spectrum-cells.json',gitshow(commit,'docs/tc-spectrum-cells.json'));sel=json.loads(Path(selection['path']).read_text());assert sel['indices']==indices
  log=gitshow(commit,'docs/research-log.md').decode();lines=log.splitlines();i=next(i for i,s in enumerate(lines)if s.startswith('## 2026-10-15 00:47 UTC:'));j=next((j for j in range(i+1,len(lines))if lines[j].startswith('## ')),len(lines));block=lines[i:j]
  safe=[];omitted=[]
  for offset,line in enumerate(block):
   if 'rate cap' in line:
    omitted.append(dict(source_line=i+offset+1,sha256=digest(line.encode()),reason='Private pricing/rate material omitted; resource durations retained from actual command/source.'));safe.append('[Private rate/cost line omitted; exact original source commit and omitted-line hash recorded in authority.]')
   else:safe.append(line)
  card=export('source/TC/original-card-scoped.md',('\n'.join(safe)+'\n').encode());selection_basis=dict(source_commit=commit,selection_file=selection,scoped_card=card,omitted_lines=omitted,original_block_sha256=digest(('\n'.join(block)+'\n').encode()),rule='Prior H2 outcome-based percentile selection fixed before TC spectra: three chaotic and one periodic control per map; distinct from later KY percentile design.')
 else:selection_basis=dict(source_commit=commit,cells_file=cells,rule='KY pre-spectrum percentile design in pinned cells file and source-metadata-001 original card; source-derived fixed group membership, no selection from current spectra.')
 q={k:v.get(k)for k in ['pod_id','gpu_type','commit','jobset','started','runner_returncode','hash_verified','local_sha256','remote_sha256','wall_seconds']};q['upload']={k:v.get('upload',{}).get(k)for k in ['code_bytes','data_bytes','code_sha256','data_sha256','extra_data','jobset_args']}
 q.update(id=rid,receipt=rp,archive=ap,stdout=sp,status_lines=statuses,ledger_records=lr,recorded_termination_verified=terminated,selected_member=dict(path=rel,member=m.name,file=pp,archive_match=True),source=source,cells_file=cells,runner_excerpt=runner_excerpt,runner_full_source_sha256=digest(runner),selection_basis=selection_basis)
 runs.append(q)
 rows.append(dict(id=rid,file=pp,cells_file=cells,source=source,source_commit=commit,requested_indices=indices,completed_indices=got,missing_indices=indices[len(got):],checkpoint_steps=checkpoints,k=4,transient=1000,record=2000,expected_complete=complete,requested_count=len(indices),completed_count=len(got),group_memberships=groups,selected_coordinates=[dict(index=i,coordinates=cv['cells'][i])for i in indices],original_args_metadata=args,original_disposition=('tool timeout 124; retrieved partial18/20; outer runner0 distinct' if rid=='KY1' else ('tool timeout 124; retrieved partial7/8, final periodic control406 missing; outer runner0 distinct' if rid=='TC' else 'retrieved selected spectrum tool completed0; other job statuses retained separately')),receipt=rp,archive=ap,stdout=sp,recorded_termination_verified=terminated,leading_spectrum_authority='Original source defines k leading QR estimates, not full ambient spectrum or exact asymptotic exponents.',ambient_dimension=896,ambient_dimension_scope='Qwen2.5-0.5B single-token soft-loop source contract; no model loaded or asset-byte audit here.',spectrum_values_decoded=False))
a=save('original-authority.json',dict(schema='task21b-original-authority-v1',runs=runs,scope='Local original receipts, allowlisted ledger events, exact source/commands/cell metadata, streamed hashes and unique archive-member equality. Target JSON reads restricted to args, index labels and final complete boolean; no exponent, entropy or checkpoint-vector values parsed.',limitations=['No new pod or model query; original recorded remote source/upload/archive/termination is finite historical authority.','No independent reconstruction of historical model/tokenizer/vector bytes or original process births.','KY1 and TC spectrum timeouts and original outer runner zero remain distinct. TC grouping comes from its prior design, not the later KY design.','Producer log/paper summaries were exposed before independent arithmetic; later computations are unblinded reproduction.']))
m=save('inputs.json',dict(schema='task21b-saved-spectrum-inputs-v1',status='prepared_unlaunchable',source_snapshot='b7ffbb7d02a31069d633b62fa1a7bac4711ca93a',authority=a,source_metadata_manifest=pin('/tmp/task21b-source-metadata-001/manifest.json'),runs=rows,selected_file_bytes=sum(x['file']['bytes']for x in rows),requested_cells=38,completed_cells=35,spectrum_values_decoded=False))
manifest=dict(files=files,hash_io=dict(bytes=sum(x['bytes']for x in io),seconds=sum(x['seconds']for x in io)),elapsed_seconds=time.monotonic()-tic,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(inputs=m,authority=a,manifest_sha256=digest((OUT/'manifest.json').read_bytes()),selected_files=len(rows),selected_bytes=sum(x['file']['bytes']for x in rows),hash_io=manifest['hash_io'],elapsed_seconds=manifest['elapsed_seconds'])))
