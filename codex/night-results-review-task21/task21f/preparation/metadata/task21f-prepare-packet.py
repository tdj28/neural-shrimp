from pathlib import Path
import ast,datetime,hashlib,io,json,re,struct,subprocess,tarfile,time,zipfile
import numpy as np
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');DATA=Path('/data2/neural-shrimp-data');D=Path('/tmp/task21f-source-001');D.mkdir(exist_ok=False)
start=time.perf_counter();count=0
H=lambda b:hashlib.sha256(b).hexdigest()
def read(p):
 global count
 b=Path(p).read_bytes();count+=len(b);return b
def pin(p):
 b=read(p);return {'path':str(p),'sha256':H(b),'bytes':len(b)}
def put(n,b):
 p=D/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);return {'path':str(p),'sha256':H(b),'bytes':len(b)}
def dump(n,v):return put(n,(json.dumps(v,indent=2)+'\n').encode())
def git(c,p):return subprocess.check_output(['git','show',c+':'+p],cwd=R)
def schema(v):
 if isinstance(v,dict):return {k:schema(x)for k,x in v.items()}
 if isinstance(v,list):return {'type':'list','length':len(v),'element':schema(v[0])if v else None}
 return type(v).__name__
def safe(v):
 if isinstance(v,dict):return {k:safe(x)for k,x in v.items()if not any(t in k.lower()for t in ['cost','price','rate'])}
 if isinstance(v,list):return [safe(x)for x in v]
 return v
current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip()
ids=['DSS1','DSS1b','DSS1c','DSS2a','DSS2b','DSS2c'];runs=[];attempts=[];commits=[]
for ident in ids:
 op=DATA/'runpod'/f'{ident}-orchestrator.log';ob=read(op);events=[]
 for l in ob.splitlines():
  try:events.append(json.loads(l))
  except ValueError:pass
 init=next(x for x in events if 'jobset_args'in x);c=init['commit'];commits.append(c);name=init['jobset_args'].split()[0]
 out=next(x for x in reversed(events)if'out'in x);base=Path(out['out']);rb=read(base/'result.json');result=json.loads(rb)
 assert result['commit']==c and result['hash_verified']
 archive=base/'results.tgz';ab=read(archive);assert H(ab)==result['local_sha256']==result['remote_sha256']
 stdout=base/'runner-stdout.txt';sb=read(stdout);lines=[l.decode()for l in sb.splitlines()if re.search(rb' start dss-| end dss-| runner done',l)]
 initpin=dump('authority/'+ident+'/initial.json',safe(init));retpin=dump('authority/'+ident+'/retrieval.json',{**safe(result),'original_record':{'path':str(base/'result.json'),'sha256':H(rb),'bytes':len(rb),'financial_fields_omitted':True}})
 term=next(x for x in reversed(events)if'terminated'in x);termpin=dump('authority/'+ident+'/termination.json',safe(term))
 statuspin=dump('authority/'+ident+'/tool-status.json',{'original_path':str(stdout),'sha256':H(sb),'bytes':len(sb),'exact_status_lines':lines,'runner_returncode':result['runner_returncode']})
 a={'id':ident,'run':name,'source_commit':c,'initial':initpin,'retrieval':retpin,'termination':termpin,'tool_status':statuspin,'orchestrator':{'path':str(op),'sha256':H(ob),'bytes':len(ob)},'archive':{'path':str(archive),'sha256':H(ab),'bytes':len(ab)},'selected_members':[],'historical_limit':'Original local source/upload/retrieval/status and recorded producer termination. Historical remotely loaded bytes, process births, model assets and native numerical premises are not independently reconstructed.'}
 r={'id':name,'attempt':ident,'source_commit':c,'original_command':init['jobset_args'],'files':{},'declared_args':None,'metadata':{},'scope':'Header/axis/argument/progress-index metadata only at preparation; numerical grid fields not decoded.'}
 with tarfile.open(fileobj=io.BytesIO(ab))as tf:
  names={m.name:m for m in tf.getmembers()if m.isfile()}
  prefix='explore/llm-soft-loop/pod64/dss/'+name+'/'
  for fname in ['grid.npz','grid.json','progress.jsonl']:
   member=prefix+fname
   if member not in names:
    r['files'][fname]=None;continue
   b=tf.extractfile(names[member]).read();original=DATA/member;assert read(original)==b
   fp=put('inputs/'+name+'/'+fname,b);r['files'][fname]=fp;a['selected_members'].append({'member':member,'original_path':str(original),**fp})
   if fname=='grid.json':
    q=json.loads(b);r['declared_args']=q['args'];r['metadata']['grid_json_schema']=schema(q);r['metadata']['grid_json_present']=True
   if fname=='progress.jsonl':
    schemas={};indices=[];nlines=0;other_schemas=[]
    for line in b.splitlines():
     q=json.loads(line);nlines+=1;s=schema(q);key=json.dumps(s,sort_keys=True);schemas[key]=s
     if 'i'in q and'j'in q:indices.append([q['i'],q['j']])
     else:other_schemas.append(s)
    r['metadata']['progress']={'line_count':nlines,'cell_record_count':len(indices),'indices':indices,'unique_index_count':len(set(map(tuple,indices))),'schemas':list(schemas.values()),'other_record_schemas':other_schemas,'scope':'BFS record coverage metadata, not numerical success/clean/token/factor assessment. NPZ may lag final progress in an interrupted attempt.'}
   if fname=='grid.npz':
    headers=[];axes={}
    with zipfile.ZipFile(io.BytesIO(b))as z:
     for zi in z.infolist():
      with z.open(zi)as f:
       v=np.lib.format.read_magic(f);assert v in [(1,0),(2,0)],v;shape,fort,dtype=(np.lib.format.read_array_header_1_0(f) if v==(1,0) else np.lib.format.read_array_header_2_0(f))
      assert not dtype.hasobject
      headers.append({'member':zi.filename,'shape':list(shape),'dtype':dtype.str,'fortran_order':fort,'array_nbytes':int(np.prod(shape,dtype=np.int64))*dtype.itemsize,'zip_uncompressed_bytes':zi.file_size,'zip_compressed_bytes':zi.compress_size,'zip_crc':zi.CRC})
      if zi.filename in ['alpha.npy','log10_temperature.npy','seed.npy','row_axis.npy']:
       with z.open(zi)as f:x=np.lib.format.read_array(f,allow_pickle=False)
       axes[zi.filename[:-4]]=x.tolist()
    r['metadata']['npz_members']=headers;r['metadata']['axis_and_seed_metadata']=axes
 r['metadata']['grid_json_present']=r['files']['grid.json']is not None
 r['original_tool_status_lines']=lines
 runs.append(r);attempts.append(a)
source=[];seen={}
for c in dict.fromkeys(commits):
 for p in ['tools/doubly_superstable.py','tools/llm_feigenbaum.py','tools/explore_llm_soft_loop.py','tools/explore_llm_attractors.py']:
  b=git(c,p);h=H(b)
  if (p,h)not in seen:seen[p,h]=put('source/'+c[:8]+'/'+p,b)
  source.append({'source_commit':c,'repository_path':p,**seen[p,h]})
 p='tools/runpod_soft_runner.sh';b=git(c,p);ls=b.splitlines(keepends=True);i=next(i for i,l in enumerate(ls)if re.match(rb'  dss\)',l));j=next(j for j in range(i+1,len(ls))if re.match(rb'  [A-Za-z0-9_]+\)',ls[j]))
 hi=next(i for i,l in enumerate(ls)if l.startswith(b'run()'));hj=next(j for j in range(hi+1,len(ls))if ls[j].strip()==b'}')+1
 for lo,end in [(hi,hj),(i,j)]:source.append({'source_commit':c,'repository_path':p,'full_source_sha256':H(b),'start_line':lo+1,'end_line':end,**put('source/'+c[:8]+f'/runner.lines{lo+1}-{end}.sh',b''.join(ls[lo:end]))})
frozen='66ffa102c56f6d5b22f116a1bb194f503c75562e'
for p in ['tools/doubly_superstable.py','tools/plot_superstable_lines.py']:
 b=git(frozen,p);source.append({'source_commit':frozen,'repository_path':p,'scope':'Previously reviewed Task27 source; historical run versions above remain authoritative.',**put('source/task27/'+Path(p).name,b)})
for a,c in [(commits[0],commits[1]),(commits[1],commits[3]),(commits[3],frozen)]:
 put('source/'+a[:8]+'-to-'+c[:8]+'.diff',subprocess.check_output(['git','diff',a,c,'--','tools/doubly_superstable.py'],cwd=R))
claims=[];log=git(frozen,'docs/research-log.md');ls=log.splitlines(keepends=True)
for stamp in ['04:02','04:37','05:50','06:54']:
 i=next(i for i,l in enumerate(ls)if l.startswith(('## 2026-10-15 '+stamp+' UTC:').encode()));j=next((j for j in range(i+1,len(ls))if ls[j].startswith(b'## ')),len(ls));claims.append({'source_commit':frozen,'repository_path':'docs/research-log.md','full_source_sha256':H(log),'start_line':i+1,'end_line':j,**put('claims/'+stamp.replace(':','')+'.md',b''.join(ls[i:j]))})
# This separate token-amendment entry is identified by source text, not outcomes.
for i,l in enumerate(ls):
 if l.startswith(b'## ') and b'DSS'in l and b'05:50'not in l and b'06:54'not in l and b'04:37'not in l and b'04:02'not in l:
  j=next((j for j in range(i+1,len(ls))if ls[j].startswith(b'## ')),len(ls));claims.append({'source_commit':frozen,'repository_path':'docs/research-log.md','full_source_sha256':H(log),'start_line':i+1,'end_line':j,**put('claims/dss-entry-'+str(i+1)+'.md',b''.join(ls[i:j]))})
paper=git(current,'paper/llm/main.tex');ls=paper.splitlines(keepends=True)
i=next(i for i,l in enumerate(ls)if b'\\label{sec:superstable}'in l)if any(b'\\label{sec:superstable}'in l for l in ls)else next(i for i,l in enumerate(ls)if b'1.3' in l and b'10^{-3}'in l)
lo=max(0,i-8);hi=min(len(ls),i+65);claims.append({'source_commit':current,'repository_path':'paper/llm/main.tex','full_source_sha256':H(paper),'start_line':lo+1,'end_line':hi,**put('claims/current-paper.tex',b''.join(ls[lo:hi]))})
q=Path('/tmp/task21c-source-001/assignment-and-queue.md');put('claims/assignment-and-queue.md',read(q))
sm=dump('source-map.json',{'schema':'task21f-source-map-v1','capture_commit':current,'source_files':source,'claims':claims,'gap_analysis_source':'No named independent gap-analysis source identified in the bounded original tool/plotter/Task27 source packet. Plotter saves segments but has no closest-gap routine; retain producer quoted gap as claim pending an explicitly defined own saved-grid distance.','scope':'Original Git source and amendments, no producer imports or execution.'})
au=dump('original-authority.json',{'schema':'task21f-original-authority-v1','attempts':attempts,'source_map':sm,'scope':'Six selected local archives/member hashes and exact original commands/source/status/recorded termination. No pod queries or model/native states. Numerical grid payloads remain unexamined.','snapshot_limit':'The interrupted original p7-blob has no final grid.json; progress can be newer than final NPZ checkpoint. Each saved source kept separate; all overlap remains.'})
inp=dump('inputs.json',{'schema':'task21f-saved-inputs-v1','prepared_unlaunchable':True,'capture_commit':current,'runs':runs,'authority':au,'source_map':sm,'run_count':6,'target_file_bytes':sum(f['bytes']for r in runs for f in r['files'].values()if f),'join_scope':'Separate originals always retained. Prospective plot comparison uses literal ordered zoom a/b/c and drops first row of b/c only after exact axis metadata association, never tolerant search or interpolation.','native_unavailable':'Parent/orientation vectors and orbit/native derivative replay absent; gauges and precision premises inherited.'})
files=[{'path':str(p.relative_to(D)),'bytes':p.stat().st_size,'sha256':H(p.read_bytes())}for p in sorted(D.rglob('*'))if p.is_file()]
man=dump('manifest.json',{'schema':'task21f-source-input-packet-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':current,'inputs':inp,'authority':au,'source_map':sm,'files':files,'packet_bytes':sum(x['bytes']for x in files),'metadata_bytes_read':count,'metadata_elapsed_seconds':time.perf_counter()-start,'measurement_scope':'Archive/member/hash/header/axis/progress-index preparation; source Git reads and final manifest hashing excluded from byte counter. Not target arithmetic runtime.','target_numerical_grid_decoded':False,'models_or_producer_code_executed':False})
print(json.dumps({'manifest':man,'inputs':inp,'authority':au,'source_map':sm,'runs':[{'id':r['id'],'shape':next(h['shape']for h in r['metadata']['npz_members']if h['member']=='mu.npy'),'fields':[h['member']for h in r['metadata']['npz_members']],'progress_records':r['metadata']['progress']['cell_record_count'],'final_json':r['metadata']['grid_json_present'],'status':r['original_tool_status_lines']}for r in runs]},indent=2))
