from pathlib import Path
import ast,datetime,hashlib,io,json,re,subprocess,tarfile,time,zipfile
import numpy as np
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');DATA=Path('/data2/neural-shrimp-data');D=Path('/tmp/task21g-source-001');D.mkdir(exist_ok=False);start=time.perf_counter();count=0
H=lambda b:hashlib.sha256(b).hexdigest()
def read(p):
 global count
 b=Path(p).read_bytes();count+=len(b);return b
def put(n,b):
 p=D/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);return {'path':str(p),'sha256':H(b),'bytes':len(b)}
def dump(n,q):return put(n,(json.dumps(q,indent=2)+'\n').encode())
def git(c,p):return subprocess.check_output(['git','show',c+':'+p],cwd=R)
def safe(v):
 if isinstance(v,dict):return {k:safe(x)for k,x in v.items()if not any(t in k.lower()for t in ['cost','rate','price'])}
 if isinstance(v,list):return [safe(x)for x in v]
 return v
def schema(v):
 if isinstance(v,dict):return {k:schema(x)for k,x in v.items()}
 if isinstance(v,list):return {'type':'list','length':len(v),'element':schema(v[0])if v else None}
 return type(v).__name__
current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip();op=DATA/'runpod/RM1-orchestrator.log';ob=read(op);ev=[]
for line in ob.splitlines():
 try:ev.append(json.loads(line))
 except ValueError:pass
init=next(x for x in ev if'jobset_args'in x);c=init['commit'];out=next(x for x in reversed(ev)if'out'in x);base=Path(out['out']);rb=read(base/'result.json');q=json.loads(rb);assert q['commit']==c and q['hash_verified'];ab=read(base/'results.tgz');assert H(ab)==q['local_sha256']==q['remote_sha256']
initial=dump('authority/initial.json',safe(init));retrieval=dump('authority/retrieval.json',{**safe(q),'original_record':{'path':str(base/'result.json'),'sha256':H(rb),'bytes':len(rb),'financial_fields_omitted':True}});term=dump('authority/termination.json',safe(next(x for x in reversed(ev)if'terminated'in x)));sb=read(base/'runner-stdout.txt');status=dump('authority/tool-status.json',{'original_path':str(base/'runner-stdout.txt'),'sha256':H(sb),'bytes':len(sb),'exact_status_lines':[l.decode()for l in sb.splitlines()if re.search(rb' start retmap-| end retmap-| runner done',l)],'runner_returncode':q['runner_returncode']})
files={};members=[];metadata={}
with tarfile.open(fileobj=io.BytesIO(ab))as tf:
 allmembers=[m for m in tf.getmembers()if m.isfile()];names={m.name:m for m in allmembers}
 inventory=[{'name':m.name,'bytes':m.size}for m in allmembers]
 for name in ['return-map.npz','return-map.json']:
  member='explore/llm-soft-loop/pod64/retmap/body-edge/'+name;b=tf.extractfile(names[member]).read();assert read(DATA/member)==b;fp=put('inputs/'+name,b);files[name]=fp;members.append({'archive_member':member,'original_path':str(DATA/member),**fp})
  if name.endswith('.json'):
   j=json.loads(b);metadata['args']=j['args'];metadata['schema']=schema(j);metadata['cell_order']=[{'index':i,'alpha':r['alpha'],'log10_temperature':r['log10_temperature'],'phase_indices':[p['phase']for p in r['phases']],'phase_schema':schema(r['phases'][0])}for i,r in enumerate(j['cells'])]
  else:
   headers=[];axes={}
   with zipfile.ZipFile(io.BytesIO(b))as z:
    for zi in z.infolist():
     with z.open(zi)as f:
      v=np.lib.format.read_magic(f);assert v in [(1,0),(2,0)];shape,fort,dtype=(np.lib.format.read_array_header_1_0(f)if v==(1,0)else np.lib.format.read_array_header_2_0(f))
     assert not dtype.hasobject
     headers.append({'member':zi.filename,'shape':list(shape),'dtype':dtype.str,'fortran_order':fort,'array_nbytes':int(np.prod(shape,dtype=np.int64))*dtype.itemsize,'zip_uncompressed_bytes':zi.file_size,'zip_compressed_bytes':zi.compress_size,'zip_crc':zi.CRC})
     if zi.filename in ['cells.npy','period.npy']:
      with z.open(zi)as f:axes[zi.filename[:-4]]=np.lib.format.read_array(f,allow_pickle=False).tolist()
   metadata['npz_members']=headers;metadata['coordinate_and_period_metadata']=axes
source=[]
for p in ['tools/return_map.py','tools/explore_llm_soft_loop.py','tools/explore_llm_attractors.py']:
 b=git(c,p);source.append({'source_commit':c,'repository_path':p,**put('source/original/'+Path(p).name,b)})
p='tools/explore_llm_phase_map.py';b=git(c,p);tree=ast.parse(b);cl=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='SteeredModel');fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef)and n.name=='__init__');lo=cl.lineno;hi=fn.end_lineno;source.append({'source_commit':c,'repository_path':p,'full_source_sha256':H(b),'start_line':lo,'end_line':hi,**put('source/original/steered-model-init.py',b''.join(b.splitlines(keepends=True)[lo-1:hi]))})
p='tools/runpod_soft_runner.sh';b=git(c,p);ls=b.splitlines(keepends=True);i=next(i for i,l in enumerate(ls)if re.match(rb'  retmap\)',l));j=next(j for j in range(i+1,len(ls))if re.match(rb'  [A-Za-z0-9_]+\)',ls[j]));hi=next(i for i,l in enumerate(ls)if l.startswith(b'run()'));hj=next(j for j in range(hi+1,len(ls))if ls[j].strip()==b'}')+1
for lo,end in [(hi,hj),(i,j)]:source.append({'source_commit':c,'repository_path':p,'full_source_sha256':H(b),'start_line':lo+1,'end_line':end,**put(f'source/original/runner.lines{lo+1}-{end}.sh',b''.join(ls[lo:end]))})
frozen='66ffa102c56f6d5b22f116a1bb194f503c75562e'
for p in ['tools/return_map.py','tools/plot_return_maps.py']:
 b=git(frozen,p);source.append({'source_commit':frozen,'repository_path':p,**put('source/task27/'+Path(p).name,b)})
put('source/original-to-task27.diff',subprocess.check_output(['git','diff',c,frozen,'--','tools/return_map.py'],cwd=R))
claims=[];log=git(frozen,'docs/research-log.md');ls=log.splitlines(keepends=True)
for stamp in ['06:00','06:08']:
 i=next(i for i,l in enumerate(ls)if l.startswith(('## 2026-10-15 '+stamp+' UTC:').encode()));j=next((j for j in range(i+1,len(ls))if ls[j].startswith(b'## ')),len(ls));claims.append({'source_commit':frozen,'repository_path':'docs/research-log.md','full_source_sha256':H(log),'start_line':i+1,'end_line':j,**put('claims/'+stamp.replace(':','')+'.md',b''.join(ls[i:j]))})
put('claims/assignment-and-queue.md',read('/tmp/task21c-source-001/assignment-and-queue.md'))
sm=dump('source-map.json',{'schema':'task21g-source-map-v1','capture_commit':current,'source_files':source,'claims':claims,'original_factor_equals_task27':git(c,'tools/return_map.py')==git(frozen,'tools/return_map.py'),'scope':'Exact original source/loader/runner association only. No native/PCA replay.'})
au=dump('original-authority.json',{'schema':'task21g-original-authority-v1','original_id':'RM1','naming_note':'Queue labels return maps RM; original local orchestrator/manifest identity is RM1, not coordinator shorthand RM0. Exact source and seven-cell command identify the selected record.','source_commit':c,'initial':initial,'retrieval':retrieval,'termination':term,'tool_status':status,'orchestrator':{'path':str(op),'sha256':H(ob),'bytes':len(ob)},'archive':{'path':str(base/'results.tgz'),'sha256':H(ab),'bytes':len(ab)},'archive_file_inventory':inventory,'selected_members':members,'source_map':sm,'scope':'Original local archive/member retrieval and recorded producer termination; no historical loaded-source/model-asset/birth reconstruction. No target scalar sequences decoded during preparation.','native_pca_limit':'Selected producer output saves scalar projections and summary shares, not native states/PCA vectors/means/singular values. Original code computes then discards native states; independent PCA explained variance/native clock/precision cannot be reconstructed from these selected artifacts.'})
inp=dump('inputs.json',{'schema':'task21g-saved-inputs-v1','prepared_unlaunchable':True,'capture_commit':current,'source_commit':c,'attempt':'RM1','files':files,'metadata':metadata,'authority':au,'source_map':sm,'requested_cell_count':7,'phase_count':7,'expected_sequence_count':49,'group_memberships':{'body_edge':[0,1,2],'hostile_line':[3,4,5,6],'original_figure_rows':[0,1,2,5]},'claims_scope':'Body-edge every-phase claim distinct from all7cell/figure-row inventory; mixed nominal-phase control retained. No outcome-selected pass threshold.','target_bytes':sum(x['bytes']for x in files.values())})
rows=[{'path':str(p.relative_to(D)),'bytes':p.stat().st_size,'sha256':H(p.read_bytes())}for p in sorted(D.rglob('*'))if p.is_file()]
man=dump('manifest.json',{'schema':'task21g-source-input-packet-v1','source_commit':current,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inputs':inp,'authority':au,'source_map':sm,'files':rows,'packet_bytes':sum(x['bytes']for x in rows),'metadata_bytes_read':count,'metadata_elapsed_seconds':time.perf_counter()-start,'measurement_scope':'Finite archive/member/hash/header/coordinate/schema metadata; excludes Git/source and finalmanifesthash bytes; not target arithmetic runtime.','target_scalar_sequences_decoded':False,'producer_code_executed':False})
print(json.dumps({'manifest':man,'inputs':inp,'authority':au,'source_map':sm,'target_bytes':sum(x['bytes']for x in files.values()),'coordinate_metadata':metadata['coordinate_and_period_metadata'],'sequence_shapes':{x['member']:x['shape']for x in metadata['npz_members']if x['member'].startswith('c')and x['member']!='cells.npy'}},indent=2))
