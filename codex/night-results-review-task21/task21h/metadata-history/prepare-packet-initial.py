from pathlib import Path
import datetime,hashlib,io,json,re,subprocess,tarfile,time,zipfile,resource,os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1');os.nice(10-os.getpriority(os.PRIO_PROCESS,0));resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2)
import numpy as np
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');DATA=Path('/data2/neural-shrimp-data');D=Path('/tmp/task21h-source-001');D.mkdir(exist_ok=False);start=time.perf_counter();count=0;H=lambda b:hashlib.sha256(b).hexdigest()
def read(p):
 global count
 b=Path(p).read_bytes();count+=len(b);return b
def put(n,b):
 p=D/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);return {'path':str(p),'sha256':H(b),'bytes':len(b)}
def dump(n,v):return put(n,(json.dumps(v,indent=2)+'\n').encode())
def git(c,p):return subprocess.check_output(['git','show',c+':'+p],cwd=R,timeout=20)
def safe(v):
 if isinstance(v,dict):return {k:safe(x)for k,x in v.items()if not any(t in k.lower()for t in ['cost','price','rate'])}
 if isinstance(v,list):return [safe(x)for x in v]
 return v
def schema(v):
 if isinstance(v,dict):return {k:schema(x)for k,x in v.items()}
 if isinstance(v,list):return {'type':'list','length':len(v),'element':schema(v[0])if v else None}
 return type(v).__name__
def npz_meta(b):
 headers=[];axes={}
 with zipfile.ZipFile(io.BytesIO(b))as z:
  for zi in z.infolist():
   with z.open(zi)as f:
    v=np.lib.format.read_magic(f);assert v in [(1,0),(2,0)];shape,fort,dtype=(np.lib.format.read_array_header_1_0(f)if v==(1,0)else np.lib.format.read_array_header_2_0(f))
   assert not dtype.hasobject
   headers.append({'member':zi.filename,'shape':list(shape),'dtype':dtype.str,'fortran_order':fort,'array_nbytes':int(np.prod(shape,dtype=np.int64))*dtype.itemsize,'zip_uncompressed_bytes':zi.file_size,'zip_compressed_bytes':zi.compress_size,'zip_crc':zi.CRC})
   if zi.filename in ['alpha.npy','alpha1.npy','alpha2.npy','log10_temperature.npy','done.npy']:
    with z.open(zi)as f:x=np.lib.format.read_array(f,allow_pickle=False)
    axes[zi.filename[:-4]]=x.tolist()
 return {'npz_members':headers,'coordinate_and_done_metadata':axes,'scientific_grid_values_decoded':False}
current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip();ids=['LS4lo','LS4hi','LS8lo','LS8hi','LS16lo','LS16hi','LS20lo','LS20hi','TB1','TB2','TBC'];runs=[];attempts=[];commits=[]
for ident in ids:
 op=DATA/'runpod'/f'{ident}-orchestrator.log';ob=read(op);events=[]
 for l in ob.splitlines():
  try:events.append(json.loads(l))
  except ValueError:pass
 init=next(x for x in events if 'jobset_args'in x);c=init['commit'];commits.append(c);base=Path(next(x for x in reversed(events)if'out'in x)['out']);rb=read(base/'result.json');result=json.loads(rb);assert result['commit']==c and result['hash_verified'];archive=base/'results.tgz';ab=read(archive);assert H(ab)==result['local_sha256']==result['remote_sha256'];sb=read(base/'runner-stdout.txt');lines=[l.decode()for l in sb.splitlines()if re.search(rb' start | end | runner done',l)]
 refs={'initial':dump('authority/'+ident+'/initial.json',safe(init)),'retrieval':dump('authority/'+ident+'/retrieval.json',{**safe(result),'original_record':{'path':str(base/'result.json'),'sha256':H(rb),'bytes':len(rb),'financial_fields_omitted':True}}),'termination':dump('authority/'+ident+'/termination.json',safe(next(x for x in reversed(events)if'terminated'in x))),'tool_status':dump('authority/'+ident+'/tool-status.json',{'original_path':str(base/'runner-stdout.txt'),'sha256':H(sb),'bytes':len(sb),'exact_status_lines':lines,'runner_returncode':result['runner_returncode']})}
 a={'id':ident,'source_commit':c,**refs,'orchestrator':{'path':str(op),'sha256':H(ob),'bytes':len(ob)},'archive':{'path':str(archive),'sha256':H(ab),'bytes':len(ab)},'selected_members':[],'scope':'Original selected local archive/member/status/retrieval and recorded termination only; historical remote loaded bytes, process births and model/native numerical premises not reconstructed.'}
 names=[x.strip().split()[0]for x in init['jobset_args'].split('::')];kind='LS'if ident.startswith('LS')else'TBC'if ident=='TBC'else'TB';folder='scan64'if kind=='LS'else'tb'
 with tarfile.open(fileobj=io.BytesIO(ab))as tf:
  members={m.name:m for m in tf.getmembers()if m.isfile()}
  for name in names:
   prefix='explore/llm-soft-loop/pod64/'+folder+'/'+name+'/';selected=sorted(k for k in members if k.startswith(prefix)and (k[len(prefix):]in ['scan.npz','scan.json','scan.partial.npz']or re.fullmatch(r'tb-t[+-][0-9.]+\.(npz|json)',k[len(prefix):])))
   assert selected,(ident,name)
   r={'id':name,'attempt':ident,'kind':kind,'source_commit':c,'original_command':init['jobset_args'],'files':{},'metadata':{},'original_tool_status_lines':lines}
   for member in selected:
    fname=member[len(prefix):];b=tf.extractfile(members[member]).read();original=DATA/member;assert read(original)==b;fp=put('inputs/'+name+'/'+fname,b);r['files'][fname]=fp;a['selected_members'].append({'member':member,'original_path':str(original),**fp})
    if fname.endswith('.npz'):r['metadata'][fname]=npz_meta(b)
    else:
     q=json.loads(b);r['metadata'][fname]={'args':q.get('args'),'schema':schema(q),'scope':'Only args/schema retained; no saved scientific summary values interpreted.'}
     if 'log10_temperature'in q:r['metadata'][fname]['declared_log10_temperature']=q['log10_temperature']
   runs.append(r)
 attempts.append(a)
# Reuse the accepted Task16 block12 authority, without re-opening its64MB historical archive.
T=R/'codex/steering-controls-crosscheck';metab=read(T/'input-metadata-001.json');meta=json.loads(metab);seg=next(x for x in meta['segments']if x['segment_id']=='segment-10');aub=read(T/'original-authority-001/authority.json');oldau=json.loads(aub);match=[x for x in oldau['records']if any(y.get('segment_id')=='segment-10'for y in x['selected_member_bindings'])];assert len(match)==1;pa=match[0];refs={'task16_header_inventory':{'path':str(T/'input-metadata-001.json'),'sha256':H(metab),'bytes':len(metab)},'task16_authority':{'path':str(T/'original-authority-001/authority.json'),'sha256':H(aub),'bytes':len(aub)}};a={'id':'Task16-block12','source_commit':pa['commit'],'inherited_accepted_task16':refs,'selected_original_record':pa,'scope':'Task16 already accepted complete source/retrieval/member/termination basis; this reuse authenticates only current selected files and saved metadata, no archive or numerical re-audit.'};dump('authority/Task16-block12/inherited-record.json',a);attempts.append(a);commits.append(pa['commit']);r={'id':'map128-philosophy','attempt':'Task16-block12','kind':'LS-reference','source_commit':pa['commit'],'original_command':pa['upload']['jobset_args'],'files':{},'metadata':{},'original_tool_status_lines':pa['status_lines']}
for fname,pin in [('scan.npz',seg),('scan.json',seg['scan_info'])]:
 b=read(pin['path']);assert len(b)==pin['bytes']and H(b)==pin['sha256'];r['files'][fname]=put('inputs/map128-philosophy/'+fname,b);r['metadata'][fname]=npz_meta(b)if fname.endswith('.npz')else {'args':seg['scan_args'],'schema':schema(json.loads(b)),'scope':'Args inherited from accepted Task16 inventory; no saved scientific summaries interpreted.'}
runs.append(r)
source=[];seen={}
for c in dict.fromkeys(commits):
 for p in ['tools/explore_llm_soft_loop.py','tools/hidden_chaos.py','tools/explore_llm_phase_map.py','tools/explore_llm_attractors.py']+(['tools/two_block_map.py']if c in [x['source_commit']for x in runs if x['kind']in ['TB','TBC']]else[]):
  b=git(c,p);h=H(b)
  if (p,h)not in seen:seen[p,h]=put('source/'+c[:8]+'/'+p,b)
  source.append({'source_commit':c,'repository_path':p,**seen[p,h]})
 b=git(c,'tools/runpod_soft_runner.sh');ls=b.splitlines(keepends=True);hi=next(i for i,l in enumerate(ls)if l.startswith(b'run()'));hj=next(i for i in range(hi+1,len(ls))if ls[i].strip()==b'}')+1
 ranges=[(hi,hj)]
 for key in ['scan64','tb','tbmulti','pairmap']:
  indices=[i for i,l in enumerate(ls)if re.match(('  '+key+r'\)').encode(),l)]
  if not indices:continue
  i=indices[0];j=next((j for j in range(i+1,len(ls))if re.match(rb'  [A-Za-z0-9_]+\)',ls[j])),len(ls));ranges.append((i,j))
 for lo,hi in ranges:source.append({'source_commit':c,'repository_path':'tools/runpod_soft_runner.sh','full_source_sha256':H(b),'start_line':lo+1,'end_line':hi,**put('source/'+c[:8]+f'/runner.lines{lo+1}-{hi}.sh',b''.join(ls[lo:hi]))})
for p in ['tools/plot_layer_sweep.py','tools/plot_two_block.py']:source.append({'source_commit':current,'repository_path':p,**put('source/current/'+Path(p).name,git(current,p))})
# Fixed-coordinate relations only; no scientific grid field is used here.
axes={r['id']:{k:v['coordinate_and_done_metadata']for k,v in r['metadata'].items()if k.endswith('.npz')}for r in runs};ref=axes['map128-philosophy']['scan.npz'];joins=[]
for block in [4,8,16,20]:
 lo=axes[f'ls-l{block}-lo']['scan.npz'];hi=axes[f'ls-l{block}-hi']['scan.npz'];v=np.array(lo['log10_temperature']+hi['log10_temperature']);rv=np.array(ref['log10_temperature']);diff=v-rv
 joins.append({'kind':'LS-fixed-ordinal','block':block,'alpha_lo_hi_equal':lo['alpha']==hi['alpha'],'alpha_equals_block12':lo['alpha']==ref['alpha'],'joined_rows':len(v),'strictly_increasing':bool(np.all(np.diff(v)>0)),'full_block12_grid_exact':bool(np.array_equal(v,rv)),'nonexact_ordinals':[{'row':int(i),'joined':float(v[i]),'block12':float(rv[i]),'difference':float(diff[i])}for i in np.flatnonzero(diff)],'scope':'Original two independently constructed halves retained by declared row order. No tolerance search/interpolation.'})
for name in ['grat4-host16','host4-grat16']:
 for lt in [-.1,.1]:
  fn=f'tb-t{lt:+.3f}.npz';p=axes[name][fn];x=axes['c-'+name+'-a1'][fn];y=axes['c-'+name+'-a2'][fn];joins.append({'kind':'TBC-exact-axis-association','plane':name,'log10_temperature':lt,'alpha1_control_exact':x['alpha1']==p['alpha1'],'alpha2_control_exact':y['alpha2']==p['alpha2'],'a1_control_alpha2':x['alpha2'],'a2_control_alpha1':y['alpha1'],'plane_has_alpha1_zero':0.0 in p['alpha1'],'plane_has_alpha2_zero':0.0 in p['alpha2'],'scope':'Exact axis association only; control and plane batches/horizon source differences remain visible. No outcome decoded.'})
joinpin=dump('coordinate-association.json',{'joins':joins,'scope':'Header/coordinate-only comparisons before scientific outcomes.'})
claims=[put('claims/assignment-and-queue.md',read(R/'codex/night-results-review-task21/task21f/source-001/claims/assignment-and-queue.md'))]
log=git(current,'docs/research-log.md');lines=log.splitlines(keepends=True)
for stamp in ['04:53','05:10','05:14','06:03','07:57','11:45']:
 hits=[i for i,l in enumerate(lines)if l.startswith(('## 2026-10-15 '+stamp+' UTC:').encode())]
 for i in hits:
  j=next((j for j in range(i+1,len(lines))if lines[j].startswith(b'## ')),len(lines));claims.append({'source_commit':current,'repository_path':'docs/research-log.md','full_source_sha256':H(log),'start_line':i+1,'end_line':j,**put('claims/'+stamp.replace(':','')+'.md',b''.join(lines[i:j]))})
sm=dump('source-map.json',{'schema':'task21h-source-map-v1','capture_commit':current,'source_files':source,'claims':claims,'scope':'Historical source/callers/classifiers and source-defined analysis; no producer imports/execution. Current plot_two_block remains the original near-axis implementation, not a saved exact-TBC join.'});au=dump('original-authority.json',{'schema':'task21h-original-authority-v1','attempts':attempts,'source_map':sm,'scope':'Eleven original local archive/retrieval/status/recorded-termination associations plus accepted Task16 block12 reuse. No remote queries/asset reconstruction. All failures/statuses retained.'});inp=dump('inputs.json',{'schema':'task21h-saved-inputs-v1','prepared_unlaunchable':True,'capture_commit':current,'runs':runs,'authority':au,'source_map':sm,'coordinate_association':joinpin,'run_count':len(runs),'target_file_bytes':sum(f['bytes']for r in runs for f in r['files'].values()),'scope':'LS half grids and inherited block12 reference; four TB planes and eight exact single-push TBC lines. No target LE/period/chaotic fields decoded. Exact control batches differ from plane batches; retain finite convention scope.'});files=[{'path':p.relative_to(D).as_posix(),'bytes':p.stat().st_size,'sha256':H(p.read_bytes())}for p in sorted(D.rglob('*'))if p.is_file()];man=dump('manifest.json',{'schema':'task21h-source-input-packet-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'capture_commit':current,'inputs':inp,'authority':au,'source_map':sm,'coordinate_association':joinpin,'files':files,'packet_bytes':sum(x['bytes']for x in files),'metadata_bytes_read':count,'metadata_elapsed_seconds':time.perf_counter()-start,'scope':'Source/JSON-args/NPZ-headers/coordinate/done metadata only; no target numerical grid values decoded. Source Git reads and final manifest hashing excluded from IO counter.','original_archives_reopened':11,'task16_archive_reopened':False});print(json.dumps({'manifest':man,'inputs':inp,'authority':au,'source_map':sm,'coordinates':joinpin,'runs':[(r['id'],list(r['files']))for r in runs],'source_count':len(source),'bytes_read':count,'seconds':time.perf_counter()-start},indent=2))
