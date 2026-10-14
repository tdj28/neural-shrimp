import os,resource
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));os.nice(max(0,10-os.nice(0)))
from pathlib import Path
import json,hashlib,time,datetime,re,ast,subprocess,tarfile
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');D=Path('/data2/neural-shrimp-data');O=Path('/tmp/task16-original-authority-001');O.mkdir(exist_ok=False)
T=time.monotonic();io=[]
def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p):
 st=p.stat();t=time.monotonic();h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024**2):h.update(b)
 en=p.stat();assert(st.st_size,st.st_mtime_ns,st.st_ino)==(en.st_size,en.st_mtime_ns,en.st_ino)
 io.append({'path':str(p),'bytes':st.st_size,'seconds':time.monotonic()-t})
 return {'path':str(p),'sha256':h.hexdigest(),'bytes':st.st_size}
def put(n,v):
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n' if not isinstance(v,str) else v);p.chmod(0o444);return pin(p)
H=Path('/tmp/task16-input-metadata-001.json');h=json.loads(H.read_bytes());assert sha(H.read_bytes())=='fdd511be93d478850fda51706fdbaa019e9b6292eb9b9accf1bb2eb619d34f19'
runs=['20261014T000119Z','20261013T235819Z','20261013T235834Z','20261014T002733Z','20261014T000034Z','20261014T005040Z','20261014T000049Z','20261014T005055Z','20261014T000104Z','20261014T005110Z','20261013T195015Z','20261013T195015Z','20261013T201917Z','20261014T001126Z','20261014T001141Z','20261013T235934Z','20261014T001156Z','20261014T001211Z','20261014T001226Z']
assert len(runs)==len(h['segments'])==19
byrun={}
for seg,run in zip(h['segments'],runs):byrun.setdefault(run,[]).append(seg)
ledger=D/'runpod/pods.jsonl';lrows=[(i+1,json.loads(b),sha(b)) for i,b in enumerate(ledger.read_bytes().splitlines())]
allow=['event','time','pod_id','commit','gpu_type','archive_bytes','data_bytes','hash_verified','sha256','seconds_after_create']
records=[];srefs={};casekeys={}
for run,segs in byrun.items():
 base=D/'runpod/soft64'/run;p=base/'result.json';v=json.loads(p.read_bytes());out={k:v.get(k) for k in ['pod_id','gpu_type','commit','jobset','started','runner_returncode','hash_verified','local_sha256','remote_sha256','wall_seconds']};out['upload']={k:v.get('upload',{}).get(k) for k in ['code_bytes','data_bytes','code_sha256','data_sha256','extra_data','jobset_args']}
 stdout=base/'runner-stdout.txt';out['run_id']=run;out['original_receipt']=pin(p);out['stdout']=pin(stdout)
 out['status_lines']=[line for line in stdout.read_text().splitlines() if re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ (?:start [A-Za-z0-9_.-]+|end [A-Za-z0-9_.-]+ status -?\d+)',line)]
 out['ledger_records']=[{'line_number':n,'original_line_sha256':s,'fields':{k:q[k] for k in allow if k in q}} for n,q,s in lrows if q.get('pod_id')==v['pod_id']]
 out['recorded_termination_verified']=any(x['fields']['event']=='terminated_verified' for x in out['ledger_records'])
 archive=base/'results.tgz';out['archive']=pin(archive);assert out['archive']['sha256']==v['local_sha256']==v['remote_sha256'];assert v['hash_verified'] is True
 wanted={str(Path(s['path']).relative_to(D)):s for s in segs}
 for s in segs:
  if s['scan_info']:wanted[str(Path(s['scan_info']['path']).relative_to(D))]=s['scan_info']
 matches=[]
 with tarfile.open(archive,'r:gz') as tf:
  for member in tf:
   name=member.name.removeprefix('./')
   if name not in wanted:continue
   assert member.isfile();m=hashlib.sha256();n=0
   f=tf.extractfile(member)
   while b:=f.read(1024**2):m.update(b);n+=len(b)
   exp=wanted[name];assert n==exp['bytes'] and m.hexdigest()==exp['sha256'],name
   matches.append({'archive_member':member.name,'bytes':n,'sha256':m.hexdigest(),'current_path':exp['path'],'segment_id':exp.get('segment_id'),'matches_header_inventory':True})
 assert len(matches)==len(wanted) and len({x['archive_member'] for x in matches})==len(matches),(run,len(matches),len(wanted))
 out['selected_member_bindings']=matches
 out['segment_ids']=[s['segment_id'] for s in segs];out['safe_receipt']=put('receipts/'+run+'.json',out)
 records.append(out);casekeys.setdefault(v['commit'],set()).add(v['jobset'])
for commit,keys in casekeys.items():
 files={}
 for rel in ['tools/explore_llm_soft_loop.py','tools/runpod_soft_runner.sh']:
  b=subprocess.check_output(['git','-C',str(R),'show',commit+':'+rel]);txt=b.decode();lines=txt.splitlines(keepends=True)
  if rel.endswith('.py'):
   node=next(x for x in ast.parse(txt).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name=='scan');ranges=[(node.lineno,node.end_lineno)]
  else:
   start=next(i for i,l in enumerate(lines,1) if l.startswith('run()'))
   end=next(i for i in range(start,len(lines)+1) if lines[i-1].strip()=='}');ranges=[(start,end)]
   for key in sorted(keys):
    st=next(i for i,l in enumerate(lines,1) if re.match(r'  '+re.escape(key)+r'\)',l));en=next(i for i in range(st,len(lines)+1) if lines[i-1].strip()==';;');ranges.append((st,en))
  excerpt=''.join('## Original lines %d–%d\n'%(a,z)+''.join(lines[a-1:z])+'\n' for a,z in ranges)
  ex=put('source/'+commit+'-'+Path(rel).name+'.txt',excerpt)
  files[rel]={'original_git_blob_sha256':sha(b),'original_bytes':len(b),'ranges':ranges,'excerpt':ex}
 srefs[commit]=files
joinrecords=[]
for k in [1,2,3]:
 q=[x for x in records if any(h['segments'][int(s.split('-')[1])]['dataset_id']==f'q05-random{k}' for s in x['segment_ids'])]
 joinrecords.append({'dataset_id':f'q05-random{k}','runs':[{'run_id':x['run_id'],'source_commit':x['commit'],'jobset':x['jobset'],'exact_jobset_args':x['upload']['jobset_args'],'original_receipt':x['original_receipt'],'safe_receipt':x['safe_receipt'],'status_lines':x['status_lines'],'recorded_termination_verified':x['recorded_termination_verified'],'archive_member_bindings':x['selected_member_bindings']} for x in q],'header_join':next(x for x in h['joins'] if x['dataset_id']==f'q05-random{k}'),'prospective_selection':{'lower_rows':[0,96],'top_rows_global':[96,128],'extra_lower_rows_nonselected':[96,112] if k>1 else [],'coordinate_mapping':'Fixed ordinal row correspondence from explicit original grid endpoints/resolutions; preserve actual coordinate floats and their nominal-grid differences. No interpolation or tolerance search.'}})
joinpin=put('random-partition-command-authority.json',{'schema':'task16-random-partition-command-authority-v1','header_inventory':pin(H),'prospective_root_review':pin(Path('/tmp/task16-root-prospective-join-review.md')),'source_commit_excerpts':{s:srefs[s] for s in ['9f23285af412f3a66627facd1128625e3190593c','881b36b1820066c44fab76d191adcc22827a1bae']},'joins':joinrecords,'scientific_values_decoded':False})
meta={'schema':'task16-original-authority-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'header_inventory':pin(H),'records':records,'historical_source_excerpts':srefs,'random_partition_authority':joinpin,'read_scope':'Original receipt safe fields, original stdout start/end status lines, scoped local ledger metadata, source and streamed original archive/member hashes only. No scientific values decoded and no pod or producer actions.','trust_limits':['Archive upload code/data hashes and source commits are original producer records; no reconstruction of uploaded archive or historical model/tokenizer/vector bytes was executed.','Original wrapper exit0 is distinct from each timeout-wrapped scientific job status; all original start/end statuses retained.','Local ledger terminated_verified entries are original recorded termination evidence, not a new pod query.','Current input files stay at original paths; before/after source-byte authentication must refuse mutation. No original files chmodded, overwritten or moved.'],'scientific_values_decoded':False,'hash_io':{'bytes':sum(x['bytes'] for x in io),'seconds':sum(x['seconds'] for x in io)},'elapsed_seconds':time.monotonic()-T}
mp=put('authority.json',meta)
manifest=put('manifest.json',{'schema':'task16-original-authority-export-v1','files':[pin(p) for p in sorted(O.rglob('*')) if p.is_file()],'capture_script':pin(Path(__file__))})
print(json.dumps({'authority':mp,'manifest':manifest,'join_authority':joinpin,'records':len(records),'segments':19,'all_recorded_terminated':all(x['recorded_termination_verified'] for x in records),'statuses':[{'run':x['run_id'],'statuses':x['status_lines']} for x in records],'hash_io':meta['hash_io'],'elapsed_seconds':meta['elapsed_seconds']},indent=2))
