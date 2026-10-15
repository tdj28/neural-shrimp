from pathlib import Path
import json,hashlib,subprocess,tarfile,io,re,datetime,time
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');D=Path('/tmp/task21i-source-001');D.mkdir(exist_ok=False);index=json.loads(Path('/tmp/task21i-attempt-index-001.json').read_text());H=lambda b:hashlib.sha256(b).hexdigest();source=[];archives=[];nbytes=0;start=time.perf_counter()
def put(n,b):p=D/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);return dict(path=str(p),sha256=H(b),bytes=len(b))
def git(c,p):return subprocess.check_output(['git','-C',str(R),'show',c+':'+p],timeout=20)
for c,paths in [('9113b64e73f6b4f9981ef45ac31fe5b6f2f02280',['tools/behavior_zones.py','docs/bz-pairs.json','docs/bz-tasks.json']),('6a7c7ac1c1cbef76a93bc798d712bb2d5d965970',['tools/behavior_prompts.py','docs/bp-tasks.json'])]:
 for path in paths:source.append({'source_commit':c,'repository_path':path,**put('source/'+c[:8]+'/'+path,git(c,path))})
 b=git(c,'tools/runpod_soft_runner.sh');lines=b.splitlines(True);ranges=[];i=next(i for i,l in enumerate(lines)if l.startswith(b'run()'));j=next(i2 for i2 in range(i+1,len(lines))if lines[i2].strip()==b'}')+1;ranges.append((i,j))
 for key in ['bz','bp','bplabel','bpgen']:
  for i,l in enumerate(lines):
   if re.match(('  '+key+r'\)').encode(),l):
    j=next((j for j in range(i+1,len(lines))if re.match(rb'  [A-Za-z0-9_]+\)',lines[j])),len(lines));ranges.append((i,j))
 for i,j in ranges:source.append({'source_commit':c,'repository_path':'tools/runpod_soft_runner.sh','full_source_sha256':H(b),'start_line':i+1,'end_line':j,**put(f'source/{c[:8]}/runner.lines{i+1}-{j}.sh',b''.join(lines[i:j]))})
for r in index['attempts']:
 if not r.get('archive'):continue
 p=Path(r['archive']['path']);b=p.read_bytes();nbytes+=len(b);ret=r['retrieval'];assert H(b)==ret['local_sha256']==ret['remote_sha256'];members=[]
 with tarfile.open(fileobj=io.BytesIO(b))as tf:
  for m in tf.getmembers():
   if m.isfile():members.append(dict(member=m.name,bytes=m.size))
 archives.append({'id':r['id'],'source_commit':r['initial']['commit'],'command':r['initial']['jobset_args'],'archive':{'path':str(p),'bytes':len(b),'sha256':H(b)},'members':members,'recorded_retrieval':r['retrieval_record'],'exact_tool_status_lines':r.get('exact_status_lines'),'recorded_termination_available':r['recorded_termination']is not None})
meta={'schema':'task21i-source-and-archive-index-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files':source,'original_attempt_index':put('original-attempt-index.json',Path('/tmp/task21i-attempt-index-001.json').read_bytes()),'selected_archives':archives,'archive_bytes_hashed':nbytes,'seconds':time.perf_counter()-start,'scope':'Historical source/card/task metadata and selected13 complete archive hashes/member names/sizes only. No selected target member payload or outcome decoded. Initial and unretrieved attempts retained in exact index; no inferred remote closure.'}
p=put('source-and-archive-index.json',(json.dumps(meta,indent=2)+'\n').encode());print(json.dumps({'index':p,'sources':source,'archives':[{'id':a['id'],'members':a['members']}for a in archives],'archive_bytes_hashed':nbytes},indent=2))
